"""Public fitted regression estimator."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from sklearn.feature_selection import f_regression
from sklearn.utils import InputTags, RegressorTags, Tags, TargetTags

from ._execution import _InferenceExecutor
from .checkpoint import load_regressor_checkpoint
from .config import InferenceManifest
from .ensemble import (
    EnsemblePlan,
    _classify_columns,
    _svd_applies,
    build_ensemble_inputs,
    fit_member_weights,
)
from .model.heads import RegressionModel
from .preprocessing import TabularPreprocessor
from .presets import released_checkpoint
from .weights import resolve_weights


_logger = logging.getLogger(__name__)

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# Fixed seed, and legacy RandomState specifically: the split has to reproduce the
# reference recipe's permutation, which default_rng would not.
_VALIDATION_SPLIT_SEED = 0

# Seed for the augmented half of a split run, so its members differ from the
# un-augmented half by their permutation as well as by the augmentation. The
# value is the reference recipe's, carried over verbatim to stay comparable.
_SVD_SPLIT_SEED = 1234567


class EXAONETabularRegressor:
    def __init__(self, manifest: InferenceManifest, *, device: torch.device | str = "cpu", model: nn.Module | None = None, max_vram_bytes: int | None = None) -> None:
        if not isinstance(manifest, InferenceManifest):
            raise TypeError("manifest must be an InferenceManifest")
        if manifest.task != "regression":
            raise ValueError("manifest task must be regression")
        if model is not None and not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module or None")
        if max_vram_bytes is not None and (isinstance(max_vram_bytes, bool) or not isinstance(max_vram_bytes, int)):
            raise TypeError("max_vram_bytes must be an integer or None")
        if max_vram_bytes is not None and max_vram_bytes <= 0:
            raise ValueError("max_vram_bytes must be positive")
        if max_vram_bytes is not None and torch.device(device).type != "cuda":
            raise ValueError("max_vram_bytes requires a CUDA device")
        self.manifest, self.device, self.max_vram_bytes = manifest, torch.device(device), max_vram_bytes
        assert manifest.regression is not None
        if model is None:
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.manual_seed(manifest.runtime.seed)
                model = RegressionModel(
                    manifest.model,
                    manifest.regression,
                    device=self.device,
                    dtype=_DTYPES[manifest.runtime.compute_dtype],
                )
        self.model = model.to(self.device)
        self._fitted_state: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        weights: str | None = None,
        *,
        device: torch.device | str = "cpu",
        revision: str | None = None,
        cache_dir: str | None = None,
        filename: str | None = None,
        manifest: InferenceManifest | None = None,
        ensemble_count: int | None = None,
        compute_dtype: str | None = None,
        seed: int | None = None,
        max_vram_bytes: int | None = None,
    ) -> "EXAONETabularRegressor":
        """Build a regressor from the released checkpoint (or your own weights).

        See :meth:`EXAONETabularClassifier.from_pretrained`; the only difference is
        that this loads the released regression checkpoint and its quantile head.
        The readout and the SVD feature augmentation are pinned by the manifest's
        :class:`RegressionConfig`; pass ``manifest=`` to change either.
        """
        checkpoint = released_checkpoint("regression")
        base_manifest = checkpoint.manifest if manifest is None else manifest
        if not isinstance(base_manifest, InferenceManifest):
            raise TypeError("manifest must be an InferenceManifest or None")
        if base_manifest.task != "regression":
            raise ValueError("manifest task must be regression")
        resolved_manifest = base_manifest.with_overrides(
            ensemble_count=ensemble_count, compute_dtype=compute_dtype, seed=seed
        )
        source = resolve_weights(
            checkpoint, weights, filename=filename, revision=revision, cache_dir=cache_dir
        )
        estimator = cls(resolved_manifest, device=device, max_vram_bytes=max_vram_bytes)
        if not source.is_default:
            _logger.warning(
                "using user-supplied weights at %s; released-checkpoint integrity "
                "pin not enforced (SHA-256 not checked)",
                source.path,
            )
        load_regressor_checkpoint(
            source.path,
            estimator.model,
            resolved_manifest,
            verify_checksum=source.is_default,
        )
        return estimator

    @property
    def is_fitted(self) -> bool:
        return self._fitted_state is not None

    def __sklearn_is_fitted__(self) -> bool:
        """Answer scikit-learn's fitted check; see the classifier's counterpart."""
        return self.is_fitted

    def __sklearn_tags__(self) -> Tags:
        """Describe this estimator to scikit-learn; see the classifier's counterpart."""
        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True),
            regressor_tags=RegressorTags(),
            input_tags=InputTags(allow_nan=True),
        )

    @property
    def n_features_in_(self) -> int:
        return int(self._state()["n_features"])

    def _state(self) -> dict[str, Any]:
        if self._fitted_state is None:
            raise RuntimeError("regressor is not fitted")
        return self._fitted_state

    def _executor(self) -> _InferenceExecutor:
        return _InferenceExecutor(
            self.manifest,
            device=self.device,
            max_vram_bytes=self.max_vram_bytes,
        )

    def _forward_chunked(
        self,
        support_batch: torch.Tensor,
        label_batch: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        return self._executor().forward(
            self.model, support_batch, label_batch, query_batch
        )

    def _collapse_members(
        self, output: torch.Tensor, query_count: int
    ) -> torch.Tensor:
        """Reduce each member's quantile bank to one point prediction per row."""
        expected = (
            self.manifest.runtime.ensemble_count,
            query_count,
            self.manifest.output_width,
        )
        if (
            not isinstance(output, torch.Tensor)
            or tuple(output.shape) != expected
            or not bool(torch.isfinite(output).all())
        ):
            raise RuntimeError("model returned invalid regression quantiles")
        assert self.manifest.regression is not None
        if self.manifest.regression.point_estimate == "trimmed":
            # Central 99.8% trapezoidal mean over the quantile function:
            #   [Q(.001)/2 + sum_{i=2..N-1} Q(.001i) + Q(.999)/2] / (N-1)
            # RMSE targets the conditional mean, which the median misses on
            # skewed targets; halving the endpoints keeps the extreme quantiles
            # from dominating. Sorting guards against tau-crossing.
            quantiles = torch.sort(output.float(), dim=-1).values
            count = quantiles.shape[-1]
            return (
                0.5 * quantiles[..., 0]
                + quantiles[..., 1:-1].sum(dim=-1)
                + 0.5 * quantiles[..., -1]
            ) / (count - 1)
        median_index = self.manifest.regression.quantile_count // 2
        # Slice before widening so the unused 998 quantiles are not copied to
        # float32 after the memory-planned model execution has completed.
        return output[..., median_index].float()

    def _member_points(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        *,
        n_svd: int,
        seed: int,
    ) -> torch.Tensor:
        """Run one ensemble forward, returning (members, rows) point predictions.

        The plan is rebuilt identically for weight fitting and for prediction:
        member m must denote the same preprocessing rule in both, or a fitted
        weight lands on a member it was never fitted against. Both ``n_svd`` and
        ``seed`` are settled by :meth:`fit` and carried in the fitted state, so a
        pass over a subset of the support cannot land in a different geometry.
        """
        plan = EnsemblePlan(
            members=self.manifest.runtime.ensemble_count,
            seed=seed,
            task="regression",
            n_svd=n_svd,
        )
        batch_xs, batch_y, batch_xq, _fitted_plan = build_ensemble_inputs(
            support_x, support_y, query_x, plan
        )
        self.model.eval()
        with torch.inference_mode():
            output = self._forward_chunked(batch_xs, batch_y, batch_xq)
        return self._collapse_members(output, query_x.shape[0])

    def _ensemble_passes(self, n_svd: int) -> tuple[tuple[int, int], ...]:
        """The ``(n_svd, seed)`` pairs whose members pool into one prediction.

        A split run pairs an un-augmented pass with an augmented one and leaves it
        to the weight fit to say how much the augmentation is worth, rather than
        the gate deciding all-or-nothing. The augmented half also gets its own
        seed, so the two halves differ by permutation as well as by augmentation
        and the fit arbitrates between genuinely distinct views. The split needs
        an augmentation to contrast against, so an exempted or disabled width
        collapses back to a single pass on the run's own seed.
        """
        assert self.manifest.regression is not None
        base_seed = self.manifest.runtime.seed
        if self.manifest.regression.svd_split and n_svd:
            return ((0, base_seed), (n_svd, _SVD_SPLIT_SEED))
        return ((n_svd, base_seed),)

    def _pooled_member_points(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        passes: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        """Concatenate every pass's members along the member axis."""
        return torch.cat(
            [
                self._member_points(support_x, support_y, query_x, n_svd=n_svd, seed=seed)
                for n_svd, seed in passes
            ],
            dim=0,
        )

    def _nnls_member_weights(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        targets: np.ndarray,
        *,
        center: float,
        scale: float,
        passes: tuple[tuple[int, int], ...],
    ) -> np.ndarray | None:
        """Weight members by their accuracy on a held-out slice of the support.

        The split is taken over the in-context support rows: the held-out rows
        are withheld from the conditioning set and predicted from the rest, so
        the fit measures each member on rows it did not see.

        Returns ``None`` when the split is too small to trust, which leaves
        prediction on the uniform mean.
        """
        assert self.manifest.regression is not None
        regression = self.manifest.regression
        rows = support_x.shape[0]
        held_out = int(regression.nnls_validation_fraction * rows)
        if held_out < regression.nnls_min_validation_rows:
            _logger.warning(
                "support of %d rows yields %d validation rows, fewer than "
                "nnls_min_validation_rows=%d; aggregating ensemble members with "
                "uniform weights instead",
                rows,
                held_out,
                regression.nnls_min_validation_rows,
            )
            return None
        order = np.random.RandomState(_VALIDATION_SPLIT_SEED).permutation(rows)
        validation, fitting = order[:held_out], order[held_out:]
        points = self._pooled_member_points(
            torch.as_tensor(support_x[fitting], dtype=torch.float32, device=self.device),
            torch.as_tensor(support_y[fitting], dtype=torch.float32, device=self.device),
            torch.as_tensor(support_x[validation], dtype=torch.float32, device=self.device),
            passes,
        )
        # NNLS fits on the original target scale, matching the de-normalized
        # predictions that prediction ultimately combines.
        return fit_member_weights(
            (points.float() * scale + center).cpu().numpy(),
            targets[validation],
            shrinkage=regression.nnls_shrinkage,
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "EXAONETabularRegressor":
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a NumPy array")
        if not isinstance(targets, np.ndarray):
            raise TypeError("targets must be a NumPy array")
        if features.ndim != 2 or targets.ndim != 1:
            raise ValueError("features must be rank two and targets rank one")
        if features.shape[0] == 0 or features.shape[0] != targets.shape[0]:
            raise ValueError("training rows must be nonempty and aligned")
        if not np.issubdtype(targets.dtype, np.number) or np.issubdtype(targets.dtype, np.complexfloating):
            raise TypeError("targets must have a real numeric dtype")
        y = np.asarray(targets, dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("targets must be finite")
        limit = self.manifest.runtime.feature_limit
        columns = np.arange(features.shape[1], dtype=np.int64)
        if columns.size > limit:
            numeric = np.asarray(features, dtype=np.float64).copy()
            numeric[np.isnan(numeric)] = np.take(np.nanmean(numeric, axis=0), np.where(np.isnan(numeric))[1])
            numeric = np.nan_to_num(numeric)
            scores, _ = f_regression(numeric, y, force_finite=True)
            columns = np.sort(np.argsort(np.nan_to_num(scores), kind="stable")[-limit:]).astype(np.int64)
        preprocessor = TabularPreprocessor(limit, gaussianize=self.manifest.preprocessing.use_quantile_map, feature_count_rescale=self.manifest.preprocessing.rescale_for_column_count, fixed_columns=columns)
        prepared = preprocessor.fit_transform(features, np.arange(features.shape[0]))
        row_limit = self.manifest.runtime.support_row_limit
        if y.size > row_limit:
            chosen = np.random.default_rng(self.manifest.runtime.seed).choice(y.size, row_limit, replace=False)
        else:
            chosen = np.arange(y.size)
        selected_y = torch.as_tensor(y[chosen], dtype=torch.float32)
        center_tensor = selected_y.mean()
        scale_tensor = (
            selected_y.std(correction=1)
            if selected_y.numel() > 1
            else selected_y.new_zeros(())
        )
        center = float(center_tensor)
        scale = float(scale_tensor)
        transformed = (selected_y - center_tensor) / (scale_tensor + 1e-8)
        support_x = np.array(prepared.values[chosen], copy=True)
        support_y = transformed.numpy().copy()
        assert self.manifest.regression is not None
        # Settle the augmentation once, against the whole support. Every later
        # ensemble build reuses the width rather than re-deriving the exemption
        # from the rows it happens to be handed.
        n_svd = self.manifest.regression.n_svd
        if n_svd and self.manifest.regression.svd_gate:
            categorical, _numeric = _classify_columns(torch.from_numpy(support_x))
            if not _svd_applies(
                support_x.shape[0], support_x.shape[1], bool(categorical)
            ):
                n_svd = 0
        passes = self._ensemble_passes(n_svd)
        # Fit the weights here rather than at prediction: the split needs only the
        # support, so paying the extra forward once keeps predict single-pass.
        weights = None
        if self.manifest.regression.member_weighting == "nnls":
            weights = self._nnls_member_weights(
                support_x, support_y, y[chosen], center=center, scale=scale, passes=passes
            )
        self._fitted_state = {
            "preprocessor": preprocessor,
            "n_features": features.shape[1],
            "support_x": support_x,
            "support_y": support_y,
            "center": center,
            "scale": scale,
            "n_svd": n_svd,
            "passes": passes,
            "member_weights": weights,
        }
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        state = self._state()
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a NumPy array")
        if features.ndim != 2 or features.shape[1] != state["n_features"]:
            raise ValueError("query features have an invalid shape")
        query = state["preprocessor"].transform(features).values
        if query.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        member_predictions = self._pooled_member_points(
            torch.as_tensor(state["support_x"], dtype=torch.float32, device=self.device),
            torch.as_tensor(state["support_y"], dtype=torch.float32, device=self.device),
            torch.as_tensor(query, dtype=torch.float32, device=self.device),
            state["passes"],
        )
        member_predictions = (
            member_predictions * state["scale"] + state["center"]
        )
        weights = state["member_weights"]
        if weights is None:
            combined = member_predictions.mean(dim=0)
        else:
            blend = torch.as_tensor(
                weights,
                dtype=member_predictions.dtype,
                device=member_predictions.device,
            )
            combined = (blend.unsqueeze(1) * member_predictions).sum(dim=0)
        return np.array(
            combined.cpu().numpy(), dtype=np.float64, order="C", copy=True
        )


__all__ = ["EXAONETabularRegressor"]
