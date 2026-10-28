"""Public fitted regression estimator."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from sklearn.utils import InputTags, RegressorTags, Tags, TargetTags

from ._execution import _InferenceExecutor
from ._inference_preprocessing import _FeatureEnsemble
from .checkpoint import load_regressor_checkpoint
from .config import InferenceManifest
from .ensemble import (
    EnsemblePlan,
    _svd_applies,
    build_ensemble_inputs,
    fit_member_weights,
)
from .model.heads import RegressionModel
from .preprocessing import TabularPreprocessor, _resolve_categorical_mask
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
        return self._readout_quantiles(output)

    def _readout_quantiles(self, output: torch.Tensor) -> torch.Tensor:
        assert self.manifest.regression is not None
        if self.manifest.regression.point_estimate == "mean":
            return output.float().mean(dim=-1)
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

    def _member_quantiles(
        self, support_x, support_y, query_x, *, n_svd, seed,
        preprocessor=None, categorical_mask=None,
    ):
        if preprocessor is None:
            preprocessor = _FeatureEnsemble(
                self.manifest.runtime.ensemble_count, seed,
                self.manifest.runtime.feature_limit, device=self.device,
                gaussianize=self.manifest.preprocessing.use_quantile_map,
                rescale=self.manifest.preprocessing.rescale_for_column_count,
                task="regression",
            ).fit(support_x.detach().cpu().numpy(), categorical_mask)
        plan = EnsemblePlan(
            members=self.manifest.runtime.ensemble_count,
            seed=seed, task="regression", n_svd=n_svd,
        )
        batch_xs, batch_y, batch_xq, fitted = build_ensemble_inputs(
            support_x, support_y, query_x, plan,
            preprocessor=preprocessor, categorical_mask=categorical_mask,
        )
        self.model.eval()
        with torch.inference_mode():
            output = self._forward_chunked(batch_xs, batch_y, batch_xq)
        expected = (plan.members, query_x.shape[0], self.manifest.output_width)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected or not bool(torch.isfinite(output).all()):
            raise RuntimeError("model returned invalid regression quantiles")
        return output.float() * fitted.target_signs[:, None, None]

    def _member_points(
        self, support_x, support_y, query_x, *, n_svd, seed,
        preprocessor=None, categorical_mask=None,
    ):
        output = self._member_quantiles(
            support_x, support_y, query_x, n_svd=n_svd, seed=seed,
            preprocessor=preprocessor, categorical_mask=categorical_mask,
        )
        return self._collapse_members(output.sort(dim=-1).values, query_x.shape[0])

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
        *, preprocessors=None, categorical_mask=None,
    ) -> torch.Tensor:
        """Concatenate every pass's members along the member axis."""
        return torch.cat(
            [
                self._member_points(
                    support_x, support_y, query_x, n_svd=n_svd, seed=seed,
                    preprocessor=None if preprocessors is None else preprocessors[seed],
                    categorical_mask=categorical_mask,
                )
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
        categorical_mask=None,
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
        preprocessors = {
            seed: _FeatureEnsemble(
                self.manifest.runtime.ensemble_count, seed,
                self.manifest.runtime.feature_limit, device=self.device,
                gaussianize=self.manifest.preprocessing.use_quantile_map,
                rescale=self.manifest.preprocessing.rescale_for_column_count,
                task="regression",
            ).fit(support_x[fitting], categorical_mask)
            for _, seed in passes
        }
        points = self._pooled_member_points(
            torch.as_tensor(support_x[fitting], dtype=torch.float64),
            torch.as_tensor(support_y[fitting], dtype=torch.float32),
            support_x[validation],
            passes, preprocessors=preprocessors,
        )
        # NNLS fits on the original target scale, matching the de-normalized
        # predictions that prediction ultimately combines.
        return fit_member_weights(
            (points.float() * scale + center).cpu().numpy(),
            targets[validation],
            shrinkage=regression.nnls_shrinkage,
        )

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        categorical_mask: np.ndarray | None = None,
    ) -> "EXAONETabularRegressor":
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
        resolved_categorical_mask = _resolve_categorical_mask(features, categorical_mask)
        row_limit = self.manifest.runtime.support_row_limit
        if y.size > row_limit:
            chosen = np.random.default_rng(self.manifest.runtime.seed).choice(y.size, row_limit, replace=False)
        else:
            chosen = np.arange(y.size)
        selected_y = torch.as_tensor(y[chosen], dtype=torch.float32)
        center_tensor = selected_y.mean()
        variance = (selected_y - center_tensor).square().mean()
        eps = torch.finfo(selected_y.dtype).eps
        constant = variance <= selected_y.numel() * eps * variance + (selected_y.numel() * center_tensor * eps) ** 2
        scale_tensor = torch.where(constant, torch.ones_like(variance), variance.sqrt())
        center, scale = float(center_tensor), float(scale_tensor)
        support_y = ((selected_y - center_tensor) / scale_tensor).numpy().copy()
        support_x = np.array(features[chosen], copy=True)
        assert self.manifest.regression is not None
        # Settle the augmentation once, against the whole support. Every later
        # ensemble build reuses the width rather than re-deriving the exemption
        # from the rows it happens to be handed.
        n_svd = self.manifest.regression.n_svd
        if n_svd and self.manifest.regression.svd_gate:
            if not _svd_applies(
                support_x.shape[0], support_x.shape[1], bool(resolved_categorical_mask.any())
            ):
                n_svd = 0
        passes = self._ensemble_passes(n_svd)
        preprocessors = {
            seed: _FeatureEnsemble(
                self.manifest.runtime.ensemble_count, seed,
                self.manifest.runtime.feature_limit, device=self.device,
                gaussianize=self.manifest.preprocessing.use_quantile_map,
                rescale=self.manifest.preprocessing.rescale_for_column_count,
                task="regression",
            ).fit(support_x, resolved_categorical_mask)
            for _, seed in passes
        }
        # Fit the weights here rather than at prediction: the split needs only the
        # support, so paying the extra forward once keeps predict single-pass.
        weights = None
        if self.manifest.regression.member_weighting == "nnls":
            weights = self._nnls_member_weights(
                support_x, support_y, y[chosen], center=center, scale=scale, passes=passes,
                categorical_mask=resolved_categorical_mask,
            )
        self._fitted_state = {
            "preprocessor": preprocessors[passes[0][1]],
            "preprocessors": preprocessors,
            "n_features": features.shape[1],
            "categorical_mask": resolved_categorical_mask,
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
        query = TabularPreprocessor._validate_features(features, allow_zero_rows=True)
        if query.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        support_x = torch.as_tensor(state["support_x"], dtype=torch.float64)
        support_y = torch.as_tensor(state["support_y"], dtype=torch.float32)
        query_x = query
        if self.manifest.regression.member_weighting == "trimmed":
            quantiles = torch.cat([
                self._member_quantiles(
                    support_x, support_y, query_x, n_svd=n_svd, seed=seed,
                    preprocessor=state["preprocessors"][seed],
                )
                for n_svd, seed in state["passes"]
            ], dim=0)
            quantiles = (quantiles * state["scale"] + state["center"]).sort(dim=-1).values
            cut = int(0.2 * quantiles.shape[0])
            if cut:
                quantiles = quantiles.sort(dim=0).values[cut:-cut]
            combined = self._readout_quantiles(quantiles.mean(dim=0))
            return np.array(combined.cpu().numpy(), dtype=np.float64, order="C", copy=True)
        member_predictions = self._pooled_member_points(
            support_x, support_y, query_x, state["passes"],
            preprocessors=state["preprocessors"],
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
