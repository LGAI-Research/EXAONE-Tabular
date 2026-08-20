"""Immutable configuration values for tabular inference."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, ClassVar, TypeVar

__all__ = [
    "ModelConfig",
    "PreprocessingConfig",
    "RuntimeConfig",
    "ClassificationConfig",
    "RegressionConfig",
    "AttentionScorer",
    "FeatureSelectionConfig",
    "FEATURE_SELECTION",
    "InferenceManifest",
]


_ValueT = TypeVar("_ValueT", bound="_StrictMappingValue")


class _StrictMappingValue:
    @classmethod
    def from_dict(cls: type[_ValueT], value: Mapping[str, Any]) -> _ValueT:
        if not isinstance(value, Mapping):
            raise TypeError("configuration input must be a mapping")
        allowed = {item.name for item in fields(cls)}
        if set(value) - allowed:
            raise ValueError("configuration contains unknown fields")
        return cls(**dict(value))


def _check_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _check_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _check_positive_int(name: str, value: Any) -> None:
    _check_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_choice(name: str, value: Any, allowed: frozenset[str]) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in allowed:
        raise ValueError(f"unsupported {name}: {value!r}")


def _check_unit_fraction(name: str, value: Any) -> float:
    """Validate a real scalar in [0, 1] and return it coerced to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return fraction


@dataclass(frozen=True, kw_only=True)
class ModelConfig(_StrictMappingValue):
    width: int = 192
    head_count: int = 6
    block_count: int = 12
    feedforward_widths: int | Sequence[int] = 768
    columns_per_group: int = 1
    item_summary_count: int = 3
    feature_summary_count: int = 32
    feature_attention_repeats: int = 2
    class_capacity: int = 20

    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = (
        "width",
        "head_count",
        "block_count",
        "columns_per_group",
        "item_summary_count",
        "feature_summary_count",
        "feature_attention_repeats",
        "class_capacity",
    )

    def __post_init__(self) -> None:
        for name in self._POSITIVE_FIELDS:
            _check_positive_int(name, getattr(self, name))
        if self.class_capacity < 2:
            raise ValueError("class_capacity must be at least two")
        if self.width % self.head_count:
            raise ValueError("width must be divisible by head_count")

        supplied = self.feedforward_widths
        if isinstance(supplied, bool):
            raise TypeError("feedforward_widths must contain integers")
        if isinstance(supplied, int):
            normalized = (supplied,) * self.block_count
        else:
            if isinstance(supplied, (str, bytes, bytearray)) or not isinstance(
                supplied, Sequence
            ):
                raise TypeError("feedforward_widths must be an integer or sequence")
            normalized = tuple(supplied)
            if len(normalized) != self.block_count:
                raise ValueError("feedforward_widths length must equal block_count")

        for width in normalized:
            _check_positive_int("feedforward width", width)
        object.__setattr__(self, "feedforward_widths", normalized)

    @property
    def head_width(self) -> int:
        return self.width // self.head_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "head_count": self.head_count,
            "block_count": self.block_count,
            "feedforward_widths": list(self.feedforward_widths),
            "columns_per_group": self.columns_per_group,
            "item_summary_count": self.item_summary_count,
            "feature_summary_count": self.feature_summary_count,
            "feature_attention_repeats": self.feature_attention_repeats,
            "class_capacity": self.class_capacity,
        }


@dataclass(frozen=True, kw_only=True)
class PreprocessingConfig(_StrictMappingValue):
    use_quantile_map: bool = False
    rescale_for_column_count: bool = False

    def __post_init__(self) -> None:
        _check_bool("use_quantile_map", self.use_quantile_map)
        _check_bool("rescale_for_column_count", self.rescale_for_column_count)

    def to_dict(self) -> dict[str, bool]:
        return {
            "use_quantile_map": self.use_quantile_map,
            "rescale_for_column_count": self.rescale_for_column_count,
        }


@dataclass(frozen=True, kw_only=True)
class RuntimeConfig(_StrictMappingValue):
    support_row_limit: int = 100000
    feature_limit: int = 100
    query_batch_limit: int = 1024
    ensemble_count: int = 8
    # fp16 over bf16: same 2-byte footprint and tensor-core throughput, but 10
    # mantissa bits to bf16's 7. Activations here stay far from fp16's 65504
    # ceiling, so bf16's wider exponent range -- the one thing fp16 gives up --
    # buys nothing; it stays available for inputs that do approach that ceiling.
    compute_dtype: str = "float16"
    seed: int = 0
    # When a chunked GPU-resident support cache cannot fit the budget (its
    # fragmentation-inflated peak overflows), offload the per-layer K/V to host
    # during the build and bulk-reload it before the query phase. Set False to
    # force GPU-resident caching only (falls through to CPU streaming or fails).
    support_cache_offload: bool = True

    _DTYPES: ClassVar[frozenset[str]] = frozenset(
        {"float32", "float16", "bfloat16"}
    )

    def __post_init__(self) -> None:
        for name in (
            "support_row_limit",
            "feature_limit",
            "query_batch_limit",
            "ensemble_count",
        ):
            _check_positive_int(name, getattr(self, name))
        if not isinstance(self.compute_dtype, str):
            raise TypeError("compute_dtype must be a string")
        if self.compute_dtype not in self._DTYPES:
            raise ValueError("unsupported compute_dtype")
        _check_int("seed", self.seed)
        if not 0 <= self.seed <= 2**63 - 1:
            raise ValueError("seed is outside the supported range")
        _check_bool("support_cache_offload", self.support_cache_offload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_row_limit": self.support_row_limit,
            "feature_limit": self.feature_limit,
            "query_batch_limit": self.query_batch_limit,
            "ensemble_count": self.ensemble_count,
            "compute_dtype": self.compute_dtype,
            "seed": self.seed,
            "support_cache_offload": self.support_cache_offload,
        }


@dataclass(frozen=True, kw_only=True)
class ClassificationConfig(_StrictMappingValue):
    """Readout values specific to classification inference."""

    # SVD components appended to every ensemble member's features; 0 disables the
    # augmentation and leaves the model's input at the preprocessed width. Unlike
    # regression this is unconditional -- there is no small-table exemption and no
    # un-augmented arm to fall back on, because probability aggregation averages
    # members rather than weighting them, so it has no way to price a second view.
    n_svd: int = 0

    def __post_init__(self) -> None:
        _check_int("n_svd", self.n_svd)
        if self.n_svd < 0:
            raise ValueError("n_svd must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"n_svd": self.n_svd}


@dataclass(frozen=True, kw_only=True)
class RegressionConfig(_StrictMappingValue):
    """Architecture and readout values specific to quantile regression inference."""

    quantile_count: int = 999
    decoder_hidden_width: int = 384
    point_estimate: str = "trimmed"
    # SVD components appended to every ensemble member's features; 0 disables the
    # augmentation and leaves the model's input at the preprocessed width.
    n_svd: int = 16
    # Withhold the augmentation from small, narrow, all-numeric tables, where a
    # near-full-rank SVD only restates the input. Off by default because the split
    # below already prices the augmentation by weight, making an all-or-nothing
    # exemption redundant; turn it on to run the gated arm instead.
    svd_gate: bool = False
    # Run two ensembles -- one un-augmented, one at n_svd -- and pool their members
    # into a single prediction, so the weight fit arbitrates between the two views.
    # Doubles members and forwards, at predict as well as fit.
    svd_split: bool = True
    # How the members' point predictions are combined. "nnls" fits non-negative
    # weights against a held-out slice of the support set; "uniform" averages.
    member_weighting: str = "nnls"
    # Blend toward the uniform 1/E. NNLS on a small validation split is sparse
    # enough to drop members outright, and this bounds how far it may stray.
    nnls_shrinkage: float = 0.75
    nnls_validation_fraction: float = 0.2
    # Held-out rows the fit must see before its weights are trusted; anything
    # smaller stays on the uniform mean. Fitting a handful of weights against a
    # few dozen rows is the regime where the solve collapses to all-zero.
    nnls_min_validation_rows: int = 2000

    def __post_init__(self) -> None:
        if self.point_estimate not in ("median", "trimmed"):
            raise ValueError("point_estimate must be 'median' or 'trimmed'")
        if self.member_weighting not in ("uniform", "nnls"):
            raise ValueError("member_weighting must be 'uniform' or 'nnls'")
        object.__setattr__(
            self,
            "nnls_shrinkage",
            _check_unit_fraction("nnls_shrinkage", self.nnls_shrinkage),
        )
        fraction = _check_unit_fraction(
            "nnls_validation_fraction", self.nnls_validation_fraction
        )
        if not 0.0 < fraction < 1.0:
            raise ValueError("nnls_validation_fraction must lie in (0, 1)")
        object.__setattr__(self, "nnls_validation_fraction", fraction)
        _check_positive_int("nnls_min_validation_rows", self.nnls_min_validation_rows)
        _check_int("n_svd", self.n_svd)
        if self.n_svd < 0:
            raise ValueError("n_svd must be non-negative")
        _check_bool("svd_gate", self.svd_gate)
        _check_bool("svd_split", self.svd_split)
        _check_positive_int("quantile_count", self.quantile_count)
        if self.quantile_count < 3 or self.quantile_count % 2 == 0:
            raise ValueError("quantile_count must be an odd integer of at least three")
        _check_positive_int("decoder_hidden_width", self.decoder_hidden_width)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantile_count": self.quantile_count,
            "decoder_hidden_width": self.decoder_hidden_width,
            "point_estimate": self.point_estimate,
            "n_svd": self.n_svd,
            "svd_gate": self.svd_gate,
            "svd_split": self.svd_split,
            "member_weighting": self.member_weighting,
            "nnls_shrinkage": self.nnls_shrinkage,
            "nnls_validation_fraction": self.nnls_validation_fraction,
            "nnls_min_validation_rows": self.nnls_min_validation_rows,
        }


@dataclass(frozen=True, kw_only=True)
class AttentionScorer(_StrictMappingValue):
    """One per-feature importance signal read out of the feature-attention blocks.

    Each scorer reduces the captured attention over heads and layers into a score
    per input column.  ``aggregation`` picks which feature-attention query tokens
    are read: ``y_attn`` uses the target-token query, while ``sumi_to_feat`` sums
    the item-summary token queries.

    Two behaviors are unconditional rather than configurable, because the released
    configuration uses the same value for both scorers: head reduction is always
    the mean, and every score is weighted by the value-vector norm, which is what
    makes this ALTI rather than raw attention probability.
    """

    aggregation: str
    layer_reduce: str

    _AGGREGATIONS: ClassVar[frozenset[str]] = frozenset({"y_attn", "sumi_to_feat"})
    _LAYER_REDUCTIONS: ClassVar[frozenset[str]] = frozenset({"last", "mean"})

    def __post_init__(self) -> None:
        _check_choice("aggregation", self.aggregation, self._AGGREGATIONS)
        _check_choice("layer_reduce", self.layer_reduce, self._LAYER_REDUCTIONS)

    def to_dict(self) -> dict[str, str]:
        return {"aggregation": self.aggregation, "layer_reduce": self.layer_reduce}


@dataclass(frozen=True, kw_only=True)
class FeatureSelectionConfig(_StrictMappingValue):
    """Attention feature selection applied to tables wider than the model's limit.

    Selection runs a single bounded forward pass over a subsample of the fitted
    table, scores every column from the captured feature attention, and keeps the
    highest-scoring ``target_feature_count``.  Scores from each scorer are min-max
    normalized and averaged.
    """

    target_feature_count: int = 100
    # Rows fed to the scoring pass.  Bounds its cost independently of
    # RuntimeConfig.support_row_limit, which governs the prediction support set.
    pre_pass_row_limit: int = 512
    scorers: Sequence[AttentionScorer] = ()
    # Deliberately *not* RuntimeConfig.seed: it seeds the pre-pass row subsample,
    # so changing it moves the selected columns.
    seed: int = 42

    def __post_init__(self) -> None:
        _check_positive_int("target_feature_count", self.target_feature_count)
        _check_positive_int("pre_pass_row_limit", self.pre_pass_row_limit)
        _check_int("seed", self.seed)
        if not 0 <= self.seed <= 2**63 - 1:
            raise ValueError("seed is outside the supported range")

        supplied = self.scorers
        if isinstance(supplied, (str, bytes, bytearray)) or not isinstance(
            supplied, Sequence
        ):
            raise TypeError("scorers must be a sequence of AttentionScorer")
        normalized = tuple(supplied)
        if not normalized:
            raise ValueError("at least one scorer is required")
        for scorer in normalized:
            if not isinstance(scorer, AttentionScorer):
                raise TypeError("scorers must contain AttentionScorer values")
        object.__setattr__(self, "scorers", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_feature_count": self.target_feature_count,
            "pre_pass_row_limit": self.pre_pass_row_limit,
            "scorers": [scorer.to_dict() for scorer in self.scorers],
            "seed": self.seed,
        }


# The one selection configuration this package implements. It is deliberately *not*
# part of InferenceManifest or ReleasedCheckpoint: what the scorers name is the
# architecture's token layout — `y_attn` is the target-token query, `sumi_to_feat`
# the item-summary token queries, and `last` the final feature-attention module —
# so it is identical
# for every checkpoint of this architecture rather than a property of any one weights
# file. Selection is unconditional; there is no opt-out.
FEATURE_SELECTION = FeatureSelectionConfig(
    target_feature_count=100,
    pre_pass_row_limit=512,
    scorers=(
        AttentionScorer(aggregation="y_attn", layer_reduce="last"),
        AttentionScorer(aggregation="sumi_to_feat", layer_reduce="mean"),
    ),
    seed=42,
)


@dataclass(frozen=True, kw_only=True)
class InferenceManifest(_StrictMappingValue):
    # None until a released file is final: the pin identifies one exact byte stream,
    # so it can only be filled in once that stream exists. Unpinned loads warn.
    checkpoint_sha256: str | None = None
    schema_version: int = 1
    task: str = "classification"
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # Left None so an existing classification manifest keeps constructing unchanged;
    # __post_init__ fills in the defaults rather than making every caller pass one.
    classification: ClassificationConfig | None = None
    regression: RegressionConfig | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_sha256 is not None:
            if not isinstance(self.checkpoint_sha256, str):
                raise TypeError("checkpoint_sha256 must be a string or None")
            if len(self.checkpoint_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in self.checkpoint_sha256
            ):
                raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        _check_int("schema_version", self.schema_version)
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")
        if not isinstance(self.task, str):
            raise TypeError("task must be a string")
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")
        if not isinstance(self.preprocessing, PreprocessingConfig):
            raise TypeError("preprocessing must be a PreprocessingConfig")
        if not isinstance(self.runtime, RuntimeConfig):
            raise TypeError("runtime must be a RuntimeConfig")
        if self.task == "classification":
            if self.regression is not None:
                raise ValueError("classification manifests cannot declare regression config")
            if self.classification is None:
                object.__setattr__(self, "classification", ClassificationConfig())
            elif not isinstance(self.classification, ClassificationConfig):
                raise TypeError("classification must be a ClassificationConfig")
        else:
            if not isinstance(self.regression, RegressionConfig):
                raise TypeError("regression manifests require a RegressionConfig")
            if self.classification is not None:
                raise ValueError("regression manifests cannot declare classification config")

    @property
    def output_width(self) -> int:
        return (
            self.model.class_capacity
            if self.regression is None
            else self.regression.quantile_count
        )

    @property
    def decoder_hidden_width(self) -> int:
        return (
            self.model.feedforward_widths[-1]
            if self.regression is None
            else self.regression.decoder_hidden_width
        )

    def with_overrides(
        self,
        *,
        ensemble_count: int | None = None,
        compute_dtype: str | None = None,
        seed: int | None = None,
    ) -> "InferenceManifest":
        """Return a copy with selected runtime knobs replaced, others untouched.

        Only the convenience runtime knobs are overridable here; the pinned model
        architecture and checkpoint digest are intentionally immutable. The result
        is re-validated through the normal constructor, so invalid overrides fail
        loudly rather than producing an inconsistent manifest.
        """
        changes: dict[str, Any] = {}
        if ensemble_count is not None:
            changes["ensemble_count"] = ensemble_count
        if compute_dtype is not None:
            changes["compute_dtype"] = compute_dtype
        if seed is not None:
            changes["seed"] = seed
        if not changes:
            return self
        return replace(self, runtime=replace(self.runtime, **changes))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "schema_version": self.schema_version,
            "task": self.task,
            "model": self.model.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "runtime": self.runtime.to_dict(),
        }
        if self.task == "regression":
            assert self.regression is not None
            result["regression"] = self.regression.to_dict()
        else:
            assert self.classification is not None
            result["classification"] = self.classification.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceManifest":
        if not isinstance(value, Mapping):
            raise TypeError("manifest input must be a mapping")
        allowed = {item.name for item in fields(cls)}
        if set(value) - allowed:
            raise ValueError("manifest contains unknown fields")
        copied = dict(value)
        converters = {
            "model": ModelConfig,
            "preprocessing": PreprocessingConfig,
            "runtime": RuntimeConfig,
            "classification": ClassificationConfig,
            "regression": RegressionConfig,
        }
        for name, config_type in converters.items():
            nested = copied.get(name)
            if isinstance(nested, Mapping):
                copied[name] = config_type.from_dict(nested)
        return cls(**copied)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, text: str) -> "InferenceManifest":
        if not isinstance(text, str):
            raise TypeError("manifest JSON must be text")

        def reject_constant(token: str) -> None:
            raise ValueError(f"invalid JSON constant: {token}")

        try:
            value = json.loads(text, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("invalid manifest JSON") from error
        if not isinstance(value, dict):
            raise ValueError("manifest JSON must contain an object")
        return cls.from_dict(value)
