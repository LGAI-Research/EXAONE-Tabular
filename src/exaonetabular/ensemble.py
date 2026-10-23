"""Deterministic test-time ensembles for tabular inference."""

from dataclasses import dataclass, replace
import logging
import math

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from ._inference_preprocessing import _FeatureEnsemble

_logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float32, torch.float64)
_LOGIT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_LABEL_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
)
_FLOOR = 1e-6


@dataclass(frozen=True, eq=False, kw_only=True)
class EnsemblePlan:
    members: int
    seed: int
    task: str = "classification"
    # SVD components appended to every member's features, already resolved by the
    # caller: 0 leaves the feature width untouched, any positive value widens the
    # model's input by that much. The small-table exemption (:func:`_svd_applies`)
    # is the estimator's call, decided once against the fitted support -- deciding
    # it here would re-derive it from whatever subset each build happens to get.
    n_svd: int = 0
    feature_permutations: torch.Tensor | None = None
    class_permutations: torch.Tensor | None = None
    class_count: int | None = None
    target_signs: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.members, bool) or not isinstance(self.members, int):
            raise TypeError("members must be an integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if isinstance(self.n_svd, bool) or not isinstance(self.n_svd, int):
            raise TypeError("n_svd must be an integer")
        if self.members < 1:
            raise ValueError("members must be positive")
        if not 0 <= self.seed <= 2**63 - 1:
            raise ValueError("seed is outside the supported range")
        if self.n_svd < 0:
            raise ValueError("n_svd must be non-negative")
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")


def _percentile(sorted_values: torch.Tensor, fraction: float) -> torch.Tensor:
    position = (sorted_values.numel() - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _support_quartiles(observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = torch.sort(observed).values
    return _percentile(ordered, 0.25), _percentile(ordered, 0.75)


def _yeo_johnson(values: torch.Tensor, lam_index: int) -> torch.Tensor:
    lam = lam_index / 8.0
    nonnegative = values >= 0
    result = torch.empty_like(values)
    positive = values[nonnegative]
    negative = values[~nonnegative]
    if lam_index == 0:
        result[nonnegative] = torch.log1p(positive)
    else:
        result[nonnegative] = (torch.pow(positive + 1, lam) - 1) / lam
    if lam_index == 16:
        result[~nonnegative] = -torch.log1p(-negative)
    else:
        exponent = 2.0 - lam
        result[~nonnegative] = -(torch.pow(1 - negative, exponent) - 1) / exponent
    return result


def _yeo_johnson_candidates(values: torch.Tensor) -> torch.Tensor:
    """Evaluate all 17 fitted lambda candidates without host synchronization."""
    indices = torch.arange(17, dtype=values.dtype, device=values.device).unsqueeze(1)
    lam = indices / 8.0
    expanded = values.unsqueeze(0)
    nonnegative = expanded >= 0

    safe_lam = torch.where(indices == 0, torch.ones_like(lam), lam)
    positive_power = (torch.pow(expanded + 1, lam) - 1) / safe_lam
    positive = torch.where(indices == 0, torch.log1p(expanded), positive_power)

    exponent = 2.0 - lam
    safe_exponent = torch.where(indices == 16, torch.ones_like(exponent), exponent)
    negative_power = -(torch.pow(1 - expanded, exponent) - 1) / safe_exponent
    negative = torch.where(indices == 16, -torch.log1p(-expanded), negative_power)
    return torch.where(nonnegative, positive, negative)


def _yeo_johnson_selected(
    values: torch.Tensor, lam_index: torch.Tensor
) -> torch.Tensor:
    """Evaluate one device-selected lambda without converting it to Python."""
    index = lam_index.to(device=values.device)
    lam = index.to(dtype=values.dtype) / 8.0
    nonnegative = values >= 0
    safe_lam = torch.where(index == 0, torch.ones_like(lam), lam)
    positive_power = (torch.pow(values + 1, lam) - 1) / safe_lam
    positive = torch.where(index == 0, torch.log1p(values), positive_power)
    exponent = 2.0 - lam
    safe_exponent = torch.where(index == 16, torch.ones_like(exponent), exponent)
    negative_power = -(torch.pow(1 - values, exponent) - 1) / safe_exponent
    negative = torch.where(index == 16, -torch.log1p(-values), negative_power)
    return torch.where(nonnegative, positive, negative)


def _numeric_column(
    support: torch.Tensor,
    query: torch.Tensor,
    rule: int,
    generator: torch.Generator,
    *,
    regression: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rule == 0:
        return support.clone(), query.clone()

    support_mask = ~torch.isnan(support)
    query_mask = ~torch.isnan(query)
    observed = support[support_mask]
    if rule == 1:
        ordered = torch.sort(observed).values
        median = ordered[(ordered.numel() - 1) // 2]
        q1, q3 = _support_quartiles(observed)
        scale = torch.clamp(q3 - q1, min=_FLOOR)

        return (
            torch.asinh((support - median) / scale),
            torch.asinh((query - median) / scale),
        )

    if observed.numel() <= 1:
        if rule == 2:
            return support.clone(), query.clone()
        support_result = torch.zeros_like(support)
        query_result = torch.zeros_like(query)
        support_result[~support_mask] = torch.nan
        query_result[~query_mask] = torch.nan
        return support_result, query_result

    if rule == 2:
        candidates = _yeo_johnson_candidates(observed)
        candidate_means = candidates.mean(dim=1)
        candidate_stds = torch.clamp(
            candidates.std(dim=1, correction=0), min=_FLOOR
        )
        candidate_scores = torch.abs(
            ((candidates - candidate_means[:, None]) ** 3).mean(dim=1)
            / candidate_stds**3
        )
        best_index = torch.argmin(candidate_scores)
        fitted = candidates[best_index]
        mean = fitted.mean()
        scale = torch.clamp(fitted.std(unbiased=False), min=_FLOOR)
        support_result = support.clone()
        query_result = query.clone()
        support_result[support_mask] = (fitted - mean) / scale
        query_result[query_mask] = (
            _yeo_johnson_selected(query[query_mask], best_index) - mean
        ) / scale
        return support_result, query_result

    q1, q3 = _support_quartiles(observed)
    noise_scale = 1e-4 * torch.clamp(q3 - q1, min=_FLOOR)
    reference_noise = torch.randn(
        observed.shape,
        dtype=support.dtype,
        device=support.device,
        generator=generator,
    )
    support_noise = torch.randn(
        observed.shape,
        dtype=support.dtype,
        device=support.device,
        generator=generator,
    )
    observed_query = query[query_mask]
    query_noise = torch.randn(
        observed_query.shape,
        dtype=query.dtype,
        device=query.device,
        generator=generator,
    )
    perturbed_support = observed + reference_noise * noise_scale
    reference = torch.sort(perturbed_support).values

    def gaussianize(
        values: torch.Tensor, mask: torch.Tensor, observed_noise: torch.Tensor
    ) -> torch.Tensor:
        output = values.clone()
        perturbed = values[mask] + observed_noise * noise_scale
        ranks = torch.searchsorted(reference, perturbed, right=False)
        probabilities = (ranks.to(values.dtype) + 0.5) / (reference.numel() + 1)
        probabilities = torch.clamp(probabilities, _FLOOR, 1.0 - _FLOOR)
        output[mask] = torch.erfinv(2 * probabilities - 1) * math.sqrt(2.0)
        return output

    return (
        gaussianize(support, support_mask, support_noise),
        gaussianize(query, query_mask, query_noise),
    )


def _categorical_column(
    support: torch.Tensor,
    query: torch.Tensor,
    generator: torch.Generator,
    *,
    forbidden_assignment: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_mask = ~torch.isnan(support)
    query_mask = ~torch.isnan(query)
    categories = torch.unique(support[support_mask], sorted=True)
    if categories.numel() == 0:
        return (
            torch.zeros_like(support),
            torch.zeros_like(query),
            torch.empty(0, dtype=torch.int64, device=support.device),
        )
    assignment = _random_permutation(
        categories.numel(),
        device=support.device,
        generator=generator,
        forbidden=forbidden_assignment,
    )
    support_positions = torch.searchsorted(categories, support[support_mask])
    support_codes = assignment[support_positions].to(support.dtype)
    mean = support_codes.mean()
    scale = torch.clamp(support_codes.std(unbiased=False), min=_FLOOR)
    support_result = support.clone()
    support_result[support_mask] = (support_codes - mean) / scale

    query_result = query.clone()
    query_positions = torch.searchsorted(categories, query[query_mask])
    bounded = torch.clamp(query_positions, max=categories.numel() - 1)
    known = (query_positions < categories.numel()) & (
        categories[bounded] == query[query_mask]
    )
    query_codes = torch.full(
        query_positions.shape,
        assignment[0],
        dtype=torch.int64,
        device=query.device,
    )
    query_codes[known] = assignment[bounded[known]]
    query_result[query_mask] = (query_codes.to(query.dtype) - mean) / scale
    return support_result, query_result, assignment


def _classify_columns(x_support: torch.Tensor) -> tuple[list[int], list[int]]:
    """Split column indices into categorical and numeric by distinct-value count."""
    category_columns: list[int] = []
    numeric_columns: list[int] = []
    for column in range(x_support.shape[1]):
        observed = x_support[:, column][~torch.isnan(x_support[:, column])]
        target = category_columns if torch.unique(observed).numel() < 10 else numeric_columns
        target.append(column)
    return category_columns, numeric_columns


def _svd_applies(rows: int, columns: int, has_categorical: bool) -> bool:
    """Exempt small all-numeric tables: a full-rank SVD only restates the input."""
    return not (rows < 1000 and columns < 10 and not has_categorical)


def _append_svd_components(
    support: torch.Tensor, query: torch.Tensor, n_svd: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the support's top-``n_svd`` SVD projections as z-normed columns.

    Missing values stand in as zero, which is already the center of the
    standardized columns, so imputation does not pull the basis.  The query is
    projected onto the support's basis rather than its own.  A support with
    fewer than ``n_svd`` components is zero-padded, fixing the width at
    ``D + n_svd`` regardless of rank.
    """
    filled_support = torch.nan_to_num(support, nan=0.0).to(torch.float32)
    filled_query = torch.nan_to_num(query, nan=0.0).to(torch.float32)
    components = min(n_svd, min(support.shape))
    if components > 0:
        # full_matrices=False makes Vh (rank, D); its leading rows are the basis.
        basis = torch.linalg.svd(filled_support, full_matrices=False).Vh[:components]
        support_projection = filled_support @ basis.transpose(0, 1)
        query_projection = filled_query @ basis.transpose(0, 1)
        center = support_projection.mean(dim=0)
        scale = torch.clamp(support_projection.std(dim=0, correction=0), min=_FLOOR)
        support_projection = ((support_projection - center) / scale).to(support.dtype)
        query_projection = ((query_projection - center) / scale).to(query.dtype)
    else:
        support_projection = support.new_zeros((support.shape[0], 0))
        query_projection = query.new_zeros((query.shape[0], 0))
    padding = n_svd - components
    if padding:
        support_projection = torch.cat(
            (support_projection, support_projection.new_zeros((support.shape[0], padding))),
            dim=1,
        )
        query_projection = torch.cat(
            (query_projection, query_projection.new_zeros((query.shape[0], padding))),
            dim=1,
        )
    return (
        torch.cat((support, support_projection), dim=1),
        torch.cat((query, query_projection), dim=1),
    )


def _random_permutation(
    width: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    forbidden: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw uniformly, excluding one paired-member permutation when possible."""
    candidate = torch.randperm(
        width, dtype=torch.int64, device=device, generator=generator
    )
    if forbidden is None or width < 2:
        return candidate
    if forbidden.shape != (width,) or forbidden.dtype != torch.int64:
        raise ValueError("forbidden permutation is invalid")
    if forbidden.device != device:
        raise ValueError("forbidden permutation is on the wrong device")
    for _ in range(31):
        if not torch.equal(candidate, forbidden):
            return candidate
        candidate = torch.randperm(
            width, dtype=torch.int64, device=device, generator=generator
        )
    if not torch.equal(candidate, forbidden):
        return candidate
    # The rejection probability after 32 draws is negligible even for binary
    # labels. Keep the path total and deterministic if a pathological random
    # stream nevertheless repeats the forbidden value every time.
    candidate = forbidden.clone()
    candidate[[0, 1]] = candidate[[1, 0]]
    return candidate


def _validate_build(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor,
    plan: EnsemblePlan,
) -> None:
    if not isinstance(plan, EnsemblePlan):
        raise TypeError("plan must be an EnsemblePlan")
    if not all(isinstance(value, torch.Tensor) for value in (x_support, y_support, x_query)):
        raise TypeError("all data inputs must be tensors")
    if x_support.dtype not in _FLOAT_DTYPES or x_query.dtype not in _FLOAT_DTYPES:
        raise TypeError("features must use float32 or float64")
    if y_support.dtype not in _LABEL_DTYPES:
        raise TypeError("labels must use a real numeric dtype")
    if x_support.ndim != 2 or x_query.ndim != 2 or y_support.ndim != 1:
        raise ValueError("unexpected tensor rank")
    if x_support.shape[0] != y_support.shape[0]:
        raise ValueError("support features and labels have different lengths")
    if x_support.shape[1] != x_query.shape[1]:
        raise ValueError("support and query feature counts differ")
    if x_support.dtype != x_query.dtype:
        raise TypeError("feature dtypes differ")
    if not (x_support.device == x_query.device == y_support.device):
        raise ValueError("all inputs must share a device")
    if any(size == 0 for size in (x_support.shape[0], x_query.shape[0], x_support.shape[1])):
        raise ValueError("empty data axes are unsupported")
    if any(
        value is not None
        for value in (
            plan.feature_permutations,
            plan.class_permutations,
            plan.class_count,
            plan.target_signs,
        )
    ):
        raise ValueError("plan is already fitted")
    labels_as_float = y_support.to(torch.float64)
    if not bool(torch.isfinite(labels_as_float).all()):
        raise ValueError("labels must be finite")
    if plan.task == "regression":
        if not y_support.is_floating_point():
            raise TypeError("regression targets must use a floating dtype")
        return
    if not bool((labels_as_float == torch.floor(labels_as_float)).all()):
        raise ValueError("labels must be integral")
    if bool((labels_as_float < 0).any()):
        raise ValueError("labels must be non-negative")
    distinct = torch.unique(y_support.to(torch.int64), sorted=True)
    expected = torch.arange(distinct.numel(), dtype=torch.int64, device=y_support.device)
    if distinct.numel() == 0 or not torch.equal(distinct, expected):
        raise ValueError("labels must form a contiguous zero-based range")


def build_ensemble_inputs(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor | np.ndarray,
    plan: EnsemblePlan,
    *,
    preprocessor: _FeatureEnsemble | None = None,
    categorical_mask: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, EnsemblePlan]:
    query_values = x_query
    if isinstance(x_query, np.ndarray) and isinstance(x_support, torch.Tensor):
        x_query = torch.as_tensor(x_query, dtype=x_support.dtype, device=x_support.device)
    _validate_build(x_support, y_support, x_query, plan)
    if isinstance(query_values, torch.Tensor):
        query_values = query_values.detach().cpu().numpy()
    device = x_support.device if preprocessor is None else preprocessor.device
    if preprocessor is None:
        preprocessor = _FeatureEnsemble(
            plan.members, plan.seed, device=device, task=plan.task,
            class_count=int(torch.unique(y_support).numel()) if plan.task == "classification" else None,
        ).fit(
            x_support.detach().cpu().numpy(), categorical_mask
        )
    if preprocessor.members != plan.members or preprocessor.support_members.shape[1] != x_support.shape[0]:
        raise ValueError("preprocessor differs from ensemble support")
    support = torch.as_tensor(preprocessor.support_members, device=device)
    query = torch.as_tensor(preprocessor.transform(query_values), device=device)
    features = torch.as_tensor(preprocessor.orders, dtype=torch.int64, device=device)
    if plan.n_svd:
        augmented = [
            _append_svd_components(support[i], query[i], plan.n_svd)
            for i in range(plan.members)
        ]
        support = torch.stack([pair[0] for pair in augmented])
        query = torch.stack([pair[1] for pair in augmented])
        extra = features.max() + 1 + torch.arange(plan.n_svd, device=device)
        features = torch.cat((features, extra.expand(plan.members, -1)), dim=1)
    generator = torch.Generator(device=device).set_state(preprocessor.target_rng_state)
    y_support = y_support.to(device)
    labels, permutations, signs = [], [], []
    count = int(torch.unique(y_support).numel()) if plan.task == "classification" else None
    offsets = torch.empty(0, dtype=torch.int64, device=device)
    for _ in range(plan.members):
        if plan.task == "regression":
            sign = torch.empty((1, 1), dtype=y_support.dtype, device=device)
            sign.bernoulli_(0.5, generator=generator).mul_(-2).add_(1)
            signs.append(sign.reshape(()))
            labels.append(y_support * sign.reshape(()))
        else:
            if offsets.numel() == 0:
                offsets = torch.randperm(count, generator=generator, device=device)
            order = (torch.arange(count, device=device) - offsets[0]) % count
            offsets = offsets[1:]
            permutations.append(order)
            labels.append(order[y_support.to(torch.int64)].to(y_support.dtype))
    fitted = replace(
        plan,
        feature_permutations=features,
        class_permutations=torch.stack(permutations) if permutations else None,
        class_count=count,
        target_signs=torch.stack(signs) if signs else None,
    )
    return support, torch.stack(labels), query, fitted


def _is_permutation_rows(values: torch.Tensor, width: int) -> bool:
    expected = torch.arange(width, dtype=torch.int64, device=values.device).expand_as(values)
    return torch.equal(torch.sort(values, dim=1).values, expected)


def aggregate_probabilities(
    member_logits: torch.Tensor, fitted_plan: EnsemblePlan
) -> torch.Tensor:
    """Decode member class order, average logits, and normalize once."""
    if not isinstance(fitted_plan, EnsemblePlan):
        raise TypeError("fitted_plan must be an EnsemblePlan")
    if fitted_plan.task != "classification":
        raise ValueError("probability aggregation requires a classification plan")
    if not isinstance(member_logits, torch.Tensor):
        raise TypeError("member_logits must be a tensor")
    if member_logits.dtype not in _LOGIT_DTYPES:
        raise TypeError("logits must use a floating dtype")
    if member_logits.ndim != 3:
        raise ValueError("logits must have three axes")
    if member_logits.shape[0] != fitted_plan.members:
        raise ValueError("member count does not match plan")
    if member_logits.shape[1] == 0 or member_logits.shape[2] == 0:
        raise ValueError("empty logits axes are unsupported")
    features = fitted_plan.feature_permutations
    classes = fitted_plan.class_permutations
    count = fitted_plan.class_count
    if features is None or classes is None or count is None:
        raise RuntimeError("plan has not been fitted")
    if not isinstance(features, torch.Tensor) or not isinstance(classes, torch.Tensor):
        raise RuntimeError("fitted metadata is invalid")
    if features.device != member_logits.device or classes.device != member_logits.device:
        raise ValueError("logits and plan metadata must share a device")
    consistent = (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        and features.dtype == torch.int64
        and classes.dtype == torch.int64
        and features.ndim == 2
        and classes.ndim == 2
        and features.shape[0] == fitted_plan.members
        and classes.shape == (fitted_plan.members, count)
        and features.shape[1] > 0
    )
    if not consistent:
        raise RuntimeError("fitted metadata is inconsistent")
    if bool((features < 0).any()) or bool((torch.diff(features.sort(dim=1).values, dim=1) == 0).any()) or not _is_permutation_rows(classes, count):
        raise RuntimeError("fitted metadata does not contain permutations")
    if member_logits.shape[2] < count:
        raise ValueError("too few class-score columns")
    restored = []
    for member in range(fitted_plan.members):
        leading = member_logits[member, :, :count].index_select(1, classes[member])
        restored.append(torch.cat((leading, member_logits[member, :, count:]), dim=1))
    decoded = torch.stack(restored)
    if decoded.dtype in (torch.float16, torch.bfloat16):
        decoded = decoded.to(torch.float32)
    return torch.softmax(decoded.mean(dim=0), dim=1)


def fit_member_weights(
    member_predictions: np.ndarray, targets: np.ndarray, *, shrinkage: float
) -> np.ndarray:
    """Fit ensemble weights against held-out targets, then shrink toward uniform.

    Solves ``min_w ||P^T w - y||`` subject to ``w >= 0`` and rescales the solution
    to sum to one, so the aggregate stays a convex combination of members -- an
    interpolation, never an extrapolation.  Unconstrained least squares on members
    this correlated would return large cancelling coefficients from a validation
    split of a few dozen rows; non-negativity is what keeps the fit usable there.

    Both arguments must already be on the original target scale.
    """
    if not isinstance(member_predictions, np.ndarray) or not isinstance(targets, np.ndarray):
        raise TypeError("member predictions and targets must be NumPy arrays")
    if member_predictions.ndim != 2 or targets.ndim != 1:
        raise ValueError("member predictions must be rank two and targets rank one")
    if member_predictions.shape[1] != targets.shape[0]:
        raise ValueError("member predictions and targets have different lengths")
    if member_predictions.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("empty data axes are unsupported")
    if isinstance(shrinkage, bool) or not isinstance(shrinkage, float):
        raise TypeError("shrinkage must be a float")
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    design = np.asarray(member_predictions, dtype=np.float64).T
    observed = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(design).all() or not np.isfinite(observed).all():
        raise ValueError("member predictions and targets must be finite")

    members = design.shape[1]
    uniform = np.full(members, 1.0 / members, dtype=np.float64)
    # positive=True dispatches to scipy.optimize.nnls for a single target, so this
    # is the NNLS solve without taking a direct dependency on scipy.
    solver = LinearRegression(positive=True, fit_intercept=False)
    solver.fit(design, observed)
    weights = np.asarray(solver.coef_, dtype=np.float64)
    if not np.isfinite(weights).all() or bool((weights < 0.0).any()):
        raise RuntimeError("non-negative least squares returned an invalid solution")
    total = float(weights.sum())
    if total <= 1e-9:
        _logger.warning(
            "non-negative least squares zeroed every ensemble member "
            "(weight sum %.3e); falling back to uniform weights",
            total,
        )
        return uniform
    return shrinkage * (weights / total) + (1.0 - shrinkage) * uniform
