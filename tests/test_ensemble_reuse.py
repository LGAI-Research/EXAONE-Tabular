from __future__ import annotations

import numpy as np
import torch

from exaonetabular.classifier import EXAONETabularClassifier
from exaonetabular.config import (
    InferenceManifest,
    ModelConfig,
    PreprocessingConfig,
    RuntimeConfig,
)
from exaonetabular.ensemble import (
    EnsemblePlan,
    _numeric_column,
    _yeo_johnson,
    build_ensemble_inputs,
)


def _manifest() -> InferenceManifest:
    return InferenceManifest(
        checkpoint_sha256="0" * 64,
        model=ModelConfig(
            width=16,
            head_count=4,
            block_count=2,
            feedforward_widths=32,
            columns_per_group=1,
            item_summary_count=2,
            feature_summary_count=3,
            feature_attention_repeats=1,
            class_capacity=3,
        ),
        preprocessing=PreprocessingConfig(),
        runtime=RuntimeConfig(
            support_row_limit=50_000,
            feature_limit=100,
            query_batch_limit=1_024,
            ensemble_count=8,
            compute_dtype="float32",
            seed=0,
        ),
    )


def _numeric_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(24, dtype=torch.float32)
    support = torch.stack((rows, rows.square() + 0.25, rows.sin()), dim=1)
    query_rows = torch.arange(24, 29, dtype=torch.float32)
    query = torch.stack(
        (query_rows, query_rows.square() + 0.25, query_rows.sin()), dim=1
    )
    labels = rows.to(torch.int64).remainder(2).to(torch.float32)
    return support, labels, query


def _restore_feature_order(
    members: torch.Tensor, permutations: torch.Tensor
) -> torch.Tensor:
    restored = []
    for member in range(members.shape[0]):
        restored.append(members[member, :, torch.argsort(permutations[member])])
    return torch.stack(restored)


def test_plan_keeps_eight_distinct_executions_and_pairs_class_orders() -> None:
    support, labels, query = _numeric_problem()
    xs, ys, xq, plan = build_ensemble_inputs(
        support, labels, query, EnsemblePlan(members=8, seed=0)
    )

    assert xs.shape[0] == ys.shape[0] == xq.shape[0] == plan.members == 8
    assert not hasattr(plan, "member_sources")
    assert plan.class_permutations is not None
    for member in range(4, 8):
        assert not torch.equal(
            plan.class_permutations[member], plan.class_permutations[member - 4]
        )

    assert plan.feature_permutations is not None
    restored = _restore_feature_order(xs, plan.feature_permutations)
    # The deterministic rules share their prepared tensors across the two
    # cycles, while noisy Gaussianization remains member-specific.
    for first, second in ((0, 4), (1, 5), (2, 6)):
        torch.testing.assert_close(restored[first], restored[second], rtol=0, atol=0)
    assert not torch.equal(restored[3], restored[7])


def test_members_depend_only_on_the_plan_seed_not_ambient_rng() -> None:
    """The ensemble draws from its own generator, so nothing outside can move it.

    Guards the whole released-inference contract: members must be reproducible
    across processes and unchanged by whatever the caller did to the global RNG
    beforehand. A draw that forgets ``generator=`` would pass every other test.
    """
    support, labels, query = _numeric_problem()

    torch.manual_seed(1234)
    first, _, first_query, first_plan = build_ensemble_inputs(
        support, labels, query, EnsemblePlan(members=8, seed=0)
    )
    torch.manual_seed(9999)
    torch.randn(64)  # perturb the global stream between the two builds
    second, _, second_query, second_plan = build_ensemble_inputs(
        support, labels, query, EnsemblePlan(members=8, seed=0)
    )

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_query, second_query, rtol=0, atol=0)
    assert torch.equal(first_plan.feature_permutations, second_plan.feature_permutations)
    assert torch.equal(first_plan.class_permutations, second_plan.class_permutations)

    # ...and the seed is genuinely wired in, not merely constant.
    other, _, _, other_plan = build_ensemble_inputs(
        support, labels, query, EnsemblePlan(members=8, seed=1)
    )
    assert not torch.equal(first_plan.feature_permutations, other_plan.feature_permutations)
    assert not torch.equal(first, other)


def test_categorical_assignments_do_not_repeat_across_member_cycles() -> None:
    support = torch.tensor(
        [[0, 0], [1, 1], [0, 2], [1, 0], [0, 1], [1, 2]],
        dtype=torch.float32,
    )
    query = torch.tensor([[0, 2], [1, 0]], dtype=torch.float32)
    labels = torch.arange(6).remainder(2).to(torch.float32)
    xs, _, _, plan = build_ensemble_inputs(
        support, labels, query, EnsemblePlan(members=8, seed=17)
    )
    assert plan.feature_permutations is not None
    restored = _restore_feature_order(xs, plan.feature_permutations)

    def standardized(values: torch.Tensor) -> torch.Tensor:
        return (values - values.mean(dim=0)) / values.std(dim=0, correction=0)

    normalized = torch.stack([standardized(member) for member in restored])
    for member in range(4, 8):
        for column in range(support.shape[1]):
            assert not torch.equal(
                normalized[member, :, column],
                normalized[member - 4, :, column],
            )


def test_classifier_executes_all_eight_members(monkeypatch) -> None:
    classifier = EXAONETabularClassifier(_manifest())
    support, labels, query = _numeric_problem()
    calls: list[int] = []

    def recorded_forward(xs, ys, xq):
        calls.append(int(xs.shape[0]))
        return torch.zeros(
            (xs.shape[0], xq.shape[1], classifier.manifest.model.class_capacity),
            dtype=xs.dtype,
            device=xs.device,
        )

    monkeypatch.setattr(classifier, "_forward_chunked", recorded_forward)
    result = classifier._infer_problem(
        support.numpy(), labels.numpy().astype(np.int64), query.numpy(), 2
    )
    assert calls == [8]
    assert result.shape == (query.shape[0], 2)


def test_rule_one_is_robust_asinh_instead_of_affine() -> None:
    support = torch.tensor([-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0])
    query = torch.tensor([-10.0, 10.0])
    transformed_support, transformed_query = _numeric_column(
        support, query, 1, torch.Generator().manual_seed(0)
    )
    expected_support = torch.asinh(support / 3.0)
    expected_query = torch.asinh(query / 3.0)
    torch.testing.assert_close(transformed_support, expected_support, rtol=0, atol=0)
    torch.testing.assert_close(transformed_query, expected_query, rtol=0, atol=0)

    original_z = (support - support.mean()) / support.std(correction=0)
    transformed_z = (
        transformed_support - transformed_support.mean()
    ) / transformed_support.std(correction=0)
    assert not torch.allclose(original_z, transformed_z)


def test_vectorized_yeo_johnson_matches_serial_lambda_search() -> None:
    support = torch.tensor(
        [-3.5, -2.0, -0.25, 0.0, 0.5, 1.25, 4.0, float("nan")],
        dtype=torch.float32,
    )
    query = torch.tensor(
        [-4.0, -0.5, 0.25, 2.5, float("nan")], dtype=torch.float32
    )

    def serial_rule_two():
        support_mask = ~torch.isnan(support)
        query_mask = ~torch.isnan(query)
        observed = support[support_mask]
        best_index = 0
        best_score = None
        for lam_index in range(17):
            candidate = _yeo_johnson(observed, lam_index)
            mean = candidate.mean()
            std = torch.clamp(candidate.std(unbiased=False), min=1e-6)
            score = float(torch.abs(((candidate - mean) ** 3).mean() / std**3))
            if best_score is None or score < best_score:
                best_score = score
                best_index = lam_index
        fitted = _yeo_johnson(observed, best_index)
        mean = fitted.mean()
        scale = torch.clamp(fitted.std(unbiased=False), min=1e-6)
        support_result = support.clone()
        query_result = query.clone()
        support_result[support_mask] = (fitted - mean) / scale
        query_result[query_mask] = (
            _yeo_johnson(query[query_mask], best_index) - mean
        ) / scale
        return support_result, query_result

    expected_support, expected_query = serial_rule_two()
    actual_support, actual_query = _numeric_column(
        support, query, 2, torch.Generator().manual_seed(0)
    )
    torch.testing.assert_close(
        actual_support, expected_support, rtol=1e-6, atol=2e-7, equal_nan=True
    )
    torch.testing.assert_close(
        actual_query, expected_query, rtol=1e-6, atol=2e-7, equal_nan=True
    )
