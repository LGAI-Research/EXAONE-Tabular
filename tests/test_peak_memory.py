from __future__ import annotations

import pytest
import torch

from exaonetabular.classifier import EXAONETabularClassifier
from exaonetabular._execution import _InferenceExecutor
from exaonetabular.config import InferenceManifest, ModelConfig, RuntimeConfig
from exaonetabular.config import RegressionConfig
from exaonetabular.model.heads import RegressionModel
from exaonetabular.model.memory import (
    PeakMemoryEstimate,
    PeakMemoryMode,
    estimated_peak,
)
from exaonetabular.runtime import FULL_FEATURE_ATTENTION_ROWS

_FFN_CHUNK = 524_288


def _config() -> ModelConfig:
    return ModelConfig(
        width=16,
        head_count=4,
        block_count=2,
        feedforward_widths=32,
        columns_per_group=1,
        item_summary_count=2,
        feature_summary_count=3,
        feature_attention_repeats=1,
        class_capacity=3,
    )


def _estimate(**overrides):
    values = {
        "support_rows": 64,
        "query_rows": 16,
        "feature_count": 8,
        "members": 1,
        "feedforward_token_chunk": _FFN_CHUNK,
        "dtype": torch.float16,
        "mode": PeakMemoryMode.JOINED,
        "max_vram_bytes": 2**50,
    }
    values.update(overrides)
    return estimated_peak(_config(), **values)


def _classifier(
    *,
    query_batch_limit: int = 16,
    ensemble_count: int = 8,
    support_cache_offload: bool = True,
):
    manifest = InferenceManifest(
        checkpoint_sha256="0" * 64,
        model=_config(),
        runtime=RuntimeConfig(
            support_row_limit=128,
            feature_limit=8,
            query_batch_limit=query_batch_limit,
            ensemble_count=ensemble_count,
            compute_dtype="float32",
            support_cache_offload=support_cache_offload,
        ),
    )
    classifier = EXAONETabularClassifier(manifest)
    classifier.model.supports_query_chunk_cache = False
    return classifier


def _linear_estimator(limit: int):
    def estimate(mode, query_rows, members, feedforward_token_chunk):
        del mode, feedforward_token_chunk
        peak = query_rows * members
        return PeakMemoryEstimate(
            allocated_bytes=peak,
            projected_reserved_bytes=peak,
            target_vram_bytes=limit,
            limiting_phase="test",
            fits=peak <= limit,
        )

    return estimate


def test_peak_estimate_is_shape_and_dtype_monotonic() -> None:
    base = _estimate()
    assert _estimate(query_rows=32).allocated_bytes > base.allocated_bytes
    assert _estimate(support_rows=128).allocated_bytes > base.allocated_bytes
    assert _estimate(feature_count=16).allocated_bytes > base.allocated_bytes
    assert _estimate(members=2).allocated_bytes == 2 * base.allocated_bytes
    assert _estimate(dtype=torch.float32).allocated_bytes > base.allocated_bytes


def test_peak_estimate_reports_budget_and_limiting_phase() -> None:
    base = _estimate(
        baseline_allocated_bytes=1024,
        baseline_reserved_bytes=2048,
    )
    assert base.allocated_bytes > 1024
    assert base.projected_reserved_bytes >= base.allocated_bytes
    assert base.limiting_phase in {
        "encoder",
        "feature_attention",
        "item_attention",
        "query_cross_attention",
        "feedforward",
    }
    rejected = _estimate(
        baseline_allocated_bytes=1024,
        baseline_reserved_bytes=2048,
        max_vram_bytes=base.projected_reserved_bytes - 1,
    )
    accepted = _estimate(
        baseline_allocated_bytes=1024,
        baseline_reserved_bytes=2048,
        max_vram_bytes=base.projected_reserved_bytes,
    )
    assert not rejected.fits
    assert accepted.fits


def test_cache_modes_include_persistent_or_transfer_state() -> None:
    joined = _estimate(mode=PeakMemoryMode.JOINED)
    gpu = _estimate(mode=PeakMemoryMode.CACHE_BUILD_GPU)
    cpu = _estimate(mode=PeakMemoryMode.CACHE_BUILD_CPU)
    assert gpu.allocated_bytes >= cpu.allocated_bytes >= joined.allocated_bytes

    compact = _estimate(
        support_rows=128,
        query_rows=16,
        mode=PeakMemoryMode.CACHED_COMPACT_GPU,
    )
    padded = _estimate(
        support_rows=128,
        query_rows=16,
        mode=PeakMemoryMode.CACHED_PADDED_GPU,
    )
    assert compact.allocated_bytes < padded.allocated_bytes


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("support_rows", 0, ValueError),
        ("query_rows", -1, ValueError),
        ("feature_count", True, TypeError),
        ("members", 0, ValueError),
        ("feedforward_token_chunk", 0, ValueError),
        ("max_vram_bytes", 0, ValueError),
    ],
)
def test_peak_estimate_rejects_invalid_dimensions(name, value, error) -> None:
    with pytest.raises(error):
        _estimate(**{name: value})


def test_safe_query_chunk_uses_band_without_extra_call() -> None:
    selected = EXAONETabularClassifier._safe_query_chunk(
        10_000, 25_334, torch.float16
    )
    assert 8_500 <= selected < 9_000
    assert selected % 8 == 0
    assert (25_334 + selected - 1) // selected == 3

    # A workload between the safe selection and the fitted limit must still
    # be split; fitting alone does not provide the requested 10-15% headroom.
    selected = EXAONETabularClassifier._safe_query_chunk(
        10_000, 9_500, torch.float16
    )
    assert 8_500 <= selected < 9_000

    # No integer exists in the strict 85-90% interval at this tiny limit.
    assert EXAONETabularClassifier._safe_query_chunk(
        10, 10, torch.float16
    ) == 10


def test_peak_estimate_uses_explicit_feedforward_chunk() -> None:
    small = _estimate(
        support_rows=50_000,
        query_rows=1,
        feature_count=100,
        feedforward_token_chunk=1_000,
    )
    large = _estimate(
        support_rows=50_000,
        query_rows=1,
        feature_count=100,
        feedforward_token_chunk=5_000_000,
    )
    assert large.projected_reserved_bytes > small.projected_reserved_bytes


def test_regression_peak_includes_quantile_decoder_width() -> None:
    config = _config()
    narrow = RegressionModel(
        config,
        RegressionConfig(quantile_count=5, decoder_hidden_width=16),
    ).estimated_peak(
        support_rows=16,
        query_rows=256,
        feature_count=2,
        members=2,
        feedforward_token_chunk=_FFN_CHUNK,
        mode=PeakMemoryMode.JOINED,
        max_vram_bytes=2**50,
    )
    wide = RegressionModel(
        config,
        RegressionConfig(quantile_count=999, decoder_hidden_width=384),
    ).estimated_peak(
        support_rows=16,
        query_rows=256,
        feature_count=2,
        members=2,
        feedforward_token_chunk=_FFN_CHUNK,
        mode=PeakMemoryMode.JOINED,
        max_vram_bytes=2**50,
    )
    assert wide.allocated_bytes > narrow.allocated_bytes


def test_outer_chunk_outputs_are_part_of_the_common_memory_budget() -> None:
    base = _estimate(
        output_width=999,
        decoder_hidden_width=384,
        outer_output_bytes=0,
    )
    retained = _estimate(
        output_width=999,
        decoder_hidden_width=384,
        outer_output_bytes=32_000_000,
    )
    assert retained.allocated_bytes >= base.allocated_bytes + 32_000_000
    assert retained.allocated_bytes >= 64_000_000
    assert retained.projected_reserved_bytes > base.projected_reserved_bytes


def test_output_concatenation_can_be_the_limiting_phase() -> None:
    estimate = _estimate(
        support_rows=1,
        query_rows=1,
        feature_count=1,
        members=1,
        feedforward_token_chunk=1,
        outer_output_bytes=1_000_000,
    )
    assert estimate.limiting_phase == "output_concatenation"
    assert estimate.allocated_bytes == 2_000_000


def test_classifier_defaults_include_its_output_head() -> None:
    implicit = _estimate()
    explicit = _estimate(
        output_width=_config().class_capacity,
        decoder_hidden_width=_config().feedforward_widths[-1],
    )
    assert implicit == explicit


def test_safe_query_band_across_common_task_memory_grid() -> None:
    total_query = 60_000
    checked = 0
    for output_width, decoder_hidden_width in ((10, 768), (999, 384)):
        for support_rows in (500, 2_000, 10_000):
            for members in (1, 2, 4, 8):
                for feature_count in (5, 20, 50, 100):
                    for budget in (2_000_000_000, 4_000_000_000, 8_000_000_000):
                        common = {
                            "support_rows": support_rows,
                            "feature_count": feature_count,
                            "members": members,
                            "feedforward_token_chunk": 100_000,
                            "dtype": torch.float16,
                            "mode": PeakMemoryMode.JOINED,
                            "max_vram_bytes": budget,
                            "baseline_allocated_bytes": 100_000_000,
                            "baseline_reserved_bytes": 128_000_000,
                            "output_width": output_width,
                            "decoder_hidden_width": decoder_hidden_width,
                            "outer_output_bytes": (
                                members * total_query * output_width * 2
                            ),
                        }
                        if not estimated_peak(
                            _config(), query_rows=1, **common
                        ).fits:
                            continue
                        low, high = 1, total_query
                        while low < high:
                            middle = (low + high + 1) // 2
                            if estimated_peak(
                                _config(), query_rows=middle, **common
                            ).fits:
                                low = middle
                            else:
                                high = middle - 1
                        if low < 100:
                            continue
                        selected = _InferenceExecutor.safe_query_chunk(
                            low, total_query, torch.float16
                        )
                        assert 85 * low <= 100 * selected < 90 * low
                        checked += 1
    assert checked >= 100


def test_cuda_planner_selects_smallest_c_on_best_call_plateau(
    monkeypatch,
) -> None:
    classifier = _classifier()
    support = torch.randn(8, 64, 8)
    queries = torch.randn(8, 100, 8)

    monkeypatch.setattr(
        "exaonetabular._execution.feedforward_token_counts",
        lambda *args, **kwargs: (1_000_000 * kwargs["members"],),
    )

    def estimate(mode, query_rows, members, feedforward_token_chunk):
        del mode, query_rows
        peak = feedforward_token_chunk + 100_000 * members
        return PeakMemoryEstimate(
            allocated_bytes=peak,
            projected_reserved_bytes=peak,
            target_vram_bytes=800_000,
            limiting_phase="test",
            fits=peak <= 800_000,
        )

    plan = classifier._plan_cuda_execution(
        support,
        queries,
        torch.float32,
        estimate,
    )

    assert plan.query_step == 100
    assert plan.feedforward_token_chunk == 500_000
    assert plan.ensemble_step == 3


def _fragmenting_gpu_cache_estimate(mode, query_rows, members, feedforward_token_chunk):
    """GPU-resident cache build overflows; offloaded build + resident query fit.

    Models the trigger the offload+reload path targets: ``CACHE_BUILD_GPU`` never
    fits (its fragmentation-inflated peak overflows even maximally chunked), while
    the host-offloaded build (``CACHE_BUILD_CPU``) always fits and the resident
    cached query (``CACHED_*``) fits only when the query is chunked <= 64.
    """
    del feedforward_token_chunk
    if members != 1:
        fits = False
    elif mode is PeakMemoryMode.JOINED:
        fits = query_rows <= 32  # forces the joined path to chunk -> cache path
    elif mode in {
        PeakMemoryMode.CACHE_BUILD_GPU,
        PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU,
    }:
        fits = False  # GPU-resident support build cannot fit
    elif mode is PeakMemoryMode.CACHE_BUILD_CPU:
        fits = True  # host-offloaded build fits
    else:  # CACHED_* query phases (resident or streamed)
        fits = query_rows <= 64
    return PeakMemoryEstimate(
        allocated_bytes=1,
        projected_reserved_bytes=1,
        target_vram_bytes=10**9,
        limiting_phase=str(mode),
        fits=fits,
    )


def test_cuda_planner_offloads_and_reloads_when_gpu_cache_fragments(monkeypatch) -> None:
    classifier = _classifier()
    classifier.model.supports_query_chunk_cache = True
    monkeypatch.setattr(classifier.model, "can_cache_support", lambda *a, **k: True)
    monkeypatch.setattr(classifier.model, "can_cache_query_chunks", lambda *a, **k: True)
    support = torch.randn(1, 60000, 8)  # large enough to clear the size heuristic
    queries = torch.randn(1, 256, 8)

    plan = classifier._plan_cuda_execution(
        support, queries, torch.float32, _fragmenting_gpu_cache_estimate
    )

    assert plan.query_cache and plan.cache_to_cpu and plan.reload_cache
    assert plan.compact_cache
    assert plan.query_step < 256  # the query is genuinely chunked over the cache


def test_cuda_planner_offload_disabled_falls_back_to_streaming(monkeypatch) -> None:
    classifier = _classifier(support_cache_offload=False)
    classifier.model.supports_query_chunk_cache = True
    monkeypatch.setattr(classifier.model, "can_cache_support", lambda *a, **k: True)
    monkeypatch.setattr(classifier.model, "can_cache_query_chunks", lambda *a, **k: True)
    support = torch.randn(1, 60000, 8)
    queries = torch.randn(1, 256, 8)

    plan = classifier._plan_cuda_execution(
        support, queries, torch.float32, _fragmenting_gpu_cache_estimate
    )

    # Offload disabled: no reload; the streaming CPU cache remains the fallback.
    assert plan.query_cache and plan.cache_to_cpu and not plan.reload_cache


def test_cuda_planner_splits_ensemble_before_query() -> None:
    classifier = _classifier()
    support = torch.randn(8, 64, 8)
    queries = torch.randn(8, 100, 8)

    plan = classifier._plan_cuda_execution(
        support,
        queries,
        torch.float32,
        _linear_estimator(350),
    )

    assert plan.ensemble_step == 3
    assert plan.query_step == 100
    assert not plan.support_only_cache
    assert not plan.query_cache


def test_cuda_planner_splits_query_only_after_single_member() -> None:
    classifier = _classifier()
    support = torch.randn(8, 64, 8)
    queries = torch.randn(8, 100, 8)

    plan = classifier._plan_cuda_execution(
        support,
        queries,
        torch.float32,
        _linear_estimator(80),
    )

    assert plan.ensemble_step == 1
    assert plan.query_step == 68


def test_cuda_planner_applies_headroom_to_fixed_support_phase() -> None:
    classifier = _classifier()
    classifier.model.supports_query_chunk_cache = True
    support = torch.randn(8, 64, 8)
    queries = torch.randn(8, 10, 8)

    def estimate(mode, query_rows, members, feedforward_token_chunk):
        del feedforward_token_chunk
        peak = (
            31 * members
            if mode is PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU
            else query_rows * members
        )
        return PeakMemoryEstimate(
            allocated_bytes=peak,
            projected_reserved_bytes=peak,
            target_vram_bytes=100,
            limiting_phase="test",
            fits=peak <= 100,
        )

    plan = classifier._plan_cuda_execution(
        support,
        queries,
        torch.float32,
        estimate,
    )

    assert plan.support_only_cache
    assert plan.ensemble_step == 2
    assert plan.query_step == 10


def test_query_heavy_regression_shape_selects_two_joined_members() -> None:
    config = ModelConfig()
    manifest = InferenceManifest(
        checkpoint_sha256="0" * 64,
        model=config,
        runtime=RuntimeConfig(
            support_row_limit=10_000,
            feature_limit=50,
            query_batch_limit=1024,
            ensemble_count=8,
            compute_dtype="float16",
        ),
    )
    classifier = EXAONETabularClassifier(manifest)
    classifier.model.supports_query_chunk_cache = False
    support = torch.empty(8, 10_000, 50, device="meta")
    queries = torch.empty(8, 44_099, 50, device="meta")

    def estimate(mode, query_rows, members, feedforward_token_chunk):
        return estimated_peak(
            config,
            support_rows=10_000,
            query_rows=query_rows,
            feature_count=50,
            members=members,
            feedforward_token_chunk=feedforward_token_chunk,
            dtype=torch.float16,
            mode=mode,
            max_vram_bytes=47_600_000_000,
        )

    plan = classifier._plan_cuda_execution(
        support,
        queries,
        torch.float16,
        estimate,
    )

    assert plan.ensemble_step == 2
    assert plan.query_step == 44_099


def test_cuda_planner_rejects_an_unusable_minimum() -> None:
    classifier = _classifier()
    support = torch.randn(8, 64, 8)
    queries = torch.randn(8, 100, 8)

    with pytest.raises(RuntimeError, match="minimum model forward"):
        classifier._plan_cuda_execution(
            support,
            queries,
            torch.float32,
            _linear_estimator(0),
        )


def test_forward_propagates_oom_without_retry() -> None:
    class OOMModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, *args, **kwargs):
            del args, kwargs
            self.calls += 1
            raise torch.cuda.OutOfMemoryError("planned forward failed")

    classifier = _classifier(query_batch_limit=3, ensemble_count=2)
    model = OOMModel()
    classifier.model = model
    support = torch.randn(2, 4, 8)
    labels = torch.zeros(2, 4)
    queries = torch.randn(2, 7, 8)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="planned forward failed"):
        classifier._forward_chunked(support, labels, queries)

    assert model.calls == 1


def test_default_cuda_budget_excludes_other_process_usage(monkeypatch) -> None:
    classifier = _classifier()
    classifier.device = torch.device("cuda:3")
    classifier.max_vram_bytes = None

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20, 100))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 7)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 11)

    # Driver-free bytes plus this process's reclaimable reservation equals
    # total VRAM minus memory owned by other processes.
    assert classifier._cuda_budget() == (31, 7, 11, 20)

    classifier.max_vram_bytes = 55
    assert classifier._cuda_budget() == (55, 7, 11, 20)


def test_max_vram_bytes_requires_cuda() -> None:
    manifest = InferenceManifest(
        checkpoint_sha256="0" * 64,
        model=_config(),
        runtime=RuntimeConfig(
            support_row_limit=128,
            feature_limit=8,
            query_batch_limit=16,
            ensemble_count=2,
            compute_dtype="float32",
        ),
    )
    EXAONETabularClassifier(manifest)
    with pytest.raises(ValueError, match="CUDA"):
        EXAONETabularClassifier(manifest, max_vram_bytes=1024)


def test_shrunk_preference_budget_keeps_the_detected_vram_ceiling() -> None:
    """A lowered preference budget does not move the feasibility ceiling.

    The support peak is not chunkable, so feasibility stays pinned to the
    physical ceiling; a forward the hardware can run must not be refused because
    the preference budget was lowered to bias chunk selection.
    """
    classifier = _classifier()
    support = torch.randn(8, 64, 8)
    queries = torch.randn(8, 100, 8)

    plan = classifier._plan_cuda_execution(
        support,
        queries,
        torch.float32,
        _linear_estimator(0),  # preference budget admits nothing
        _linear_estimator(1),  # ceiling admits the minimum forward
    )

    assert plan.query_step >= 1
    assert plan.ensemble_step >= 1


def test_minimum_over_the_ceiling_still_raises() -> None:
    """Overflowing the ceiling is a real capacity failure, not a retry artifact."""
    classifier = _classifier()
    support = torch.randn(8, 64, 8)
    queries = torch.randn(8, 100, 8)

    with pytest.raises(RuntimeError, match="minimum model forward"):
        classifier._plan_cuda_execution(
            support,
            queries,
            torch.float32,
            _linear_estimator(0),
            _linear_estimator(0),
        )


def test_feature_attention_chunk_lowers_the_modeled_peak_monotonically() -> None:
    """Row-chunking must shrink only the feature-attention transient.

    In a support-heavy build feature attention is the limiting phase, so the
    modeled peak is non-increasing as the chunk shrinks and floors on the next
    phase; a chunk at or above the row count reproduces the unchunked estimate.
    """
    common = dict(
        mode=PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU,
        support_rows=2048,
        query_rows=0,
        feature_count=32,
        feedforward_token_chunk=64,
    )
    full = _estimate(**common)
    assert full.limiting_phase == "feature_attention"

    # chunk >= rows reproduces the unchunked estimate exactly.
    assert (
        _estimate(**common, feature_attention_row_chunk=2048).allocated_bytes
        == full.allocated_bytes
    )

    peaks = [
        _estimate(**common, feature_attention_row_chunk=chunk).allocated_bytes
        for chunk in (2048, 1024, 256, 64, 16, 1)
    ]
    assert peaks == sorted(peaks, reverse=True)  # monotonic non-increasing
    assert peaks[-1] < peaks[0]  # chunking genuinely lowers the peak


def _chunk_estimate(reserved: int, target: int) -> PeakMemoryEstimate:
    return PeakMemoryEstimate(
        allocated_bytes=reserved,
        projected_reserved_bytes=reserved,
        target_vram_bytes=target,
        limiting_phase="feature_attention",
        fits=reserved <= target,
    )


def _has_headroom(result: PeakMemoryEstimate) -> bool:
    return result.fits and (
        10 * result.projected_reserved_bytes < 9 * result.target_vram_bytes
    )


def test_feature_attention_chunk_selection_picks_largest_fitting_chunk() -> None:
    """The planner picks the largest chunk that clears the ~90% headroom."""
    select = _InferenceExecutor._plan_feature_attention_chunk

    # Full rows clear the budget with headroom -> no chunk (byte-identical path).
    loose = select(
        1000, lambda chunk: _chunk_estimate(10 + chunk, 10**9), accepts=_has_headroom
    )
    assert loose == FULL_FEATURE_ATTENTION_ROWS

    # Tight budget: largest chunk with 10*reserved < 9*target, i.e. chunk < 900.
    tight = select(
        1000, lambda chunk: _chunk_estimate(chunk, 1000), accepts=_has_headroom
    )
    assert tight == 899
    assert 1 <= tight < 1000  # sub-full: chunking has engaged

    # Even a single-row chunk overflows -> return 1 so the caller surfaces the
    # honest ceiling failure through the ordinary plan path.
    starved = select(
        1000,
        lambda chunk: _chunk_estimate(10**9 + chunk, 1000),
        accepts=_has_headroom,
    )
    assert starved == 1


def test_feature_attention_chunk_respects_the_candidate_s_own_bar() -> None:
    """A build judged on ``fits`` alone must not be chunked to reach headroom.

    Query-dependent builds are admitted when they merely fit -- the query chunk
    is their lever. Holding them to the support-only phase's 90% margin would
    hand them a needlessly small chunk (and its kernel launches) to buy slack
    nothing checks.
    """
    select = _InferenceExecutor._plan_feature_attention_chunk
    peak = lambda chunk: _chunk_estimate(950 + chunk // 20, 1000)  # noqa: E731

    assert select(1000, peak, accepts=lambda r: r.fits) == FULL_FEATURE_ATTENTION_ROWS
    assert select(1000, peak, accepts=_has_headroom) == 1


def test_engaging_the_chunk_costs_a_destination_buffer() -> None:
    """A chunked pass pays for one full-row destination the un-chunked path lacks.

    ``runtime.run_in_chunks`` writes its blocks into a preallocated tensor, so
    chunking is not free: just under full rows it costs more than it saves. The
    estimator has to say so, or the planner would engage a chunk that raises the
    peak it was asked to lower.
    """
    common = dict(
        mode=PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU,
        support_rows=2048,
        query_rows=0,
        feature_count=32,
        feedforward_token_chunk=64,
    )
    bypassed = _estimate(**common, feature_attention_row_chunk=2048).allocated_bytes
    engaged = _estimate(**common, feature_attention_row_chunk=2047).allocated_bytes
    floor = _estimate(**common, feature_attention_row_chunk=1).allocated_bytes

    # One row of the state||item_summary block the chunk loop writes out:
    # (32 feature groups + label) + 2 item-summary tokens, float16.
    row_bytes = (32 + 1 + 2) * 16 * 2
    destination = 2048 * row_bytes
    transient = 6 * 2048 * row_bytes

    # One row off full rows: a whole destination bought for six rows of relief.
    assert engaged - bypassed == destination - 6 * row_bytes
    # So the most chunking can ever save is the transient less that buffer.
    assert bypassed - floor == transient - destination - 6 * row_bytes


def test_each_candidate_derives_its_chunk_from_its_own_build_phase() -> None:
    """Candidates differ in build peak, so one shared chunk is not enough.

    The support-only build here peaks well above the joined build. A chunk sized
    for the joined build leaves it overflowing, which would drop the cache plan
    the budget can in fact afford once its own chunk is chosen.
    """
    classifier = _classifier(ensemble_count=1)
    classifier.model.supports_query_chunk_cache = True
    support = torch.randn(1, 999, 8)
    queries = torch.randn(1, 10, 8)

    def chunked(chunk: int):
        def estimate(mode, query_rows, members, feedforward_token_chunk):
            del query_rows, members, feedforward_token_chunk
            # The support-only build is twice the joined build at equal chunks;
            # the cached query phase is negligible either way.
            if mode is PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU:
                return _chunk_estimate(2 * min(chunk, 1000), 1000)
            if mode is PeakMemoryMode.JOINED:
                return _chunk_estimate(min(chunk, 1000), 1000)
            return _chunk_estimate(1, 1000)

        return estimate

    plan = classifier._plan_cuda_execution(
        support,
        queries,
        torch.float32,
        chunked(FULL_FEATURE_ATTENTION_ROWS),
        chunked_estimate=chunked,
        chunked_hard_estimate=chunked,
    )

    # The support-only build is what this plan runs, so its chunk is the one
    # that clears 90% of the budget for *that* phase: 2*chunk < 900.
    assert plan.support_only_cache
    assert plan.feature_attention_row_chunk == 449
