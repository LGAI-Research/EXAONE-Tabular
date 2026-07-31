from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

from exaonetabular._execution import (
    _CudaExecutionPlan,
    _InferenceExecutor,
)
from exaonetabular.runtime import FULL_FEATURE_ATTENTION_ROWS
from exaonetabular.classifier import EXAONETabularClassifier
from exaonetabular.config import (
    InferenceManifest,
    ModelConfig,
    RegressionConfig,
    RuntimeConfig,
)
from exaonetabular.model.heads import ClassificationModel, RegressionModel
from exaonetabular.model.memory import estimated_peak
from exaonetabular.regressor import EXAONETabularRegressor


def _model_config() -> ModelConfig:
    return ModelConfig(
        width=16,
        head_count=4,
        block_count=2,
        feedforward_widths=32,
        columns_per_group=1,
        item_summary_count=2,
        feature_summary_count=3,
        feature_attention_repeats=1,
        class_capacity=5,
    )


def _regression_config() -> RegressionConfig:
    return RegressionConfig(
        quantile_count=9,
        decoder_hidden_width=12,
        fixed_head_scale=0.43,
    )


def _manifest(task: str) -> InferenceManifest:
    options = {
        "checkpoint_sha256": "0" * 64,
        "task": task,
        "model": _model_config(),
        "runtime": RuntimeConfig(
            support_row_limit=64,
            feature_limit=8,
            query_batch_limit=3,
            ensemble_count=2,
            compute_dtype="float32",
        ),
    }
    if task == "regression":
        options["regression"] = _regression_config()
    return InferenceManifest(**options)


def _model(task: str):
    if task == "classification":
        return ClassificationModel(_model_config()).eval()
    return RegressionModel(_model_config(), _regression_config()).eval()


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize(
    "row_chunk",
    [1, 3, FULL_FEATURE_ATTENTION_ROWS],
    ids=["row_chunk=1", "row_chunk=3", "row_chunk=full"],
)
@pytest.mark.parametrize(
    "plan",
    [
        _CudaExecutionPlan(ensemble_step=1, query_step=3, feedforward_token_chunk=256),
        _CudaExecutionPlan(
            ensemble_step=1,
            query_step=7,
            feedforward_token_chunk=256,
            support_only_cache=True,
            compact_cache=True,
        ),
        _CudaExecutionPlan(
            ensemble_step=1,
            query_step=3,
            feedforward_token_chunk=256,
            query_cache=True,
            compact_cache=True,
        ),
        _CudaExecutionPlan(
            ensemble_step=1,
            query_step=3,
            feedforward_token_chunk=256,
            query_cache=True,
        ),
        _CudaExecutionPlan(
            ensemble_step=1,
            query_step=3,
            feedforward_token_chunk=256,
            query_cache=True,
            cache_to_cpu=True,
            compact_cache=True,
        ),
    ],
)
def test_all_execution_strategies_preserve_both_model_outputs(
    task: str,
    plan: _CudaExecutionPlan,
    row_chunk: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(101)
    baseline = _model(task)
    candidate = copy.deepcopy(baseline)
    support = torch.randn(2, 11, 5)
    query = torch.randn(2, 7, 5)
    labels = (
        torch.arange(11).remainder(3).float().expand(2, -1).clone()
        if task == "classification"
        else torch.randn(2, 11)
    )
    # Rows are a batch axis in feature attention, so chunking them must not move
    # the output. The baseline runs un-chunked (default full rows); the executor
    # drives the model with feature_attention_row_chunk in {1, 3, full}. On CPU
    # (this test executes on CPU tensors) the chunked path is bit-identical.
    plan = dataclasses.replace(plan, feature_attention_row_chunk=row_chunk)
    executor = _InferenceExecutor(
        _manifest(task), device="cuda:0", max_vram_bytes=None
    )
    monkeypatch.setattr(executor, "cuda_budget", lambda: (2**50, 0, 0, 2**50))
    monkeypatch.setattr(executor, "empty_cuda_cache", lambda: None)
    # Pin the whole plan, not just plan_cuda_execution: the row chunk is chosen
    # inside _plan_with_budget, so patching only the inner call would hand the
    # model a re-derived (here: full-rows) chunk and silently test nothing.
    monkeypatch.setattr(
        executor, "_plan_with_budget", lambda *args, **options: plan
    )
    # Record what the model is actually driven with, so a plan field that stops
    # reaching the forward fails here instead of passing vacuously.
    executed_chunks: list[int] = []
    model_forward = candidate.forward

    def recording_forward(*args, **options):
        executed_chunks.append(options["feature_attention_row_chunk"])
        return model_forward(*args, **options)

    monkeypatch.setattr(candidate, "forward", recording_forward)

    with torch.inference_mode():
        expected = baseline(
            support,
            labels,
            query,
            feedforward_token_chunk=256,
            query_chunk_size=7,
        )
        actual = executor.forward(candidate, support, labels, query)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert executed_chunks and set(executed_chunks) == {row_chunk}
    assert not candidate.transformer.has_cached_support


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_max_vram_requires_cuda_for_both_estimators(task: str) -> None:
    manifest = _manifest(task)
    estimator = (
        EXAONETabularClassifier
        if task == "classification"
        else EXAONETabularRegressor
    )
    with pytest.raises(ValueError, match="CUDA"):
        estimator(manifest, max_vram_bytes=1024)


def test_output_memory_coefficients_are_task_neutral() -> None:
    common_model = ModelConfig(
        width=16,
        head_count=4,
        block_count=2,
        feedforward_widths=12,
        columns_per_group=1,
        item_summary_count=2,
        feature_summary_count=3,
        feature_attention_repeats=1,
        class_capacity=9,
    )
    regression = RegressionConfig(
        quantile_count=9,
        decoder_hidden_width=12,
        fixed_head_scale=0.43,
    )
    classifier = ClassificationModel(common_model)
    regressor = RegressionModel(common_model, regression)
    classification_manifest = InferenceManifest(
        checkpoint_sha256="0" * 64,
        model=common_model,
    )
    regression_manifest = InferenceManifest(
        checkpoint_sha256="0" * 64,
        task="regression",
        model=common_model,
        regression=regression,
    )
    assert classification_manifest.output_width == 9
    assert regression_manifest.output_width == 9
    assert classification_manifest.decoder_hidden_width == 12
    assert regression_manifest.decoder_hidden_width == 12
    classification_options = _InferenceExecutor(
        classification_manifest, device="cuda:0", max_vram_bytes=None
    ).output_memory_options(
        ensemble_count=3,
        query_count=128,
        compute_dtype=torch.float16,
    )
    regression_options = _InferenceExecutor(
        regression_manifest, device="cuda:0", max_vram_bytes=None
    ).output_memory_options(
        ensemble_count=3,
        query_count=128,
        compute_dtype=torch.float16,
    )
    assert classification_options == regression_options == {
        "output_width": 9,
        "decoder_hidden_width": 12,
        "outer_output_bytes": 3 * 128 * 9 * 2,
    }
    options = {
        "support_rows": 32,
        "query_rows": 128,
        "feature_count": 7,
        "members": 3,
        "feedforward_token_chunk": 2_048,
        "mode": "joined",
        "max_vram_bytes": 2**50,
    }
    assert classifier.estimated_peak(**options) == regressor.estimated_peak(**options)


def test_regression_planner_chunks_for_wide_quantile_decoder() -> None:
    config = ModelConfig()
    regression = RegressionConfig()
    manifest = InferenceManifest(
        checkpoint_sha256="0" * 64,
        task="regression",
        model=config,
        runtime=RuntimeConfig(ensemble_count=4, compute_dtype="float16"),
        regression=regression,
    )
    model = RegressionModel(config, regression)
    model.supports_query_chunk_cache = False
    executor = _InferenceExecutor(
        manifest, device="cuda:0", max_vram_bytes=None
    )
    support = torch.empty(4, 1_000, 50, device="meta")
    query = torch.empty(4, 5_000, 50, device="meta")
    outer_output = 4 * 5_000 * 999 * 2
    common = {
        "support_rows": 1_000,
        "feature_count": 50,
        "dtype": torch.float16,
        "output_width": 999,
        "decoder_hidden_width": 384,
        "outer_output_bytes": outer_output,
    }
    target = estimated_peak(
        config,
        query_rows=1_000,
        members=1,
        feedforward_token_chunk=1,
        mode="joined",
        max_vram_bytes=2**50,
        **common,
    ).projected_reserved_bytes

    def estimate(mode, query_rows, members, feedforward_token_chunk):
        return estimated_peak(
            config,
            query_rows=query_rows,
            members=members,
            feedforward_token_chunk=feedforward_token_chunk,
            mode=mode,
            max_vram_bytes=target,
            **common,
        )

    plan = executor.plan_cuda_execution(
        model, support, query, torch.float16, estimate
    )
    assert plan.ensemble_step == 1
    assert 1 <= plan.query_step < query.shape[1]
    assert plan.feedforward_token_chunk > 0


class _RecordingRegressionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.query_rows: list[int] = []

    def forward(self, support, labels, query, **options):
        del labels, options
        self.query_rows.append(query.shape[1])
        return support.new_zeros(support.shape[0], query.shape[1], 9)


def test_regressor_chunks_the_entire_model_on_cpu() -> None:
    manifest = _manifest("regression")
    model = _RecordingRegressionModel()
    regressor = EXAONETabularRegressor(manifest, model=model)
    features = np.arange(24, dtype=np.float64).reshape(12, 2)
    targets = np.linspace(-2.0, 3.0, 12)
    query = np.arange(20, dtype=np.float64).reshape(10, 2)
    regressor.fit(features, targets)
    predictions = regressor.predict(query)

    assert model.query_rows == [3, 3, 3, 1]
    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()


class _FailingCacheModel(torch.nn.Module):
    supports_query_chunk_cache = True

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cleared = 0

    def forward(self, support, labels, query, **options):
        del labels
        if options.get("use_cached_support"):
            raise RuntimeError("cached query failed")
        return support.new_zeros(support.shape[0], query.shape[1], 3)

    def clear_support_cache(self) -> None:
        self.cleared += 1


def test_cache_is_cleared_when_a_cached_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _InferenceExecutor(
        _manifest("classification"), device="cuda:0", max_vram_bytes=None
    )
    plan = _CudaExecutionPlan(
        ensemble_step=1,
        query_step=2,
        feedforward_token_chunk=16,
        query_cache=True,
        compact_cache=True,
    )
    monkeypatch.setattr(executor, "cuda_budget", lambda: (2**50, 0, 0, 2**50))
    monkeypatch.setattr(executor, "empty_cuda_cache", lambda: None)
    monkeypatch.setattr(
        executor, "plan_cuda_execution", lambda *args, **options: plan
    )
    model = _FailingCacheModel()
    support = torch.randn(1, 4, 2)
    labels = torch.zeros(1, 4)
    query = torch.randn(1, 5, 2)

    with pytest.raises(RuntimeError, match="cached query failed"):
        executor.forward(model, support, labels, query)
    assert model.cleared == 1


def _oom(message: str = "CUDA out of memory (test)") -> torch.cuda.OutOfMemoryError:
    return torch.cuda.OutOfMemoryError(message)


def _cuda_executor(monkeypatch: pytest.MonkeyPatch) -> _InferenceExecutor:
    executor = _InferenceExecutor(
        _manifest("classification"), device="cuda:0", max_vram_bytes=None
    )
    monkeypatch.setattr(executor, "cuda_budget", lambda: (1000, 0, 0, 1000))
    monkeypatch.setattr(executor, "empty_cuda_cache", lambda: None)
    return executor


def test_cuda_forward_raises_oom_to_the_caller_without_replanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OOM is the caller's to catch: one plan, one attempt, no swallowing.

    This package is consumed as a library, so a CUDA OOM must surface as the
    original exception rather than being absorbed by a shrink-and-retry loop that
    spends the caller's VRAM and wall-clock on a recovery they did not choose.
    """
    executor = _cuda_executor(monkeypatch)

    plans = {"n": 0}

    def count_plan(*_args, **_kwargs) -> _CudaExecutionPlan:
        plans["n"] += 1
        return _CudaExecutionPlan(
            ensemble_step=1, query_step=1, feedforward_token_chunk=1
        )

    monkeypatch.setattr(executor, "_plan_with_budget", count_plan)

    attempts = {"n": 0}
    raised = _oom("planned forward failed")

    def always_oom(*_args) -> torch.Tensor:
        attempts["n"] += 1
        raise raised

    monkeypatch.setattr(executor, "_run_execution_plan", always_oom)

    support = torch.randn(2, 4, 3)
    query = torch.randn(2, 5, 3)
    labels = torch.zeros(2, 4)

    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        executor.forward(_model("classification"), support, labels, query)

    assert caught.value is raised  # the original error, not a re-wrap
    assert attempts["n"] == 1
    assert plans["n"] == 1


# --- the support-cache offload, which only exists on CUDA -----------------------------


_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the support-cache offload and reload only engage on CUDA",
)


def _cuda_manifest(task: str) -> InferenceManifest:
    """As _manifest, but bf16: the flash pin has no fp32 CUDA kernel for the long
    item-attention sequences these plans exercise."""
    options = {
        "checkpoint_sha256": "0" * 64,
        "task": task,
        "model": _model_config(),
        "runtime": RuntimeConfig(
            support_row_limit=64,
            feature_limit=8,
            query_batch_limit=3,
            ensemble_count=2,
            compute_dtype="bfloat16",
        ),
    }
    if task == "regression":
        options["regression"] = _regression_config()
    return InferenceManifest(**options)


@_needs_cuda
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cuda_support_cache_offload_and_reload_preserve_outputs(
    task: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-tripping the support K/V through host memory must not move the numbers.

    Offload and reload are pure device transfers at a fixed dtype, so this is exact by
    construction -- but asserting `reload_cache` in planner-selection tests alone would
    never catch a reload that stopped reaching the model, so it is checked against a real
    output here. A CPU test cannot close that gap: `_execution` gates the reload on
    `device.type == "cuda"`, and `reload_cached_context` is a no-op when the cache
    already sits on a device of that type.
    """
    torch.manual_seed(101)
    build = {"device": "cuda", "dtype": torch.bfloat16}
    model = (
        ClassificationModel(_model_config(), **build)
        if task == "classification"
        else RegressionModel(_model_config(), _regression_config(), **build)
    ).eval()
    support = torch.randn(2, 11, 5, device="cuda")
    query = torch.randn(2, 7, 5, device="cuda")
    labels = (
        torch.arange(11).remainder(3).float().expand(2, -1).clone().cuda()
        if task == "classification"
        else torch.randn(2, 11, device="cuda")
    )

    common = {"ensemble_step": 1, "query_step": 3, "feedforward_token_chunk": 256}
    plans = {
        # GPU-resident cache: the control, no host round-trip at all.
        "gpu_cache": _CudaExecutionPlan(**common, query_cache=True),
        # Offloaded and streamed back per chunk (the pre-existing fallback).
        "offload_stream": _CudaExecutionPlan(
            **common, query_cache=True, cache_to_cpu=True, compact_cache=True
        ),
        # Offloaded during the build, bulk-reloaded before queries.
        "offload_reload": _CudaExecutionPlan(
            **common,
            query_cache=True,
            cache_to_cpu=True,
            compact_cache=True,
            reload_cache=True,
        ),
    }

    results: dict[str, torch.Tensor] = {}
    reloaded: dict[str, int] = {}
    for name, plan in plans.items():
        candidate = copy.deepcopy(model)
        calls = {"n": 0}
        real_load = candidate.load_support_cache

        def counting_load(device, _calls=calls, _real=real_load):
            _calls["n"] += 1
            return _real(device)

        monkeypatch.setattr(candidate, "load_support_cache", counting_load)
        executor = _InferenceExecutor(
            _cuda_manifest(task), device="cuda:0", max_vram_bytes=None
        )
        monkeypatch.setattr(executor, "cuda_budget", lambda: (2**50, 0, 0, 2**50))
        monkeypatch.setattr(executor, "empty_cuda_cache", lambda: None)
        monkeypatch.setattr(
            executor, "_plan_with_budget", lambda *args, _p=plan, **options: _p
        )
        with torch.inference_mode():
            results[name] = executor.forward(candidate, support, labels, query)
        reloaded[name] = calls["n"]

    # Without this the test would still pass if the reload never ran at all.
    assert reloaded["offload_reload"] > 0, "reload_cache never reached the model"
    assert reloaded["gpu_cache"] == 0 and reloaded["offload_stream"] == 0

    assert torch.equal(results["gpu_cache"], results["offload_reload"]), (
        "offload+reload changed the output relative to a GPU-resident cache"
    )
    assert torch.equal(results["gpu_cache"], results["offload_stream"]), (
        "offload+stream changed the output relative to a GPU-resident cache"
    )
