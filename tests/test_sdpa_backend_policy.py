"""The per-call SDPA kernel choice: which call sites get which kernel, and where.

The policy itself is pure and exhaustively testable anywhere; the kernels it
names are CUDA-only, so the tests that would actually run the mem-efficient
kernel are skipped off a GPU. What a CPU run *can* pin is the mapping from real
forward shapes to kernel choices -- the spy tests below record the sequence
lengths a genuine inference produces and resolve them as if on CUDA.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from exaonetabular.classifier import EXAONETabularClassifier
from exaonetabular.config import InferenceManifest, ModelConfig, RuntimeConfig
from exaonetabular.model import attention as attention_module
from exaonetabular.model.attention import (
    MEM_EFFICIENT_MAX_SEQUENCE,
    TensorAttention,
    _select_sdpa_backend,
)
from exaonetabular.model.heads import ClassificationModel

_FLASH = SDPBackend.FLASH_ATTENTION
_MEM_EFFICIENT = SDPBackend.EFFICIENT_ATTENTION
_LIMIT = MEM_EFFICIENT_MAX_SEQUENCE
_FFN_CHUNK = 2**19

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the pinned kernels are CUDA-only"
)


def _model_config(**overrides) -> ModelConfig:
    values = {
        "width": 16,
        "head_count": 4,
        "block_count": 2,
        "feedforward_widths": 32,
        "columns_per_group": 1,
        "item_summary_count": 2,
        "feature_summary_count": 3,
        "feature_attention_repeats": 1,
        "class_capacity": 5,
    }
    values.update(overrides)
    return ModelConfig(**values)


# The released model is 192 wide over 6 heads. Anything not divisible by 8 is
# refused by the mem-efficient kernel -- see the dedicated test below.
_SERVED_HEAD_WIDTH = 32


def _choose(
    query_length: int,
    context_length: int,
    *,
    policy: str = "auto",
    head_width: int = _SERVED_HEAD_WIDTH,
):
    """Resolve as if on CUDA, which is the only device that has a choice."""
    return _select_sdpa_backend(
        query_length, context_length, head_width, on_cuda=True, policy=policy
    )


def _cuda_choice(call: tuple[int, int, str]):
    """Resolve one recorded call, honouring the policy its module was built with."""
    query_length, context_length, policy = call
    return _choose(query_length, context_length, policy=policy)


def _shapes(calls) -> set[tuple[int, int]]:
    return {(query, context) for query, context, _ in calls}


# --- the policy ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("query_length", "context_length", "expected"),
    [
        (1, 1, _MEM_EFFICIENT),
        (_LIMIT, _LIMIT, _MEM_EFFICIENT),
        (_LIMIT + 1, _LIMIT + 1, _FLASH),
        (2**20, 2**20, _FLASH),
    ],
)
def test_the_cliff_is_one_token_wide(query_length, context_length, expected) -> None:
    """The measured boundary is inclusive: L=64 still wins, L=65 has fallen off."""
    assert _choose(query_length, context_length) is expected


@pytest.mark.parametrize(
    ("query_length", "context_length"),
    [(1, _LIMIT + 1), (_LIMIT + 1, 1), (_LIMIT, _LIMIT + 1), (_LIMIT + 1, _LIMIT)],
)
def test_cross_attention_is_judged_by_its_longer_side(
    query_length, context_length
) -> None:
    """One long side is enough: the kernel runs the whole call, not the short axis."""
    assert _choose(query_length, context_length) is _FLASH
    assert _choose(min(query_length, context_length), _LIMIT) is _MEM_EFFICIENT


@pytest.mark.parametrize("policy", ["auto", "flash", "mem_efficient"])
@pytest.mark.parametrize("length", [1, _LIMIT, _LIMIT + 1, 2**20])
def test_no_device_but_cuda_ever_leaves_flash(policy, length) -> None:
    """Including under an explicit pin -- see the companion test below for why."""
    assert (
        _select_sdpa_backend(
            length, length, _SERVED_HEAD_WIDTH, on_cuda=False, policy=policy
        )
        is _FLASH
    )


def test_pinning_the_mem_efficient_kernel_off_cuda_would_raise() -> None:
    """What the CPU guard is actually protecting, stated as an executable fact.

    Without it every CPU forward in this runtime raises, because SDPA is left
    with no viable kernel rather than falling back to one that exists.
    """
    query = torch.randn(2, 2, 4, 8)
    with pytest.raises(RuntimeError, match="No viable backend"):
        with sdpa_kernel([_MEM_EFFICIENT]):
            F.scaled_dot_product_attention(query, query, query)
    with sdpa_kernel([_FLASH]):  # the pin this runtime has always used
        F.scaled_dot_product_attention(query, query, query)


@pytest.mark.parametrize("head_width", [1, 2, 4, 6, 12, 20])
@pytest.mark.parametrize("policy", ["auto", "mem_efficient"])
def test_a_head_width_the_kernel_refuses_stays_on_flash(head_width, policy) -> None:
    """The mem-efficient kernel rejects head widths not divisible by 8 ("last
    dimension of inputs to be divisible by 8") while flash serves them.

    So this is not a speed preference. Pinning mem-efficient at those widths leaves
    SDPA with no viable kernel and the forward dies with "No available kernel". The
    released checkpoints are head_width 32, so only a custom architecture reaches
    this, and it must reach flash rather than a failed forward.
    """
    assert _choose(1, 1, policy=policy, head_width=head_width) is _FLASH


@pytest.mark.parametrize("head_width", [8, 16, 32, 64, 128])
def test_a_served_head_width_still_takes_the_short_kernel(head_width) -> None:
    assert _choose(1, 1, head_width=head_width) is _MEM_EFFICIENT


def test_an_explicit_pin_overrides_the_length_rule() -> None:
    assert _choose(1, 1, policy="flash") is _FLASH
    assert _choose(2**20, 2**20, policy="mem_efficient") is _MEM_EFFICIENT


@pytest.mark.parametrize(
    ("backend", "error"),
    [(b"flash", TypeError), (None, TypeError), ("cudnn", ValueError), ("", ValueError)],
)
def test_an_unusable_backend_name_is_rejected_at_construction(backend, error) -> None:
    with pytest.raises(error):
        TensorAttention(8, 2, sdpa_backend=backend)


def test_the_supported_feature_range_straddles_the_cliff() -> None:
    """Why this runtime chooses per call instead of pinning a module.

    A module-level pin is only safe when every feature-attention call a runtime
    makes lands on one side of the cliff. The supported feature range straddles it,
    so the choice has to read the shape. If these ever stop straddling -- the limit
    drops, or the crossover is re-measured higher -- a plain pin becomes the simpler
    correct answer again.
    """
    default_limit = RuntimeConfig().feature_limit
    item_summary = ModelConfig().item_summary_count
    assert item_summary + default_limit + 1 > MEM_EFFICIENT_MAX_SEQUENCE
    assert item_summary + 1 + 1 <= MEM_EFFICIENT_MAX_SEQUENCE


def test_the_default_policy_is_per_call() -> None:
    assert TensorAttention(8, 2).sdpa_backend == "auto"
    assert TensorAttention(8, 2, sdpa_backend="flash").sdpa_backend == "flash"


# --- the call sites, under a real forward -------------------------------------


def _record_forward(
    monkeypatch: pytest.MonkeyPatch,
    *,
    support: int,
    query: int,
    features: int,
    config: ModelConfig | None = None,
    **forward_options,
) -> Counter:
    """Run one inference and return the (query, context, policy) calls it made."""
    config = _model_config() if config is None else config
    model = ClassificationModel(config, dtype=torch.float32).eval()
    calls: Counter = Counter()
    real = attention_module._select_sdpa_backend

    def spy(query_length, context_length, head_width, *, on_cuda, policy):
        # These models are built on CPU; the flag has to arrive that way or the
        # guard tested above is never the thing keeping this forward alive.
        assert on_cuda is False
        calls[(query_length, context_length, policy)] += 1
        return real(
            query_length, context_length, head_width, on_cuda=on_cuda, policy=policy
        )

    monkeypatch.setattr(attention_module, "_select_sdpa_backend", spy)
    generator = torch.Generator().manual_seed(0)
    x_support = torch.randn(1, support, features, generator=generator)
    x_query = torch.randn(1, query, features, generator=generator)
    y_support = torch.randint(0, 3, (1, support), generator=generator).float()
    with torch.no_grad():
        model(
            x_support,
            y_support,
            x_query,
            feedforward_token_chunk=_FFN_CHUNK,
            **forward_options,
        )
    assert calls, "the forward made no attention calls"
    return calls


def test_a_narrow_dataset_splits_the_call_sites_across_both_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature attention and item-summary cross-attention take the mem-efficient
    kernel; the three row-length call sites stay on flash. Shapes are derived,
    not spelled, so a reshaped call site fails here rather than silently landing
    on the wrong side of the cliff."""
    config = _model_config()
    support, query, features = 80, 7, 8
    groups = features + 1  # the label group rides alongside the feature groups
    calls = _record_forward(
        monkeypatch, support=support, query=query, features=features, config=config
    )
    rows = support + query
    feature_tokens = config.item_summary_count + groups
    context_rows = config.feature_summary_count + support
    expected = {
        (feature_tokens, feature_tokens): _MEM_EFFICIENT,  # feature self-attention
        (groups, config.item_summary_count): _MEM_EFFICIENT,  # item-summary cross
        (rows, config.feature_summary_count): _FLASH,  # feature-summary cross
        (context_rows, context_rows): _FLASH,  # item self-attention
        (query, context_rows): _FLASH,  # query cross-attention
    }
    resolved = {(call[0], call[1]): _cuda_choice(call) for call in calls}
    for shape, backend in expected.items():
        assert shape in resolved, f"the forward never made a {shape} call"
        assert resolved[shape] is backend, f"{shape} took the wrong kernel"


def test_a_wide_dataset_moves_feature_attention_back_to_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason the choice reads the shape instead of pinning the module.

    RuntimeConfig.feature_limit admits 100 features, so feature attention runs
    on both sides of the cliff: the same call site that took the mem-efficient
    kernel on a narrow table crosses back to flash on a wide one.
    """
    config = _model_config()
    features = _LIMIT  # tokens = item_summary_count + features + 1 > the limit
    calls = _record_forward(
        monkeypatch, support=20, query=7, features=features, config=config
    )
    feature_tokens = config.item_summary_count + features + 1

    assert feature_tokens > _LIMIT
    assert (feature_tokens, feature_tokens) in _shapes(calls)
    assert _choose(feature_tokens, feature_tokens) is _FLASH


def test_a_tiny_support_set_moves_item_attention_to_the_short_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliberate consequence of reading the shape: with few support rows every
    unpinned call sits below the cliff, on the faster side of the measured curve."""
    calls = _record_forward(monkeypatch, support=4, query=3, features=2)
    unpinned = [call for call in calls if call[2] == "auto"]
    assert unpinned and all(
        _cuda_choice(call) is _MEM_EFFICIENT for call in unpinned
    ), f"expected every short call to take the mem-efficient kernel: {sorted(calls)}"


@pytest.mark.parametrize(
    ("support", "query", "features"),
    [(1, 1, 1), (1, 64, 1), (64, 1, 1), (3, 2, 100)],
)
def test_degenerate_shapes_still_produce_one_choice_per_call(
    monkeypatch: pytest.MonkeyPatch, support, query, features
) -> None:
    """Single rows, single features and the widest allowed input all forward
    cleanly, and every call resolves to a kernel rather than to None."""
    calls = _record_forward(
        monkeypatch, support=support, query=query, features=features
    )
    assert all(_cuda_choice(call) in (_FLASH, _MEM_EFFICIENT) for call in calls)


def test_row_chunking_never_changes_a_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predictions must not depend on the memory budget.

    The row chunk is chosen from available VRAM, and for the feature-summary
    cross-attention rows *are* the sequence axis -- so a chunk small enough to
    cross the cliff would otherwise pick a different kernel, and a bf16 GPU run
    would answer differently on a fuller card. That call site is pinned instead,
    which is why the shapes below may move while the kernels may not.
    """
    shared = {"support": 40, "query": 9, "features": 8}
    whole = _record_forward(monkeypatch, **shared)
    chunked = _record_forward(monkeypatch, **shared, feature_attention_row_chunk=7)

    moved = _shapes(whole) ^ _shapes(chunked)
    assert moved, "chunk too coarse to move any sequence length; test proves nothing"

    def resolved(calls, shapes):
        return {
            (call[0], call[1]): _cuda_choice(call)
            for call in calls
            if (call[0], call[1]) in shapes
        }

    assert set(resolved(set(whole) | set(chunked), moved).values()) == {_FLASH}
    assert resolved(whole, _shapes(whole) - moved) == resolved(
        chunked, _shapes(chunked) - moved
    )


# --- results ------------------------------------------------------------------


def _attention_pair(
    embedding_dim: int = 16, head_count: int = 4, **options
) -> tuple[TensorAttention, TensorAttention]:
    torch.manual_seed(0)
    policy = TensorAttention(embedding_dim, head_count, **options)
    pinned = TensorAttention(
        embedding_dim, head_count, sdpa_backend="flash", **options
    )
    pinned.load_state_dict(policy.state_dict())
    return policy.eval(), pinned.eval()


@pytest.mark.parametrize("sequence", [8, _LIMIT, _LIMIT + 1])
def test_cpu_results_are_untouched_by_the_policy(sequence) -> None:
    """Every CPU call resolves to flash, so the policy is bit-for-bit inert here
    -- a CPU forward answers identically with or without it."""
    policy, pinned = _attention_pair()
    query = torch.randn(3, sequence, 16)
    with torch.no_grad():
        assert torch.equal(policy(query), pinned(query))


def test_the_flat_batch_chunk_agrees_with_the_unchunked_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chunk exists for flash's launch grid; a per-call kernel choice must
    not have made it disagree with the path it stands in for."""
    policy, _ = _attention_pair()
    query = torch.randn(9, 4, 16)
    with torch.no_grad():
        whole = policy(query)
        monkeypatch.setattr(attention_module, "FLASH_ATTENTION_BATCH_LIMIT", 2)
        chunked = policy(query)
    assert torch.equal(whole, chunked)


@_needs_cuda
def test_a_mem_efficient_call_is_not_split_by_the_flash_grid_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split serves flash's launch grid. Mem-efficient has no such ceiling,
    so splitting it would only add a concatenation to the largest tensor here."""
    attention = TensorAttention(32, 4, sdpa_backend="mem_efficient").cuda().eval()
    query = torch.randn(9, 4, 32, device="cuda")
    calls = 0
    unpatched = F.scaled_dot_product_attention

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return unpatched(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", counted)
    monkeypatch.setattr(attention_module, "FLASH_ATTENTION_BATCH_LIMIT", 2)
    with torch.no_grad():
        attention(query)
    assert calls == 1


def test_a_config_can_pin_the_whole_model_back_to_flash() -> None:
    """The documented way out: the previous behaviour stays reachable without a
    code edit, for a GPU where the measured crossover does not hold."""
    import dataclasses

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class PinnedConfig(ModelConfig):
        sdpa_backend: str = "flash"

    values = _model_config().to_dict()
    values["feedforward_widths"] = list(values["feedforward_widths"])
    model = ClassificationModel(PinnedConfig(**values), dtype=torch.float32)
    backends = {
        module.sdpa_backend
        for module in model.modules()
        if isinstance(module, TensorAttention)
    }
    assert backends == {"flash"}


def test_only_the_row_sequence_call_site_is_pinned_by_default() -> None:
    layers = ClassificationModel(_model_config()).transformer.layers
    assert {
        module.sdpa_backend
        for layer in layers
        for module in layer.feature_context_attentions
    } == {"flash"}
    assert {
        module.sdpa_backend
        for layer in layers
        for module in layer.feature_attentions
    } == {"auto"}
    assert {layer.item_attention.sdpa_backend for layer in layers} == {"auto"}


def test_example_inference_runs_under_the_policy() -> None:
    """End to end through the public estimator, on both sides of the cliff."""
    manifest = InferenceManifest(
        checkpoint_sha256="0" * 64,
        model=_model_config(),
        runtime=RuntimeConfig(
            support_row_limit=128,
            feature_limit=100,
            query_batch_limit=16,
            ensemble_count=2,
            compute_dtype="float32",
        ),
    )
    generator = np.random.default_rng(0)
    for features in (8, _LIMIT + 8):
        classifier = EXAONETabularClassifier(manifest)
        x = generator.normal(size=(40, features))
        y = generator.integers(0, 3, size=40)
        probabilities = classifier.fit(x, y).predict_proba(generator.normal(size=(6, features)))
        assert probabilities.shape == (6, 3)
        assert np.all(np.isfinite(probabilities))
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


# --- CUDA ---------------------------------------------------------------------


# The released model is 192 wide over 6 heads; these keep that head width rather
# than the toy 4 the CPU tests use, so the GPU kernels see a shape they ship with.
_CUDA_WIDTH, _CUDA_HEADS = 64, 2


@_needs_cuda
@pytest.mark.parametrize("sequence", [8, _LIMIT, _LIMIT + 1])
def test_cuda_agrees_with_the_flash_pin(sequence) -> None:
    """The kernels are not bit-identical in bf16, so this is a tolerance check,
    at the ~1e-2 relative floor two bf16 implementations already share."""
    policy, pinned = _attention_pair(
        _CUDA_WIDTH, _CUDA_HEADS, device="cuda", dtype=torch.bfloat16
    )
    query = torch.randn(512, sequence, _CUDA_WIDTH, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        result, reference = policy(query), pinned(query)
    expected = _MEM_EFFICIENT if sequence <= _LIMIT else _FLASH
    assert _choose(sequence, sequence) is expected
    torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)


@_needs_cuda
@pytest.mark.parametrize("features", [8, _LIMIT + 8])
def test_cuda_runs_a_full_forward_on_both_sides_of_the_cliff(features) -> None:
    """No call site trips "no viable kernel" on a real GPU forward -- the failure
    mode a strict pin produces when a shape it cannot serve reaches it."""
    config = _model_config(
        width=_CUDA_WIDTH, head_count=_CUDA_HEADS, feedforward_widths=_CUDA_WIDTH
    )
    model = ClassificationModel(config, device="cuda", dtype=torch.bfloat16).eval()
    x_support = torch.randn(1, 40, features, device="cuda")
    x_query = torch.randn(1, 6, features, device="cuda")
    y_support = torch.randint(0, 3, (1, 40), device="cuda").float()
    with torch.no_grad():
        output = model(
            x_support, y_support, x_query, feedforward_token_chunk=_FFN_CHUNK
        )
    assert torch.isfinite(output).all()
