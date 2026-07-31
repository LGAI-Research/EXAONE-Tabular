from __future__ import annotations

import copy
import types

import torch

from exaonetabular.config import ModelConfig
from exaonetabular.model.attention import TensorAttention
from exaonetabular.model.heads import ClassificationModel
from exaonetabular.model.memory import PeakMemoryMode

_FFN_CHUNK = 524_288


def _small_model() -> ClassificationModel:
    config = ModelConfig(
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
    return ClassificationModel(config).eval()


def test_cached_query_chunks_match_repeated_full_support() -> None:
    torch.manual_seed(11)
    baseline = _small_model()
    cached = copy.deepcopy(baseline)
    support = torch.randn(2, 8, 3)
    labels = torch.arange(8).remainder(2).float().expand(2, -1).clone()
    queries = torch.randn(2, 7, 3)

    with torch.inference_mode():
        expected = torch.cat(
            [
                baseline(
                    support,
                    labels,
                    queries[:, start : start + 3],
                    feedforward_token_chunk=_FFN_CHUNK,
                    query_chunk_size=3,
                )
                for start in range(0, queries.shape[1], 3)
            ],
            dim=1,
        )
        cached(
            support,
            labels,
            queries[:, :3],
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=3,
            cache_support=True,
        )
        actual = torch.cat(
            [
                cached(
                    support,
                    labels,
                    queries[:, start : start + 3],
                    feedforward_token_chunk=_FFN_CHUNK,
                    query_chunk_size=3,
                    use_cached_support=True,
                )
                for start in range(0, queries.shape[1], 3)
            ],
            dim=1,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert all(
        layer.item_attention._cached_key.shape[-3] == 1
        for layer in cached.transformer.layers
    )
    cached.clear_support_cache()
    assert not cached.transformer.has_cached_support


def test_cpu_offloaded_cache_matches_device_cache() -> None:
    torch.manual_seed(17)
    device_cached = _small_model()
    cpu_cached = copy.deepcopy(device_cached)
    support = torch.randn(1, 9, 4)
    labels = torch.arange(9).remainder(3).float().unsqueeze(0)
    queries = torch.randn(1, 6, 4)

    with torch.inference_mode():
        first_device = device_cached(
            support,
            labels,
            queries[:, :3],
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=3,
            cache_support=True,
        )
        first_cpu = cpu_cached(
            support,
            labels,
            queries[:, :3],
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=3,
            cache_support=True,
            cache_support_to_cpu=True,
        )
        next_device = device_cached(
            support,
            labels,
            queries[:, 3:],
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=3,
            use_cached_support=True,
        )
        next_cpu = cpu_cached(
            support,
            labels,
            queries[:, 3:],
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=3,
            use_cached_support=True,
        )

    torch.testing.assert_close(first_cpu, first_device, rtol=0, atol=0)
    torch.testing.assert_close(next_cpu, next_device, rtol=0, atol=0)
    assert all(
        layer.item_attention._cached_key.device.type == "cpu"
        for layer in cpu_cached.transformer.layers
    )


def test_compact_cached_execution_matches_full_query_rows() -> None:
    torch.manual_seed(23)
    baseline = _small_model()
    compact = copy.deepcopy(baseline)
    support = torch.randn(1, 9, 4)
    labels = torch.arange(9).remainder(3).float().unsqueeze(0)
    queries = torch.randn(1, 7, 4)

    with torch.inference_mode():
        expected = baseline(
            support,
            labels,
            queries[:, 6:],
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=3,
        )
        compact(
            support,
            labels,
            queries[:, :3],
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=3,
            cache_support=True,
            cache_support_to_cpu=True,
        )
        actual = compact(
            support,
            labels,
            queries[:, 6:],
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=3,
            use_cached_support=True,
            compact_cached_support=True,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_support_cache_guard_rejects_query_dependent_feature_mask() -> None:
    support = torch.randn(1, 8, 3)
    assert ClassificationModel.can_cache_support(support)

    support[:, :, 1] = 4.0
    assert not ClassificationModel.can_cache_support(support)


def test_query_chunk_guard_accepts_stable_constant_feature_mask() -> None:
    model = _small_model()
    support = torch.randn(1, 8, 3)
    support[:, :, 0] = 4.0
    queries = torch.randn(1, 6, 3)
    queries[:, :, 0] = 5.0
    assert model.can_cache_query_chunks(support, queries, 3)

    queries[:, 3:, 0] = 4.0
    assert not model.can_cache_query_chunks(support, queries, 3)

    queries = torch.randn(1, 7, 3)
    queries[:, :, 0] = 5.0
    assert model.can_cache_query_chunks(support, queries, 3)
    queries[:, -1, 0] = 4.0
    assert not model.can_cache_query_chunks(support, queries, 3)

    queries = torch.randn(1, 5, 3)
    queries[:, :, 0] = 5.0
    assert model.can_cache_query_chunks(support, queries, 3)
    queries[:, 3:, 0] = 4.0
    assert not model.can_cache_query_chunks(support, queries, 3)


def test_support_cache_peak_estimate_scales_with_member_count() -> None:
    model = _small_model()
    one_member = model.estimated_peak(
        support_rows=8,
        query_rows=4,
        feature_count=3,
        members=1,
        feedforward_token_chunk=_FFN_CHUNK,
        mode=PeakMemoryMode.CACHE_BUILD_GPU,
        max_vram_bytes=2**50,
    )
    two_members = model.estimated_peak(
        support_rows=8,
        query_rows=4,
        feature_count=3,
        members=2,
        feedforward_token_chunk=_FFN_CHUNK,
        mode=PeakMemoryMode.CACHE_BUILD_GPU,
        max_vram_bytes=2**50,
    )
    assert one_member.allocated_bytes > 0
    assert two_members.allocated_bytes == 2 * one_member.allocated_bytes


def test_standard_output_skips_only_unused_final_support_attention() -> None:
    torch.manual_seed(31)
    model = _small_model()
    support = torch.randn(1, 9, 64)
    labels = torch.arange(9).remainder(3).float().unsqueeze(0)
    queries = torch.randn(1, 5, 64)

    joined = torch.cat((support, queries), dim=1).reshape(1, 14, 64, 1)
    feature_state = model.feature_encoder(joined[0], 9).unsqueeze(0)
    label_input = torch.cat(
        (
            labels.transpose(0, 1).unsqueeze(-1),
            labels.new_full((5, 1, 1), float("nan")),
        ),
        dim=0,
    )
    label_state = model.label_encoder(label_input, 9).transpose(0, 1)
    state = torch.cat((feature_state, label_state.unsqueeze(2)), dim=2)

    with torch.inference_mode():
        complete = model.transformer(
            state.clone(),
            9,
            feedforward_token_chunk=_FFN_CHUNK,
            output="all",
        )
        standard = model.transformer(
            state.clone(),
            9,
            feedforward_token_chunk=_FFN_CHUNK,
            output="standard",
        )

    torch.testing.assert_close(
        standard, complete["standard"], rtol=1e-5, atol=1e-6
    )


def test_trusted_internal_inputs_preserve_public_forward_output() -> None:
    torch.manual_seed(37)
    model = _small_model()
    support = torch.randn(1, 11, 4)
    labels = torch.arange(11).remainder(3).float().unsqueeze(0)
    queries = torch.randn(1, 5, 4)

    with torch.inference_mode():
        expected = model(
            support,
            labels,
            queries,
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=5,
        )
        actual = model(
            support,
            labels,
            queries,
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=5,
            trusted_internal_inputs=True,
        )

    assert torch.equal(actual, expected)


def test_final_group_pruning_preserves_feature_attention_capture() -> None:
    torch.manual_seed(41)
    model = _small_model()
    attention = model.transformer.layers[-1].feature_attentions[-1]
    original = attention.forward
    contexts: list[torch.Tensor | None] = []

    def captured(self, query, context=None, **kwargs):
        contexts.append(context)
        return original(query, context, **kwargs)

    attention.forward = types.MethodType(captured, attention)
    support = torch.randn(1, 9, 4)
    labels = torch.arange(9).remainder(3).float().unsqueeze(0)
    queries = torch.randn(1, 5, 4)
    try:
        with torch.inference_mode():
            output = model(
                support,
                labels,
                queries,
                feedforward_token_chunk=_FFN_CHUNK,
                query_chunk_size=5,
            )
    finally:
        delattr(attention, "forward")

    assert output.shape == (1, 5, 3)
    assert contexts
    assert all(context is None for context in contexts)


def test_packed_inference_projections_match_separate_reference() -> None:
    torch.manual_seed(43)
    attention = TensorAttention(16, 4).eval()
    query = torch.randn(2, 7, 16)
    context = torch.randn(2, 11, 16)

    with torch.enable_grad():
        self_reference = attention(query).detach()
        cross_reference = attention(
            query, context, reuse_first_context_head=True
        ).detach()
        attention.cache_context_projections(context, first_head_only=True)
        key_reference = attention._cached_key.detach().clone()
        value_reference = attention._cached_value.detach().clone()
    attention.clear_cache()

    with torch.inference_mode():
        self_packed = attention(query)
        cross_packed = attention(
            query, context, reuse_first_context_head=True
        )
        attention.cache_context_projections(context, first_head_only=True)

    torch.testing.assert_close(self_packed, self_reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cross_packed, cross_reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        attention._cached_key, key_reference, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        attention._cached_value, value_reference, rtol=1e-5, atol=1e-6
    )


def test_support_only_cache_matches_joined_query_execution() -> None:
    torch.manual_seed(47)
    baseline = _small_model()
    cached = copy.deepcopy(baseline)
    support = torch.randn(1, 9, 4)
    labels = torch.arange(9).remainder(3).float().unsqueeze(0)
    queries = torch.randn(1, 7, 4)

    with torch.inference_mode():
        expected = baseline(
            support,
            labels,
            queries,
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=7,
        )
        cached(
            support,
            labels,
            queries[:, :0],
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=7,
            cache_support=True,
            trusted_internal_inputs=True,
        )
        actual = cached(
            support,
            labels,
            queries,
            feedforward_token_chunk=_FFN_CHUNK,
            query_chunk_size=7,
            use_cached_support=True,
            compact_cached_support=True,
            trusted_internal_inputs=True,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert cached.estimated_peak(
        support_rows=9,
        query_rows=0,
        feature_count=4,
        members=1,
        feedforward_token_chunk=_FFN_CHUNK,
        mode=PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU,
        max_vram_bytes=2**50,
    ).allocated_bytes > 0
