from __future__ import annotations

import torch

from exaonetabular.feature_selection.selection import _AttentionCapture


def _reference_full_reduction(
    attention: torch.Tensor, values: torch.Tensor, item_summary_count: int
) -> dict[str, torch.Tensor]:
    """The same reduction, taken over the full attention matrix.

    Kept here rather than in the package: shipped code only ever needs the reduced
    statistics, so this exists solely to pin the reduced path to the naive math.
    """
    group_count = attention.size(-1) - item_summary_count - 1
    feature_slice = slice(item_summary_count, item_summary_count + group_count)
    return {
        "sumi_to_feat": attention[:, :, :item_summary_count, feature_slice]
        .sum(dim=-2)
        .float(),
        "y_attn": attention[:, :, -1, feature_slice].float(),
        "v_norm_feat": values.float().norm(dim=-1)[:, :, feature_slice],
    }


def _capture(attention: torch.Tensor, values: torch.Tensor) -> dict[str, torch.Tensor]:
    capture = _AttentionCapture(1, 3)
    capture.consume_selected_rows(attention, values, 0)
    return capture.per_layer[0]


def test_capture_matches_the_reference_full_reduction() -> None:
    """The exact chunked path feeds the whole attention matrix to the capture."""
    generator = torch.Generator().manual_seed(11)
    attention = torch.softmax(torch.randn(2, 4, 9, 9, generator=generator), dim=-1)
    values = torch.randn(2, 4, 9, 6, generator=generator)

    expected = _reference_full_reduction(attention, values, 3)
    captured = _capture(attention, values)

    for name in ("sumi_to_feat", "y_attn", "v_norm_feat"):
        assert torch.equal(captured[name], expected[name])


def test_selected_row_capture_matches_full_reduction() -> None:
    """The cuDNN path feeds only the summary and target rows, and must agree."""
    generator = torch.Generator().manual_seed(11)
    attention = torch.softmax(torch.randn(2, 4, 9, 9, generator=generator), dim=-1)
    values = torch.randn(2, 4, 9, 6, generator=generator)

    expected = _reference_full_reduction(attention, values, 3)
    selected_rows = torch.cat((attention[:, :, :3], attention[:, :, -1:]), dim=-2)
    captured = _capture(selected_rows, values)

    for name in ("sumi_to_feat", "y_attn", "v_norm_feat"):
        assert torch.equal(captured[name], expected[name])
