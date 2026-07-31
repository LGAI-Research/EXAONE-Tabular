from __future__ import annotations

import torch

from exaonetabular.model.encoders import (
    FeatureEncoder,
    _finite_support_mean,
    _nonfinite_codes,
)


def _loop_reference(
    encoder: FeatureEncoder, features: torch.Tensor, support_size: int
) -> torch.Tensor:
    rows, groups, _ = features.shape
    values = features.new_zeros((rows, groups, 1))
    codes = features.new_zeros((rows, groups, 1))
    for group in range(groups):
        source = features[:, group, 0]
        retained = rows > 1 and bool((source[1:] != source[0]).any())
        if not retained:
            continue
        replacement = _finite_support_mean(source[:support_size])
        replaced = torch.where(torch.isfinite(source), source, replacement)
        support = replaced[:support_size]
        mean = support.mean()
        deviation = support.std(correction=1)
        if bool(deviation == 0):
            deviation = torch.ones_like(deviation)
        normalized = ((replaced - mean) / (deviation + 1e-16)).clamp(-100, 100)
        values[:, group, 0] = torch.nan_to_num(
            normalized, nan=0.0, posinf=100.0, neginf=-100.0
        )
        codes[:, group, 0] = _nonfinite_codes(source)
    return encoder._project(torch.cat((values, codes), dim=-1))


def test_single_coordinate_vectorization_matches_loop_with_fp_tolerance() -> None:
    torch.manual_seed(5)
    encoder = FeatureEncoder(1, 16).eval()
    features = torch.randn(31, 7, 1)
    features[:, 1] = 2.0
    features[::5, 3] = torch.nan
    features[:, 6] = torch.nan

    with torch.inference_mode():
        expected = _loop_reference(encoder, features, 23)
        actual = encoder(features, 23)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
