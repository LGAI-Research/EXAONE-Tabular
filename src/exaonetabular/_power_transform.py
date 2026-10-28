# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .preprocessing import TabularPreprocessor


@dataclass(frozen=True)
class _PowerState:
    lambdas: np.ndarray
    centers: np.ndarray
    divisors: np.ndarray


def _power_values(values: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    magnitude = np.log1p(np.abs(values))
    exponent = np.where(values >= 0, lambdas, 2 - lambdas)
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.expm1(exponent * magnitude)
        np.divide(
            transformed, exponent, out=transformed,
            where=np.abs(exponent) >= np.finfo(np.float64).eps,
        )
    transformed = np.where(
        np.abs(exponent) < np.finfo(np.float64).eps, magnitude, transformed
    )
    return np.copysign(transformed, values)


def _constant_columns(
    variance: np.ndarray, centers: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    return variance <= counts * eps * variance + (counts * centers * eps) ** 2


def _power_score(
    values: np.ndarray,
    lambdas: np.ndarray,
    counts: np.ndarray,
    log_jacobian: np.ndarray,
) -> np.ndarray:
    transformed = _power_values(values, lambdas)
    centers = np.nansum(transformed, axis=0) / counts
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        variance = np.nansum((transformed - centers) ** 2, axis=0) / counts
        score = -counts / 2 * np.log(variance) + (lambdas - 1) * log_jacobian
    valid = np.isfinite(variance) & (variance >= np.finfo(np.float64).tiny)
    return np.where(valid, score, -np.inf)


def _fit_power_transform(features: np.ndarray) -> _PowerState:
    values = TabularPreprocessor._validate_features(
        features, allow_zero_rows=False
    ).astype(np.float64)
    counts = np.maximum(np.isfinite(values).sum(axis=0), 1)
    centers = np.nansum(values, axis=0) / counts
    variance = np.nansum((values - centers) ** 2, axis=0) / counts
    constant = _constant_columns(variance, centers, counts)

    max_abs = np.max(np.nan_to_num(np.abs(values), nan=0.0), axis=0)
    log_range = np.where(max_abs == 0, 1.0, np.log1p(20 * max_abs))
    finfo = np.finfo(np.float64)
    log_eps = math.log(finfo.eps)
    lower = (math.log(finfo.tiny) - log_eps) / (2 * log_range)
    upper = (math.log(finfo.max) + log_eps) / (2 * log_range)
    negative = values < 0
    mixed_lower = np.maximum(2 - upper, lower)
    mixed_upper = np.minimum(2 - mixed_lower, upper)
    left = np.where(negative.any(axis=0), mixed_lower, lower)
    right = np.where(negative.any(axis=0), mixed_upper, upper)
    all_negative = (negative | np.isnan(values)).all(axis=0)
    left = np.where(all_negative, 2 - upper, left)
    right = np.where(all_negative, 2 - lower, right)
    left = np.where(constant, 1.0, left)
    right = np.where(constant, 1.0, right)

    log_jacobian = np.nansum(np.copysign(np.log1p(np.abs(values)), values), axis=0)
    ratio = (math.sqrt(5) - 1) / 2
    span = (right - left) * ratio
    c, d = right - span, left + span
    fc = _power_score(values, c, counts, log_jacobian)
    fd = _power_score(values, d, counts, log_jacobian)
    for _ in range(44):
        choose_right = fc < fd
        left = np.where(choose_right, c, left)
        right = np.where(choose_right, right, d)
        span = (right - left) * ratio
        next_c = np.where(choose_right, d, right - span)
        next_d = np.where(choose_right, left + span, c)
        score = _power_score(
            values, np.where(choose_right, next_d, next_c), counts, log_jacobian
        )
        fc, fd = np.where(choose_right, fd, score), np.where(choose_right, score, fc)
        c, d = next_c, next_d
    lambdas = np.where(constant, 1.0, (left + right) / 2)

    transformed = _power_values(values, lambdas)
    centers = np.nansum(transformed, axis=0) / counts
    variance = np.nansum((transformed - centers) ** 2, axis=0) / counts
    divisors = np.sqrt(variance)
    divisors[_constant_columns(variance, centers, counts)] = 1.0
    return _PowerState(lambdas, centers, divisors)


def _transform_power_transform(features: np.ndarray, state: _PowerState) -> np.ndarray:
    values = TabularPreprocessor._validate_features(features, allow_zero_rows=True)
    if values.shape[1] != state.lambdas.size:
        raise ValueError("feature width differs from fitted width")
    transformed = _power_values(values.astype(np.float64), state.lambdas)
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = (transformed - state.centers) / state.divisors
    bound = np.finfo(np.float64).max
    return np.clip(transformed, -bound, bound)
