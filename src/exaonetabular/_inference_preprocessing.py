# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from ._power_transform import _fit_power_transform, _transform_power_transform
from .preprocessing import (
    TabularPreprocessor,
    _fit_numeric_base,
    _resolve_categorical_mask,
    _transform_numeric_base,
)


def _fit_robust(values):
    valid = np.isfinite(values).any(axis=0)
    median = np.full(values.shape[1], np.nan)
    scale = median.copy()
    if valid.any():
        lower, middle, upper = np.nanquantile(values[:, valid], [0.25, 0.5, 0.75], axis=0)
        median[valid] = middle
        scale[valid] = np.where(lower == upper, 1.0, upper - lower)
    return median, scale


def _transform_robust(values, state):
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = (values - state[0]) / state[1]
        unit = scaled / np.hypot(3.0, scaled)
    unit = np.where(np.isinf(scaled), np.sign(scaled), unit)
    return 3.0 * unit


def _ndtri(values):
    return torch.special.ndtri(torch.as_tensor(values, dtype=torch.float64)).numpy()


def _fit_rank(values, max_knots=8192):
    knots = []
    for column in values.T:
        observed = column[np.isfinite(column)]
        x, counts = np.unique(observed, return_counts=True)
        if not x.size:
            knots.append((np.array([np.nan]), np.array([np.nan])))
            continue
        probabilities = (np.cumsum(counts) - counts / 2) / observed.size
        probabilities = np.minimum(probabilities, 1 - np.finfo(np.float64).eps)
        if x.size > max_knots:
            normal = np.linspace(_ndtri(probabilities[0]), _ndtri(probabilities[-1]), max_knots)
            target = torch.special.ndtr(torch.from_numpy(normal)).numpy()
            above = np.minimum(np.searchsorted(probabilities, target), x.size - 1)
            below = np.maximum(above - 1, 0)
            closer = target - probabilities[below] < probabilities[above] - target
            positions = np.where(closer, below, above)
            positions[0], positions[-1] = 0, x.size - 1
            x, probabilities = x[positions], probabilities[positions]
        knots.append((x, probabilities))
    return tuple(knots)


def _transform_rank(values, state):
    output = np.empty_like(values, dtype=np.float64)
    for index, (knots, probabilities) in enumerate(state):
        output[:, index] = _ndtri(np.interp(values[:, index], knots, probabilities))
    output[np.isnan(values)] = np.nan
    return output


def _fit_sigma(values):
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    mean = np.where(finite, values, 0.0).sum(axis=0) / np.maximum(counts, 1)
    centered = np.where(finite, values - mean, 0.0)
    scale = np.maximum(np.sqrt(np.square(centered).sum(axis=0) / np.maximum(counts - 1, 1)), 1e-6)
    keep = finite & (values >= mean - 4 * scale) & (values <= mean + 4 * scale)
    kept_counts = keep.sum(axis=0)
    kept_mean = np.where(keep, values, 0.0).sum(axis=0) / np.maximum(kept_counts, 1)
    centered = np.where(keep, values - kept_mean, 0.0)
    kept_scale = np.maximum(
        np.sqrt(np.square(centered).sum(axis=0) / np.maximum(kept_counts - 1, 1)), 1e-6
    )
    mean = np.where(kept_counts > 0, kept_mean, mean)
    scale = np.where(kept_counts > 0, kept_scale, scale)
    return mean - 4 * scale, mean + 4 * scale


def _transform_sigma(values, state):
    margin = np.log1p(np.abs(values))
    return np.minimum(np.maximum(values, state[0] - margin), state[1] + margin)


class _NumericBlock:
    def __init__(self, values, *, gaussianize=False):
        self.mapper = None
        self.base = None
        self.rules = []
        if values.shape[1] == 0:
            self.width = 0
            return
        if gaussianize:
            self.mapper = QuantileTransformer(
                n_quantiles=min(values.shape[0], 1000), output_distribution="normal", random_state=0
            ).fit(values)
            values = self.mapper.transform(values)
        self.base = _fit_numeric_base(values)
        normalized = _transform_numeric_base(values, self.base).values
        self.width = normalized.shape[1]
        if not self.width:
            return
        for rule in range(4):
            state = (
                None if rule == 0 else
                _fit_power_transform(normalized) if rule == 1 else
                _fit_robust(normalized) if rule == 2 else
                _fit_rank(normalized)
            )
            transformed = self._apply(normalized, rule, state)
            self.rules.append((state, _fit_sigma(transformed)))

    @staticmethod
    def _apply(values, rule, state):
        if rule == 1:
            return _transform_power_transform(values, state)
        if rule == 2:
            return _transform_robust(values, state)
        if rule == 3:
            return _transform_rank(values, state)
        return values

    def transform(self, values, rule):
        if not self.width:
            return np.empty((values.shape[0], 0), dtype=np.float64)
        if self.mapper is not None and values.shape[0]:
            values = self.mapper.transform(values)
        normalized = _transform_numeric_base(values, self.base).values
        state, sigma = self.rules[rule]
        return _transform_sigma(self._apply(normalized, rule, state), sigma)


class _FeatureEnsemble:
    def __init__(self, members, seed, feature_limit=500, *, device="cpu", gaussianize=False, rescale=False, task=None, class_count=None):
        self.members, self.seed = members, seed
        self.feature_limit = min(feature_limit, 500)
        self.device = torch.device(device)
        self.gaussianize, self.rescale = gaussianize, rescale
        self.task, self.class_count = task, class_count

    def fit(self, features, categorical_mask=None):
        matrix = TabularPreprocessor._validate_features(features, allow_zero_rows=False)
        self.input_width = matrix.shape[1]
        self.categorical_mask = _resolve_categorical_mask(matrix, categorical_mask)
        self.categories = []
        self.counts = []
        for column in matrix[:, self.categorical_mask].T:
            categories, counts = np.unique(column[~np.isnan(column)], return_counts=True)
            self.categories.append(categories)
            self.counts.append(np.append(counts, np.isnan(column).sum()))
        self.numeric = _NumericBlock(matrix[:, ~self.categorical_mask], gaussianize=self.gaussianize)
        self.categorical = _NumericBlock(self._encode_categories(matrix))
        width = self.numeric.width + self.categorical.width
        if width == 0:
            raise ValueError("features has no nonconstant columns")
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.target_rng_state = generator.get_state()
        if self.task == "regression":
            for _ in range(self.members):
                torch.empty((1, 1), device=self.device, dtype=torch.float32).bernoulli_(0.5, generator=generator)
        elif self.task == "classification" and self.class_count is not None and self.class_count > 1:
            for _ in range(0, self.members, self.class_count):
                torch.randperm(self.class_count, generator=generator, device=self.device)
        signs = []
        for _ in range(self.members):
            sign = torch.empty((1, self.numeric.width), dtype=torch.float64, device=self.device)
            if matrix[:, ~self.categorical_mask].shape[1]:
                sign.bernoulli_(0.5, generator=generator).mul_(-2).add_(1)
            signs.append(sign.cpu().numpy()[0])
        self.signs = np.stack(signs)
        base = torch.randperm(width, generator=generator, device=self.device).cpu().numpy()
        rows = torch.randperm(width, generator=generator, device=self.device).cpu().numpy()
        self.orders = np.stack([
            base[(member % width - rows) % width][:self.feature_limit]
            for member in range(self.members)
        ])
        self.rng_state = generator.get_state()
        self.support_members = self.transform(matrix)
        return self

    def _encode_categories(self, matrix):
        encoded, frequencies = [], []
        for column, categories, counts in zip(
            matrix[:, self.categorical_mask].T, self.categories, self.counts
        ):
            codes = np.full(column.size, -1.0)
            slots = np.full(column.size, categories.size, dtype=np.int64)
            if categories.size:
                positions = np.searchsorted(categories, column)
                bounded = np.minimum(positions, categories.size - 1)
                known = ~np.isnan(column) & (positions < categories.size) & (categories[bounded] == column)
                codes[known] = positions[known]
                slots[known] = positions[known]
            encoded.append(codes)
            if categories.size > 50:
                frequencies.append(np.log1p(counts[slots].astype(np.float32)).astype(np.float64))
        columns = frequencies + encoded
        return np.column_stack(columns) if columns else np.empty((matrix.shape[0], 0))

    def transform(self, features):
        matrix = TabularPreprocessor._validate_features(features, allow_zero_rows=True)
        if matrix.shape[1] != self.input_width:
            raise ValueError("feature width differs from fitted width")
        numeric = matrix[:, ~self.categorical_mask]
        categorical = self._encode_categories(matrix)
        cache = {}
        output = []
        for member in range(self.members):
            rule = member % 4
            if rule not in cache:
                cache[rule] = (self.numeric.transform(numeric, rule), self.categorical.transform(categorical, rule))
            num, cat = cache[rule]
            values = np.concatenate((num * self.signs[member], cat), axis=1)
            values = values[:, self.orders[member]]
            if self.rescale:
                values = values * (self.feature_limit / values.shape[1])
            output.append(values.astype(np.float32))
        return np.stack(output)
