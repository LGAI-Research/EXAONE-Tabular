from __future__ import annotations

import numpy as np
import pytest

from exaonetabular._power_transform import (
    _PowerState,
    _fit_power_transform,
    _power_values,
    _transform_power_transform,
)


def test_power_transform_at_zero_one_and_two() -> None:
    values = np.repeat(np.array([-3., -1., 0., 1., 3.])[:, None], 3, axis=1)
    expected = np.array([
        [-7.5, -3, -np.log(4)],
        [-1.5, -1, -np.log(2)],
        [0, 0, 0],
        [np.log(2), 1, 1.5],
        [np.log(4), 3, 7.5],
    ])

    np.testing.assert_allclose(_power_values(values, np.array([0., 1., 2.])), expected)


@pytest.mark.parametrize(
    "values, expected_lambda, expected",
    [
        (
            [0., 1., 2., 4., 8., 16.], -0.03210997922467354,
            [-1.5173000699, -0.7528219191, -0.3134553282, 0.2319998833, 0.8486624003, 1.5029150335],
        ),
        (
            [0., -1., -2., -4., -8., -16.], 2.032109930253125,
            [1.5173000397, 0.7528219286, 0.3134553487, -0.2319998614, -0.8486623935, -1.5029150621],
        ),
        (
            [-8., -2., -1., 0., 1., 4., 16., np.nan], 0.7962448859410352,
            [-1.8308336584, -0.3981466073, -0.1971401091, -0.0163657179, 0.1380789836, 0.5292334643, 1.7751736449, np.nan],
        ),
    ],
)
def test_power_fit_matches_reference_values(values, expected_lambda, expected) -> None:
    support = np.array(values)[:, None]
    original = support.copy()
    state = _fit_power_transform(support)

    np.testing.assert_allclose(state.lambdas, [expected_lambda], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        _transform_power_transform(support, state)[:, 0], expected, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_array_equal(support, original)


def test_power_fit_handles_constant_and_missing_columns() -> None:
    support = np.array([[3., 0., np.nan], [3., 0., np.nan]])
    state = _fit_power_transform(support)

    np.testing.assert_array_equal(state.lambdas, [1, 1, 1])
    np.testing.assert_allclose(state.centers, [3, 0, 0])
    np.testing.assert_array_equal(state.divisors, [1, 1, 1])
    np.testing.assert_array_equal(
        _transform_power_transform(support, state), [[0, 0, np.nan], [0, 0, np.nan]]
    )
    np.testing.assert_allclose(_transform_power_transform(np.array([[4., 2., 1.]]), state), [[1, 2, 1]])


def test_power_transform_reuses_fit_for_query_batches() -> None:
    support = np.array([0., 1., 2., 4., 8., 16.], dtype=np.float32)[:, None]
    state = _fit_power_transform(support)
    original = tuple(value.copy() for value in (state.lambdas, state.centers, state.divisors))
    query = np.array([[-5.], [np.nan], [3.], [50.]])
    original_query = query.copy()

    whole = _transform_power_transform(query, state)
    split = np.concatenate([_transform_power_transform(chunk, state) for chunk in np.split(query, 2)])

    assert whole.dtype == np.float64
    np.testing.assert_array_equal(split, whole)
    np.testing.assert_array_equal(_transform_power_transform(query, state), whole)
    np.testing.assert_array_equal(query, original_query)
    for actual, expected in zip((state.lambdas, state.centers, state.divisors), original):
        np.testing.assert_array_equal(actual, expected)


def test_power_transform_clamps_overflowing_queries() -> None:
    state = _PowerState(np.array([3., -1.]), np.zeros(2), np.ones(2))
    result = _transform_power_transform(np.array([[1e300, -1e300]]), state)

    np.testing.assert_array_equal(result, [[np.finfo(np.float64).max, -np.finfo(np.float64).max]])


def test_power_fit_handles_single_row() -> None:
    support = np.array([[5., np.nan]])
    state = _fit_power_transform(support)

    np.testing.assert_array_equal(_transform_power_transform(support, state), [[0, np.nan]])


def test_power_transform_accepts_empty_query_rows() -> None:
    state = _fit_power_transform(np.array([[0.], [2.]]))

    assert _transform_power_transform(np.empty((0, 1)), state).shape == (0, 1)


def test_power_transform_rejects_query_width_mismatch() -> None:
    state = _fit_power_transform(np.array([[0.], [2.]]))

    with pytest.raises(ValueError, match="feature width"):
        _transform_power_transform(np.ones((2, 2)), state)
