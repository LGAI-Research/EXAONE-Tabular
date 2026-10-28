from __future__ import annotations

import numpy as np
import pytest

from exaonetabular.preprocessing import _fit_numeric_base, _transform_numeric_base


def test_numeric_base_uses_observed_support_statistics() -> None:
    support = np.array([[1, 7, np.nan, 2], [3, 7, np.nan, np.nan], [5, 7, np.nan, 2]])
    query = np.array([[7, 999, 42, np.nan], [-1, -8, np.nan, 3]])
    original_support, original_query = support.copy(), query.copy()

    state = _fit_numeric_base(support)
    prepared = _transform_numeric_base(query, state)

    np.testing.assert_array_equal(state.columns, [0, 3])
    np.testing.assert_array_equal(state.centers, [3, 2])
    divisor = np.sqrt(8 / 3) + 1e-6
    np.testing.assert_allclose(state.divisors, [divisor, 1e-6])
    np.testing.assert_allclose(
        prepared.values, [[4 / divisor, np.nan], [-4 / divisor, 100]]
    )
    np.testing.assert_array_equal(prepared.missing_mask, [[False, True], [False, False]])
    np.testing.assert_array_equal(prepared.source_columns, [0, 3])
    prepared.source_columns[:] = 0
    np.testing.assert_array_equal(state.columns, [0, 3])
    np.testing.assert_array_equal(support, original_support)
    np.testing.assert_array_equal(query, original_query)


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.float32, np.float64])
def test_numeric_base_returns_double_precision(dtype) -> None:
    support = np.array([[0, 1], [1, 0]], dtype=dtype)
    state = _fit_numeric_base(support)
    prepared = _transform_numeric_base(support, state)

    assert state.centers.dtype == np.float64
    assert state.divisors.dtype == np.float64
    assert prepared.values.dtype == np.float64
    magnitude = 0.5 / (0.5 + 1e-6)
    np.testing.assert_allclose(prepared.values, [[-magnitude, magnitude], [magnitude, -magnitude]])


def test_numeric_base_avoids_single_precision_variance_overflow() -> None:
    support = np.array([[1e20], [2e20], [3e20]], dtype=np.float32)
    prepared = _transform_numeric_base(support, _fit_numeric_base(support))

    np.testing.assert_allclose(
        prepared.values[:, 0], [-np.sqrt(1.5), 0, np.sqrt(1.5)], atol=1e-6
    )


def test_numeric_base_reuses_fit_for_query_batches() -> None:
    state = _fit_numeric_base(np.array([[0.0], [2.0]]))
    query = np.array([[-1e9], [np.nan], [2.0], [1e9]])
    expected = np.array([[-100], [np.nan], [1 / (1 + 1e-6)], [100]])

    whole = _transform_numeric_base(query, state).values
    split = np.concatenate(
        [_transform_numeric_base(chunk, state).values for chunk in np.split(query, 2)]
    )

    np.testing.assert_allclose(whole, expected)
    np.testing.assert_array_equal(split, whole)
    np.testing.assert_array_equal(_transform_numeric_base(query, state).values, whole)


def test_numeric_base_preserves_single_row_columns() -> None:
    support = np.array([[5.0, np.nan]])
    state = _fit_numeric_base(support)

    np.testing.assert_array_equal(state.columns, [0, 1])
    np.testing.assert_array_equal(state.centers, [5, 0])
    np.testing.assert_array_equal(state.divisors, [1e-6, 1e-6])
    np.testing.assert_array_equal(_transform_numeric_base(support, state).values, [[0, np.nan]])


def test_numeric_base_can_drop_all_columns() -> None:
    support = np.array([[1.0, np.nan], [1.0, np.nan]])
    state = _fit_numeric_base(support)
    prepared = _transform_numeric_base(np.array([[4.0, 2.0]]), state)

    assert prepared.values.shape == prepared.missing_mask.shape == (1, 0)
    assert prepared.source_columns.size == 0


def test_numeric_base_accepts_empty_query_rows() -> None:
    state = _fit_numeric_base(np.array([[0.0, 3.0], [2.0, 3.0]]))
    prepared = _transform_numeric_base(np.empty((0, 2)), state)

    assert prepared.values.shape == prepared.missing_mask.shape == (0, 1)


def test_numeric_base_rejects_query_width_mismatch() -> None:
    state = _fit_numeric_base(np.array([[0.0], [2.0]]))

    with pytest.raises(ValueError, match="feature width"):
        _transform_numeric_base(np.ones((2, 2)), state)
