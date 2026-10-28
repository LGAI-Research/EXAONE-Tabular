from __future__ import annotations

import numpy as np
import pytest

from exaonetabular.classifier import EXAONETabularClassifier
from exaonetabular.config import InferenceManifest, ModelConfig, RegressionConfig, RuntimeConfig
from exaonetabular.preprocessing import _resolve_categorical_mask
from exaonetabular.regressor import EXAONETabularRegressor


def test_inference_ignores_nan_and_preserves_numeric_precision() -> None:
    support = np.column_stack(
        (
            np.arange(12) % 9,
            np.arange(12) % 10,
            np.full(12, np.nan),
            np.ones(12),
            np.arange(12, dtype=np.float64) + 2**24,
        )
    )
    support[9:, 0] = np.nan
    support[11, 1] = np.nan
    original = support.copy()

    mask = _resolve_categorical_mask(support)

    np.testing.assert_array_equal(mask, [True, False, True, True, False])
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(support, original)


@pytest.mark.parametrize("dtype", [np.bool_, np.int64])
def test_inference_accepts_boolean_and_integer_features(dtype) -> None:
    support = np.array([[0, 1], [1, 0]], dtype=dtype)

    np.testing.assert_array_equal(_resolve_categorical_mask(support), [True, True])


def test_explicit_mask_overrides_inference_and_is_copied() -> None:
    support = np.column_stack((np.arange(12), np.zeros(12)))
    original = support.copy()
    explicit = np.array([True, False])

    mask = _resolve_categorical_mask(support, explicit)

    np.testing.assert_array_equal(mask, explicit)
    mask[:] = False
    np.testing.assert_array_equal(explicit, [True, False])
    np.testing.assert_array_equal(support, original)


@pytest.mark.parametrize(
    "mask, error",
    [
        ([True, False], TypeError),
        (np.array([1, 0]), TypeError),
        (np.array([1.0, 0.0]), TypeError),
        (np.array(True), ValueError),
        (np.array([[True, False]]), ValueError),
        (np.array([True]), ValueError),
        (np.array([], dtype=np.bool_), ValueError),
    ],
)
def test_invalid_mask_is_rejected(mask, error) -> None:
    with pytest.raises(error, match="categorical_mask"):
        _resolve_categorical_mask(np.ones((3, 2)), mask)


@pytest.fixture
def classifier() -> EXAONETabularClassifier:
    return EXAONETabularClassifier(
        InferenceManifest(
            model=ModelConfig(width=8, head_count=2, block_count=1, feedforward_widths=16),
            runtime=RuntimeConfig(
                feature_limit=2, ensemble_count=1, compute_dtype="float32"
            ),
        )
    )


@pytest.mark.parametrize("explicit", [None, np.array([False, True])])
def test_classifier_fit_stores_resolved_mask(classifier, explicit) -> None:
    support = np.column_stack((np.arange(12) % 3, np.arange(12)))
    expected = np.array([True, False]) if explicit is None else explicit.copy()

    assert classifier.fit(support, np.arange(12) % 2, categorical_mask=explicit) is classifier
    if explicit is not None:
        explicit[:] = ~explicit

    np.testing.assert_array_equal(classifier._state()["categorical_mask"], expected)


def test_classifier_mask_uses_original_columns(classifier) -> None:
    support = np.column_stack(
        (np.arange(12) % 3, np.arange(12), np.arange(12) % 2, np.arange(12) ** 2)
    )

    classifier.fit(
        support, np.arange(12) % 2, categorical_mask=np.array([True, False, True, False])
    )

    np.testing.assert_array_equal(
        classifier._state()["categorical_mask"], [True, False, True, False]
    )
    assert classifier._state()["preprocessor"].support_members.shape == (1, 12, 2)


def test_invalid_classifier_mask_preserves_fit_and_skips_selection(
    classifier, monkeypatch
) -> None:
    classifier.fit(np.column_stack((np.arange(12), np.arange(12) ** 2)), np.arange(12) % 2)
    previous = classifier._state()

    def unexpected_selection(*args, **kwargs):
        pytest.fail("selection ran before mask validation")

    monkeypatch.setattr("exaonetabular.classifier._FeatureEnsemble.fit", unexpected_selection)

    with pytest.raises(ValueError, match="categorical_mask"):
        classifier.fit(
            np.ones((12, 4)), np.arange(12) % 2, categorical_mask=np.array([True, False])
        )

    assert classifier._state() is previous


@pytest.fixture
def regressor() -> EXAONETabularRegressor:
    return EXAONETabularRegressor(
        InferenceManifest(
            task="regression",
            model=ModelConfig(width=8, head_count=2, block_count=1, feedforward_widths=16),
            runtime=RuntimeConfig(
                feature_limit=2, ensemble_count=1, compute_dtype="float32"
            ),
            regression=RegressionConfig(
                quantile_count=9, decoder_hidden_width=12,
                n_svd=0, member_weighting="uniform",
            ),
        )
    )


@pytest.mark.parametrize("explicit", [None, np.array([False, True])])
def test_regressor_fit_stores_resolved_mask(regressor, explicit) -> None:
    support = np.column_stack((np.arange(12) % 3, np.arange(12)))
    expected = np.array([True, False]) if explicit is None else explicit.copy()

    assert regressor.fit(support, np.arange(12), categorical_mask=explicit) is regressor
    if explicit is not None:
        explicit[:] = ~explicit

    np.testing.assert_array_equal(regressor._state()["categorical_mask"], expected)


def test_regressor_mask_uses_original_columns(regressor) -> None:
    support = np.column_stack(
        (np.arange(12) % 3, np.arange(12), np.arange(12) % 2, np.arange(12) ** 2)
    )

    regressor.fit(
        support, np.arange(12), categorical_mask=np.array([True, False, True, False])
    )

    np.testing.assert_array_equal(
        regressor._state()["categorical_mask"], [True, False, True, False]
    )
    assert regressor._state()["preprocessor"].support_members.shape == (1, 12, 2)


def test_invalid_regressor_mask_preserves_fit_and_skips_selection(
    regressor, monkeypatch
) -> None:
    regressor.fit(np.column_stack((np.arange(12), np.arange(12) ** 2)), np.arange(12))
    previous = regressor._state()

    def unexpected_selection(*args, **kwargs):
        pytest.fail("selection ran before mask validation")

    monkeypatch.setattr("exaonetabular.regressor._FeatureEnsemble.fit", unexpected_selection)

    with pytest.raises(ValueError, match="categorical_mask"):
        regressor.fit(
            np.ones((12, 4)), np.arange(12), categorical_mask=np.array([True, False])
        )

    assert regressor._state() is previous
