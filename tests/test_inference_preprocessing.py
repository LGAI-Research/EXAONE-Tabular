from __future__ import annotations

import numpy as np
import pytest
import torch

from exaonetabular import (
    EXAONETabularClassifier, EXAONETabularRegressor, InferenceManifest,
    ModelConfig, RegressionConfig, RuntimeConfig, released_manifest,
)
from exaonetabular._inference_preprocessing import (
    _FeatureEnsemble, _fit_rank, _fit_robust, _fit_sigma,
    _transform_rank, _transform_robust, _transform_sigma,
)
from exaonetabular.ensemble import EnsemblePlan, build_ensemble_inputs


def test_robust_scaling_and_saturation():
    state = _fit_robust(np.array([[0.], [1.], [2.], [3.], [100.], [np.nan]]))
    np.testing.assert_array_equal(state[0], [2])
    np.testing.assert_array_equal(state[1], [2])
    actual = _transform_robust(np.array([[2.], [4.], [8.], [np.nan]]), state)
    np.testing.assert_allclose(actual[:, 0], [0, 3 / np.sqrt(10), 3 / np.sqrt(2), np.nan])


def test_rank_normalization_uses_tied_mid_ranks_and_interpolation():
    state = _fit_rank(np.array([[0.], [0.], [2.], [6.], [np.nan]]))
    query = np.array([[0.], [1.], [2.], [6.], [-99.], [99.], [np.nan]])
    probabilities = torch.tensor([.25, .4375, .625, .875, .25, .875, float("nan")], dtype=torch.float64)
    np.testing.assert_allclose(_transform_rank(query, state)[:, 0], torch.special.ndtri(probabilities).numpy())


def test_rank_grid_is_bounded_and_preserves_endpoints():
    state = _fit_rank(np.arange(10000, dtype=np.float64)[:, None])
    assert len(state[0][0]) == 8192
    np.testing.assert_array_equal(state[0][0][[0, -1]], [0, 9999])
    assert np.all(np.diff(_transform_rank(np.arange(10000)[:, None], state)[:, 0]) >= 0)


def test_sigma_clipping_refits_without_outlier():
    values = np.array([0.] * 100 + [100.])[:, None]
    state = _fit_sigma(values)
    np.testing.assert_allclose(state, [[-4e-6], [4e-6]])
    result = _transform_sigma(np.array([[-100.], [100.], [np.nan]]), state)
    np.testing.assert_allclose(result[:, 0], [-np.log1p(100) - 4e-6, np.log1p(100) + 4e-6, np.nan])


def test_category_counts_and_unknown_values():
    support = (np.arange(122) % 51).astype(np.float64)[:, None]
    support[-2:] = np.nan
    processor = _FeatureEnsemble(8, 0).fit(support, np.array([True]))
    encoded = processor._encode_categories(np.array([[0.], [999.], [np.nan]]))
    expected_counts = np.log1p(np.array([3, 2, 2], dtype=np.float32))
    np.testing.assert_allclose(encoded[:, 0], expected_counts)
    np.testing.assert_array_equal(encoded[:, 1], [0, -1, -1])
    assert processor.support_members.shape == (8, 122, 2)


def test_feature_state_is_reused_for_query_batches():
    rng = np.random.default_rng(9)
    support = rng.normal(size=(40, 4))
    support[:, 1] = np.arange(40) % 3
    processor = _FeatureEnsemble(8, 17).fit(support, np.array([False, True, False, False]))
    query = rng.normal(size=(8, 4))
    query[0] = np.nan
    whole = processor.transform(query)
    split = np.concatenate([processor.transform(chunk) for chunk in np.split(query, 2)], axis=1)
    np.testing.assert_array_equal(split, whole)
    np.testing.assert_array_equal(processor.transform(query), whole)


def test_all_constant_features_are_rejected():
    with pytest.raises(ValueError, match="no nonconstant columns"):
        _FeatureEnsemble(8, 0).fit(np.ones((12, 3)))


def _regressor(**options):
    return EXAONETabularRegressor(InferenceManifest(
        task="regression",
        model=ModelConfig(width=8, head_count=2, block_count=1, feedforward_widths=16),
        runtime=RuntimeConfig(ensemble_count=8, compute_dtype="float32", query_batch_limit=3),
        regression=RegressionConfig(quantile_count=3, decoder_hidden_width=12, **options),
    ))


def test_regression_trims_members_at_each_restored_quantile(monkeypatch):
    regressor = _regressor()
    support = np.arange(12, dtype=np.float64)[:, None]
    regressor.fit(support, np.arange(12) + 10.)
    state = regressor._state()
    assert state["center"] == 15.5
    np.testing.assert_allclose(state["scale"], np.sqrt(143 / 12), rtol=1e-6)
    query = np.array([[3.]])
    _, labels, _, plan = build_ensemble_inputs(
        torch.from_numpy(support), torch.from_numpy(state["support_y"]), torch.from_numpy(query),
        EnsemblePlan(members=8, seed=0, task="regression"), preprocessor=state["preprocessor"],
    )
    expected_quantiles = torch.tensor([
        [100., 0., -100.], [1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
        [10., 11., 12.], [13., 14., 15.], [16., 17., 18.], [19., 20., 1000.],
    ])[:, None, :]
    def forward(xs, ys, xq):
        torch.testing.assert_close(ys, labels)
        return expected_quantiles * plan.target_signs[:, None, None]
    monkeypatch.setattr(regressor, "_forward_chunked", forward)
    result = regressor.predict(query)
    expected = ((8.5 + 9.5 + 160 / 6) / 3) * state["scale"] + state["center"]
    np.testing.assert_allclose(result, [expected], rtol=1e-6)


def test_constant_regression_target_round_trips(monkeypatch):
    regressor = _regressor()
    regressor.fit(np.arange(12, dtype=np.float64)[:, None], np.full(12, 7.))
    assert regressor._state()["scale"] == 1
    np.testing.assert_array_equal(regressor._state()["support_y"], np.zeros(12))
    monkeypatch.setattr(regressor, "_forward_chunked", lambda xs, ys, xq: torch.zeros((8, xq.shape[1], 3)))
    np.testing.assert_array_equal(regressor.predict(np.array([[2.], [3.]])), [7, 7])


@pytest.mark.parametrize("readout", ["mean", "median", "trimmed"])
def test_svd_split_and_nnls_remain_available(readout):
    regressor = _regressor(
        n_svd=1, svd_split=True, member_weighting="nnls",
        nnls_min_validation_rows=2, point_estimate=readout,
    )
    rng = np.random.default_rng(7)
    regressor.fit(rng.normal(size=(30, 3)), rng.normal(size=30), categorical_mask=np.zeros(3, bool))
    state = regressor._state()
    assert len(state["passes"]) == 2
    assert state["member_weights"].shape == (16,)
    np.testing.assert_allclose(state["member_weights"].sum(), 1)
    assert (state["member_weights"] >= 0).all()
    assert np.isfinite(regressor.predict(rng.normal(size=(4, 3)))).all()


def test_classifier_ecoc_and_repeated_queries(monkeypatch):
    classifier = EXAONETabularClassifier(InferenceManifest(
        model=ModelConfig(width=8, head_count=2, block_count=1, feedforward_widths=16, class_capacity=3),
        runtime=RuntimeConfig(ensemble_count=8, compute_dtype="float32"),
    ))
    rng = np.random.default_rng(8)
    classifier.fit(rng.normal(size=(30, 3)), np.arange(30) % 7)
    def unexpected_fit(*args, **kwargs):
        pytest.fail("prediction refitted preprocessing")
    monkeypatch.setattr(_FeatureEnsemble, "fit", unexpected_fit)
    query = rng.normal(size=(4, 3))
    probabilities = classifier.predict_proba(query)
    assert probabilities.shape == (4, 7)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(classifier.predict_proba(query), probabilities)


def test_released_defaults_and_config_round_trip():
    for task in ("classification", "regression"):
        manifest = released_manifest(task)
        assert manifest.runtime.feature_limit == 500
        assert InferenceManifest.from_dict(manifest.to_dict()) == manifest
    config = released_manifest("regression").regression
    assert config.n_svd == 0 and not config.svd_split
    assert config.member_weighting == "trimmed" and config.point_estimate == "mean"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_large_integer_categories_keep_distinct_query_codes(task, monkeypatch):
    estimator = _regressor() if task == "regression" else EXAONETabularClassifier(InferenceManifest(
        model=ModelConfig(width=8, head_count=2, block_count=1, feedforward_widths=16, class_capacity=2),
        runtime=RuntimeConfig(ensemble_count=8, compute_dtype="float32"),
    ))
    support = (2**60 + np.arange(12, dtype=np.int64) % 3)[:, None]
    targets = np.arange(12, dtype=np.float64) if task == "regression" else np.arange(12) % 2
    estimator.fit(support, targets, categorical_mask=np.array([True]))
    seen = []
    def forward(xs, ys, xq):
        seen.append(xq)
        return torch.zeros((8, xq.shape[1], 3 if task == "regression" else 2))
    monkeypatch.setattr(estimator, "_forward_chunked", forward)
    query = (2**60 + np.array([0, 1, 100], dtype=np.int64))[:, None]
    if task == "regression":
        estimator.predict(query)
    else:
        estimator.predict_proba(query)
    assert (seen[0][:, 0] != seen[0][:, 1]).all()
