"""The frozen feature-selection config, and how the classifier applies it."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from exaonetabular import (
    FEATURE_SELECTION,
    AttentionScorer,
    EXAONETabularClassifier,
    FeatureSelectionConfig,
    InferenceManifest,
    ModelConfig,
    RuntimeConfig,
)


def _manifest(feature_limit: int = 100) -> InferenceManifest:
    return InferenceManifest(
        checkpoint_sha256="0" * 64,
        task="classification",
        model=ModelConfig(
            width=12,
            head_count=3,
            block_count=2,
            feedforward_widths=24,
            item_summary_count=3,
            feature_summary_count=4,
            class_capacity=4,
        ),
        runtime=RuntimeConfig(
            ensemble_count=2,
            compute_dtype="float32",
            seed=0,
            feature_limit=feature_limit,
        ),
    )


def _table(rows: int, columns: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, columns)), rng.integers(0, 3, size=rows)


# --- the frozen values -------------------------------------------------------


def test_frozen_config_matches_the_released_selection() -> None:
    """Pinned field-by-field: these values define the released selection behavior."""
    assert FEATURE_SELECTION.target_feature_count == 100
    assert FEATURE_SELECTION.pre_pass_row_limit == 512
    assert FEATURE_SELECTION.seed == 42
    assert FEATURE_SELECTION.scorers == (
        AttentionScorer(aggregation="y_attn", layer_reduce="last"),
        AttentionScorer(aggregation="sumi_to_feat", layer_reduce="mean"),
    )


def test_selection_is_not_checkpoint_scoped() -> None:
    """It describes the architecture's token layout, so no checkpoint record owns it."""
    from exaonetabular import presets

    assert not hasattr(presets, "RELEASED_FEATURE_SELECTION")
    assert not hasattr(presets, "released_feature_selection")
    assert not any(
        f.name == "feature_selection" for f in fields(presets.ReleasedCheckpoint)
    )


def test_frozen_seed_is_not_the_runtime_seed() -> None:
    assert FEATURE_SELECTION.seed != _manifest().runtime.seed


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"aggregation": "col_sum", "layer_reduce": "mean"},
        {"aggregation": "y_attn", "layer_reduce": "max"},
        {"aggregation": 3, "layer_reduce": "mean"},
    ],
)
def test_scorer_rejects_unsupported_values(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        AttentionScorer(**kwargs)


def test_config_rejects_empty_scorers() -> None:
    with pytest.raises(ValueError):
        FeatureSelectionConfig(scorers=())


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        FeatureSelectionConfig.from_dict({"rollout": {"enabled": True}})


def test_config_rejects_non_scorer_entries() -> None:
    with pytest.raises(TypeError):
        FeatureSelectionConfig(scorers=({"aggregation": "y_attn"},))


def test_feature_selection_package_surface_is_one_function() -> None:
    """The package exposes one function, and no stale module can resurface beside it.

    A setuptools `build/` staging directory left over from an earlier layout can put
    modules into the wheel that the source tree no longer has, so assert the shipped
    surface rather than trusting the tree.
    """
    from pathlib import Path

    import exaonetabular.feature_selection as fs

    assert fs.__all__ == ["select_features"]
    shipped = sorted(p.name for p in Path(fs.__file__).parent.glob("*.py"))
    assert shipped == ["__init__.py", "selection.py"], shipped


# --- how the classifier applies it -------------------------------------------


def test_narrow_table_skips_selection() -> None:
    features, targets = _table(40, 20)
    classifier = EXAONETabularClassifier(_manifest(), device="cpu").fit(
        features, targets
    )
    assert classifier.selected_feature_indices_ is None
    columns = classifier._state()["preprocessor"].selected_columns
    assert np.array_equal(columns, np.arange(20))


def test_wide_table_is_narrowed_to_the_target() -> None:
    features, targets = _table(60, 130)
    classifier = EXAONETabularClassifier(_manifest(), device="cpu").fit(
        features, targets
    )
    selected = classifier.selected_feature_indices_
    assert selected.shape == (100,)
    assert selected.dtype == np.int64
    assert np.all(np.diff(selected) > 0)          # sorted, no duplicates
    assert selected.max() < 130
    # The preprocessor was handed exactly those columns rather than running its own
    # f_classif truncation.
    assert np.array_equal(
        classifier._state()["preprocessor"].selected_columns, selected
    )


def test_selection_does_not_change_the_public_width() -> None:
    features, targets = _table(60, 130)
    classifier = EXAONETabularClassifier(_manifest(), device="cpu").fit(
        features, targets
    )
    assert classifier.n_features_in_ == 130
    probabilities = classifier.predict_proba(features[:5])
    assert probabilities.shape == (5, 3)
    with pytest.raises(ValueError):
        classifier.predict_proba(features[:5, :100])


def test_selection_target_respects_a_tighter_feature_limit() -> None:
    """feature_limit is the model's hard width, so it wins when it is smaller."""
    features, targets = _table(40, 30)
    classifier = EXAONETabularClassifier(_manifest(feature_limit=8), device="cpu").fit(
        features, targets
    )
    assert classifier.selected_feature_indices_.shape == (8,)


def test_selected_indices_are_a_copy() -> None:
    features, targets = _table(60, 130)
    classifier = EXAONETabularClassifier(_manifest(), device="cpu").fit(
        features, targets
    )
    first = classifier.selected_feature_indices_
    first[0] = -1
    assert classifier.selected_feature_indices_[0] != -1
