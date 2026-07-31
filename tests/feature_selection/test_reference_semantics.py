"""Focused regression tests for behavior shared with the reference fs_cfg."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from exaonetabular.config import (
    AttentionScorer,
    FeatureSelectionConfig,
    ModelConfig,
)
from exaonetabular.feature_selection import select_features
from exaonetabular.feature_selection import selection
from exaonetabular.model.heads import ClassificationModel


def _config(target: int) -> FeatureSelectionConfig:
    return FeatureSelectionConfig(
        target_feature_count=target,
        scorers=(AttentionScorer(aggregation="y_attn", layer_reduce="last"),),
    )


class ReferenceSemanticsTests(unittest.TestCase):
    @staticmethod
    def model() -> ClassificationModel:
        config = ModelConfig(
            width=12,
            head_count=3,
            block_count=2,
            feedforward_widths=(24, 24),
            columns_per_group=1,
            item_summary_count=3,
            feature_summary_count=4,
            feature_attention_repeats=2,
            class_capacity=4,
        )
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(11)
            return ClassificationModel(config, dtype=torch.float32).eval()

    def test_value_weighted_layer_reduction(self) -> None:
        """Scores are attention x ||V||, reduced over heads then layers."""
        capture = selection._AttentionCapture(2, 1)
        for slot, scale in enumerate((1.0, 3.0)):
            capture.per_layer[slot] = {
                "y_attn": torch.full((1, 2, 2), scale),
                "v_norm_feat": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            }

        last = selection._compute_capture_score(
            capture, AttentionScorer(aggregation="y_attn", layer_reduce="last")
        )
        mean = selection._compute_capture_score(
            capture, AttentionScorer(aggregation="y_attn", layer_reduce="mean")
        )
        # head means of (attention x norm): layer 0 -> (2, 3), layer 1 -> (6, 9)
        self.assertTrue(np.allclose(last, [6.0, 9.0]))
        self.assertTrue(np.allclose(mean, [4.0, 6.0]))

    def test_probability_mean_normalizes_each_scorer(self) -> None:
        combined = selection._probability_mean(
            [np.array([0.0, 1.0, 2.0]), np.array([100.0, 50.0, 0.0])]
        )
        self.assertTrue(np.allclose(combined, [0.5, 0.5, 0.5]))

    def test_narrow_table_is_a_noop(self) -> None:
        rng = np.random.default_rng(30)
        features = rng.normal(size=(20, 5)).astype(np.float32)
        targets = (features[:, 0] > 0).astype(np.int64)
        with mock.patch.object(selection, "_attention_scores") as scores:
            observed = select_features(
                model=self.model(),
                features=features,
                targets=targets,
                config=_config(5),
            )
        scores.assert_not_called()
        self.assertTrue(np.array_equal(observed, np.arange(5)))

    def test_hard_failure_propagates(self) -> None:
        rng = np.random.default_rng(31)
        features = rng.normal(size=(20, 5)).astype(np.float32)
        targets = (features[:, 0] > 0).astype(np.int64)
        with mock.patch.object(
            selection, "_attention_scores", side_effect=RuntimeError("capture failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                select_features(
                    model=self.model(),
                    features=features,
                    targets=targets,
                    config=_config(2),
                )

    def test_cuda_oom_propagates(self) -> None:
        rng = np.random.default_rng(38)
        features = rng.normal(size=(20, 5)).astype(np.float32)
        targets = (features[:, 0] > 0).astype(np.int64)
        with mock.patch.object(
            selection,
            "_attention_scores",
            side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory"),
        ):
            with self.assertRaisesRegex(
                torch.cuda.OutOfMemoryError, "CUDA out of memory"
            ):
                select_features(
                    model=self.model(),
                    features=features,
                    targets=targets,
                    config=_config(2),
                )

    def test_attention_selection_plans_required_feedforward_chunk(self) -> None:
        rng = np.random.default_rng(33)
        features = rng.normal(size=(20, 101)).astype(np.float32)
        targets = (features[:, 0] > 0).astype(np.int64)
        model = self.model()
        with mock.patch.object(model, "forward", wraps=model.forward) as forward:
            selected = select_features(
                model=model,
                features=features,
                targets=targets,
                config=_config(100),
            )
        self.assertEqual(selected.shape, (100,))
        self.assertGreater(forward.call_args.kwargs["feedforward_token_chunk"], 0)

    def test_pre_pass_row_limit_bounds_the_forward(self) -> None:
        rng = np.random.default_rng(34)
        features = rng.normal(size=(60, 101)).astype(np.float32)
        targets = (features[:, 0] > 0).astype(np.int64)
        config = FeatureSelectionConfig(
            target_feature_count=100,
            pre_pass_row_limit=16,
            scorers=(AttentionScorer(aggregation="y_attn", layer_reduce="last"),),
        )
        model = self.model()
        with mock.patch.object(model, "forward", wraps=model.forward) as forward:
            select_features(
                model=model, features=features, targets=targets, config=config
            )
        self.assertEqual(forward.call_args.args[0].shape[1], 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
