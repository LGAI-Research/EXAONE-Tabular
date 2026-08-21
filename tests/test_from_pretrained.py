"""Offline coverage for the ``from_pretrained`` weight-loading path.

Every test avoids the network: Hub downloads are exercised by monkeypatching the
one download seam (``weights._hf_download``), and the end-to-end estimator wiring
uses a tiny locally-saved checkpoint.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from exaonetabular import (
    EXAONETabularClassifier,
    ModelConfig,
    released_manifest,
)
from exaonetabular import weights as weights_module
from exaonetabular.checkpoint import load_classifier_checkpoint
from exaonetabular.config import InferenceManifest, RuntimeConfig
from exaonetabular.model.heads import ClassificationModel
from exaonetabular.presets import released_checkpoint


# --- resolution precedence ------------------------------------------------


def _record_downloads(monkeypatch):
    calls = []

    def fake_download(repo_id, filename, revision, cache_dir):
        calls.append((repo_id, filename, revision, cache_dir))
        return f"/fake/cache/{repo_id}/{filename}"

    monkeypatch.setattr(weights_module, "_hf_download", fake_download)
    return calls


def test_default_source_downloads_pinned_release(monkeypatch):
    calls = _record_downloads(monkeypatch)
    checkpoint = released_checkpoint("classification")

    source = weights_module.resolve_weights(checkpoint)

    assert source.is_default is True
    assert calls == [
        (checkpoint.repo_id, "config.json", checkpoint.revision, None),
        (checkpoint.repo_id, checkpoint.filename, checkpoint.revision, None)
    ]


def test_explicit_local_path_is_used_verbatim(tmp_path, monkeypatch):
    calls = _record_downloads(monkeypatch)
    file = tmp_path / "my.safetensors"
    file.write_bytes(b"weights")

    source = weights_module.resolve_weights(
        released_checkpoint("classification"), str(file)
    )

    assert source == weights_module.WeightSource(path=str(file), is_default=False)
    assert calls == []  # a local file must never trigger a download


def test_env_override_takes_precedence_over_default(tmp_path, monkeypatch):
    calls = _record_downloads(monkeypatch)
    checkpoint = released_checkpoint("classification")
    file = tmp_path / "env.safetensors"
    file.write_bytes(b"weights")
    monkeypatch.setenv(checkpoint.weights_env_var, str(file))

    source = weights_module.resolve_weights(checkpoint)

    assert source.is_default is False
    assert source.path == str(file)
    assert calls == []


def test_repo_id_source_downloads_with_canonical_filename(monkeypatch):
    calls = _record_downloads(monkeypatch)
    checkpoint = released_checkpoint("classification")

    source = weights_module.resolve_weights(checkpoint, "some-org/some-repo")

    assert source.is_default is False
    assert calls == [("some-org/some-repo", checkpoint.filename, None, None)]


def test_filename_override_is_forwarded(monkeypatch):
    calls = _record_downloads(monkeypatch)

    weights_module.resolve_weights(
        released_checkpoint("classification"),
        "some-org/some-repo",
        filename="custom.safetensors",
    )

    assert calls == [("some-org/some-repo", "custom.safetensors", None, None)]


def test_unresolvable_source_fails_loudly(monkeypatch):
    _record_downloads(monkeypatch)
    with pytest.raises(ValueError, match="local file or a Hub repo id"):
        weights_module.resolve_weights(
            released_checkpoint("classification"), "not a repo or path"
        )


# --- manifest overrides & preset fidelity ---------------------------------


def test_with_overrides_replaces_only_named_knobs():
    manifest = released_manifest("classification")
    updated = manifest.with_overrides(compute_dtype="float32", ensemble_count=4)

    assert updated.runtime.compute_dtype == "float32"
    assert updated.runtime.ensemble_count == 4
    assert updated.runtime.seed == manifest.runtime.seed  # untouched
    assert updated.model == manifest.model  # architecture is immutable
    assert manifest.runtime.compute_dtype == "float16"  # original unchanged


def test_with_overrides_noop_returns_same_object():
    manifest = released_manifest("regression")
    assert manifest.with_overrides() is manifest


def test_with_overrides_revalidates():
    with pytest.raises(ValueError):
        released_manifest("classification").with_overrides(compute_dtype="float8")


def test_released_manifests_are_frozen_expected_shapes():
    clf = released_manifest("classification")
    assert clf.task == "classification"
    assert clf.model.class_capacity == 10
    assert clf.runtime.ensemble_count == 8

    reg = released_manifest("regression")
    assert reg.task == "regression"
    assert reg.regression is not None
    assert reg.regression.quantile_count == 999
    assert reg.model.feature_attention_repeats == 2


# --- checksum gate & end-to-end wiring ------------------------------------


def _tiny_manifest() -> InferenceManifest:
    """A small architecture whose checkpoint is cheap to build and save."""
    return InferenceManifest(
        checkpoint_sha256="0" * 64,  # deliberately wrong; skipped for user weights
        task="classification",
        model=ModelConfig(
            width=16,
            head_count=4,
            block_count=2,
            feedforward_widths=32,
            item_summary_count=2,
            feature_summary_count=3,
            feature_attention_repeats=1,
            class_capacity=3,
        ),
        runtime=RuntimeConfig(ensemble_count=2, compute_dtype="float32", seed=0),
    )


def _save_tiny_checkpoint(path, manifest) -> None:
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(manifest.runtime.seed)
        model = ClassificationModel(manifest.model, dtype=torch.float32)
    save_file(model.state_dict(), str(path))


def test_load_gate_skips_digest_when_disabled(tmp_path):
    manifest = _tiny_manifest()
    ckpt = tmp_path / "tiny.safetensors"
    _save_tiny_checkpoint(ckpt, manifest)
    model = ClassificationModel(manifest.model, dtype=torch.float32)

    # The pinned digest is wrong, so enforcing it must fail...
    with pytest.raises(ValueError, match="digest"):
        load_classifier_checkpoint(str(ckpt), model, manifest, verify_checksum=True)
    # ...but with the gate off the same file loads cleanly.
    load_classifier_checkpoint(str(ckpt), model, manifest, verify_checksum=False)


def test_load_gate_warns_when_manifest_pins_no_digest(tmp_path, caplog):
    """An unpinned manifest is the released state until publication — never silent."""
    manifest = replace(_tiny_manifest(), checkpoint_sha256=None)
    ckpt = tmp_path / "tiny.safetensors"
    _save_tiny_checkpoint(ckpt, manifest)
    model = ClassificationModel(manifest.model, dtype=torch.float32)

    with caplog.at_level(logging.WARNING, logger="exaonetabular.checkpoint"):
        load_classifier_checkpoint(str(ckpt), model, manifest, verify_checksum=True)

    assert any("pins no checkpoint_sha256" in rec.message for rec in caplog.records)


def test_released_manifests_pin_no_digest_yet():
    """Guards against re-introducing a placeholder digest for unpublished weights."""
    for task in ("classification", "regression"):
        assert released_manifest(task).checkpoint_sha256 is None


def test_from_pretrained_user_weights_load_and_warn(tmp_path, monkeypatch, caplog):
    manifest = _tiny_manifest()
    ckpt = tmp_path / "tiny.safetensors"
    _save_tiny_checkpoint(ckpt, manifest)

    with caplog.at_level(logging.WARNING, logger="exaonetabular.classifier"):
        clf = EXAONETabularClassifier.from_pretrained(
            weights=str(ckpt), manifest=manifest
        )

    assert isinstance(clf, EXAONETabularClassifier)
    assert any("integrity pin not enforced" in rec.message for rec in caplog.records)

    # The loaded estimator is functional.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 5)).astype(np.float64)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    clf.fit(x, y)
    assert clf.predict(x).shape == (8,)
