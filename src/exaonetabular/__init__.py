"""Public classification interface for EXAONETabular."""

from importlib import import_module
from importlib.util import find_spec


for _module in ("numpy", "sklearn", "torch"):
    if find_spec(_module) is None:
        import_module(_module)
del _module

__version__ = "1.0.0"

__all__ = [
    "EXAONETabularClassifier",
    "EXAONETabularRegressor",
    "ClassificationModel",
    "RegressionModel",
    "build_model",
    "create_model",
    "build_model_from_manifest",
    "create_model_from_manifest",
    "model_from_manifest",
    "InferenceManifest",
    "ModelConfig",
    "PreprocessingConfig",
    "RuntimeConfig",
    "RegressionConfig",
    "AttentionScorer",
    "FeatureSelectionConfig",
    "FEATURE_SELECTION",
    "load_classifier_checkpoint",
    "load_regressor_checkpoint",
    "load_checkpoint",
    "released_manifest",
    "released_checkpoint",
    "__version__",
]

_PUBLIC_BINDINGS = {
    "EXAONETabularClassifier": (".classifier", "EXAONETabularClassifier"),
    "EXAONETabularRegressor": (".regressor", "EXAONETabularRegressor"),
    "ClassificationModel": (".model.heads", "ClassificationModel"),
    "RegressionModel": (".model.heads", "RegressionModel"),
    "build_model": (".model.heads", "build_model"),
    "create_model": (".model.heads", "create_model"),
    "build_model_from_manifest": (".model.heads", "build_model_from_manifest"),
    "create_model_from_manifest": (".model.heads", "create_model_from_manifest"),
    "model_from_manifest": (".model.heads", "model_from_manifest"),
    "InferenceManifest": (".config", "InferenceManifest"),
    "ModelConfig": (".config", "ModelConfig"),
    "PreprocessingConfig": (".config", "PreprocessingConfig"),
    "RuntimeConfig": (".config", "RuntimeConfig"),
    "RegressionConfig": (".config", "RegressionConfig"),
    "AttentionScorer": (".config", "AttentionScorer"),
    "FeatureSelectionConfig": (".config", "FeatureSelectionConfig"),
    "FEATURE_SELECTION": (".config", "FEATURE_SELECTION"),
    "load_classifier_checkpoint": (".checkpoint", "load_classifier_checkpoint"),
    "load_regressor_checkpoint": (".checkpoint", "load_regressor_checkpoint"),
    "load_checkpoint": (".checkpoint", "load_checkpoint"),
    "released_manifest": (".presets", "released_manifest"),
    "released_checkpoint": (".presets", "released_checkpoint"),
}


def __getattr__(name: str):
    """Resolve a public binding without eagerly importing unrelated dependencies."""
    try:
        module_name, attribute_name = _PUBLIC_BINDINGS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
