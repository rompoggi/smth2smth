"""Shared model definitions and registry.

Importing this package triggers self-registration of all bundled models
(``cnn_baseline``, ``cnn_lstm``, ``avanced_resnet50_tsm``).
"""

# Importing modules below has the side effect of populating MODEL_REGISTRY
# via their @register_model decorators.
from smth2smth.shared.models import avanced_resnet50_tsm as _avanced_resnet50_tsm  # noqa: F401
from smth2smth.shared.models import cnn_baseline as _cnn_baseline  # noqa: F401
from smth2smth.shared.models import cnn_lstm as _cnn_lstm  # noqa: F401
from smth2smth.shared.models.avanced_resnet50_tsm import AvancedResNet50TSM
from smth2smth.shared.models.cnn_baseline import CNNBaseline
from smth2smth.shared.models.cnn_lstm import CNNLSTM
from smth2smth.shared.models.registry import (
    MODEL_REGISTRY,
    ModelAlreadyRegisteredError,
    UnknownModelError,
    build_model,
    list_registered_models,
    register_model,
)
from smth2smth.shared.models.vjepa_ssl import (
    VJepaModel,
    VJepaPredictor,
    VJepaTrunk,
    apply_frame_mask,
    make_frame_mask,
    update_vjepa_teacher_ema,
    vjepa_feature_loss,
)

__all__ = [
    "CNNBaseline",
    "CNNLSTM",
    "AvancedResNet50TSM",
    "MODEL_REGISTRY",
    "ModelAlreadyRegisteredError",
    "UnknownModelError",
    "VJepaModel",
    "VJepaPredictor",
    "VJepaTrunk",
    "apply_frame_mask",
    "build_model",
    "list_registered_models",
    "make_frame_mask",
    "register_model",
    "update_vjepa_teacher_ema",
    "vjepa_feature_loss",
]
