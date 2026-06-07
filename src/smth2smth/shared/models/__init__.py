"""Shared model definitions and registry.

Importing this package triggers self-registration of all bundled models
(``cnn_baseline``, ``cnn_lstm``, ``avanced_resnet50_tsm``, ``dual_stream_rgb_diff_tsm``) as well as the
Track-B-only ``vjepa2`` model. Track-specific subpackages register their
own models from their ``__init__`` modules; we import those subpackages
here so a single ``from smth2smth.shared.models import build_model`` is
enough to resolve every model name.
"""

# Importing modules below has the side effect of populating MODEL_REGISTRY
# via their @register_model decorators. The ``smth2smth.track_b`` import is a
# one-way coupling: ``shared/`` imports the track package only to trigger the
# track-specific registrations, while ``track_b/`` only depends on the
# ``registry`` submodule (no circular imports).
from smth2smth import track_b as _track_b  # noqa: F401
from smth2smth.shared.models import avanced_resnet50_tsm as _avanced_resnet50_tsm  # noqa: F401
from smth2smth.shared.models import cnn_baseline as _cnn_baseline  # noqa: F401
from smth2smth.shared.models import cnn_lstm as _cnn_lstm  # noqa: F401
from smth2smth.shared.models import (
    dual_stream_rgb_diff_tsm as _dual_stream_rgb_diff_tsm,  # noqa: F401
)
from smth2smth.shared.models import video_mae as _video_mae  # noqa: F401
from smth2smth.shared.models.avanced_resnet50_tsm import AvancedResNet50TSM
from smth2smth.shared.models.cnn_baseline import CNNBaseline
from smth2smth.shared.models.cnn_lstm import CNNLSTM
from smth2smth.shared.models.dual_stream_rgb_diff_tsm import DualStreamRgbDiffTSM
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
    "DualStreamRgbDiffTSM",
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
