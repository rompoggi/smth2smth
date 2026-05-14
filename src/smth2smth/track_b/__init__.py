"""Track B specific code (overrides only; the rest of the logic lives in shared/).

Importing this package triggers self-registration of every Track-B-only
model (currently :class:`smth2smth.track_b.vjepa2.VJEPA2Probe`) into the
shared model registry, so ``build_model(cfg)`` resolves ``cfg.model.name``
to the right builder regardless of which track the experiment targets.
"""

# Importing the submodule has the side effect of populating MODEL_REGISTRY
# via the @register_model decorator inside.
from smth2smth.track_b import vjepa2 as _vjepa2  # noqa: F401
from smth2smth.track_b.vjepa2 import (
    AttentiveProbe,
    MeanLinearHead,
    VJEPA2Probe,
)

__all__ = [
    "AttentiveProbe",
    "MeanLinearHead",
    "VJEPA2Probe",
]
