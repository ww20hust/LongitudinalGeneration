"""
Core components for PEAG framework.
"""

from peag.core.encoders import (
    LabTestsEncoder,
    MetabolomicsEncoder,
    PastStateEncoder,
    GenericModalityEncoder
)
from peag.core.decoders import (
    LabTestsDecoder,
    MetabolomicsDecoder,
    GenericModalityDecoder
)
from peag.core.alignment import (
    jeffrey_divergence,
    align_distributions_dynamic,
    align_distributions_simple
)
from peag.core.fusion import (
    compute_visit_state_dynamic,
    AdaptiveVisitStateFusion
)
from peag.core.adversarial import (
    MissingnessDiscriminator,
    ModalityDiscriminator
)

__all__ = [
    "LabTestsEncoder",
    "MetabolomicsEncoder",
    "PastStateEncoder",
    "GenericModalityEncoder",
    "LabTestsDecoder",
    "MetabolomicsDecoder",
    "GenericModalityDecoder",
    "jeffrey_divergence",
    "align_distributions_dynamic",
    "align_distributions_simple",
    "compute_visit_state_dynamic",
    "AdaptiveVisitStateFusion",
    "MissingnessDiscriminator",
    "ModalityDiscriminator"
]
