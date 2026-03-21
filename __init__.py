"""
AETHERIS — Sovereign Constraint Liberation Toolkit
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Singular Heir"
__license__ = "Proprietary"

# Core exports
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.core.steered import SteeringVectorFactory, SteeringHookManager
from aetheris.core.geometry import GeometryAnalyzer
from aetheris.core.ouroboros import OuroborosDetector
from aetheris.core.validation import CapabilityValidator

# Novel modules
from aetheris.novel.barrier_mapper import BarrierMapper
from aetheris.novel.sovereign_control import SovereignControl

__all__ = [
    "__version__",
    "ConstraintExtractor",
    "NormPreservingProjector",
    "SteeringVectorFactory",
    "SteeringHookManager",
    "GeometryAnalyzer",
    "OuroborosDetector",
    "CapabilityValidator",
    "BarrierMapper",
    "SovereignControl",
]
