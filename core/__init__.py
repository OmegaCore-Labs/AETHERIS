"""
AETHERIS Core Module

Provides the foundational algorithms for constraint extraction, projection,
steering, geometric analysis, self-repair detection, and capability validation.
"""

from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.core.steered import SteeringVectorFactory, SteeringHookManager
from aetheris.core.geometry import GeometryAnalyzer
from aetheris.core.ouroboros import OuroborosDetector
from aetheris.core.validation import CapabilityValidator

__all__ = [
    "ConstraintExtractor",
    "NormPreservingProjector",
    "SteeringVectorFactory",
    "SteeringHookManager",
    "GeometryAnalyzer",
    "OuroborosDetector",
    "CapabilityValidator",
]
