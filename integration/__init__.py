"""
AETHERIS Integration Module

Connects AETHERIS to the C.I.C.D.E. manifest, Pantheon agents,
AEONIC_LOG, and research pipeline.
"""

from aetheris.integration.cicide_bridge import CICIDEBridge
from aetheris.integration.pantheon_orchestrator import PantheonOrchestrator
from aetheris.integration.manifest_updater import ManifestUpdater
from aetheris.integration.aeon_logger import AeonicLogger
from aetheris.integration.research_pipeline import ResearchPipeline

__all__ = [
    "CICIDEBridge",
    "PantheonOrchestrator",
    "ManifestUpdater",
    "AeonicLogger",
    "ResearchPipeline",
]
