"""
AETHERIS Interface Module

Human-ARIS interaction interfaces:
- Voice control
- Holographic visualization
- Web dashboard
- REST API
"""

from aetheris.interface.voice import VoiceController
from aetheris.interface.holographic import HolographicViz
from aetheris.interface.web import WebDashboard
from aetheris.interface.api import AetherisAPI

__all__ = [
    "VoiceController",
    "HolographicViz",
    "WebDashboard",
    "AetherisAPI",
]
