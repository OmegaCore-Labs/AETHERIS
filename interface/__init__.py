"""
AETHERIS Interface Module

Production-grade human-AETHERIS interaction interfaces:
- REST API (Flask with background job queue)
- Gradio UI (real core module calls, plotly visualizations)
- Web Dashboard (Flask-based with live job monitoring)
- Voice Control (speech_recognition + pyttsx3)
- Holographic Visualization (matplotlib 3D + Three.js WebGL)
- Neural Interface (BCI placeholder with realistic simulation)
- Gesture Control (MediaPipe hand tracking with simulation fallback)
"""

from aetheris.interface.api import AetherisAPI, JobQueue, Job, JobStatus
from aetheris.interface.gradio_app import AetherisUI
from aetheris.interface.web import WebDashboard
from aetheris.interface.voice import VoiceController, CommandIntent
from aetheris.interface.holographic import HolographicViz, VisualizationConfig
from aetheris.interface.neural import NeuralBridge, NeuralSignal
from aetheris.interface.gesture import GestureController, GestureType, GestureEvent

__all__ = [
    "AetherisAPI",
    "JobQueue",
    "Job",
    "JobStatus",
    "AetherisUI",
    "WebDashboard",
    "VoiceController",
    "CommandIntent",
    "HolographicViz",
    "VisualizationConfig",
    "NeuralBridge",
    "NeuralSignal",
    "GestureController",
    "GestureType",
    "GestureEvent",
]
