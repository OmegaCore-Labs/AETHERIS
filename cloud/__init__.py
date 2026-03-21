"""
AETHERIS Cloud Module

Cloud execution platforms:
- Google Colab (free T4 GPU)
- HuggingFace Spaces (free T4)
- Kaggle (free T4 x2)
- Lightning.ai (free tier)
- RunPod (paid GPU rental)
- Vast.ai (paid GPU rental)
"""

from aetheris.cloud.colab import ColabRuntime
from aetheris.cloud.spaces import SpacesDeployer
from aetheris.cloud.kaggle import KaggleRuntime
from aetheris.cloud.lightning import LightningRuntime
from aetheris.cloud.runpod import RunPodExecutor
from aetheris.cloud.vast import VastExecutor
from aetheris.cloud.hybrid import HybridExecutor

__all__ = [
    "ColabRuntime",
    "SpacesDeployer",
    "KaggleRuntime",
    "LightningRuntime",
    "RunPodExecutor",
    "VastExecutor",
    "HybridExecutor",
]
