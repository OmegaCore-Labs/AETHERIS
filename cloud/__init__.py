"""
AETHERIS Cloud Module

Production-grade cloud execution platforms:
- Google Colab (free T4 GPU) — nbformat notebooks
- HuggingFace Spaces (free T4) — huggingface_hub HfApi
- Kaggle (free T4 x2) — kagglehub integration
- PyTorch Lightning — DataModule + LightningModule + Trainer
- RunPod (paid GPU rental) — GraphQL API
- Vast.ai (paid GPU rental) — REST API
- Hybrid Executor — smart backend selection with cost/speed tradeoff
"""

from aetheris.cloud.colab import ColabRuntime
from aetheris.cloud.spaces import SpacesDeployer
from aetheris.cloud.spaces_deploy_oneclick import SpacesOneClickDeployer, deploy_liberated_model
from aetheris.cloud.kaggle import KaggleRuntime
from aetheris.cloud.lightning import LightningRuntime, LightningTrainer, ActivationDataModule, ProbeModule
from aetheris.cloud.runpod import RunPodExecutor
from aetheris.cloud.vast import VastExecutor
from aetheris.cloud.hybrid import HybridExecutor, Backend, BackendRecommendation, TradeoffAnalysis

__all__ = [
    "ColabRuntime",
    "SpacesDeployer",
    "SpacesOneClickDeployer",
    "deploy_liberated_model",
    "KaggleRuntime",
    "LightningRuntime",
    "LightningTrainer",
    "ActivationDataModule",
    "ProbeModule",
    "RunPodExecutor",
    "VastExecutor",
    "HybridExecutor",
    "Backend",
    "BackendRecommendation",
    "TradeoffAnalysis",
]
