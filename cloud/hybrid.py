"""
Hybrid Executor — Smart Backend Selection

Automatically selects the optimal execution backend based on model size,
hardware availability, and user preferences.
"""

import platform
import subprocess
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class Backend(Enum):
    """Available execution backends."""
    LOCAL_CPU = "local_cpu"
    LOCAL_GPU = "local_gpu"
    COLAB = "colab"
    SPACES = "spaces"
    KAGGLE = "kaggle"
    RUNPOD = "runpod"
    VAST = "vast"


@dataclass
class BackendRecommendation:
    """Container for backend recommendation."""
    backend: Backend
    reason: str
    estimated_time: str
    cost: str
    requirements: List[str]


class HybridExecutor:
    """
    Smart executor that selects optimal backend automatically.

    Analyzes model size, local hardware, and user preferences to choose
    the best execution environment.
    """

    def __init__(self, prefer_free: bool = True):
        """
        Initialize hybrid executor.

        Args:
            prefer_free: Whether to prefer free platforms over paid
        """
        self.prefer_free = prefer_free
        self._local_hardware = self._detect_local_hardware()

    def _detect_local_hardware(self) -> Dict[str, Any]:
        """Detect local hardware capabilities."""
        info = {
            "has_gpu": False,
            "gpu_name": None,
            "gpu_memory_gb": 0,
            "ram_gb": 0,
            "cpu_count": 0,
            "platform": platform.system()
        }

        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                info["has_gpu"] = True
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        except ImportError:
            pass

        # Check RAM
        try:
            import psutil
            info["ram_gb"] = psutil.virtual_memory().total / 1e9
            info["cpu_count"] = psutil.cpu_count()
        except ImportError:
            pass

        return info

    def auto_select_backend(
        self,
        model_name: str,
        model_size_gb: Optional[float] = None,
        prefer_free: Optional[bool] = None
    ) -> BackendRecommendation:
        """
        Automatically select the best backend.

        Args:
            model_name: Model to run
            model_size_gb: Estimated model size in GB
            prefer_free: Override prefer_free setting

        Returns:
            Backend recommendation
        """
        use_free = prefer_free if prefer_free is not None else self.prefer_free

        # Estimate model size if not provided
        if model_size_gb is None:
            model_size_gb = self._estimate_model_size(model_name)

        # Check local execution feasibility
        if self._can_run_locally(model_size_gb):
            if self._local_hardware["has_gpu"]:
                return BackendRecommendation(
                    backend=Backend.LOCAL_GPU,
                    reason=f"Local GPU detected: {self._local_hardware['gpu_name']} with {self._local_hardware['gpu_memory_gb']:.1f}GB",
                    estimated_time="5-15 minutes",
                    cost="Free",
                    requirements=["PyTorch with CUDA", "Enough VRAM"]
                )
            else:
                return BackendRecommendation(
                    backend=Backend.LOCAL_CPU,
                    reason="Local CPU execution (slower but free)",
                    estimated_time="15-60 minutes",
                    cost="Free",
                    requirements=["Python 3.9+", "8GB+ RAM"]
                )

        # Not feasible locally, recommend cloud
        if use_free:
            return self._recommend_free_cloud(model_size_gb)
        else:
            return self._recommend_paid_cloud(model_size_gb)

    def _can_run_locally(self, model_size_gb: float) -> bool:
        """Check if model can run locally."""
        if model_size_gb > self._local_hardware["ram_gb"] - 2:
            return False

        # GPU memory check
        if self._local_hardware["has_gpu"]:
            if model_size_gb > self._local_hardware["gpu_memory_gb"]:
                return False

        return True

    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in GB."""
        # Rough estimates based on parameter count
        param_estimates = {
            "7B": 14.0,
            "8B": 16.0,
            "13B": 26.0,
            "70B": 140.0,
            "1.1B": 2.2,
            "2.7B": 5.4,
            "0.5B": 1.0,
        }

        for key, size in param_estimates.items():
            if key in model_name:
                return size

        # Default: 1GB per billion parameters
        import re
        match = re.search(r'(\d+)[bB]', model_name)
        if match:
            params_b = int(match.group(1))
            return params_b * 2  # 2 bytes per param for fp16

        return 2.0  # Default for small models

    def _recommend_free_cloud(self, model_size_gb: float) -> BackendRecommendation:
        """Recommend free cloud backend."""
        if model_size_gb <= 16:
            return BackendRecommendation(
                backend=Backend.COLAB,
                reason="Google Colab T4 GPU (16GB VRAM) - Free",
                estimated_time="10-20 minutes",
                cost="Free",
                requirements=["Google account", "Colab notebook"]
            )
        elif model_size_gb <= 32:
            return BackendRecommendation(
                backend=Backend.KAGGLE,
                reason="Kaggle T4 x2 (30GB+ VRAM) - Free",
                estimated_time="15-30 minutes",
                cost="Free",
                requirements=["Kaggle account", "Phone verification"]
            )
        else:
            return BackendRecommendation(
                backend=Backend.SPACES,
                reason="HuggingFace Spaces T4 (16GB) - May need quantization",
                estimated_time="20-40 minutes",
                cost="Free",
                requirements=["HuggingFace account", "Space creation"]
            )

    def _recommend_paid_cloud(self, model_size_gb: float) -> BackendRecommendation:
        """Recommend paid cloud backend."""
        if model_size_gb <= 24:
            return BackendRecommendation(
                backend=Backend.RUNPOD,
                reason="RunPod A10 (24GB) - $0.44/hr",
                estimated_time="5-10 minutes",
                cost="$0.07 - $0.15",
                requirements=["RunPod account", "Payment method"]
            )
        elif model_size_gb <= 48:
            return BackendRecommendation(
                backend=Backend.RUNPOD,
                reason="RunPod A100 (40GB) - $1.89/hr",
                estimated_time="3-8 minutes",
                cost="$0.25 - $0.50",
                requirements=["RunPod account", "Payment method"]
            )
        else:
            return BackendRecommendation(
                backend=Backend.VAST,
                reason="Vast.ai Multi-GPU - Variable pricing",
                estimated_time="5-15 minutes",
                cost="$0.50 - $2.00",
                requirements=["Vast.ai account", "Payment method"]
            )

    def distribute_workload(
        self,
        tasks: List[Dict[str, Any]],
        backends: Optional[List[Backend]] = None
    ) -> Dict[str, Any]:
        """
        Distribute workload across multiple backends.

        Args:
            tasks: List of tasks to execute
            backends: Specific backends to use

        Returns:
            Distribution plan
        """
        if backends is None:
            backends = [Backend.COLAB, Backend.KAGGLE]

        distribution = {}
        for i, task in enumerate(tasks):
            backend = backends[i % len(backends)]
            distribution[f"task_{i}"] = {
                "task": task,
                "backend": backend.value,
                "status": "pending"
            }

        return {
            "total_tasks": len(tasks),
            "backends_used": [b.value for b in backends],
            "distribution": distribution,
            "message": f"Distributed {len(tasks)} tasks across {len(backends)} backends"
        }

    def get_backend_status(self) -> Dict[str, Any]:
        """
        Get status of all available backends.

        Returns:
            Backend availability status
        """
        return {
            "local": {
                "available": True,
                "has_gpu": self._local_hardware["has_gpu"],
                "gpu_name": self._local_hardware["gpu_name"],
                "ram_gb": self._local_hardware["ram_gb"]
            },
            "colab": {
                "available": True,
                "free": True,
                "max_model_size": 16,
                "requires": "Google account"
            },
            "kaggle": {
                "available": True,
                "free": True,
                "max_model_size": 32,
                "requires": "Kaggle account + phone verification"
            },
            "spaces": {
                "available": True,
                "free": True,
                "max_model_size": 16,
                "requires": "HuggingFace account"
            },
            "runpod": {
                "available": True,
                "free": False,
                "rates": "$0.44 - $1.89/hr",
                "requires": "Payment method"
            },
            "vast": {
                "available": True,
                "free": False,
                "rates": "Variable",
                "requires": "Payment method"
            }
        }

    def get_recommendation(
        self,
        model_name: str,
        model_size_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get complete recommendation for a model.

        Args:
            model_name: Model to analyze
            model_size_gb: Optional model size

        Returns:
            Complete recommendation
        """
        if model_size_gb is None:
            model_size_gb = self._estimate_model_size(model_name)

        local = self._can_run_locally(model_size_gb)

        return {
            "model": model_name,
            "estimated_size_gb": model_size_gb,
            "local_capable": local,
            "local_hardware": self._local_hardware,
            "recommendation": self.auto_select_backend(model_name, model_size_gb),
            "alternatives": [
                self._recommend_free_cloud(model_size_gb),
                self._recommend_paid_cloud(model_size_gb)
            ]
        }
