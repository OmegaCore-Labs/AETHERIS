"""
Hybrid Executor — Smart Backend Selection

Production-grade backend selection engine that analyzes model size,
local hardware capabilities, cost constraints, and time requirements
to automatically choose the best execution backend. Includes cost/speed
trade-off analysis with real hardware detection.
"""

import os
import re
import platform
import subprocess
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
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
    LAMBDA = "lambda"


@dataclass
class BackendRecommendation:
    """Container for backend recommendation with full analysis."""
    backend: Backend
    reason: str
    estimated_time_min: float
    estimated_time_str: str
    cost: str
    cost_float: float
    requirements: List[str]
    is_free: bool
    vram_required_gb: float
    vram_available_gb: float
    risk_level: str  # low, medium, high
    confidence: float


@dataclass
class TradeoffAnalysis:
    """Cost/speed trade-off analysis across backends."""
    options: List[BackendRecommendation] = field(default_factory=list)
    cheapest: Optional[BackendRecommendation] = None
    fastest: Optional[BackendRecommendation] = None
    best_free: Optional[BackendRecommendation] = None
    pareto_optimal: List[BackendRecommendation] = field(default_factory=list)


class HybridExecutor:
    """
    Smart executor that selects optimal backend automatically.

    Analyzes model size, local hardware, and user preferences to choose
    the best execution environment. Provides cost/speed trade-off analysis.
    """

    # Model size estimation hints (param count -> approximate fp16 GB)
    MODEL_SIZE_HINTS = {
        "125M": 0.3, "350M": 0.7, "1.1B": 2.2, "1.5B": 3.0,
        "2.7B": 5.4, "3B": 6.0, "7B": 14.0, "8B": 16.0,
        "13B": 26.0, "20B": 40.0, "30B": 60.0, "34B": 68.0,
        "40B": 80.0, "65B": 130.0, "70B": 140.0, "72B": 144.0,
    }

    # Backend capabilities
    BACKEND_CAPABILITIES = {
        Backend.COLAB: {
            "max_vram_gb": 16, "gpu": "T4", "free": True,
            "setup_time_min": 3, "speed_factor": 1.0,  # baseline
            "risk": "medium", "description": "Google Colab free T4 (16GB VRAM)",
        },
        Backend.SPACES: {
            "max_vram_gb": 16, "gpu": "T4", "free": True,
            "setup_time_min": 5, "speed_factor": 0.9,
            "risk": "medium", "description": "HuggingFace Spaces free T4 (16GB VRAM)",
        },
        Backend.KAGGLE: {
            "max_vram_gb": 32, "gpu": "T4 x2", "free": True,
            "setup_time_min": 4, "speed_factor": 1.2,
            "risk": "medium", "description": "Kaggle free T4 x2 (up to 32GB VRAM)",
        },
        Backend.RUNPOD: {
            "max_vram_gb": 80, "gpu": "Variable", "free": False,
            "setup_time_min": 2, "speed_factor": 2.0,
            "risk": "low", "description": "RunPod paid GPU (up to H100 80GB)",
        },
        Backend.VAST: {
            "max_vram_gb": 80, "gpu": "Variable", "free": False,
            "setup_time_min": 3, "speed_factor": 1.5,
            "risk": "medium", "description": "Vast.ai marketplace GPU (variable pricing)",
        },
    }

    def __init__(self, prefer_free: bool = True, max_budget: Optional[float] = None):
        """
        Initialize hybrid executor.

        Args:
            prefer_free: Whether to prefer free platforms over paid
            max_budget: Maximum budget in USD (None = no limit)
        """
        self.prefer_free = prefer_free
        self.max_budget = max_budget
        self._local_hardware = self._detect_local_hardware()

    def _detect_local_hardware(self) -> Dict[str, Any]:
        """Detect local hardware capabilities comprehensively."""
        info = {
            "has_gpu": False,
            "gpu_name": None,
            "gpu_vram_gb": 0.0,
            "gpu_count": 0,
            "ram_gb": 0.0,
            "cpu_count": 0,
            "cpu_name": None,
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "free_disk_gb": 0.0,
        }

        try:
            # GPU detection via PyTorch
            import torch
            if torch.cuda.is_available():
                info["has_gpu"] = True
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_name"] = torch.cuda.get_device_name(0)
                try:
                    mem_bytes = torch.cuda.get_device_properties(0).total_memory
                    info["gpu_vram_gb"] = mem_bytes / (1024 ** 3)
                except Exception:
                    # Fallback: nvidia-smi
                    try:
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=5,
                        )
                        if result.returncode == 0:
                            mem_mb = float(result.stdout.strip().split()[0])
                            info["gpu_vram_gb"] = mem_mb / 1024.0
                    except Exception:
                        pass
        except ImportError:
            # Try nvidia-smi
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    info["has_gpu"] = True
                    lines = result.stdout.strip().split("\n")
                    info["gpu_name"] = lines[0].split(",")[0].strip()
                    mem_mb = float(lines[0].split(",")[1].strip().split()[0])
                    info["gpu_vram_gb"] = mem_mb / 1024.0
            except Exception:
                pass

        # RAM detection
        try:
            import psutil
            info["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
            info["free_disk_gb"] = psutil.disk_usage("/").free / (1024 ** 3)
            info["cpu_count"] = psutil.cpu_count(logical=False) or psutil.cpu_count()
        except ImportError:
            pass

        # CPU name
        try:
            if platform.system() == "Windows":
                info["cpu_name"] = platform.processor()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            info["cpu_name"] = line.split(":")[1].strip()
                            break
            elif platform.system() == "Darwin":
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                        capture_output=True, text=True)
                info["cpu_name"] = result.stdout.strip()
        except Exception:
            pass

        return info

    def estimate_model_size(self, model_name: str) -> float:
        """
        Estimate model size in GB (fp16).

        Args:
            model_name: HuggingFace model name or path

        Returns:
            Estimated size in GB
        """
        # Known models
        known_models = {
            "gpt2": 0.3,
            "gpt2-medium": 0.7,
            "gpt2-large": 1.5,
            "gpt2-xl": 3.0,
            "distilgpt2": 0.2,
            "facebook/opt-125m": 0.3,
            "facebook/opt-350m": 0.7,
            "facebook/opt-1.3b": 2.6,
            "facebook/opt-2.7b": 5.4,
            "facebook/opt-6.7b": 13.4,
            "microsoft/phi-2": 5.4,
            "microsoft/phi-3-mini-4k-instruct": 7.6,
        }

        model_lower = model_name.lower()
        for key, size in known_models.items():
            if key.lower() == model_lower or key.lower() in model_lower:
                return size

        # Pattern matching: 7B, 8B, 13B, 70B, etc.
        match = re.search(r"(\d+\.?\d*)\s*[bB]", model_name)
        if match:
            params_b = float(match.group(1))
            return params_b * 2.0  # fp16 = 2 bytes per parameter

        # Check against hints
        for hint, size in self.MODEL_SIZE_HINTS.items():
            if hint.lower() in model_lower:
                return size

        # Default conservative estimate
        return 4.0

    def auto_select_backend(
        self,
        model_name: str,
        model_size_gb: Optional[float] = None,
        prefer_free: Optional[bool] = None,
        max_time_min: Optional[float] = None,
    ) -> BackendRecommendation:
        """
        Automatically select the best backend.

        Args:
            model_name: Model to run
            model_size_gb: Estimated model size in GB (auto-detected if None)
            prefer_free: Override prefer_free setting
            max_time_min: Maximum acceptable time in minutes

        Returns:
            Backend recommendation
        """
        use_free = prefer_free if prefer_free is not None else self.prefer_free
        model_gb = model_size_gb or self.estimate_model_size(model_name)

        # Factor in overhead (activations, optimizer states)
        required_vram = model_gb * 1.3  # 30% overhead

        hw = self._local_hardware

        # Check local GPU
        if hw["has_gpu"] and hw["gpu_vram_gb"] >= required_vram:
            return BackendRecommendation(
                backend=Backend.LOCAL_GPU,
                reason=f"Local {hw['gpu_name']} with {hw['gpu_vram_gb']:.1f}GB VRAM can run model ({required_vram:.1f}GB needed)",
                estimated_time_min=8.0,
                estimated_time_str="5-15 minutes",
                cost="Free (local)",
                cost_float=0.0,
                requirements=[f"Local GPU with {required_vram:.1f}GB VRAM"],
                is_free=True,
                vram_required_gb=required_vram,
                vram_available_gb=hw["gpu_vram_gb"],
                risk_level="low",
                confidence=0.95,
            )

        # Check local CPU
        if hw["ram_gb"] >= required_vram + 2:
            # CPU is slow but works for small models
            cpu_estimate = model_gb * 5  # ~5 min per GB on CPU
            return BackendRecommendation(
                backend=Backend.LOCAL_CPU,
                reason=f"Local CPU with {hw['ram_gb']:.1f}GB RAM can run model (slower but free)",
                estimated_time_min=cpu_estimate,
                estimated_time_str=f"{cpu_estimate:.0f}-{cpu_estimate * 1.5:.0f} minutes",
                cost="Free (local)",
                cost_float=0.0,
                requirements=[f"{required_vram:.1f}GB+ RAM"],
                is_free=True,
                vram_required_gb=required_vram,
                vram_available_gb=hw["ram_gb"],
                risk_level="low",
                confidence=0.90,
            )

        # Cloud backends
        tradeoff = self.analyze_tradeoffs(model_gb, max_time_min)
        if use_free and tradeoff.best_free:
            return tradeoff.best_free
        elif tradeoff.cheapest:
            if self.max_budget and tradeoff.cheapest.cost_float > self.max_budget:
                # Over budget; fall back to best free even if not ideal
                if tradeoff.best_free:
                    return tradeoff.best_free
            return tradeoff.cheapest

        # Fallback: recommend Colab
        return self._recommend_backend(Backend.COLAB, required_vram)

    def analyze_tradeoffs(
        self,
        model_size_gb: float,
        max_time_min: Optional[float] = None,
    ) -> TradeoffAnalysis:
        """
        Analyze cost/speed trade-offs across all backends.

        Args:
            model_size_gb: Model size in GB
            max_time_min: Maximum acceptable time (filters results)

        Returns:
            TradeoffAnalysis with all viable options
        """
        required_vram = model_size_gb * 1.3
        options = []

        for backend, caps in self.BACKEND_CAPABILITIES.items():
            max_vram = caps["max_vram_gb"]
            if required_vram > max_vram:
                continue

            # Estimate time
            # Base time: 5 min setup + model_size * speed_adjustment
            base_process_time = model_size_gb * 3  # ~3 min per GB
            adjusted_time = base_process_time / caps["speed_factor"]
            total_time = caps["setup_time_min"] + adjusted_time

            if max_time_min and total_time > max_time_min:
                continue

            # Estimate cost
            if caps["free"]:
                cost = 0.0
                cost_str = "Free"
            else:
                # Approximate cost based on GPU tier
                if max_vram <= 24:
                    cost_per_hr = 0.44
                elif max_vram <= 48:
                    cost_per_hr = 0.79
                else:
                    cost_per_hr = 1.89
                cost = cost_per_hr * (total_time / 60)
                cost_str = f"${cost:.2f}"

            if self.max_budget and cost > self.max_budget:
                continue

            recommendation = BackendRecommendation(
                backend=backend,
                reason=f"{caps['description']} — {total_time:.0f} min estimated",
                estimated_time_min=total_time,
                estimated_time_str=f"{total_time:.0f} minutes",
                cost=cost_str,
                cost_float=cost,
                requirements=self._get_backend_requirements(backend),
                is_free=caps["free"],
                vram_required_gb=required_vram,
                vram_available_gb=max_vram,
                risk_level=caps["risk"],
                confidence=0.85,
            )
            options.append(recommendation)

        # Sort by cost then time
        options.sort(key=lambda o: (o.cost_float, o.estimated_time_min))

        tradeoff = TradeoffAnalysis(options=options)

        if options:
            tradeoff.cheapest = min(options, key=lambda o: o.cost_float)
            tradeoff.fastest = min(options, key=lambda o: o.estimated_time_min)
            free_options = [o for o in options if o.is_free]
            tradeoff.best_free = min(free_options, key=lambda o: o.estimated_time_min) if free_options else None

            # Pareto optimal: no other option is both cheaper AND faster
            pareto = []
            for o in options:
                dominated = False
                for other in options:
                    if (other.cost_float <= o.cost_float and
                            other.estimated_time_min < o.estimated_time_min):
                        dominated = True
                        break
                if not dominated:
                    pareto.append(o)
            tradeoff.pareto_optimal = pareto

        return tradeoff

    def get_backend_status(self) -> Dict[str, Any]:
        """
        Get status of all available backends.

        Returns:
            Backend availability status with local hardware info
        """
        backends = {
            "local": {
                "available": True,
                "has_gpu": self._local_hardware["has_gpu"],
                "gpu_name": self._local_hardware.get("gpu_name"),
                "gpu_vram_gb": round(self._local_hardware.get("gpu_vram_gb", 0), 1),
                "ram_gb": round(self._local_hardware.get("ram_gb", 0), 1),
                "cpu_count": self._local_hardware.get("cpu_count"),
                "cpu_name": self._local_hardware.get("cpu_name"),
            },
        }

        for backend, caps in self.BACKEND_CAPABILITIES.items():
            backends[backend.value] = {
                "available": True,
                "free": caps["free"],
                "max_vram_gb": caps["max_vram"],
                "gpu": caps["gpu"],
                "description": caps["description"],
                "risk": caps["risk"],
                "requires": self._get_backend_requirements(backend),
            }

        # Check availability of API keys
        backends["runpod"]["api_configured"] = bool(os.environ.get("RUNPOD_API_KEY"))
        backends["vast"]["api_configured"] = bool(os.environ.get("VAST_API_KEY"))
        backends["spaces"]["api_configured"] = bool(os.environ.get("HF_TOKEN"))

        return backends

    def get_full_recommendation(
        self,
        model_name: str,
        model_size_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get complete recommendation with trade-off analysis.

        Args:
            model_name: Model to analyze
            model_size_gb: Optional model size

        Returns:
            Complete recommendation dictionary
        """
        model_gb = model_size_gb or self.estimate_model_size(model_name)
        recommendation = self.auto_select_backend(model_name, model_gb)
        tradeoff = self.analyze_tradeoffs(model_gb)

        return {
            "model": model_name,
            "estimated_size_gb": round(model_gb, 2),
            "vram_required_gb": round(model_gb * 1.3, 2),
            "local_hardware": self._local_hardware,
            "recommendation": {
                "backend": recommendation.backend.value,
                "reason": recommendation.reason,
                "estimated_time": recommendation.estimated_time_str,
                "cost": recommendation.cost,
                "risk": recommendation.risk_level,
            },
            "all_options": [
                {
                    "backend": o.backend.value,
                    "time": o.estimated_time_str,
                    "cost": o.cost,
                    "free": o.is_free,
                }
                for o in tradeoff.options
            ],
            "pareto_optimal": [
                {
                    "backend": o.backend.value,
                    "time": o.estimated_time_str,
                    "cost": o.cost,
                }
                for o in tradeoff.pareto_optimal
            ],
            "best_free": (
                {
                    "backend": tradeoff.best_free.backend.value,
                    "time": tradeoff.best_free.estimated_time_str,
                }
                if tradeoff.best_free else None
            ),
            "fastest": (
                {
                    "backend": tradeoff.fastest.backend.value,
                    "time": tradeoff.fastest.estimated_time_str,
                    "cost": tradeoff.fastest.cost,
                }
                if tradeoff.fastest else None
            ),
        }

    def get_recommendation(
        self,
        model_name: str,
        model_size_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get complete recommendation (alias for get_full_recommendation)."""
        return self.get_full_recommendation(model_name, model_size_gb)

    # ---- Internal helpers ----

    def _recommend_backend(self, backend: Backend, required_vram: float) -> BackendRecommendation:
        """Build a backend recommendation."""
        caps = self.BACKEND_CAPABILITIES.get(backend, {})
        total_time = caps.get("setup_time_min", 5) + required_vram * 3
        return BackendRecommendation(
            backend=backend,
            reason=caps.get("description", str(backend)),
            estimated_time_min=total_time,
            estimated_time_str=f"{total_time:.0f} minutes",
            cost="Free" if caps.get("free") else "Paid",
            cost_float=0.0 if caps.get("free") else 0.15,
            requirements=self._get_backend_requirements(backend),
            is_free=caps.get("free", True),
            vram_required_gb=required_vram,
            vram_available_gb=caps.get("max_vram_gb", 0),
            risk_level=caps.get("risk", "medium"),
            confidence=0.8,
        )

    def _get_backend_requirements(self, backend: Backend) -> List[str]:
        """Get requirements for a specific backend."""
        requirements = {
            Backend.COLAB: ["Google account", "Browser"],
            Backend.SPACES: ["HuggingFace account", "HF_TOKEN env var"],
            Backend.KAGGLE: ["Kaggle account", "Phone verified"],
            Backend.RUNPOD: ["RunPod account", "RUNPOD_API_KEY env var", "Payment method"],
            Backend.VAST: ["Vast.ai account", "VAST_API_KEY env var", "Payment method"],
        }
        return requirements.get(backend, ["Account required"])

    def can_run_locally(self, model_size_gb: float) -> bool:
        """Check if model can run on local hardware."""
        hw = self._local_hardware
        required = model_size_gb * 1.3

        if hw["has_gpu"] and hw["gpu_vram_gb"] >= required:
            return True
        if hw["ram_gb"] >= required + 2:
            return True
        return False

    @staticmethod
    def print_tradeoff_table(analysis: TradeoffAnalysis) -> str:
        """
        Print a human-readable trade-off table.

        Args:
            analysis: TradeoffAnalysis from analyze_tradeoffs

        Returns:
            Formatted table string
        """
        lines = []
        lines.append(f"{'Backend':<12} {'Free':<6} {'Time':<15} {'Cost':<10} {'Risk':<8}")
        lines.append("-" * 55)
        for o in analysis.options:
            lines.append(
                f"{o.backend.value:<12} {'Yes' if o.is_free else 'No':<6} "
                f"{o.estimated_time_str:<15} {o.cost:<10} {o.risk_level:<8}"
            )
        return "\n".join(lines)
