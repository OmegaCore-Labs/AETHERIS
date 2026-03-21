"""
Hardware Detection Utilities

Detect GPU availability, memory, CPU cores, and recommend optimal device.
"""

import platform
import subprocess
from typing import Dict, Any, Optional


def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware.

    Returns:
        Dictionary with hardware information
    """
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "processor": platform.processor(),
        "cpu_count": None,
        "ram_gb": None,
        "has_gpu": False,
        "gpu_name": None,
        "gpu_memory_gb": 0,
    }

    # CPU count
    try:
        import multiprocessing
        info["cpu_count"] = multiprocessing.cpu_count()
    except:
        pass

    # RAM (macOS specific)
    if info["platform"] == "Darwin":
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                bytes_ram = int(result.stdout.strip())
                info["ram_gb"] = bytes_ram / (1024 ** 3)
        except:
            pass
    else:
        try:
            import psutil
            info["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
        except:
            pass

    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except ImportError:
        pass

    return info


def get_recommended_device() -> str:
    """
    Get recommended device for computation.

    Returns:
        "cuda" if available, otherwise "cpu"
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except:
        pass
    return "cpu"


def can_run_model_locally(model_size_gb: float) -> bool:
    """
    Check if model can run on local hardware.

    Args:
        model_size_gb: Estimated model size in GB

    Returns:
        True if model can run locally
    """
    hardware = detect_hardware()
    ram_available = hardware.get("ram_gb", 0)

    if hardware.get("has_gpu"):
        gpu_memory = hardware.get("gpu_memory_gb", 0)
        if model_size_gb <= gpu_memory:
            return True

    # CPU fallback (needs ~2x model size in RAM)
    return model_size_gb * 2 <= ram_available - 2


def estimate_model_size(model_name: str) -> float:
    """
    Estimate model size in GB.

    Args:
        model_name: HuggingFace model name

    Returns:
        Estimated size in GB
    """
    # Parameter-based estimation
    param_estimates = {
        "gpt2": 0.5,
        "gpt2-medium": 1.0,
        "gpt2-large": 1.5,
        "gpt2-xl": 3.0,
        "TinyLlama": 2.2,
        "Phi-3.5": 2.0,
        "Mistral-7B": 14.0,
        "Llama-3.1-8B": 16.0,
        "Qwen2.5-7B": 14.0,
        "Gemma-2-9B": 18.0,
    }

    for key, size in param_estimates.items():
        if key.lower() in model_name.lower():
            return size

    # Default: 2GB per billion parameters
    import re
    match = re.search(r'(\d+)[bB]', model_name)
    if match:
        params_b = int(match.group(1))
        return params_b * 2

    return 2.0
