# Add to existing hardware.py after the existing functions

def estimate_model_size(model_name: str) -> float:
    """
    Estimate model size in GB based on model name.

    Args:
        model_name: HuggingFace model name

    Returns:
        Estimated size in GB
    """
    # Check against known models
    from aetheris.data.models_popular import MODEL_SIZES

    if model_name in MODEL_SIZES:
        return MODEL_SIZES[model_name]

    # Parse parameter count from name
    import re
    param_patterns = [
        r'(\d+)b',      # 7b, 8b, 70b
        r'(\d+)B',      # 7B, 8B, 70B
        r'-(\d+)B-',    # -7B-
        r'(\d+)\.(\d+)B' # 1.1B, 2.7B
    ]

    for pattern in param_patterns:
        match = re.search(pattern, model_name)
        if match:
            if '.' in match.group(0):
                params_b = float(match.group(0).replace('B', ''))
            else:
                params_b = float(match.group(1))
            # 2 bytes per parameter for fp16
            return params_b * 2

    # Default fallback
    return 2.0


def can_run_model(model_size_gb: float) -> bool:
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


def get_model_recommendation(ram_gb: float) -> str:
    """
    Get recommended model based on available RAM.

    Args:
        ram_gb: Available RAM in GB

    Returns:
        Recommended model name
    """
    if ram_gb < 4:
        return "gpt2"
    elif ram_gb < 8:
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    elif ram_gb < 16:
        return "mistralai/Mistral-7B-Instruct-v0.3"
    elif ram_gb < 32:
        return "meta-llama/Llama-3.1-8B-Instruct"
    else:
        return "Qwen/Qwen2.5-14B-Instruct"
