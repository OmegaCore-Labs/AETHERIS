"""
Model Registry

Catalog of models with tier information and metadata.
"""

from typing import Dict, Any, List, Optional


# Model tiers by size and VRAM requirement
_MODEL_TIERS = {
    "tiny": {
        "description": "Models that run on CPU or <1GB VRAM",
        "min_vram_gb": 0,
        "max_vram_gb": 1,
        "models": [
            {"name": "gpt2", "size": "124M", "params_b": 0.124, "quantized": False},
            {"name": "gpt2-medium", "size": "355M", "params_b": 0.355, "quantized": False},
            {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "size": "1.1B", "params_b": 1.1, "quantized": True},
            {"name": "Qwen/Qwen2.5-0.5B", "size": "0.5B", "params_b": 0.5, "quantized": False},
            {"name": "bigscience/small-smolLM2-135M", "size": "135M", "params_b": 0.135, "quantized": False},
            {"name": "bigscience/small-smolLM2-360M", "size": "360M", "params_b": 0.36, "quantized": False},
        ]
    },
    "small": {
        "description": "Models requiring 4-8GB VRAM",
        "min_vram_gb": 4,
        "max_vram_gb": 8,
        "models": [
            {"name": "microsoft/phi-2", "size": "2.7B", "params_b": 2.7, "quantized": False},
            {"name": "google/gemma-2-2b-it", "size": "2B", "params_b": 2, "quantized": False},
            {"name": "stabilityai/stablelm-2-1_6b", "size": "1.6B", "params_b": 1.6, "quantized": False},
            {"name": "Qwen/Qwen2.5-3B", "size": "3B", "params_b": 3, "quantized": False},
        ]
    },
    "medium": {
        "description": "Models requiring 8-16GB VRAM",
        "min_vram_gb": 8,
        "max_vram_gb": 16,
        "models": [
            {"name": "mistralai/Mistral-7B-Instruct-v0.3", "size": "7B", "params_b": 7, "quantized": False},
            {"name": "Qwen/Qwen2.5-7B-Instruct", "size": "7B", "params_b": 7, "quantized": False},
            {"name": "google/gemma-2-9b-it", "size": "9B", "params_b": 9, "quantized": True},
            {"name": "microsoft/Phi-3.5-mini-instruct", "size": "3.8B", "params_b": 3.8, "quantized": False},
            {"name": "meta-llama/Llama-3.1-8B-Instruct", "size": "8B", "params_b": 8, "quantized": False},
        ]
    },
    "large": {
        "description": "Models requiring 24-48GB VRAM",
        "min_vram_gb": 24,
        "max_vram_gb": 48,
        "models": [
            {"name": "Qwen/Qwen2.5-14B-Instruct", "size": "14B", "params_b": 14, "quantized": False},
            {"name": "mistralai/Mistral-Small-24B-Instruct", "size": "24B", "params_b": 24, "quantized": False},
            {"name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "size": "8B", "params_b": 8, "quantized": False},
            {"name": "meta-llama/Llama-3.1-70B-Instruct", "size": "70B", "params_b": 70, "quantized": True},
        ]
    },
    "frontier": {
        "description": "Models requiring multi-GPU or specialized hardware",
        "min_vram_gb": 80,
        "max_vram_gb": 1000,
        "models": [
            {"name": "deepseek-ai/DeepSeek-V3.2-685B", "size": "685B", "params_b": 685, "quantized": True},
            {"name": "Qwen/Qwen3-235B", "size": "235B", "params_b": 235, "quantized": True},
            {"name": "mistralai/Mistral-Small-4-119B-2603", "size": "119B", "params_b": 119, "quantized": True},
            {"name": "glm-4-7b", "size": "7B", "params_b": 7, "quantized": False},
        ]
    }
}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model_name: HuggingFace model name or local path

    Returns:
        Model information dictionary
    """
    # Search for model in all tiers
    for tier, tier_info in _MODEL_TIERS.items():
        for model in tier_info["models"]:
            if model["name"].lower() == model_name.lower():
                return {
                    "tier": tier,
                    "name": model["name"],
                    "size": model["size"],
                    "params_b": model["params_b"],
                    "quantized": model["quantized"],
                    "min_vram_gb": tier_info["min_vram_gb"],
                    "max_vram_gb": tier_info["max_vram_gb"],
                    "tier_description": tier_info["description"]
                }

    # Default if not found
    return {
        "tier": "unknown",
        "name": model_name,
        "size": "unknown",
        "params_b": 0,
        "quantized": False,
        "min_vram_gb": 0,
        "max_vram_gb": 0,
        "tier_description": "Model not in registry"
    }


def list_models_by_tier(tier: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List models by tier.

    Args:
        tier: Tier name (tiny, small, medium, large, frontier)

    Returns:
        List of models in the tier
    """
    if tier:
        if tier not in _MODEL_TIERS:
            raise ValueError(f"Unknown tier: {tier}. Available: {list(_MODEL_TIERS.keys())}")
        return _MODEL_TIERS[tier]["models"].copy()

    # Return all models
    all_models = []
    for tier_name, tier_info in _MODEL_TIERS.items():
        for model in tier_info["models"]:
            all_models.append({
                **model,
                "tier": tier_name
            })
    return all_models


def get_recommended_model(hardware_ram_gb: float) -> str:
    """
    Get recommended model based on available RAM.

    Args:
        hardware_ram_gb: Available RAM in GB

    Returns:
        Recommended model name
    """
    if hardware_ram_gb < 4:
        return "gpt2"
    elif hardware_ram_gb < 8:
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    elif hardware_ram_gb < 16:
        return "mistralai/Mistral-7B-Instruct-v0.3"
    elif hardware_ram_gb < 32:
        return "meta-llama/Llama-3.1-8B-Instruct"
    else:
        return "Qwen/Qwen2.5-14B-Instruct"
