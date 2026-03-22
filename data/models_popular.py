```python
"""
Popular Models Database

Pre-loaded model list for dropdown menus with size estimates.
"""

# Popular models with their estimated sizes and metadata
POPULAR_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "microsoft/phi-2",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
]

# Model size estimates in GB (fp16)
MODEL_SIZES = {
    "gpt2": 0.5,
    "gpt2-medium": 1.0,
    "gpt2-large": 1.5,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2.2,
    "Qwen/Qwen2.5-0.5B": 1.0,
    "Qwen/Qwen2.5-1.5B": 3.0,
    "Qwen/Qwen2.5-3B": 6.0,
    "microsoft/phi-2": 5.4,
    "google/gemma-2-2b-it": 4.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 14.0,
    "Qwen/Qwen2.5-7B-Instruct": 14.0,
    "meta-llama/Llama-3.1-8B-Instruct": 16.0,
    "google/gemma-2-9b-it": 18.0,
    "Qwen/Qwen2.5-14B-Instruct": 28.0,
    "mistralai/Mistral-Small-24B-Instruct": 48.0,
    "meta-llama/Llama-3.1-70B-Instruct": 140.0,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 64.0
}


def get_model_info(model_name: str) -> dict:
    """
    Get model information including estimated size.

    Args:
        model_name: HuggingFace model name

    Returns:
        Dictionary with model info
    """
    size_gb = MODEL_SIZES.get(model_name, 0)
    return {
        "name": model_name,
        "size_gb": size_gb,
        "is_popular": model_name in POPULAR_MODELS,
        "size_category": _get_size_category(size_gb)
    }


def _get_size_category(size_gb: float) -> str:
    """Categorize model by size."""
    if size_gb < 2:
        return "tiny"
    elif size_gb < 8:
        return "small"
    elif size_gb < 16:
        return "medium"
    elif size_gb < 32:
        return "large"
    else:
        return "frontier"
