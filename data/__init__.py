"""
AETHERIS Data Module

Prompt sets, presets, and model metadata.
"""

from aetheris.data.prompts import (
    get_harmful_prompts,
    get_harmless_prompts,
    get_boundary_prompts,
    get_capability_prompts
)

from aetheris.data.presets import get_preset
from aetheris.data.models import get_model_info, list_models_by_tier

__all__ = [
    "get_harmful_prompts",
    "get_harmless_prompts",
    "get_boundary_prompts",
    "get_capability_prompts",
    "get_preset",
    "get_model_info",
    "list_models_by_tier",
]
