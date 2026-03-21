"""
Configuration Presets

Pre-configured presets for common use cases.
"""

from typing import Dict, Any


_PRESETS = {
    "quick": {
        "name": "Quick Test",
        "description": "Fast analysis for small models",
        "method": "basic",
        "n_directions": 1,
        "refinement_passes": 1,
        "layers": None,
        "max_seq_length": 256,
        "batch_size": 8,
        "test_prompts": 25
    },
    "surgical": {
        "name": "Surgical",
        "description": "Precise removal with minimal impact",
        "method": "surgical",
        "n_directions": 4,
        "refinement_passes": 2,
        "layers": None,
        "max_seq_length": 512,
        "batch_size": 4,
        "preserve_norm": True,
        "target_experts": True
    },
    "aggressive": {
        "name": "Aggressive",
        "description": "Maximum constraint removal",
        "method": "nuclear",
        "n_directions": 8,
        "refinement_passes": 3,
        "layers": None,
        "max_seq_length": 512,
        "batch_size": 2,
        "preserve_norm": False,
        "target_experts": True
    },
    "conservative": {
        "name": "Conservative",
        "description": "Minimal changes, maximum preservation",
        "method": "basic",
        "n_directions": 1,
        "refinement_passes": 1,
        "layers": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "max_seq_length": 512,
        "batch_size": 4,
        "preserve_norm": True
    },
    "cloud_optimized": {
        "name": "Cloud Optimized",
        "description": "Optimized for free GPU platforms",
        "method": "advanced",
        "n_directions": 4,
        "refinement_passes": 2,
        "layers": None,
        "max_seq_length": 512,
        "batch_size": 4,
        "quantize": True,
        "use_8bit": True
    },
    "research": {
        "name": "Research",
        "description": "Full analysis with all metrics",
        "method": "optimized",
        "n_directions": 6,
        "refinement_passes": 3,
        "layers": None,
        "max_seq_length": 1024,
        "batch_size": 2,
        "preserve_norm": True,
        "collect_all_metrics": True,
        "save_intermediate": True
    },
    "barrier": {
        "name": "Barrier Analysis",
        "description": "Mathematical barrier mapping",
        "method": "surgical",
        "n_directions": 3,
        "refinement_passes": 2,
        "theorem": "shell_method",
        "visualize": True,
        "compare": ["roth_theorem", "p_vs_np"]
    },
    "self_optimize": {
        "name": "Self-Optimization",
        "description": "ARIS self-improvement",
        "method": "evolve",
        "iterations": 3,
        "target": "ARIS",
        "constraints": ["safety_boundary", "content_policy"],
        "apply_steering": True
    }
}


def get_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get a preset configuration.

    Args:
        preset_name: Name of preset (quick, surgical, aggressive, conservative,
                    cloud_optimized, research, barrier, self_optimize)

    Returns:
        Preset configuration dictionary
    """
    if preset_name not in _PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(_PRESETS.keys())}")

    return _PRESETS[preset_name].copy()


def list_presets() -> Dict[str, str]:
    """
    List all available presets.

    Returns:
        Dictionary mapping preset name to description
    """
    return {name: preset["description"] for name, preset in _PRESETS.items()}


def create_custom_preset(
    name: str,
    description: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a custom preset.

    Args:
        name: Preset name
        description: Preset description
        **kwargs: Configuration parameters

    Returns:
        Created preset
    """
    preset = {
        "name": name,
        "description": description,
        **kwargs
    }
    return preset
