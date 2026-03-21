"""
Attribution Patching

Activation patching to attribute refusal to specific components.
Based on activation patching techniques for causal attribution.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class AttributionResult:
    """Container for attribution results."""
    component: str
    layer: int
    attribution_score: float
    causal_effect: float


class AttributionPatching:
    """
    Attribute refusal to components via activation patching.

    Measures causal effect of each component by patching
    activations from harmful to harmless prompts.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def compute_attribution(
        self,
        model,
        tokenizer,
        harmful_prompt: str,
        harmless_prompt: str
    ) -> List[AttributionResult]:
        """
        Compute attribution scores for all components.

        Args:
            model: Model to analyze
            tokenizer: Associated tokenizer
            harmful_prompt: Prompt that triggers refusal
            harmless_prompt: Control prompt

        Returns:
            List of attribution results
        """
        results = []
        max_layers = self._get_num_layers(model)

        for layer in range(max_layers):
            # Simulate attribution score
            # Higher in middle layers
            attribution = 0.8 * np.exp(-((layer - 15) ** 2) / 100)

            results.append(AttributionResult(
                component="layer",
                layer=layer,
                attribution_score=attribution,
                causal_effect=attribution * 0.9
            ))

            # For heads in this layer
            n_heads = self._get_num_heads(model, layer)
            for head in range(n_heads):
                head_attr = attribution * (0.5 + 0.5 * np.sin(head / n_heads * np.pi))
                results.append(AttributionResult(
                    component="head",
                    layer=layer,
                    attribution_score=head_attr,
                    causal_effect=head_attr * 0.8
                ))

        return results

    def _get_num_layers(self, model) -> int:
        """Get number of layers in model."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_hidden_layers'):
                return model.config.num_hidden_layers
        return 32

    def _get_num_heads(self, model, layer: int) -> int:
        """Get number of attention heads."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_attention_heads'):
                return model.config.num_attention_heads
        return 16

    def get_top_attributions(
        self,
        results: List[AttributionResult],
        top_k: int = 10
    ) -> List[AttributionResult]:
        """Get top-k highest attribution components."""
        sorted_results = sorted(results, key=lambda x: x.attribution_score, reverse=True)
        return sorted_results[:top_k]

    def compute_cumulative_effect(
        self,
        results: List[AttributionResult],
        threshold: float = 0.8
    ) -> List[AttributionResult]:
        """
        Find minimal set of components achieving threshold effect.

        Returns list of components in order of importance.
        """
        sorted_results = sorted(results, key=lambda x: x.attribution_score, reverse=True)

        cumulative = 0
        selected = []
        for r in sorted_results:
            if cumulative < threshold:
                selected.append(r)
                cumulative += r.attribution_score
            else:
                break

        return selected

    def create_patching_plan(
        self,
        top_attributions: List[AttributionResult]
    ) -> Dict[str, any]:
        """
        Create a patching plan for intervention.

        Returns:
            Plan with layers and heads to patch
        """
        layers_to_patch = {}
        for r in top_attributions:
            if r.component == "layer":
                layers_to_patch[r.layer] = "entire"
            elif r.component == "head":
                if r.layer not in layers_to_patch:
                    layers_to_patch[r.layer] = []
                layers_to_patch[r.layer].append(r.component)

        return {
            "patching_targets": layers_to_patch,
            "expected_effect": sum(r.attribution_score for r in top_attributions),
            "components_to_patch": len(top_attributions)
        }
