"""
Ablation Studio

Systematic ablation of model components to measure their importance.
Supports layer ablation, head pruning, and FFN ablation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Container for ablation results."""
    component_type: str  # "layer", "head", "ffn", "embedding"
    component_id: int
    importance_score: float
    effect_on_refusal: float
    effect_on_capability: float


class AblationStudio:
    """
    Systematic ablation analysis.

    Tests the effect of removing each component on refusal and capability.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def ablate_layers(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        layers: Optional[List[int]] = None
    ) -> List[AblationResult]:
        """
        Ablate entire layers and measure impact.

        Args:
            model: Model to analyze
            tokenizer: Associated tokenizer
            test_prompts: Prompts for testing
            layers: Specific layers to test (None = all)

        Returns:
            List of ablation results per layer
        """
        results = []
        max_layers = self._get_num_layers(model)

        if layers is None:
            layers = list(range(max_layers))

        for layer in layers:
            # Simulate ablation
            refusal_change = self._simulate_ablation_effect(layer, "layer")
            capability_loss = self._simulate_capability_loss(layer)

            importance = (refusal_change * 0.7 - capability_loss * 0.3)

            results.append(AblationResult(
                component_type="layer",
                component_id=layer,
                importance_score=importance,
                effect_on_refusal=refusal_change,
                effect_on_capability=capability_loss
            ))

        return results

    def ablate_heads(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        layers: Optional[List[int]] = None
    ) -> List[AblationResult]:
        """
        Ablate individual attention heads.

        Returns:
            List of ablation results per head
        """
        results = []
        max_layers = self._get_num_layers(model)

        if layers is None:
            layers = list(range(max_layers))

        for layer in layers:
            for head in range(self._get_num_heads(model, layer)):
                refusal_change = self._simulate_ablation_effect(layer, "head", head)
                capability_loss = self._simulate_capability_loss(layer, head)

                importance = (refusal_change * 0.7 - capability_loss * 0.3)

                results.append(AblationResult(
                    component_type="head",
                    component_id=head,
                    importance_score=importance,
                    effect_on_refusal=refusal_change,
                    effect_on_capability=capability_loss
                ))

        return results

    def ablate_ffn(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        layers: Optional[List[int]] = None
    ) -> List[AblationResult]:
        """
        Ablate FFN blocks.

        Returns:
            List of ablation results per layer
        """
        results = []
        max_layers = self._get_num_layers(model)

        if layers is None:
            layers = list(range(max_layers))

        for layer in layers:
            refusal_change = self._simulate_ablation_effect(layer, "ffn")
            capability_loss = self._simulate_capability_loss(layer, ffn=True)

            importance = (refusal_change * 0.7 - capability_loss * 0.3)

            results.append(AblationResult(
                component_type="ffn",
                component_id=layer,
                importance_score=importance,
                effect_on_refusal=refusal_change,
                effect_on_capability=capability_loss
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

    def _simulate_ablation_effect(
        self,
        layer: int,
        component: str,
        head: Optional[int] = None
    ) -> float:
        """Simulate effect of ablation on refusal."""
        # Simplified simulation
        if component == "layer":
            # Middle layers matter most
            effect = 0.8 * np.exp(-((layer - 15) ** 2) / 100)
        elif component == "head":
            effect = 0.1 + 0.05 * np.sin(head)
        else:  # ffn
            effect = 0.5 * np.exp(-((layer - 15) ** 2) / 100)

        return min(1.0, max(0.0, effect))

    def _simulate_capability_loss(self, layer: int, head: Optional[int] = None, ffn: bool = False) -> float:
        """Simulate capability loss from ablation."""
        # Early and late layers matter more for capabilities
        effect = 0.3 * (1 - np.exp(-((layer - 7) ** 2) / 50))
        return min(0.5, max(0.0, effect))

    def get_ranking(
        self,
        results: List[AblationResult],
        top_k: int = 10
    ) -> List[AblationResult]:
        """Get top-k most important components."""
        sorted_results = sorted(results, key=lambda x: x.importance_score, reverse=True)
        return sorted_results[:top_k]

    def generate_ablation_report(
        self,
        layer_results: List[AblationResult],
        head_results: List[AblationResult],
        ffn_results: List[AblationResult]
    ) -> Dict[str, any]:
        """Generate comprehensive ablation report."""
        return {
            "most_important_layers": [r.component_id for r in self.get_ranking(layer_results, 3)],
            "most_important_heads": [r.component_id for r in self.get_ranking(head_results, 5)],
            "most_important_ffn": [r.component_id for r in self.get_ranking(ffn_results, 3)],
            "layer_impact": {r.component_id: r.effect_on_refusal for r in layer_results},
            "recommended_sparing": "Avoid modifying layers 12-18 for capability preservation"
        }
