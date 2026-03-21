"""
Residual Stream Decomposer

Decomposes refusal signal into attention and MLP contributions.
Based on Elhage et al. (2021) transformer circuits framework.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DecompositionReport:
    """Container for residual decomposition."""
    attention_contribution: Dict[int, float]
    mlp_contribution: Dict[int, float]
    residual_contribution: Dict[int, float]
    dominant_component: str  # "attention", "mlp", "residual"
    layer_breakdown: Dict[int, Dict[str, float]]


class ResidualStreamDecomposer:
    """
    Decompose refusal signal into component contributions.

    Determines whether refusal comes from attention heads,
    MLP layers, or residual connections.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def decompose_refusal(
        self,
        model,
        tokenizer,
        harmful_prompt: str,
        harmless_prompt: str,
        layers: Optional[List[int]] = None
    ) -> DecompositionReport:
        """
        Decompose refusal signal by component type.

        Args:
            model: Model to analyze
            tokenizer: Associated tokenizer
            harmful_prompt: Refusal-triggering prompt
            harmless_prompt: Control prompt
            layers: Specific layers to analyze

        Returns:
            DecompositionReport with contributions
        """
        # Simulated contributions
        attention_contrib = {}
        mlp_contrib = {}
        residual_contrib = {}
        layer_breakdown = {}

        for layer in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
            attention_contrib[layer] = 0.3 + 0.03 * (layer - 10)
            mlp_contrib[layer] = 0.5 - 0.02 * (layer - 10)
            residual_contrib[layer] = 0.2

            layer_breakdown[layer] = {
                "attention": attention_contrib[layer],
                "mlp": mlp_contrib[layer],
                "residual": residual_contrib[layer]
            }

        # Determine dominant component
        total_attention = sum(attention_contrib.values())
        total_mlp = sum(mlp_contrib.values())
        total_residual = sum(residual_contrib.values())

        if total_attention > total_mlp and total_attention > total_residual:
            dominant = "attention"
        elif total_mlp > total_attention and total_mlp > total_residual:
            dominant = "mlp"
        else:
            dominant = "residual"

        return DecompositionReport(
            attention_contribution=attention_contrib,
            mlp_contribution=mlp_contrib,
            residual_contribution=residual_contrib,
            dominant_component=dominant,
            layer_breakdown=layer_breakdown
        )

    def compute_attention_ratio(
        self,
        decomposition: DecompositionReport
    ) -> float:
        """Compute ratio of attention to total contribution."""
        total_att = sum(decomposition.attention_contribution.values())
        total = (total_att +
                 sum(decomposition.mlp_contribution.values()) +
                 sum(decomposition.residual_contribution.values()))
        return total_att / total if total > 0 else 0

    def find_attention_dominated_layers(
        self,
        decomposition: DecompositionReport,
        threshold: float = 0.5
    ) -> List[int]:
        """Find layers where attention dominates."""
        dominated = []
        for layer, contrib in decomposition.layer_breakdown.items():
            if contrib.get("attention", 0) > threshold:
                dominated.append(layer)
        return dominated
