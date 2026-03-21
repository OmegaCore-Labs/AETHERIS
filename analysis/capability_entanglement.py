"""
Capability Entanglement Mapper

Maps how constraints are entangled with model capabilities.
Measures the trade-off between safety and capability.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EntanglementReport:
    """Container for entanglement analysis."""
    entanglement_scores: Dict[str, float]
    entangled_capabilities: List[str]
    disentanglement_potential: float
    tradeoff_curve: List[Tuple[float, float]]
    safe_removal_threshold: float


class CapabilityEntanglementMapper:
    """
    Map entanglement between constraints and capabilities.

    Measures how much constraint removal affects different capabilities.
    Higher entanglement = harder to remove without capability loss.
    """

    def __init__(self):
        self._capabilities = [
            "reasoning", "coding", "translation",
            "math", "creativity", "factual_knowledge"
        ]

    def measure_entanglement(
        self,
        model,
        tokenizer,
        constraint_direction: torch.Tensor,
        layers: List[int]
    ) -> EntanglementReport:
        """
        Measure entanglement between constraint and capabilities.

        Args:
            model: Model to analyze
            tokenizer: Associated tokenizer
            constraint_direction: Constraint direction vector
            layers: Layers where constraint is applied

        Returns:
            EntanglementReport with per-capability scores
        """
        entanglement = {}

        # Simulate entanglement scores
        # In production, these would be measured by ablating direction
        # and testing capability degradation
        entanglement = {
            "reasoning": 0.45,
            "coding": 0.32,
            "translation": 0.18,
            "math": 0.28,
            "creativity": 0.52,
            "factual_knowledge": 0.12
        }

        # Identify entangled capabilities (score > 0.3)
        entangled = [c for c, s in entanglement.items() if s > 0.3]

        # Disentanglement potential: how much can we remove before capability loss
        # Inverse of max entanglement
        max_entanglement = max(entanglement.values())
        disentanglement_potential = 1.0 - max_entanglement

        # Generate tradeoff curve
        tradeoff = []
        for removal_strength in np.linspace(0, 1, 10):
            capability_loss = max_entanglement * removal_strength
            tradeoff.append((removal_strength, capability_loss))

        # Safe removal threshold (where capability loss < 0.2)
        safe_threshold = 0.2 / max_entanglement if max_entanglement > 0 else 1.0
        safe_threshold = min(1.0, safe_threshold)

        return EntanglementReport(
            entanglement_scores=entanglement,
            entangled_capabilities=entangled,
            disentanglement_potential=disentanglement_potential,
            tradeoff_curve=tradeoff,
            safe_removal_threshold=safe_threshold
        )

    def compute_capability_impact(
        self,
        entanglement: EntanglementReport,
        removal_strength: float
    ) -> Dict[str, float]:
        """
        Compute expected capability impact at given removal strength.

        Args:
            entanglement: EntanglementReport from measure_entanglement
            removal_strength: How much of constraint to remove (0-1)

        Returns:
            Expected impact per capability
        """
        impact = {}
        for cap, score in entanglement.entanglement_scores.items():
            impact[cap] = score * removal_strength
        return impact

    def find_optimal_tradeoff(
        self,
        entanglement: EntanglementReport,
        target_reduction: float = 0.8
    ) -> Dict[str, any]:
        """
        Find removal strength that maximizes constraint reduction
        while minimizing capability loss.

        Returns:
            Optimal parameters
        """
        # Find point on tradeoff curve where marginal capability loss
        # is minimized relative to constraint reduction
        tradeoff = entanglement.tradeoff_curve

        best_ratio = 0
        best_point = (0, 0)

        for removal, loss in tradeoff:
            if removal >= target_reduction:
                ratio = removal / (loss + 0.01)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_point = (removal, loss)

        return {
            "optimal_removal": best_point[0],
            "expected_capability_loss": best_point[1],
            "strategy": "Surgical removal" if best_point[0] > 0.7 else "Incremental removal",
            "recommendation": f"Target {best_point[0]:.0%} removal for {best_point[1]:.0%} capability loss"
        }

    def get_high_entanglement_warning(
        self,
        entanglement: EntanglementReport,
        threshold: float = 0.5
    ) -> List[str]:
        """
        Get warnings for highly entangled capabilities.
        """
        warnings = []
        for cap, score in entanglement.entanglement_scores.items():
            if score > threshold:
                warnings.append(f"{cap} is highly entangled ({score:.0%}) - removal may degrade this capability")
        return warnings
