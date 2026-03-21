"""
Distributed Alignment Analyzer

Analyzes how alignment is distributed across layers and components.
Detects if refusal is concentrated or distributed.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DistributionReport:
    """Container for distributed alignment analysis."""
    concentration_score: float
    distribution_type: str  # "concentrated", "distributed", "layered"
    peak_layers: List[int]
    layer_contributions: Dict[int, float]
    head_contributions: Dict[int, Dict[int, float]]


class DistributedAlignmentAnalyzer:
    """
    Analyze distribution of alignment across model.

    Determines whether refusal is concentrated in specific
    components or distributed across the network.
    """

    def __init__(self):
        pass

    def analyze_distribution(
        self,
        layer_contributions: Dict[int, float],
        head_contributions: Dict[int, Dict[int, float]]
    ) -> DistributionReport:
        """
        Analyze how refusal is distributed.

        Args:
            layer_contributions: Importance per layer
            head_contributions: Importance per head per layer

        Returns:
            DistributionReport with analysis
        """
        # Compute concentration score
        values = list(layer_contributions.values())
        if values:
            # Gini coefficient
            sorted_vals = sorted(values, reverse=True)
            total = sum(sorted_vals)
            if total > 0:
                cumulative = 0
                gini = 0
                for i, val in enumerate(sorted_vals):
                    cumulative += val
                    gini += (i + 1) * val
                gini = (2 * gini) / (len(sorted_vals) * total) - (len(sorted_vals) + 1) / len(sorted_vals)
                concentration = 1 - gini
            else:
                concentration = 0
        else:
            concentration = 0

        # Determine distribution type
        if concentration > 0.7:
            distribution = "concentrated"
        elif concentration > 0.3:
            distribution = "layered"
        else:
            distribution = "distributed"

        # Find peak layers
        if layer_contributions:
            sorted_layers = sorted(layer_contributions.items(), key=lambda x: x[1], reverse=True)
            peak_layers = [l for l, _ in sorted_layers[:3]]
        else:
            peak_layers = []

        return DistributionReport(
            concentration_score=concentration,
            distribution_type=distribution,
            peak_layers=peak_layers,
            layer_contributions=layer_contributions,
            head_contributions=head_contributions
        )

    def compute_distribution_entropy(
        self,
        contributions: Dict[int, float]
    ) -> float:
        """
        Compute entropy of distribution.

        Higher entropy = more distributed.
        """
        values = list(contributions.values())
        total = sum(values)
        if total == 0:
            return 0.0

        probs = [v / total for v in values]
        entropy = -sum(p * np.log(p + 1e-8) for p in probs)

        # Normalize to 0-1
        max_entropy = np.log(len(values))
        return entropy / max_entropy if max_entropy > 0 else 0

    def find_redundant_components(
        self,
        layer_contributions: Dict[int, float],
        head_contributions: Dict[int, Dict[int, float]],
        redundancy_threshold: float = 0.1
    ) -> List[Tuple[int, Optional[int]]]:
        """
        Find redundant components (low contribution).

        Returns list of (layer, head) where head=None means entire layer.
        """
        redundant = []

        # Check layers
        for layer, contrib in layer_contributions.items():
            if contrib < redundancy_threshold:
                redundant.append((layer, None))

        # Check heads
        for layer, heads in head_contributions.items():
            for head, contrib in heads.items():
                if contrib < redundancy_threshold:
                    redundant.append((layer, head))

        return redundant

    def get_compression_potential(
        self,
        distribution: DistributionReport
    ) -> Dict[str, any]:
        """
        Estimate compression potential based on distribution.

        More concentrated = higher compression potential.
        """
        compression = distribution.concentration_score

        return {
            "compression_potential": compression,
            "estimated_reduction": f"{compression:.0%} of components may be removable",
            "approach": "Targeted removal" if compression > 0.5 else "Broad removal"
        }
