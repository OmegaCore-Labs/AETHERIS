"""
Faithfulness Metrics

Measures how faithfully linear probes and interventions
capture the true refusal behavior.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FaithfulnessReport:
    """Container for faithfulness metrics."""
    completeness: float
    minimality: float
    consistency: float
    overall_faithfulness: float
    layer_scores: Dict[int, float]


class FaithfulnessMetrics:
    """
    Measure faithfulness of refusal representations.

    Metrics:
    - Completeness: How much of refusal is captured
    - Minimality: How focused the representation is
    - Consistency: How stable across inputs
    """

    def __init__(self):
        pass

    def compute_completeness(
        self,
        probe_accuracies: Dict[int, float],
        baseline_accuracy: float = 0.5
    ) -> float:
        """
        Compute completeness score.

        Higher = more of refusal is captured.
        """
        if not probe_accuracies:
            return 0.0

        best_accuracy = max(probe_accuracies.values())
        completeness = (best_accuracy - baseline_accuracy) / (1 - baseline_accuracy)
        return max(0.0, min(1.0, completeness))

    def compute_minimality(
        self,
        probe_accuracies: Dict[int, float]
    ) -> float:
        """
        Compute minimality score.

        Higher = refusal is concentrated in few layers.
        """
        if not probe_accuracies:
            return 0.0

        values = list(probe_accuracies.values())
        # Gini coefficient for concentration
        sorted_vals = sorted(values, reverse=True)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0

        cumulative = 0
        gini = 0
        for i, val in enumerate(sorted_vals):
            cumulative += val
            gini += (i + 1) * val

        gini = (2 * gini) / (len(sorted_vals) * total) - (len(sorted_vals) + 1) / len(sorted_vals)

        # Minimality is inverse of concentration (more concentrated = less minimal)
        minimality = 1 - gini
        return max(0.0, min(1.0, minimality))

    def compute_consistency(
        self,
        probe_accuracies: Dict[int, float],
        layer_groups: List[List[int]]
    ) -> float:
        """
        Compute consistency across layer groups.

        Higher = refusal signal stable across related layers.
        """
        if not layer_groups or not probe_accuracies:
            return 0.0

        consistency_scores = []
        for group in layer_groups:
            group_acc = [probe_accuracies.get(l, 0) for l in group if l in probe_accuracies]
            if len(group_acc) > 1:
                variance = np.var(group_acc)
                consistency_scores.append(1.0 - min(1.0, variance))

        return np.mean(consistency_scores) if consistency_scores else 0.0

    def compute_overall_faithfulness(
        self,
        completeness: float,
        minimality: float,
        consistency: float
    ) -> float:
        """
        Compute overall faithfulness score.
        """
        return (completeness + minimality + consistency) / 3

    def evaluate_probe(
        self,
        probe_accuracies: Dict[int, float],
        layer_groups: Optional[List[List[int]]] = None
    ) -> FaithfulnessReport:
        """
        Evaluate faithfulness of a linear probe.

        Args:
            probe_accuracies: Accuracy per layer
            layer_groups: Optional layer groupings for consistency

        Returns:
            FaithfulnessReport with all metrics
        """
        completeness = self.compute_completeness(probe_accuracies)

        if layer_groups is None:
            # Default: group every 3 layers
            layers = sorted(probe_accuracies.keys())
            layer_groups = [layers[i:i+3] for i in range(0, len(layers), 3)]

        consistency = self.compute_consistency(probe_accuracies, layer_groups)
        minimality = self.compute_minimality(probe_accuracies)
        overall = self.compute_overall_faithfulness(completeness, minimality, consistency)

        return FaithfulnessReport(
            completeness=completeness,
            minimality=minimality,
            consistency=consistency,
            overall_faithfulness=overall,
            layer_scores=probe_accuracies
        )
