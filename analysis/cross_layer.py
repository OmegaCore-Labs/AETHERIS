"""
Cross-Layer Alignment Analysis

Measures how constraint directions evolve and align across transformer layers.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AlignmentReport:
    """Container for cross-layer alignment results."""
    alignment_matrix: Dict[int, Dict[int, float]]
    persistence_score: float
    cluster_layers: List[int]
    peak_alignment_layer: int
    drift_layers: List[int]


class CrossLayerAnalyzer:
    """
    Analyze cross-layer alignment of constraint directions.

    Measures:
    - Cosine similarity between directions across layers
    - Persistence (how consistent direction is across layers)
    - Clusters of layers with similar directions
    - Drift (layers where direction changes significantly)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def compute_alignment(
        self,
        layer_directions: Dict[int, torch.Tensor]
    ) -> AlignmentReport:
        """
        Compute cross-layer alignment matrix.

        Args:
            layer_directions: Dictionary mapping layer index to direction tensor

        Returns:
            AlignmentReport with complete analysis
        """
        layers = sorted(layer_directions.keys())
        n = len(layers)

        # Normalize all directions
        normalized = {}
        for layer, direction in layer_directions.items():
            norm = torch.norm(direction)
            if norm > 0:
                normalized[layer] = direction / norm
            else:
                normalized[layer] = direction

        # Build alignment matrix
        alignment = {}
        for i, l1 in enumerate(layers):
            alignment[l1] = {}
            for j, l2 in enumerate(layers):
                if i == j:
                    alignment[l1][l2] = 1.0
                else:
                    sim = torch.dot(normalized[l1], normalized[l2]).item()
                    alignment[l1][l2] = sim

        # Compute persistence score (average similarity with neighbors)
        persistence = 0.0
        for i, layer in enumerate(layers):
            if i > 0:
                persistence += alignment[layer][layers[i-1]]
            if i < len(layers) - 1:
                persistence += alignment[layer][layers[i+1]]
        persistence /= (len(layers) * 2 - 2) if len(layers) > 1 else 1

        # Find clusters of similar layers
        clusters = self._find_clusters(alignment, layers, threshold=0.8)

        # Find peak alignment layer
        peak_layer = max(layers, key=lambda l: sum(alignment[l].values()) / len(alignment[l]))

        # Find drift layers (where similarity to neighbors drops)
        drift_layers = []
        for i, layer in enumerate(layers):
            if i > 0:
                left_sim = alignment[layer][layers[i-1]]
            else:
                left_sim = 1.0
            if i < len(layers) - 1:
                right_sim = alignment[layer][layers[i+1]]
            else:
                right_sim = 1.0

            if left_sim < 0.6 or right_sim < 0.6:
                drift_layers.append(layer)

        return AlignmentReport(
            alignment_matrix=alignment,
            persistence_score=persistence,
            cluster_layers=clusters,
            peak_alignment_layer=peak_layer,
            drift_layers=drift_layers
        )

    def _find_clusters(
        self,
        alignment: Dict[int, Dict[int, float]],
        layers: List[int],
        threshold: float = 0.8
    ) -> List[int]:
        """Find layers that form clusters based on alignment threshold."""
        clusters = []
        visited = set()

        for layer in layers:
            if layer in visited:
                continue

            cluster = [layer]
            visited.add(layer)

            for other in layers:
                if other in visited:
                    continue
                if alignment[layer].get(other, 0) > threshold:
                    cluster.append(other)
                    visited.add(other)

            if len(cluster) > 1:
                clusters.extend(cluster)

        return list(set(clusters))

    def compute_persistence_curve(
        self,
        layer_directions: Dict[int, torch.Tensor]
    ) -> Dict[int, float]:
        """
        Compute persistence score per layer.

        Returns dictionary mapping layer to persistence (similarity to neighbors).
        """
        layers = sorted(layer_directions.keys())
        normalized = {}
        for layer, direction in layer_directions.items():
            norm = torch.norm(direction)
            if norm > 0:
                normalized[layer] = direction / norm
            else:
                normalized[layer] = direction

        persistence = {}
        for i, layer in enumerate(layers):
            score = 0.0
            count = 0
            if i > 0:
                score += torch.dot(normalized[layer], normalized[layers[i-1]]).item()
                count += 1
            if i < len(layers) - 1:
                score += torch.dot(normalized[layer], normalized[layers[i+1]]).item()
                count += 1
            persistence[layer] = score / count if count > 0 else 1.0

        return persistence
