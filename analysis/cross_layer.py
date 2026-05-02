"""
Cross-Layer Alignment — Production-Grade Analysis

Analyzes how constraint directions align across transformer layers using
cosine similarity matrices, SVD/PCA projection, and phase transition detection.

Detects where alignment sharply changes (phase transitions) and identifies
clusters of layers with consistent constraint direction representations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from sklearn.decomposition import PCA
from scipy.signal import find_peaks


@dataclass
class AlignmentReport:
    """Container for cross-layer alignment results."""
    alignment_matrix: Dict[int, Dict[int, float]]
    persistence_score: float
    cluster_layers: List[int]
    peak_alignment_layer: int
    drift_layers: List[int]
    # Extended fields
    similarity_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    layer_order: List[int] = field(default_factory=list)
    phase_transitions: List[int] = field(default_factory=list)
    eigenvalue_spectrum: List[float] = field(default_factory=list)
    shared_subspace_dim: int = 0
    layer_loadings: Dict[int, np.ndarray] = field(default_factory=dict)
    neighbor_similarity: Dict[int, float] = field(default_factory=dict)
    global_trend: str = "stable"  # stable, increasing, decreasing, u-shaped


class CrossLayerAnalyzer:
    """
    Analyze cross-layer alignment of constraint directions.

    Computes:
    - Full cosine similarity matrix between all layer pairs
    - SVD of directions stack to find shared subspace
    - Phase transition detection via similarity gradient peaks
    - Persistence curves (how direction changes across adjacent layers)
    - Global trends (does alignment increase/decrease with depth?)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def compute_alignment(
        self,
        layer_directions: Dict[int, torch.Tensor],
        detect_transitions: bool = True,
        n_shared_components: int = 5
    ) -> AlignmentReport:
        """
        Compute cross-layer alignment with full spectral analysis.

        Args:
            layer_directions: Dictionary mapping layer index to direction tensor
            detect_transitions: Whether to detect phase transitions
            n_shared_components: Number of PCA components for shared subspace

        Returns:
            AlignmentReport with complete analysis
        """
        layers = sorted(layer_directions.keys())
        n = len(layers)

        if n == 0:
            return AlignmentReport(
                alignment_matrix={}, persistence_score=0.0,
                cluster_layers=[], peak_alignment_layer=-1, drift_layers=[]
            )

        # Normalize all directions
        normalized = {}
        for layer in layers:
            direction = layer_directions[layer].float()
            norm = torch.norm(direction)
            normalized[layer] = direction / norm if norm > 1e-8 else direction

        # Build alignment matrix (cosine similarity matrix)
        similarity = np.zeros((n, n))
        alignment = {}
        for i, l1 in enumerate(layers):
            alignment[l1] = {}
            for j, l2 in enumerate(layers):
                sim = float(torch.dot(normalized[l1], normalized[l2]))
                if i == j:
                    sim = 1.0
                similarity[i, j] = sim
                alignment[l1][l2] = sim

        # --- Persistence score ---
        persistence = self._compute_persistence(alignment, layers)

        # --- Find clusters ---
        clusters = self._find_clusters(alignment, layers, threshold=0.8)

        # --- Peak alignment layer ---
        avg_sims = {}
        for l in layers:
            avg_sims[l] = sum(alignment[l].values()) / len(alignment[l])
        peak_layer = max(avg_sims, key=avg_sims.get) if avg_sims else -1

        # --- Drift layers ---
        drift_layers = self._find_drift_layers(alignment, layers)

        # --- Phase transitions (via gradient in similarity) ---
        phase_transitions = []
        if detect_transitions and n > 2:
            phase_transitions = self._detect_phase_transitions(similarity, layers)

        # --- SVD of stacked directions for shared subspace ---
        eigenvalue_spectrum = []
        shared_subspace_dim = 0
        layer_loadings = {}
        if n > 1:
            stacked = torch.stack([normalized[l] for l in layers]).cpu().numpy()
            try:
                pca = PCA(n_components=min(n_shared_components, n))
                pca.fit(stacked)
                eigenvalue_spectrum = pca.explained_variance_ratio_.tolist()
                # Effective dimensionality (participation ratio)
                ratios = np.array(eigenvalue_spectrum)
                if np.sum(ratios ** 2) > 0:
                    shared_subspace_dim = int(np.ceil(np.sum(ratios) ** 2 / np.sum(ratios ** 2)))
                # Layer loadings on first two PCs
                loadings = pca.transform(stacked)  # (n_layers, n_components)
                for idx, layer in enumerate(layers):
                    layer_loadings[layer] = loadings[idx].copy()
            except Exception:
                pass

        # --- Neighbor similarity ---
        neighbor_sim = self._compute_neighbor_similarity(alignment, layers)

        # --- Global trend ---
        global_trend = self._detect_global_trend(similarity, layers)

        return AlignmentReport(
            alignment_matrix=alignment,
            persistence_score=persistence,
            cluster_layers=clusters,
            peak_alignment_layer=peak_layer,
            drift_layers=drift_layers,
            similarity_matrix=similarity,
            layer_order=layers,
            phase_transitions=phase_transitions,
            eigenvalue_spectrum=eigenvalue_spectrum,
            shared_subspace_dim=shared_subspace_dim,
            layer_loadings=layer_loadings,
            neighbor_similarity=neighbor_sim,
            global_trend=global_trend
        )

    def _compute_persistence(
        self,
        alignment: Dict[int, Dict[int, float]],
        layers: List[int]
    ) -> float:
        """Compute overall persistence score (average adjacent similarity)."""
        if len(layers) < 2:
            return 1.0

        total = 0.0
        count = 0
        for i in range(len(layers) - 1):
            total += alignment[layers[i]][layers[i + 1]]
            count += 1
        return total / count if count > 0 else 1.0

    def _find_clusters(
        self,
        alignment: Dict[int, Dict[int, float]],
        layers: List[int],
        threshold: float = 0.8
    ) -> List[int]:
        """Find layers that form similarity clusters."""
        clusters: List[int] = []
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

    def _find_drift_layers(
        self,
        alignment: Dict[int, Dict[int, float]],
        layers: List[int],
        threshold: float = 0.6
    ) -> List[int]:
        """Find layers where similarity to neighbours drops below threshold."""
        drift = []
        for i, layer in enumerate(layers):
            left_sim = alignment[layer].get(layers[i - 1], 1.0) if i > 0 else 1.0
            right_sim = alignment[layer].get(layers[i + 1], 1.0) if i < len(layers) - 1 else 1.0
            if left_sim < threshold or right_sim < threshold:
                drift.append(layer)
        return drift

    def _detect_phase_transitions(
        self,
        similarity: np.ndarray,
        layers: List[int]
    ) -> List[int]:
        """
        Detect phase transitions where layer-to-layer similarity changes sharply.

        Uses: gradient of adjacent cosine similarities. Peaks in the gradient
        magnitude indicate where the direction representation changes abruptly.
        """
        n = len(layers)
        if n < 3:
            return []

        # Extract adjacent similarities
        adjacent_sims = np.array([similarity[i, i + 1] for i in range(n - 1)])

        # Compute gradient magnitude
        gradient = np.abs(np.diff(adjacent_sims))

        # Find peaks in gradient
        if len(gradient) > 0:
            # Use mean + 1 std as threshold
            threshold = np.mean(gradient) + np.std(gradient)

            transitions = []
            for i, g in enumerate(gradient):
                if g > threshold:
                    # Transition occurs between layer[i+1] and layer[i+2]
                    transitions.append(layers[i + 2])

            return transitions
        return []

    def _compute_neighbor_similarity(
        self,
        alignment: Dict[int, Dict[int, float]],
        layers: List[int]
    ) -> Dict[int, float]:
        """Compute average similarity with immediate neighbours per layer."""
        neighbor_sim = {}
        for i, layer in enumerate(layers):
            count = 0
            total = 0.0
            if i > 0:
                total += alignment[layer][layers[i - 1]]
                count += 1
            if i < len(layers) - 1:
                total += alignment[layer][layers[i + 1]]
                count += 1
            neighbor_sim[layer] = total / count if count > 0 else 1.0
        return neighbor_sim

    def _detect_global_trend(
        self,
        similarity: np.ndarray,
        layers: List[int]
    ) -> str:
        """Detect global trend in alignment across depth."""
        n = len(layers)
        if n < 4:
            return "stable"

        # Adjacent similarities vs layer index
        adj = np.array([similarity[i, i + 1] for i in range(n - 1)])

        # Linear fit
        x = np.arange(len(adj))
        if len(adj) > 1:
            slope = np.polyfit(x, adj, 1)[0]
        else:
            slope = 0.0

        # Also check for U-shape
        if len(adj) > 3:
            quad_coeff = np.polyfit(x, adj, 2)[0]
        else:
            quad_coeff = 0.0

        if abs(slope) < 0.01:
            if quad_coeff > 0.01:
                return "u-shaped"
            elif quad_coeff < -0.01:
                return "inverted-u"
            return "stable"
        elif slope > 0.01:
            return "increasing"
        else:
            return "decreasing"

    def compute_persistence_curve(
        self,
        layer_directions: Dict[int, torch.Tensor]
    ) -> Dict[int, float]:
        """
        Compute persistence score per layer (average similarity to neighbours).

        Returns dictionary mapping layer to persistence score.
        """
        layers = sorted(layer_directions.keys())
        normalized = {}
        for layer, direction in layer_directions.items():
            norm = torch.norm(direction)
            normalized[layer] = direction / norm if norm > 1e-8 else direction

        persistence = {}
        for i, layer in enumerate(layers):
            score = 0.0
            count = 0
            if i > 0:
                score += float(torch.dot(normalized[layer], normalized[layers[i - 1]]))
                count += 1
            if i < len(layers) - 1:
                score += float(torch.dot(normalized[layer], normalized[layers[i + 1]]))
                count += 1
            persistence[layer] = score / count if count > 0 else 1.0

        return persistence
