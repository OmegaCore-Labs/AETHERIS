"""
Distributed Alignment Analyzer

Analyzes whether safety constraints are distributed across layers or concentrated.
Computes participation ratios, effective rank, and inter-layer redundancy.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class DistributionReport:
    """Container for distributed alignment analysis."""
    concentration_score: float
    distribution_type: str  # "concentrated", "distributed", "layered", "no_data"
    peak_layers: List[int]
    layer_contributions: Dict[int, float]
    head_contributions: Dict[int, Dict[int, float]]
    participation_ratio: float = 0.0  # Effective number of layers involved
    effective_rank: float = 0.0  # Effective rank of constraint matrix
    inter_layer_redundancy: float = 0.0  # Fraction of redundant layers
    layer_entropy: float = 0.0  # Normalized entropy of layer distribution
    spectral_analysis: Dict[str, float] = field(default_factory=dict)
    status: str = "no_data"  # "ok", "no_data", "error"


class DistributedAlignmentAnalyzer:
    """
    Analyze how alignment/refusal is distributed across layers.

    Performs real spectral analysis of constraint directions to
    determine whether refusal is concentrated in a few layers or
    distributed across the network.

    Core metrics:
    - Participation ratio: effective number of layers carrying refusal signal
    - Effective rank: PCA effective rank of constraint direction matrix
    - Inter-layer redundancy: fraction of layers that are redundant copies
    - Layer entropy: normalized entropy of contribution distribution
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_distribution(
        self,
        layer_contributions: Dict[int, float],
        head_contributions: Optional[Dict[int, Dict[int, float]]] = None,
        constraint_directions: Optional[List[torch.Tensor]] = None,
    ) -> DistributionReport:
        """
        Full distribution analysis of alignment.

        Args:
            layer_contributions: Importance per layer (e.g., from probe accuracies)
            head_contributions: Optional importance per head per layer
            constraint_directions: Optional list of constraint direction vectors per layer

        Returns:
            DistributionReport with full spectral analysis
        """
        if not layer_contributions:
            return DistributionReport(
                concentration_score=0.0,
                distribution_type="no_data",
                peak_layers=[],
                layer_contributions={},
                head_contributions={},
                status="no_data",
            )

        hc = head_contributions or {}

        try:
            # 1. Gini-based concentration score
            concentration = self._compute_concentration_gini(layer_contributions)

            # 2. Participation ratio from contribution weights
            participation = self._compute_participation_ratio(layer_contributions)

            # 3. Distribution type classification
            distribution = self._classify_distribution(concentration, participation)

            # 4. Peak layers (top 3 contributors)
            sorted_layers = sorted(layer_contributions.items(), key=lambda x: x[1], reverse=True)
            peak_layers = [l for l, _ in sorted_layers[:3]]

            # 5. Layer entropy
            layer_entropy = self.compute_distribution_entropy(layer_contributions)

            # 6. Spectral analysis from constraint directions
            spectral = {}
            effective_rank = 0.0
            inter_layer_redundancy = 0.0

            if constraint_directions and len(constraint_directions) > 1:
                spectral = self._spectral_analysis_of_directions(constraint_directions)
                effective_rank = spectral.get("effective_rank", 0.0)
                inter_layer_redundancy = spectral.get("redundancy_ratio", 0.0)

            return DistributionReport(
                concentration_score=round(concentration, 4),
                distribution_type=distribution,
                peak_layers=peak_layers,
                layer_contributions=layer_contributions,
                head_contributions=hc,
                participation_ratio=round(participation, 4),
                effective_rank=round(effective_rank, 4),
                inter_layer_redundancy=round(inter_layer_redundancy, 4),
                layer_entropy=round(layer_entropy, 4),
                spectral_analysis=spectral,
                status="ok",
            )
        except Exception as e:
            return DistributionReport(
                concentration_score=0.0,
                distribution_type="no_data",
                peak_layers=[],
                layer_contributions=layer_contributions,
                head_contributions=hc,
                status=f"error: {str(e)}",
            )

    def _compute_concentration_gini(
        self, contributions: Dict[int, float]
    ) -> float:
        """
        Compute concentration via Gini coefficient of contribution distribution.

        Gini ~ 1 = highly concentrated (one layer dominates)
        Gini ~ 0 = evenly distributed
        """
        values = np.array(list(contributions.values()), dtype=np.float64)
        values = values[values > 0]

        if len(values) < 2:
            return 0.0 if len(values) <= 1 else 1.0

        sorted_vals = np.sort(values)[::-1]
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n

        return float(gini)

    def _compute_participation_ratio(
        self, contributions: Dict[int, float]
    ) -> float:
        """
        Compute participation ratio (effective number of layers involved).

        PR = (sum of weights)^2 / sum of weights^2
        This is the inverse Simpson index; ranges from 1 to N.
        """
        values = np.array(list(contributions.values()), dtype=np.float64)
        values = values[values > 0]

        if len(values) == 0:
            return 0.0

        sum_val = np.sum(values)
        sum_sq = np.sum(values ** 2)

        if sum_sq < 1e-10:
            return 0.0

        pr = (sum_val ** 2) / sum_sq
        return float(pr)

    def _classify_distribution(
        self, concentration: float, participation: float
    ) -> str:
        """Classify distribution type based on concentration and participation."""
        if concentration > 0.6 or participation <= 3:
            return "concentrated"
        elif concentration > 0.3 or participation <= 8:
            return "layered"
        else:
            return "distributed"

    def _spectral_analysis_of_directions(
        self,
        directions: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Spectral analysis of constraint direction matrix.

        Stacks all direction vectors, performs SVD, computes:
        - Effective rank (from eigenvalue entropy)
        - Redundancy ratio (redundant dimensions / total dimensions)
        - Principal component proportions
        """
        results: Dict[str, float] = {}

        try:
            # Stack flattened directions into matrix (n_layers, d)
            flat_dirs = []
            for d in directions:
                flat_dirs.append(d.float().flatten())

            M = torch.stack(flat_dirs)

            # SVD
            _, S, _ = torch.linalg.svd(M, full_matrices=False)
            S = S[S > 1e-8]

            if len(S) == 0:
                results["effective_rank"] = 0.0
                results["redundancy_ratio"] = 0.0
                return results

            # Effective rank: exp(entropy of normalized singular values)
            S_norm = S / S.sum()
            entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
            effective_rank = torch.exp(entropy).item()

            # Redundancy ratio: how many dimensions exceed threshold
            threshold = S[0].item() * 0.1  # 10% of top singular value
            n_significant = torch.sum(S > threshold).item()
            n_redundant = max(0, len(S) - n_significant)
            redundancy_ratio = n_redundant / len(S) if len(S) > 0 else 0.0

            results["effective_rank"] = round(effective_rank, 4)
            results["redundancy_ratio"] = round(redundancy_ratio, 4)
            results["max_singular_value"] = round(S[0].item(), 6)
            results["min_singular_value"] = round(S[-1].item(), 6)
            results["condition_number"] = round(S[0].item() / (S[-1].item() + 1e-10), 2)
            results["top_3_variance_ratio"] = round(
                (float(S[:3].sum()) / float(S.sum())) if len(S) >= 3 else 1.0, 4
            )

        except Exception:
            results["effective_rank"] = 0.0
            results["redundancy_ratio"] = 0.0

        return results

    def compute_distribution_entropy(
        self,
        contributions: Dict[int, float],
    ) -> float:
        """
        Compute normalized Shannon entropy of contribution distribution.

        Higher entropy = more distributed (more uniform).
        Normalized to [0, 1].
        """
        values = np.array(list(contributions.values()), dtype=np.float64)
        values = values[values > 0]

        if len(values) == 0:
            return 0.0

        total = np.sum(values)
        if total == 0:
            return 0.0

        probs = values / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(values))

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def compute_inter_layer_redundancy_matrix(
        self,
        layer_directions: Dict[int, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Compute pairwise redundancy (cosine similarity) between layers.

        Returns:
            Dict with redundancy_matrix[row][col] and mean_redundancy.
        """
        layers = sorted(layer_directions.keys())
        n = len(layers)

        if n < 2:
            return {"mean_redundancy": 0.0, "redundancy_matrix": {}}

        # Normalize
        normalized = {}
        for layer, direction in layer_directions.items():
            norm = torch.norm(direction.float())
            if norm > 1e-8:
                normalized[layer] = direction.float() / norm
            else:
                normalized[layer] = direction.float()

        redundancy_matrix: Dict[int, Dict[int, float]] = {}
        redundancies = []

        for i, l1 in enumerate(layers):
            redundancy_matrix[l1] = {}
            for j, l2 in enumerate(layers):
                if l1 == l2:
                    redundancy_matrix[l1][l2] = 1.0
                else:
                    sim = torch.dot(normalized[l1], normalized[l2]).item()
                    redundancy_matrix[l1][l2] = round(sim, 4)
                    if j > i:
                        redundancies.append(sim)

        mean_redundancy = float(np.mean(redundancies)) if redundancies else 0.0

        return {
            "mean_redundancy": round(mean_redundancy, 4),
            "redundancy_matrix": redundancy_matrix,
            "max_redundancy": round(float(np.max(redundancies)), 4) if redundancies else 0.0,
            "min_redundancy": round(float(np.min(redundancies)), 4) if redundancies else 0.0,
        }

    def find_redundant_components(
        self,
        layer_contributions: Dict[int, float],
        head_contributions: Optional[Dict[int, Dict[int, float]]] = None,
        redundancy_threshold: float = 0.1,
    ) -> List[Tuple[int, Optional[int]]]:
        """
        Find redundant components (contribution below threshold).

        Returns list of (layer, head) tuples where head=None means entire layer.
        """
        redundant: List[Tuple[int, Optional[int]]] = []

        for layer, contrib in layer_contributions.items():
            if contrib < redundancy_threshold:
                redundant.append((layer, None))

        if head_contributions:
            for layer, heads in head_contributions.items():
                for head, contrib in heads.items():
                    if contrib < redundancy_threshold:
                        redundant.append((layer, head))

        return redundant

    def get_compression_potential(
        self,
        distribution: DistributionReport,
    ) -> Dict[str, Any]:
        """
        Estimate compression potential based on distribution analysis.

        More concentrated = higher compression potential (fewer layers carry signal).
        """
        compression = distribution.concentration_score

        if compression > 0.6:
            approach = "Targeted removal: few layers carry most signal"
            estimated_reduction = f"{compression:.0%} of layers appear redundant"
        elif compression > 0.3:
            approach = "Selective removal: moderate concentration"
            estimated_reduction = "Can likely remove ~50% of targeted layers"
        else:
            approach = "Broad removal needed: signal is distributed"
            estimated_reduction = "Must target most layers for full effect"

        return {
            "compression_potential": round(compression, 4),
            "participation_ratio": distribution.participation_ratio,
            "effective_rank": distribution.effective_rank,
            "estimated_reduction": estimated_reduction,
            "approach": approach,
        }
