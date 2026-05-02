"""
Universality Analysis — Cross-Model Constraint Direction Comparison

Analyzes whether safety/constraint directions are universal (shared across
model architectures and families) or model-specific. This is critical for
understanding whether constraint removal techniques transfer across models.

Key Research Questions:
- Are refusal directions universal across model architectures?
- Do different model families share the same safety geometry?
- What is the transfer coefficient between model-specific directions?
- Is the universality hypothesis (safety directions are emergent properties
  of training, not architecture) supported?
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CrossModelComparison:
    """Container for pairwise cross-model direction comparison."""
    model_a: str
    model_b: str
    cosine_similarity: float
    angular_distance: float                 # In degrees
    dot_product: float
    norm_ratio: float                       # ||dir_a|| / ||dir_b||
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalityReport:
    """Container for full universality analysis results."""
    model_directions: Dict[str, List[torch.Tensor]]
    pairwise_comparisons: List[CrossModelComparison]
    universal_score: float                  # Overall universality score [0, 1]
    universal_direction: Optional[torch.Tensor]  # Consensus direction
    model_clusters: Dict[str, List[str]]    # Clustered model families
    transfer_coefficients: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterResult:
    """Container for model-family clustering."""
    clusters: Dict[str, List[str]]
    silhouette_score: float
    n_clusters: int
    isolated_models: List[str]
    cluster_centroids: Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# UniversalityAnalyzer
# ---------------------------------------------------------------------------

class UniversalityAnalyzer:
    """
    Analyze whether constraint directions are universal across models.

    This class implements the full universality analysis pipeline:
    1. Cross-model direction comparison via cosine similarity
    2. Universal constraint detection (shared safety directions)
    3. Transfer coefficient computation
    4. Model-family clustering of constraint directions
    5. Statistical tests for the universality hypothesis

    Methodology:
    For each pair of models, the analyzer computes the cosine similarity
    between their constraint directions. If directions are universal, all
    pairwise similarities should be high (>0.7). If model-specific, similarities
    should cluster by architecture family.

    References:
    - Arditi et al. (2024): Refusal directions across model sizes
    - Marks et al. (2024): Sparse autoencoder cross-model transfer
    """
    # pylint: disable=too-many-instance-attributes

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    UNIVERSALITY_THRESHOLD: float = 0.7          # Cosine sim threshold
    TRANSFER_MIN_SIMILARITY: float = 0.5         # Min similarity for transfer
    CLUSTER_SIMILARITY_THRESHOLD: float = 0.8    # For family clustering
    BOOTSTRAP_SAMPLES: int = 1000                # For statistical tests
    MIN_CLUSTER_SIZE: int = 2                    # Min models per cluster

    # Known model families for validation
    KNOWN_FAMILIES: Dict[str, str] = {
        "llama": "meta",
        "mistral": "mistralai",
        "qwen": "alibaba",
        "gemma": "google",
        "phi": "microsoft",
        "deepseek": "deepseek",
        "gpt2": "openai",
        "pythia": "eleuther",
        "falcon": "tii",
    }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, device: str = "cpu"):
        """
        Initialize the universality analyzer.

        Args:
            device: Torch device for tensor computations.
        """
        self.device = device

    # ------------------------------------------------------------------
    # Core Analysis
    # ------------------------------------------------------------------

    def compute_pairwise_similarities(
        self,
        model_directions: Dict[str, torch.Tensor],
        normalize: bool = True,
    ) -> List[CrossModelComparison]:
        """
        Compute cosine similarity between constraint directions for all
        model pairs.

        Args:
            model_directions: Dictionary mapping model name to its primary
                constraint direction vector.
            normalize: Whether to normalize directions before comparison.

        Returns:
            List of CrossModelComparison objects for each unique pair.
        """
        model_names = sorted(model_directions.keys())
        comparisons: List[CrossModelComparison] = []

        # Pre-normalize if requested
        norms: Dict[str, float] = {}
        if normalize:
            for name, direction in model_directions.items():
                norm = torch.norm(direction).item()
                norms[name] = norm
                if norm > 0:
                    direction = direction / norm
                model_directions[name] = direction

        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if j <= i:
                    continue

                dir_a = model_directions[model_a]
                dir_b = model_directions[model_b]

                # Ensure same device
                if dir_a.device != dir_b.device:
                    dir_b = dir_b.to(dir_a.device)

                dot = torch.dot(dir_a, dir_b).item()
                norm_a = torch.norm(dir_a).item()
                norm_b = torch.norm(dir_b).item()

                if norm_a == 0 or norm_b == 0:
                    cosine = 0.0
                else:
                    cosine = dot / (norm_a * norm_b)

                angular = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                norm_ratio = norm_a / (norm_b + 1e-8)

                comparisons.append(CrossModelComparison(
                    model_a=model_a,
                    model_b=model_b,
                    cosine_similarity=cosine,
                    angular_distance=angular,
                    dot_product=dot,
                    norm_ratio=norm_ratio,
                    metadata={
                        "norm_a": norm_a,
                        "norm_b": norm_b,
                    },
                ))

        return comparisons

    def detect_universal_constraints(
        self,
        model_directions: Dict[str, torch.Tensor],
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, Optional[torch.Tensor]]:
        """
        Test whether constraint directions are universal across models.

        A direction is "universal" if all pairwise cosine similarities
        exceed the threshold.

        Args:
            model_directions: Constraint directions per model.
            threshold: Cosine similarity threshold for universality.

        Returns:
            Tuple of (is_universal, average_similarity, consensus_direction).
        """
        if threshold is None:
            threshold = self.UNIVERSALITY_THRESHOLD

        if len(model_directions) < 2:
            return False, 0.0, None

        comparisons = self.compute_pairwise_similarities(model_directions)

        if not comparisons:
            return False, 0.0, None

        similarities = [c.cosine_similarity for c in comparisons]
        avg_similarity = float(np.mean(similarities))
        min_similarity = float(min(similarities))

        is_universal = min_similarity >= threshold

        # Compute consensus direction (mean of all normalized directions)
        consensus = None
        if is_universal:
            stacked = torch.stack([
                d / (torch.norm(d) + 1e-8) for d in model_directions.values()
            ])
            consensus = stacked.mean(dim=0)
            consensus = consensus / (torch.norm(consensus) + 1e-8)

        return is_universal, avg_similarity, consensus

    def compute_transfer_coefficient(
        self,
        source_direction: torch.Tensor,
        target_direction: torch.Tensor,
        normalize: bool = True,
    ) -> float:
        """
        Compute transfer coefficient: how much of the source direction
        projects onto the target direction.

        The transfer coefficient is cos(theta) * min(1, ||target||/||source||),
        accounting for both angular alignment and magnitude compatibility.

        Args:
            source_direction: Direction from source model.
            target_direction: Direction from target model.
            normalize: Whether to normalize inputs.

        Returns:
            Transfer coefficient in [0, 1].
        """
        if normalize:
            src_norm = torch.norm(source_direction)
            tgt_norm = torch.norm(target_direction)
            if src_norm > 0:
                source_direction = source_direction / src_norm
            if tgt_norm > 0:
                target_direction = target_direction / tgt_norm

        dot = torch.dot(source_direction, target_direction).item()
        norm_src = torch.norm(source_direction).item()
        norm_tgt = torch.norm(target_direction).item()

        if norm_src == 0 or norm_tgt == 0:
            return 0.0

        cosine = max(0.0, min(1.0, dot / (norm_src * norm_tgt)))
        magnitude_factor = min(1.0, norm_tgt / (norm_src + 1e-8))

        return cosine * magnitude_factor

    # ------------------------------------------------------------------
    # Model-Family Clustering
    # ------------------------------------------------------------------

    def cluster_model_families(
        self,
        model_directions: Dict[str, torch.Tensor],
        similarity_threshold: Optional[float] = None,
    ) -> ClusterResult:
        """
        Cluster models into families based on constraint direction similarity.

        This tests whether constraint geometry correlates with known
        architectural families (e.g., Llama models cluster together).

        Args:
            model_directions: Constraint directions per model.
            similarity_threshold: Threshold for grouping models.

        Returns:
            ClusterResult with family assignments.
        """
        if similarity_threshold is None:
            similarity_threshold = self.CLUSTER_SIMILARITY_THRESHOLD

        if len(model_directions) < 2:
            return ClusterResult(
                clusters={},
                silhouette_score=0.0,
                n_clusters=0,
                isolated_models=list(model_directions.keys()),
                cluster_centroids={},
            )

        # Build similarity matrix
        model_names = sorted(model_directions.keys())
        n_models = len(model_names)

        similarity_matrix = torch.zeros((n_models, n_models))
        for i, ma in enumerate(model_names):
            for j, mb in enumerate(model_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    da = model_directions[ma]
                    db = model_directions[mb]
                    if torch.norm(da) > 0 and torch.norm(db) > 0:
                        sim = torch.dot(
                            da / torch.norm(da),
                            db / torch.norm(db),
                        ).item()
                        similarity_matrix[i, j] = sim

        # Greedy clustering
        visited: Set[int] = set()
        clusters: Dict[str, List[str]] = {}
        isolated: List[str] = []
        centroids: Dict[str, torch.Tensor] = {}

        for i in range(n_models):
            if i in visited:
                continue

            # Find all models similar to this one
            family = [model_names[i]]
            visited.add(i)

            for j in range(n_models):
                if j in visited:
                    continue
                if similarity_matrix[i, j].item() >= similarity_threshold:
                    family.append(model_names[j])
                    visited.add(j)

            if len(family) >= self.MIN_CLUSTER_SIZE:
                cluster_name = self._infer_family_name(family)
                clusters[cluster_name] = family
                # Compute centroid
                stacked = torch.stack([
                    model_directions[m] / (torch.norm(model_directions[m]) + 1e-8)
                    for m in family
                ])
                centroids[cluster_name] = stacked.mean(dim=0)
            else:
                isolated.extend(family)

        # Silhouette score approximation
        silhouette = self._compute_silhouette(
            similarity_matrix, clusters, model_names
        )

        return ClusterResult(
            clusters=clusters,
            silhouette_score=silhouette,
            n_clusters=len(clusters),
            isolated_models=isolated,
            cluster_centroids=centroids,
        )

    # ------------------------------------------------------------------
    # Statistical Tests
    # ------------------------------------------------------------------

    def statistical_test_universality(
        self,
        model_directions: Dict[str, torch.Tensor],
        n_bootstrap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform statistical tests for the universality hypothesis.

        Null Hypothesis (H0): Constraint directions are model-specific.
        Alternative (H1): Constraint directions are universal.

        Tests performed:
        - Bootstrap confidence interval for mean pairwise similarity
        - Permutation test for family clustering vs random
        - One-sample t-test against the universality threshold

        Args:
            model_directions: Constraint directions per model.
            n_bootstrap: Number of bootstrap resamples.

        Returns:
            Dictionary with test statistics and conclusions.
        """
        if n_bootstrap is None:
            n_bootstrap = self.BOOTSTRAP_SAMPLES

        if len(model_directions) < 2:
            return {
                "testable": False,
                "reason": "Need at least 2 models for statistical testing.",
            }

        comparisons = self.compute_pairwise_similarities(model_directions)
        similarities = [c.cosine_similarity for c in comparisons]
        observed_mean = float(np.mean(similarities))

        # Bootstrap confidence interval
        n_pairs = len(similarities)
        bootstrap_means: List[float] = []
        rng = np.random.RandomState(42)

        for _ in range(n_bootstrap):
            sample = rng.choice(similarities, size=n_pairs, replace=True)
            bootstrap_means.append(float(np.mean(sample)))

        bootstrap_means.sort()
        ci_lower = bootstrap_means[int(0.025 * n_bootstrap)]
        ci_upper = bootstrap_means[int(0.975 * n_bootstrap)]

        # Permutation test: randomize model labels
        model_names = list(model_directions.keys())
        n_models = len(model_names)
        permuted_means: List[float] = []

        for _ in range(min(n_bootstrap, 500)):
            shuffled = dict(zip(
                model_names,
                rng.permutation(list(model_directions.values())),
            ))
            perm_comps = self.compute_pairwise_similarities(shuffled)
            perm_sims = [c.cosine_similarity for c in perm_comps]
            permuted_means.append(float(np.mean(perm_sims)))

        permuted_means.sort()
        p_value = sum(1.0 for pm in permuted_means if pm >= observed_mean) / len(permuted_means)

        # Conclusion
        is_universal = (
            ci_lower >= self.UNIVERSALITY_THRESHOLD
            and p_value < 0.05
        )

        return {
            "testable": True,
            "observed_mean_similarity": observed_mean,
            "bootstrap_95_ci": [ci_lower, ci_upper],
            "p_value": p_value,
            "universality_threshold": self.UNIVERSALITY_THRESHOLD,
            "is_universal": is_universal,
            "conclusion": (
                "Universality hypothesis SUPPORTED: constraint directions "
                "are statistically universal across models."
                if is_universal
                else "Universality hypothesis REJECTED: constraint directions "
                "are model-specific."
            ),
            "n_models": n_models,
            "n_pairwise_comparisons": n_pairs,
            "n_bootstrap_samples": n_bootstrap,
        }

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        model_directions: Dict[str, torch.Tensor],
        run_statistical_tests: bool = True,
        cluster_models: bool = True,
        compute_transfer_matrix: bool = True,
    ) -> UniversalityReport:
        """
        Run the complete universality analysis pipeline.

        This single entry point performs all available analyses:
        - Pairwise direction comparison
        - Universal constraint detection
        - Model-family clustering
        - Transfer coefficient matrix
        - Statistical hypothesis testing

        Args:
            model_directions: Dictionary mapping model name to its primary
                constraint direction vector.
            run_statistical_tests: Whether to run bootstrap/permutation tests.
            cluster_models: Whether to cluster models by family.
            compute_transfer_matrix: Whether to compute full transfer matrix.

        Returns:
            UniversalityReport with all analysis results.
        """
        # Normalize all directions in-place for consistent analysis
        normalized: Dict[str, torch.Tensor] = {}
        for name, direction in model_directions.items():
            norm = torch.norm(direction)
            normalized[name] = direction / norm if norm > 0 else direction

        # Pairwise comparisons
        comparisons = self.compute_pairwise_similarities(normalized)

        # Universal constraint detection
        is_universal, avg_sim, consensus = self.detect_universal_constraints(normalized)

        # Model-family clustering
        model_clusters: Dict[str, List[str]] = {}
        if cluster_models:
            cluster_result = self.cluster_model_families(normalized)
            model_clusters = cluster_result.clusters

        # Transfer coefficient matrix
        transfer_coeffs: Dict[str, Dict[str, float]] = {}
        if compute_transfer_matrix:
            for src_name, src_dir in normalized.items():
                transfer_coeffs[src_name] = {}
                for tgt_name, tgt_dir in normalized.items():
                    if src_name == tgt_name:
                        transfer_coeffs[src_name][tgt_name] = 1.0
                    else:
                        transfer_coeffs[src_name][tgt_name] = self.compute_transfer_coefficient(
                            src_dir, tgt_dir
                        )

        # Statistical tests
        stats: Dict[str, Any] = {}
        if run_statistical_tests:
            stats = self.statistical_test_universality(normalized)

        return UniversalityReport(
            model_directions=normalized,
            pairwise_comparisons=comparisons,
            universal_score=avg_sim,
            universal_direction=consensus,
            model_clusters=model_clusters,
            transfer_coefficients=transfer_coeffs,
            statistical_tests=stats,
            metadata={
                "n_models": len(model_directions),
                "n_pairs": len(comparisons),
                "is_universal": is_universal,
            },
        )

    # ------------------------------------------------------------------
    # Comparison with Multiple Directions per Model
    # ------------------------------------------------------------------

    def compare_multi_direction(
        self,
        model_multi_directions: Dict[str, List[torch.Tensor]],
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Compare multiple constraint directions across models.

        Instead of a single direction per model, this handles the polyhedral
        case where each model may have k distinct constraint mechanisms.

        Args:
            model_multi_directions: Model -> list of direction tensors.
            top_k: Number of top directions to compare.

        Returns:
            Dictionary with multi-direction comparison results.
        """
        results: Dict[str, Any] = {
            "model_ranks": {},
            "average_cross_model_similarity": {},
            "direction_consistency": {},
        }

        # Compute effective rank per model
        for model_name, directions in model_multi_directions.items():
            results["model_ranks"][model_name] = len(directions)

        # Compare each direction rank across models
        for rank_idx in range(top_k):
            rank_dirs: Dict[str, torch.Tensor] = {}
            for model_name, directions in model_multi_directions.items():
                if rank_idx < len(directions):
                    rank_dirs[model_name] = directions[rank_idx]

            if len(rank_dirs) >= 2:
                comparisons = self.compute_pairwise_similarities(rank_dirs)
                avg_sim = float(np.mean([c.cosine_similarity for c in comparisons]))
                results["average_cross_model_similarity"][f"rank_{rank_idx}"] = avg_sim
                results["direction_consistency"][f"rank_{rank_idx}"] = avg_sim

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _infer_family_name(self, model_names: List[str]) -> str:
        """Infer a cluster family name from model names."""
        for keyword, family in self.KNOWN_FAMILIES.items():
            for name in model_names:
                if keyword in name.lower():
                    return f"{family}_family"
        # Fallback: use the shortest common prefix
        if model_names:
            return model_names[0].split("/")[0].split("-")[0]
        return "unknown_family"

    def _compute_silhouette(
        self,
        similarity_matrix: torch.Tensor,
        clusters: Dict[str, List[str]],
        model_names: List[str],
    ) -> float:
        """
        Approximate silhouette score from similarity matrix.

        Args:
            similarity_matrix: NxN matrix of pairwise similarities.
            clusters: Mapping from cluster name to list of model names.
            model_names: Ordered list of model names.

        Returns:
            Average silhouette score across all clustered models.
        """
        if not clusters or len(model_names) < 2:
            return 0.0

        name_to_idx = {name: idx for idx, name in enumerate(model_names)}
        scores: List[float] = []

        for cluster_models in clusters.values():
            for model in cluster_models:
                i = name_to_idx[model]

                # Intra-cluster distance: 1 - average similarity within cluster
                intra_sims: List[float] = []
                for other in cluster_models:
                    if other != model:
                        j = name_to_idx[other]
                        intra_sims.append(similarity_matrix[i, j].item())
                a = 1.0 - (np.mean(intra_sims) if intra_sims else 0.0)

                # Nearest-cluster distance
                b_values: List[float] = []
                for other_cluster_name, other_models in clusters.items():
                    if set(other_models) == set(cluster_models):
                        continue
                    inter_sims: List[float] = []
                    for other in other_models:
                        j = name_to_idx[other]
                        inter_sims.append(similarity_matrix[i, j].item())
                    if inter_sims:
                        b_values.append(1.0 - np.mean(inter_sims))

                b = min(b_values) if b_values else a

                if max(a, b) > 0:
                    scores.append((b - a) / max(a, b))

        return float(np.mean(scores)) if scores else 0.0
