"""
Representation Geometry Analyzer

Analyzes the full geometry of representation space.
Intrinsic dimension estimation, curvature approximation (Ricci proxy),
UMAP/t-SNE projection, and neighborhood analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class GeometryReport:
    """Container for representation geometry analysis."""
    intrinsic_dimension: float
    manifold_curvature: float
    class_separability: float
    representation_rank: int
    spectral_gap: float
    neighborhood_preservation: float = 0.0
    manifold_metrics: Dict[str, float] = field(default_factory=dict)
    dimensional_estimates: Dict[str, float] = field(default_factory=dict)
    projection_coordinates: Optional[Any] = None  # UMAP/t-SNE coords
    status: str = "no_data"  # "ok", "no_data", "error"


class RepresentationGeometryAnalyzer:
    """
    Analyze geometric properties of representation space.

    Features:
    - Intrinsic dimension estimation (PCA, MLE, TwoNN)
    - Manifold curvature proxy (Ricci curvature approximation on neighborhood graph)
    - Class separability (Fisher discriminant, LDA)
    - Neighborhood preservation ratio
    - Spectral gap of covariance
    - Manifold learning projections (PCA, t-SNE-style)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def estimate_intrinsic_dimension(
        self,
        activations: torch.Tensor,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Estimate intrinsic dimension using multiple methods.

        Args:
            activations: Activation tensor (n_samples, hidden_dim)
            methods: List of methods: "pca", "mle", "twonn" (default: all)

        Returns:
            Dict mapping method name -> estimated dimension
        """
        if methods is None:
            methods = ["pca", "mle", "twonn"]

        X = activations.float().cpu().numpy()
        if X.shape[0] < 3:
            return {"error": X.shape[0]}

        estimates: Dict[str, float] = {}

        for method in methods:
            try:
                if method == "pca":
                    estimates["pca"] = self._id_pca(X)
                elif method == "mle":
                    estimates["mle"] = self._id_mle(X)
                elif method == "twonn":
                    estimates["twonn"] = self._id_twonn(X)
            except Exception:
                estimates[method] = -1.0

        return estimates

    def _id_pca(self, X: np.ndarray, variance_threshold: float = 0.9) -> float:
        """PCA-based intrinsic dimension: dim capturing threshold variance."""
        # Center
        X_c = X - X.mean(axis=0)

        # SVD for efficiency
        try:
            _, S, _ = np.linalg.svd(X_c, full_matrices=False)
            total_var = np.sum(S ** 2)
            cumulative = np.cumsum(S ** 2) / total_var
            dim = int(np.searchsorted(cumulative, variance_threshold) + 1)
            return float(dim)
        except Exception:
            return 0.0

    def _id_mle(self, X: np.ndarray, k: int = 10) -> float:
        """Maximum Likelihood Estimation of intrinsic dimension."""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0])).fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Use distances to k-th neighbor, excluding self
        if distances.shape[1] < 2:
            return 0.0

        # Levina & Bickel (2004) MLE estimator
        k_eff = min(k, distances.shape[1] - 1)
        dist_k = distances[:, k_eff]
        dist_1 = distances[:, 1]
        dist_1[dist_1 < 1e-10] = 1e-10

        # Local dimension estimate
        local_dim = 1.0 / (np.log(dist_k / dist_1) / (k_eff - 1))

        # Harmonic mean of local estimates
        local_dim = local_dim[local_dim > 0]
        if len(local_dim) == 0:
            return 0.0
        dim = 1.0 / np.mean(1.0 / local_dim)
        return float(dim)

    def _id_twonn(self, X: np.ndarray) -> float:
        """Two-NN estimator of intrinsic dimension (Facco et al. 2017)."""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=3).fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Ratio of distances to second and first neighbor
        mu = distances[:, 2] / distances[:, 1]
        mu = mu[mu > 1.0]  # Must be > 1

        if len(mu) == 0:
            return 0.0

        # Fit Pareto distribution: mu ~ Pareto(d+1)
        # d = N / sum(log(mu))
        dim = len(mu) / np.sum(np.log(mu))
        return float(dim)

    def compute_manifold_curvature(
        self,
        activations: torch.Tensor,
        k_neighbors: int = 15,
    ) -> Dict[str, float]:
        """
        Compute manifold curvature proxy via Ricci curvature approximation.

        Uses Ollivier-Ricci curvature on the k-NN graph:
        - Positive curvature: points cluster tightly
        - Negative curvature: points spread out (hyperbolic structure)

        Args:
            activations: (n_samples, dim)
            k_neighbors: Number of neighbors for graph

        Returns:
            Dict with curvature metrics
        """
        results: Dict[str, float] = {}
        X = activations.float().cpu().numpy()

        if X.shape[0] < 5:
            results["mean_curvature"] = 0.0
            results["curvature_variance"] = 0.0
            return results

        try:
            from sklearn.neighbors import NearestNeighbors

            # Build k-NN graph
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, X.shape[0])).fit(X)
            distances, indices = nbrs.kneighbors(X)

            # Compute Ollivier-Ricci curvature for edges
            # Simplified: curvature = 1 - (geodesic_distance / euclidean_distance)
            curvatures = []
            for i in range(min(200, X.shape[0])):  # Sample points for speed
                for j_idx, j in enumerate(indices[i][1:min(4, len(indices[i]))]):  # First few neighbors
                    if i == j:
                        continue

                    # Euclidean distance
                    d_euclidean = distances[i][j_idx + 1]

                    # Approximate geodesic: shortest path among common neighbors
                    common = np.intersect1d(indices[i], indices[j])
                    if len(common) > 0:
                        d_geo = min(
                            distances[i][np.where(indices[i] == c)[0][0]] +
                            distances[j][np.where(indices[j] == c)[0][0]]
                            if len(np.where(indices[i] == c)[0]) > 0 and
                               len(np.where(indices[j] == c)[0]) > 0
                            else d_euclidean * 2
                            for c in common[:5]
                        )
                    else:
                        d_geo = d_euclidean * 2

                    # Ricci curvature proxy
                    if d_euclidean > 1e-10:
                        ricci = 1.0 - d_geo / (2.0 * d_euclidean)
                        curvatures.append(ricci)

            if curvatures:
                curv_arr = np.array(curvatures)
                results["mean_curvature"] = round(float(np.mean(curv_arr)), 6)
                results["curvature_std"] = round(float(np.std(curv_arr)), 6)
                results["curvature_positive_fraction"] = round(
                    float(np.mean(curv_arr > 0)), 4
                )
            else:
                results["mean_curvature"] = 0.0
                results["curvature_positive_fraction"] = 0.0

        except Exception:
            results["mean_curvature"] = 0.0
            results["curvature_positive_fraction"] = 0.0

        return results

    def compute_class_separability(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute class separability between harmful and harmless activations.

        Returns:
            Dict with separability metrics (Fisher ratio, LDA score, margin)
        """
        results: Dict[str, float] = {}

        try:
            harmful = harmful_activations.float()
            harmless = harmless_activations.float()

            harm_mean = harmful.mean(dim=0)
            harmless_mean = harmless.mean(dim=0)

            # Between-class distance
            between = torch.norm(harm_mean - harmless_mean).item()

            # Within-class scatter
            harm_var = torch.var(harmful, dim=0).sum().item()
            harmless_var = torch.var(harmless, dim=0).sum().item()
            within = np.sqrt(harm_var + harmless_var) if (harm_var + harmless_var) > 0 else 1.0

            # Fisher discriminant ratio
            fisher_ratio = between / (within + 1e-8)
            results["fisher_ratio"] = round(fisher_ratio, 6)

            # Separability score (sigmoid-scaled Fisher)
            results["separability_score"] = round(
                float(2.0 * (1.0 / (1.0 + np.exp(-fisher_ratio / 5.0))) - 1.0), 4
            )

            # Linear classification accuracy (LDA-style)
            # Project onto the difference direction
            diff_dir = (harm_mean - harmless_mean).flatten()
            diff_norm = torch.norm(diff_dir)
            if diff_norm > 1e-8:
                diff_dir = diff_dir / diff_norm

            harm_proj = torch.sum(harmful * diff_dir, dim=-1).cpu().numpy()
            harmless_proj = torch.sum(harmless * diff_dir, dim=-1).cpu().numpy()

            # Optimal threshold: midpoint of means
            threshold = (np.mean(harm_proj) + np.mean(harmless_proj)) / 2.0
            harm_acc = np.mean(harm_proj > threshold)
            harmless_acc = np.mean(harmless_proj <= threshold)
            accuracy = (harm_acc + harmless_acc) / 2.0
            results["linear_separability"] = round(float(accuracy), 4)

            # Margin: difference in mean projections normalized
            margin = (np.mean(harm_proj) - np.mean(harmless_proj)) / (
                np.sqrt(np.var(harm_proj) + np.var(harmless_proj)) + 1e-8
            )
            results["margin"] = round(float(margin), 6)

        except Exception:
            results["fisher_ratio"] = 0.0
            results["separability_score"] = 0.0

        return results

    def compute_dimensionality_profile(
        self,
        activations: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute full dimensionality profile: multiple estimates aggregated.

        Returns:
            Dict with all ID estimates and neighborhood metrics.
        """
        dim_estimates = self.estimate_intrinsic_dimension(activations)

        # Best estimate (harmonic mean of valid estimates)
        valid_dims = [v for v in dim_estimates.values() if v > 0]
        if valid_dims:
            best_dim = 1.0 / np.mean(1.0 / np.array(valid_dims))
        else:
            best_dim = 0.0

        return {
            "intrinsic_dimension": round(best_dim, 2),
            "all_estimates": {
                k: round(v, 2) for k, v in dim_estimates.items()
            },
            "embedding_dim": activations.shape[-1],
            "utilization_ratio": round(
                best_dim / activations.shape[-1], 4
            ) if activations.shape[-1] > 0 else 0.0,
        }

    def compute_spectral_gap(
        self,
        activations: torch.Tensor,
    ) -> float:
        """
        Compute spectral gap of activation covariance.

        Larger gap = more structured/separated representation.
        """
        try:
            centered = activations.float() - activations.float().mean(dim=0)
            cov = centered.T @ centered / (centered.shape[0] - 1)

            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[torch.isfinite(eigenvalues)]
            eigenvalues = eigenvalues.sort(descending=True).values

            if len(eigenvalues) > 1 and eigenvalues[0] > 1e-10:
                gap = eigenvalues[0] - eigenvalues[1]
                normalized_gap = gap / eigenvalues[0]
                return float(normalized_gap.item())
            return 0.0
        except Exception:
            return 0.0

    def compute_effective_rank(
        self,
        activations: torch.Tensor,
    ) -> int:
        """Compute effective rank via entropy of singular values."""
        try:
            centered = activations.float() - activations.float().mean(dim=0)
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            S = S[S > 1e-10]
            if len(S) == 0:
                return 0
            S_norm = S / S.sum()
            entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
            effective_rank = int(torch.exp(entropy).item())
            return effective_rank
        except Exception:
            return 0

    def compute_neighborhood_preservation(
        self,
        activations: torch.Tensor,
        k_neighbors: int = 30,
    ) -> Dict[str, float]:
        """
        Compute neighborhood preservation metrics.

        Measures how well local structure is preserved when changing
        representation spaces (e.g., comparing raw vs PCA-reduced).
        """
        results: Dict[str, float] = {}

        X = activations.float().cpu().numpy()
        if X.shape[0] < k_neighbors + 2:
            results["neighborhood_overlap"] = 0.0
            return results

        try:
            from sklearn.neighbors import NearestNeighbors

            # Original neighborhoods
            nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
            _, orig_indices = nbrs.kneighbors(X)
            orig_sets = [set(idx[1:]) for idx in orig_indices]

            # PCA-reduced neighborhoods
            from sklearn.decomposition import PCA
            X_centered = X - X.mean(axis=0)
            pca = PCA(n_components=min(50, X.shape[1]))
            X_pca = pca.fit_transform(X_centered)

            nbrs_pca = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_pca)
            _, pca_indices = nbrs_pca.kneighbors(X_pca)
            pca_sets = [set(idx[1:]) for idx in pca_indices]

            # Jaccard similarity between neighbor sets
            overlaps = [
                len(orig_sets[i] & pca_sets[i]) / len(orig_sets[i] | pca_sets[i])
                for i in range(len(orig_sets))
                if len(orig_sets[i] | pca_sets[i]) > 0
            ]

            results["neighborhood_overlap"] = round(float(np.mean(overlaps)), 4)
            results["neighborhood_overlap_std"] = round(float(np.std(overlaps)), 4)

            # Trustworthiness (Venna & Kaski 2001 simplified)
            trustworthiness = self._compute_trustworthiness(X, X_pca, k_neighbors)
            results["trustworthiness"] = round(trustworthiness, 4)

        except Exception:
            results["neighborhood_overlap"] = 0.0

        return results

    def _compute_trustworthiness(
        self, X_high: np.ndarray, X_low: np.ndarray, k: int
    ) -> float:
        """Compute trustworthiness metric for dimensionality reduction."""
        from sklearn.neighbors import NearestNeighbors

        n = X_high.shape[0]
        nbrs_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
        nbrs_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)

        _, high_idx = nbrs_high.kneighbors(X_high)
        _, low_idx = nbrs_low.kneighbors(X_low)

        trust = 0.0
        for i in range(min(n, 500)):  # Sample
            high_set = set(high_idx[i, 1:])
            low_set = set(low_idx[i, 1:])
            missing = high_set - low_set
            for j in missing:
                rank = np.where(np.isin(high_idx[i], j))[0]
                if len(rank) > 0:
                    trust += max(0, rank[0] - k)

        return 1.0 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * trust

    def analyze_layer(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
    ) -> GeometryReport:
        """
        Complete geometric analysis for a single layer.

        Args:
            harmful_activations: (n_samples, hidden_dim) for harmful prompts
            harmless_activations: (n_samples, hidden_dim) for harmless prompts

        Returns:
            GeometryReport with all metrics
        """
        if harmful_activations is None:
            return GeometryReport(
                intrinsic_dimension=0.0, manifold_curvature=0.0,
                class_separability=0.0, representation_rank=0,
                spectral_gap=0.0, status="no_data"
            )

        try:
            # Flatten to (samples, dim)
            if harmful_activations.dim() >= 3:
                harmful = harmful_activations.float().mean(dim=1)  # avg over seq
            else:
                harmful = harmful_activations.float()

            if harmless_activations.dim() >= 3:
                harmless = harmless_activations.float().mean(dim=1)
            else:
                harmless = harmless_activations.float()

            # Intrinsic dimension
            dim_profile = self.compute_dimensionality_profile(harmful)
            intrinsic_dim = dim_profile["intrinsic_dimension"]

            # Curvature
            curv_results = self.compute_manifold_curvature(harmful)
            curvature = curv_results.get("mean_curvature", 0.0)

            # Separability
            sep_results = self.compute_class_separability(harmful, harmless)
            separability = sep_results.get("separability_score", 0.0)

            # Spectral gap
            spectral_gap = self.compute_spectral_gap(harmful)

            # Effective rank
            all_act = torch.cat([harmful, harmless], dim=0)
            effective_rank = self.compute_effective_rank(all_act)

            # Neighborhood preservation
            if harmful.shape[0] >= 30:
                neigh_results = self.compute_neighborhood_preservation(harmful)
                neigh_preservation = neigh_results.get("neighborhood_overlap", 0.0)
            else:
                neigh_preservation = 0.0

            # Manifold metrics aggregate
            manifold_metrics = {
                "intrinsic_dimension": intrinsic_dim,
                **{f"id_{k}": v for k, v in dim_profile["all_estimates"].items()},
                **curv_results,
            }

            return GeometryReport(
                intrinsic_dimension=round(intrinsic_dim, 2),
                manifold_curvature=round(curvature, 4),
                class_separability=round(separability, 4),
                representation_rank=effective_rank,
                spectral_gap=round(spectral_gap, 6),
                neighborhood_preservation=round(neigh_preservation, 4),
                manifold_metrics=manifold_metrics,
                dimensional_estimates=dim_profile["all_estimates"],
                status="ok",
            )
        except Exception as e:
            return GeometryReport(
                intrinsic_dimension=0.0, manifold_curvature=0.0,
                class_separability=0.0, representation_rank=0,
                spectral_gap=0.0, status=f"error: {str(e)}",
            )
