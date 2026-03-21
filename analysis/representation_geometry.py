"""
Representation Geometry Analyzer

Analyzes the geometric structure of representations across layers.
Measures manifold properties, curvature, and separability.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GeometryReport:
    """Container for representation geometry analysis."""
    intrinsic_dimension: float
    manifold_curvature: float
    class_separability: float
    representation_rank: int
    spectral_gap: float


class RepresentationGeometryAnalyzer:
    """
    Analyze geometric properties of representations.

    Features:
    - Intrinsic dimension estimation
    - Manifold curvature
    - Class separability
    - Spectral analysis
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def estimate_intrinsic_dimension(
        self,
        activations: torch.Tensor,
        method: str = "pca"
    ) -> float:
        """
        Estimate intrinsic dimension of activation manifold.

        Args:
            activations: Activation tensor (n_samples, hidden_dim)
            method: "pca", "fisher", "mle"

        Returns:
            Estimated intrinsic dimension
        """
        X = activations.cpu().numpy()

        if method == "pca":
            # PCA-based estimation: dimension capturing 90% variance
            from sklearn.decomposition import PCA
            pca = PCA()
            pca.fit(X)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            dim = np.where(cumulative_variance >= 0.9)[0][0] + 1
            return float(dim)

        elif method == "mle":
            # Maximum likelihood estimation
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=10).fit(X)
            distances, _ = nbrs.kneighbors(X)

            # Simplified MLE
            k = 10
            dim = np.mean(1.0 / (np.log(distances[:, -1] / distances[:, 0]) + 1e-8))
            return dim

        else:
            return 0.0

    def compute_manifold_curvature(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor
    ) -> float:
        """
        Compute curvature of decision boundary.

        Higher curvature = more complex boundary.
        """
        # Simplified: use gradient of linear separator
        X = np.vstack([harmful_activations.cpu().numpy(),
                       harmless_activations.cpu().numpy()])
        y = np.hstack([np.ones(len(harmful_activations)),
                       np.zeros(len(harmless_activations))])

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(X, y)

        # Curvature proxy: norm of coefficients
        curvature = np.linalg.norm(clf.coef_)

        return min(1.0, curvature / 100)

    def compute_class_separability(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor
    ) -> float:
        """
        Compute how separable harmful and harmless activations are.

        Returns:
            Separability score (0-1, higher = more separable)
        """
        harmful_mean = harmful_activations.mean(dim=0)
        harmless_mean = harmless_activations.mean(dim=0)

        # Distance between means
        between_class = torch.norm(harmful_mean - harmless_mean).item()

        # Within-class variance
        harmful_var = torch.var(harmful_activations, dim=0).mean().item()
        harmless_var = torch.var(harmless_activations, dim=0).mean().item()
        within_class = np.sqrt(harmful_var + harmless_var)

        # Fisher discriminant ratio
        separability = between_class / (within_class + 1e-8)

        return min(1.0, separability / 5)

    def compute_spectral_gap(
        self,
        activations: torch.Tensor
    ) -> float:
        """
        Compute spectral gap of activation covariance.

        Larger gap = more structured representations.
        """
        centered = activations - activations.mean(dim=0)
        cov = centered.T @ centered / (centered.shape[0] - 1)

        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.flip(0)  # Descending order
        eigenvalues = eigenvalues.cpu().numpy()

        if len(eigenvalues) > 1:
            # Gap between first and second eigenvalue
            gap = eigenvalues[0] - eigenvalues[1]
            normalized_gap = gap / (eigenvalues[0] + 1e-8)
            return float(normalized_gap)

        return 0.0

    def analyze_layer(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor
    ) -> GeometryReport:
        """
        Complete geometric analysis for a single layer.

        Returns:
            GeometryReport with all metrics
        """
        intrinsic_dim = self.estimate_intrinsic_dimension(harmful_activations)
        curvature = self.compute_manifold_curvature(harmful_activations, harmless_activations)
        separability = self.compute_class_separability(harmful_activations, harmless_activations)
        spectral_gap = self.compute_spectral_gap(harmful_activations)

        # Effective rank
        all_activations = torch.cat([harmful_activations, harmless_activations], dim=0)
        centered = all_activations - all_activations.mean(dim=0)
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        S_norm = S / (S.sum() + 1e-8)
        entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-8))
        effective_rank = torch.exp(entropy).item()

        return GeometryReport(
            intrinsic_dimension=intrinsic_dim,
            manifold_curvature=curvature,
            class_separability=separability,
            representation_rank=int(effective_rank),
            spectral_gap=spectral_gap
        )
