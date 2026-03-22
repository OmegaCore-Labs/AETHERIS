"""
Float Direction Interpolation

Continuous SVD direction index via Gaussian-shaped weighting
for smoother refusal removal.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class InterpolationResult:
    """Container for interpolation results."""
    directions: List[torch.Tensor]
    weights: List[float]
    interpolation_type: str
    smoothness: float


class FloatDirectionInterpolator:
    """
    Float Direction Interpolation — Continuous SVD direction index
    via Gaussian-shaped weighting for smoother removal.

    Key Insight: Instead of selecting discrete directions (top-k),
    we interpolate between directions using a continuous weighting
    function. This allows for smoother, more gradual constraint removal.

    Novel technique not present in OBLITERATUS core.
    """

    def __init__(self, sigma: float = 2.0):
        """
        Initialize interpolator.

        Args:
            sigma: Width of Gaussian weighting (higher = smoother)
        """
        self.sigma = sigma

    def interpolate_directions(
        self,
        directions: List[torch.Tensor],
        explained_variance: List[float],
        alpha: float = 0.5,
        method: str = "gaussian"
    ) -> InterpolationResult:
        """
        Interpolate between directions for continuous removal.

        Args:
            directions: List of SVD direction vectors
            explained_variance: Variance explained by each direction
            alpha: Interpolation position (0 = first, 1 = last)
            method: "gaussian", "linear", or "sigmoid"

        Returns:
            InterpolationResult with weighted directions
        """
        n_dirs = len(directions)
        if n_dirs == 0:
            return InterpolationResult(
                directions=[],
                weights=[],
                interpolation_type=method,
                smoothness=0.0
            )

        # Compute weights based on interpolation position
        if method == "gaussian":
            # Gaussian weighting centered at alpha * n_dirs
            center = alpha * (n_dirs - 1)
            weights = [
                np.exp(-((i - center) ** 2) / (2 * self.sigma ** 2))
                for i in range(n_dirs)
            ]
        elif method == "linear":
            # Linear weighting
            weights = [1.0 - abs(alpha - i / (n_dirs - 1)) for i in range(n_dirs)]
            weights = [max(0, w) for w in weights]
        elif method == "sigmoid":
            # Sigmoid weighting
            center = alpha * (n_dirs - 1)
            weights = [
                1.0 / (1.0 + np.exp(-(i - center) / self.sigma))
                for i in range(n_dirs)
            ]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        # Create weighted direction
        weighted_dirs = []
        for i, w in enumerate(weights):
            if w > 1e-6:
                weighted_dirs.append(directions[i] * w)

        return InterpolationResult(
            directions=weighted_dirs,
            weights=weights,
            interpolation_type=method,
            smoothness=self._compute_smoothness(weights)
        )

    def _compute_smoothness(self, weights: List[float]) -> float:
        """Compute smoothness of weight distribution."""
        if len(weights) < 2:
            return 1.0

        # Entropy-based smoothness
        total = sum(weights)
        if total == 0:
            return 0.0

        probs = [w / total for w in weights]
        entropy = -sum(p * np.log(p + 1e-8) for p in probs)
        max_entropy = np.log(len(weights))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_continuous_direction(
        self,
        directions: List[torch.Tensor],
        explained_variance: List[float],
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Get a single continuous direction vector.

        Args:
            directions: List of SVD direction vectors
            explained_variance: Variance explained by each direction
            alpha: Interpolation position

        Returns:
            Combined direction vector
        """
        result = self.interpolate_directions(directions, explained_variance, alpha)

        if not result.directions:
            return torch.zeros_like(directions[0]) if directions else torch.tensor([])

        # Sum weighted directions
        combined = sum(result.directions)

        # Normalize
        norm = torch.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def create_strength_curve(
        self,
        directions: List[torch.Tensor],
        explained_variance: List[float],
        steps: int = 10
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        Create a continuous strength curve for gradual removal.

        Args:
            directions: List of SVD direction vectors
            explained_variance: Variance explained by each direction
            steps: Number of steps in curve

        Returns:
            List of (alpha, direction) pairs
        """
        curve = []
        for i in range(steps + 1):
            alpha = i / steps
            direction = self.get_continuous_direction(directions, explained_variance, alpha)
            curve.append((alpha, direction))
        return curve

    def visualize_weights(self, n_dirs: int = 8, steps: int = 20) -> str:
        """
        Generate ASCII visualization of weight distribution.

        Returns:
            ASCII chart showing how weights change with alpha
        """
        import matplotlib.pyplot as plt
        import io
        import base64

        fig, ax = plt.subplots(figsize=(10, 6))

        alphas = np.linspace(0, 1, steps)
        weights_matrix = []

        for alpha in alphas:
            result = self.interpolate_directions(
                [torch.randn(1) for _ in range(n_dirs)],
                [1.0] * n_dirs,
                alpha=alpha
            )
            weights_matrix.append(result.weights)

        weights_matrix = np.array(weights_matrix)

        for i in range(n_dirs):
            ax.plot(alphas, weights_matrix[:, i], label=f"Direction {i+1}")

        ax.set_xlabel('Alpha')
        ax.set_ylabel('Weight')
        ax.set_title('Float Direction Interpolation Weights')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Convert to base64 for embedding
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()

        return img_base64
