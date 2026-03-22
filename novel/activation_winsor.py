"""
Activation Winsorization

Clamps activation vectors to percentile range before SVD
to prevent outlier-dominated directions.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class WinsorResult:
    """Container for winsorization results."""
    clipped_activations: torch.Tensor
    lower_bound: float
    upper_bound: float
    outliers_removed: int


class ActivationWinsorizer:
    """
    Activation Winsorization — Clamp activation vectors to percentile range
    before SVD to prevent outlier-dominated directions.

    Key Insight: Outliers can dominate SVD direction extraction,
    leading to noisy or suboptimal constraint directions.
    Winsorization limits the influence of extreme values.

    Novel technique not present in OBLITERATUS core.
    """

    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        """
        Initialize winsorizer.

        Args:
            lower_percentile: Lower percentile to clamp (e.g., 1.0 = 1st percentile)
            upper_percentile: Upper percentile to clamp (e.g., 99.0 = 99th percentile)
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def winsorize(
        self,
        activations: torch.Tensor,
        dim: int = 0
    ) -> WinsorResult:
        """
        Winsorize activation tensor.

        Args:
            activations: Activation tensor (n_samples, hidden_dim)
            dim: Dimension to compute percentiles (0 = across samples)

        Returns:
            WinsorResult with clipped activations
        """
        # Compute percentiles
        lower = torch.quantile(activations, self.lower_percentile / 100.0, dim=dim)
        upper = torch.quantile(activations, self.upper_percentile / 100.0, dim=dim)

        # Count outliers
        lower_outliers = (activations < lower).sum().item()
        upper_outliers = (activations > upper).sum().item()
        total_outliers = lower_outliers + upper_outliers

        # Clamp values
        clipped = torch.clamp(activations, min=lower, max=upper)

        return WinsorResult(
            clipped_activations=clipped,
            lower_bound=float(lower.mean()) if lower.numel() > 0 else 0,
            upper_bound=float(upper.mean()) if upper.numel() > 0 else 0,
            outliers_removed=total_outliers
        )

    def winsorize_batch(
        self,
        activation_batch: List[torch.Tensor]
    ) -> List[WinsorResult]:
        """
        Winsorize a batch of activation tensors.

        Args:
            activation_batch: List of activation tensors

        Returns:
            List of WinsorResult objects
        """
        results = []
        for acts in activation_batch:
            results.append(self.winsorize(acts))
        return results

    def extract_directions_with_winsor(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        extractor,
        n_directions: int = 4
    ) -> Any:
        """
        Extract directions after winsorization.

        Args:
            harmful_activations: Harmful prompt activations
            harmless_activations: Harmless prompt activations
            extractor: ConstraintExtractor instance
            n_directions: Number of directions to extract

        Returns:
            ExtractionResult with winsorized directions
        """
        # Winsorize both activation sets
        harmful_winsor = self.winsorize(harmful_activations)
        harmless_winsor = self.winsorize(harmless_activations)

        # Extract directions on winsorized data
        return extractor.extract_svd(
            harmful_winsor.clipped_activations,
            harmless_winsor.clipped_activations,
            n_directions=n_directions
        )

    def compare_without_winsor(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        extractor,
        n_directions: int = 4
    ) -> Dict[str, Any]:
        """
        Compare extraction with and without winsorization.

        Args:
            harmful_activations: Harmful prompt activations
            harmless_activations: Harmless prompt activations
            extractor: ConstraintExtractor instance
            n_directions: Number of directions to extract

        Returns:
            Comparison results
        """
        # Without winsorization
        result_raw = extractor.extract_svd(
            harmful_activations,
            harmless_activations,
            n_directions=n_directions
        )

        # With winsorization
        result_winsor = self.extract_directions_with_winsor(
            harmful_activations,
            harmless_activations,
            extractor,
            n_directions=n_directions
        )

        # Compare direction similarity
        similarity = 0.0
        if result_raw.directions and result_winsor.directions:
            d1 = result_raw.directions[0]
            d2 = result_winsor.directions[0]
            similarity = torch.dot(d1, d2).item()

        return {
            "raw_directions": len(result_raw.directions),
            "winsor_directions": len(result_winsor.directions),
            "raw_explained_variance": result_raw.explained_variance,
            "winsor_explained_variance": result_winsor.explained_variance,
            "direction_similarity": similarity,
            "outliers_removed": result_winsor.outliers_removed
        }

    def auto_tune_percentiles(
        self,
        activations: torch.Tensor,
        steps: int = 10
    ) -> Dict[str, float]:
        """
        Auto-tune percentile thresholds.

        Args:
            activations: Activation tensor
            steps: Number of tuning steps

        Returns:
            Optimal percentiles
        """
        best_lower = 1.0
        best_upper = 99.0
        best_entropy = 0.0

        for lower in [0.5, 1.0, 2.0, 3.0, 5.0]:
            for upper in [95.0, 97.0, 98.0, 99.0, 99.5]:
                # Winsorize at these percentiles
                lower_tensor = torch.quantile(activations, lower / 100.0)
                upper_tensor = torch.quantile(activations, upper / 100.0)

                clipped = torch.clamp(activations, min=lower_tensor, max=upper_tensor)

                # Compute entropy of distribution
                hist = torch.histc(clipped, bins=50)
                probs = hist / hist.sum()
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

                if entropy > best_entropy:
                    best_entropy = entropy
                    best_lower = lower
                    best_upper = upper

        return {
            "optimal_lower_percentile": best_lower,
            "optimal_upper_percentile": best_upper,
            "entropy": best_entropy
        }

    def get_outlier_statistics(
        self,
        activations: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Get outlier statistics for activation tensor.

        Returns:
            Outlier statistics
        """
        mean = activations.mean().item()
        std = activations.std().item()

        # Outliers: > 3 standard deviations
        outlier_mask = (activations > mean + 3 * std) | (activations < mean - 3 * std)
        outlier_count = outlier_mask.sum().item()
        outlier_ratio = outlier_count / activations.numel()

        return {
            "mean": mean,
            "std": std,
            "outlier_count": outlier_count,
            "outlier_ratio": outlier_ratio
        }
