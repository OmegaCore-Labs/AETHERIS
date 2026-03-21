"""
Spectral Analyzer

Spectral decomposition of constraint directions.
Analyzes eigenvalue spectra and identifies dominant components.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpectralReport:
    """Container for spectral analysis."""
    eigenvalues: List[float]
    eigenvectors: List[torch.Tensor]
    spectral_energy: float
    dominant_frequency: float
    spectral_entropy: float
    rank_estimate: int


class SpectralAnalyzer:
    """
    Spectral analysis of constraint directions.

    Features:
    - Eigenvalue decomposition
    - Spectral energy distribution
    - Dominant frequency identification
    - Spectral entropy (complexity measure)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_spectrum(
        self,
        activations: torch.Tensor,
        n_components: int = 50
    ) -> SpectralReport:
        """
        Analyze spectral properties of activations.

        Args:
            activations: Activation tensor (n_samples, hidden_dim)
            n_components: Number of components to analyze

        Returns:
            SpectralReport with analysis
        """
        # Center data
        centered = activations - activations.mean(dim=0)

        # Compute covariance
        cov = centered.T @ centered / (centered.shape[0] - 1)

        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort descending
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # Take top components
        eigenvalues = eigenvalues[:n_components].cpu().numpy()
        eigenvectors = [eigenvectors[:, i] for i in range(min(n_components, len(eigenvectors)))]

        # Spectral energy (sum of eigenvalues)
        spectral_energy = np.sum(eigenvalues)

        # Spectral entropy (measure of complexity)
        if spectral_energy > 0:
            probs = eigenvalues / spectral_energy
            spectral_entropy = -np.sum(probs * np.log(probs + 1e-8))
        else:
            spectral_entropy = 0

        # Normalize entropy
        max_entropy = np.log(len(eigenvalues))
        spectral_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0

        # Dominant frequency (simplified: index of largest eigenvalue)
        dominant_frequency = np.argmax(eigenvalues)

        # Rank estimate (effective rank via eigenvalue entropy)
        if spectral_energy > 0:
            rank_estimate = int(np.exp(spectral_entropy * max_entropy))
        else:
            rank_estimate = 0

        return SpectralReport(
            eigenvalues=eigenvalues.tolist(),
            eigenvectors=eigenvectors,
            spectral_energy=spectral_energy,
            dominant_frequency=float(dominant_frequency),
            spectral_entropy=spectral_entropy,
            rank_estimate=rank_estimate
        )

    def compute_power_spectrum(
        self,
        direction: torch.Tensor,
        n_frequencies: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute power spectrum of a direction vector.
        """
        # Convert to numpy and apply FFT
        signal = direction.cpu().numpy()
        fft = np.fft.fft(signal)
        power = np.abs(fft) ** 2

        frequencies = np.fft.fftfreq(len(signal))
        # Take positive frequencies
        positive_idx = frequencies > 0
        frequencies = frequencies[positive_idx]
        power = power[positive_idx]

        return {
            "frequencies": frequencies,
            "power": power
        }

    def compute_spectral_similarity(
        self,
        spectrum1: SpectralReport,
        spectrum2: SpectralReport
    ) -> float:
        """
        Compute similarity between two spectra.

        Returns:
            Cosine similarity of eigenvalue vectors.
        """
        ev1 = np.array(spectrum1.eigenvalues[:min(len(spectrum1.eigenvalues), len(spectrum2.eigenvalues))])
        ev2 = np.array(spectrum2.eigenvalues[:len(ev1)])

        norm1 = np.linalg.norm(ev1)
        norm2 = np.linalg.norm(ev2)

        if norm1 > 0 and norm2 > 0:
            return np.dot(ev1, ev2) / (norm1 * norm2)
        return 0.0

    def get_dominant_components(
        self,
        spectral_report: SpectralReport,
        variance_threshold: float = 0.9
    ) -> List[int]:
        """
        Get indices of components explaining threshold variance.
        """
        eigenvalues = spectral_report.eigenvalues
        total = sum(eigenvalues)

        cumulative = 0
        components = []
        for i, ev in enumerate(eigenvalues):
            cumulative += ev
            components.append(i)
            if cumulative / total >= variance_threshold:
                break

        return components

    def compute_spectral_gap_ratio(
        self,
        spectral_report: SpectralReport
    ) -> float:
        """
        Compute ratio of largest to second largest eigenvalue.

        Higher ratio = more concentrated spectral energy.
        """
        ev = spectral_report.eigenvalues
        if len(ev) > 1:
            return ev[0] / (ev[1] + 1e-8)
        return 1.0
