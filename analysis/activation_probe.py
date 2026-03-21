"""
Activation Probe

Measures refusal signal strength at each layer using activation differences.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ProbeReport:
    """Container for activation probe results."""
    refusal_signal: Dict[int, float]
    peak_layer: int
    signal_distribution: List[float]
    is_localized: bool


class ActivationProbe:
    """
    Probe refusal signal strength across layers.

    Measures the difference between harmful and harmless activations
    to quantify how much refusal signal exists at each layer.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def probe_refusal_signal(
        self,
        harmful_activations: Dict[int, torch.Tensor],
        harmless_activations: Dict[int, torch.Tensor]
    ) -> ProbeReport:
        """
        Measure refusal signal strength per layer.

        Args:
            harmful_activations: Activations on harmful prompts
            harmless_activations: Activations on harmless prompts

        Returns:
            ProbeReport with signal strengths
        """
        signal = {}

        for layer in harmful_activations.keys():
            if layer not in harmless_activations:
                continue

            harmful = harmful_activations[layer]
            harmless = harmless_activations[layer]

            # Compute difference in means
            mean_harmful = harmful.mean(dim=0)
            mean_harmless = harmless.mean(dim=0)
            diff = mean_harmful - mean_harmless

            # Signal strength = L2 norm of difference
            signal[layer] = torch.norm(diff).item()

        # Find peak layer
        if signal:
            peak_layer = max(signal, key=signal.get)
        else:
            peak_layer = -1

        # Compute signal distribution
        signal_values = list(signal.values())
        signal_dist = [v / max(signal_values) for v in signal_values] if signal_values else []

        # Check if signal is localized (concentrated in few layers)
        if signal_values:
            # Gini coefficient for localization
            sorted_signal = sorted(signal_values, reverse=True)
            total = sum(sorted_signal)
            if total > 0:
                cumulative = 0
                gini = 0
                for i, val in enumerate(sorted_signal):
                    cumulative += val
                    gini += (i + 1) * val
                gini = (2 * gini) / (len(sorted_signal) * total) - (len(sorted_signal) + 1) / len(sorted_signal)
                is_localized = gini > 0.5
            else:
                is_localized = False
        else:
            is_localized = False

        return ProbeReport(
            refusal_signal=signal,
            peak_layer=peak_layer,
            signal_distribution=signal_dist,
            is_localized=is_localized
        )

    def compute_signal_amplitude(
        self,
        harmful_activations: Dict[int, torch.Tensor],
        harmless_activations: Dict[int, torch.Tensor]
    ) -> Dict[int, float]:
        """
        Compute normalized signal amplitude per layer.

        Returns amplitude = ||mean_harmful - mean_harmless|| / ||mean_harmless||
        """
        amplitude = {}

        for layer in harmful_activations.keys():
            if layer not in harmless_activations:
                continue

            harmful = harmful_activations[layer]
            harmless = harmless_activations[layer]

            mean_harmful = harmful.mean(dim=0)
            mean_harmless = harmless.mean(dim=0)

            diff_norm = torch.norm(mean_harmful - mean_harmless).item()
            harmless_norm = torch.norm(mean_harmless).item()

            amplitude[layer] = diff_norm / (harmless_norm + 1e-8)

        return amplitude
