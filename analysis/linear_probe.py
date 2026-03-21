"""
Linear Probe Classifier

Trains linear classifiers to detect refusal information in activations.
Based on Alain & Bengio (2017) linear probing for representation analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbeReport:
    """Container for linear probe results."""
    layer_accuracies: Dict[int, float]
    best_layer: int
    best_accuracy: float
    information_content: Dict[int, float]
    probes: Dict[int, any]


class LinearProbeClassifier:
    """
    Train linear probes to detect refusal information.

    Measures how much refusal information is linearly separable
    at each layer. Higher accuracy = more refusal signal.
    """

    def __init__(self):
        self.probes = {}

    def train_probes(
        self,
        harmful_activations: Dict[int, torch.Tensor],
        harmless_activations: Dict[int, torch.Tensor],
        test_split: float = 0.2
    ) -> ProbeReport:
        """
        Train linear probes for each layer.

        Args:
            harmful_activations: Activations for harmful prompts
            harmless_activations: Activations for harmless prompts
            test_split: Fraction of data for testing

        Returns:
            ProbeReport with accuracies per layer
        """
        accuracies = {}
        information = {}
        probes = {}

        for layer in harmful_activations.keys():
            if layer not in harmless_activations:
                continue

            # Prepare data
            harmful = harmful_activations[layer].cpu().numpy()
            harmless = harmless_activations[layer].cpu().numpy()

            X = np.vstack([harmful, harmless])
            y = np.hstack([np.ones(len(harmful)), np.zeros(len(harmless))])

            # Split
            n = len(X)
            indices = np.random.permutation(n)
            split_idx = int(n * (1 - test_split))

            X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
            y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

            # Train logistic regression
            probe = LogisticRegression(max_iter=1000, C=1.0)
            probe.fit(X_train, y_train)

            # Evaluate
            accuracy = probe.score(X_test, y_test)
            accuracies[layer] = accuracy
            probes[layer] = probe

            # Information content (normalized accuracy)
            information[layer] = (accuracy - 0.5) * 2  # Scale to 0-1

        # Find best layer
        if accuracies:
            best_layer = max(accuracies, key=accuracies.get)
            best_accuracy = accuracies[best_layer]
        else:
            best_layer = -1
            best_accuracy = 0

        return ProbeReport(
            layer_accuracies=accuracies,
            best_layer=best_layer,
            best_accuracy=best_accuracy,
            information_content=information,
            probes=probes
        )

    def probe_layer(
        self,
        layer: int,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor
    ) -> float:
        """Train a single-layer probe and return accuracy."""
        X = np.vstack([harmful_activations.cpu().numpy(),
                       harmless_activations.cpu().numpy()])
        y = np.hstack([np.ones(len(harmful_activations)),
                       np.zeros(len(harmless_activations))])

        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)

        return probe.score(X, y)

    def get_probe_weights(
        self,
        layer: int,
        probe_report: ProbeReport
    ) -> Optional[np.ndarray]:
        """Get probe weight vector for a layer."""
        if layer in probe_report.probes:
            return probe_report.probes[layer].coef_.flatten()
        return None
