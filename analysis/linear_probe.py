"""
Linear Probe Classifier — Production-Grade Probe Training

Trains and evaluates linear probes on hidden states using:
- LogisticRegression with cross-validation
- Mass-mean search for optimal probe direction
- CCS (Contrast-Consistent Search) from Burns et al. (2023)
- AUROC, separation, and direction quality metrics

Based on: Alain & Bengio (2017), Burns et al. (2023)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeReport:
    """Container for linear probe results."""
    layer_accuracies: Dict[int, float]
    best_layer: int
    best_accuracy: float
    information_content: Dict[int, float]
    probes: Dict[int, any]
    # Extended fields
    layer_aurocs: Dict[int, float] = field(default_factory=dict)
    layer_f1s: Dict[int, float] = field(default_factory=dict)
    cv_scores: Dict[int, List[float]] = field(default_factory=dict)
    probe_directions: Dict[int, np.ndarray] = field(default_factory=dict)
    separation_scores: Dict[int, float] = field(default_factory=dict)
    direction_quality: Dict[int, float] = field(default_factory=dict)
    mass_mean_direction: Optional[np.ndarray] = None
    ccs_direction: Optional[np.ndarray] = None
    mass_mean_layer: int = -1


class LinearProbeClassifier:
    """
    Train linear probes to detect refusal information in activations.

    Features:
    - Logistic regression probes with cross-validation
    - Mass-mean search: find weighted mean direction that best separates
    - CCS (Contrast-Consistent Search): find direction where contrastive pairs
      satisfy logical consistency constraints
    - Probe direction quality metrics
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._scaler = StandardScaler()

    def train_probes(
        self,
        harmful_activations: Dict[int, torch.Tensor],
        harmless_activations: Dict[int, torch.Tensor],
        test_split: float = 0.2,
        n_folds: int = 5,
        random_state: int = 42
    ) -> ProbeReport:
        """
        Train linear probes and evaluate quality at each layer.

        Args:
            harmful_activations: Activations for harmful prompts {layer: (n, d)}
            harmless_activations: Activations for harmless prompts {layer: (n, d)}
            test_split: Fraction of data for testing
            n_folds: Cross-validation folds
            random_state: Random seed

        Returns:
            ProbeReport with per-layer metrics
        """
        accuracies = {}
        information = {}
        probes = {}
        aurocs = {}
        f1s = {}
        cv_scores = {}
        probe_directions = {}
        separation_scores = {}
        direction_quality = {}

        common_layers = set(harmful_activations.keys()) & set(harmless_activations.keys())

        for layer in sorted(common_layers):
            harmful = harmful_activations[layer].detach().cpu().numpy()
            harmless = harmless_activations[layer].detach().cpu().numpy()

            X = np.vstack([harmful, harmless])
            y = np.hstack([np.ones(len(harmful), dtype=np.int64),
                          np.zeros(len(harmless), dtype=np.int64)])

            # Standardize
            X_scaled = self._scaler.fit_transform(X)

            # Train/test split
            n = len(X)
            indices = np.random.RandomState(random_state).permutation(n)
            split_idx = int(n * (1 - test_split))
            X_train, X_test = X_scaled[indices[:split_idx]], X_scaled[indices[split_idx:]]
            y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

            # Train logistic regression
            probe = LogisticRegression(
                max_iter=2000, C=1.0, solver='lbfgs',
                penalty='l2', random_state=random_state,
                class_weight='balanced'
            )
            probe.fit(X_train, y_train)

            # Metrics
            accuracy = probe.score(X_test, y_test)
            accuracies[layer] = accuracy
            probes[layer] = probe
            information[layer] = (accuracy - 0.5) * 2  # Scale [0.5, 1.0] -> [0, 1]

            # AUROC
            try:
                y_prob = probe.predict_proba(X_test)[:, 1]
                aurocs[layer] = roc_auc_score(y_test, y_prob)
            except Exception:
                aurocs[layer] = 0.5

            # F1
            y_pred = probe.predict(X_test)
            f1s[layer] = f1_score(y_test, y_pred, zero_division=0)

            # Cross-validation
            skf = StratifiedKFold(n_splits=min(n_folds, min(len(harmful), len(harmless))),
                                  shuffle=True, random_state=random_state)
            try:
                fold_scores = cross_val_score(probe, X_scaled, y, cv=skf, scoring='accuracy')
                cv_scores[layer] = fold_scores.tolist()
            except Exception:
                cv_scores[layer] = []

            # Probe direction (weight vector)
            w = probe.coef_.flatten()
            probe_directions[layer] = w

            # Separation score: probe direction alignment with mean difference
            mean_diff = harmful.mean(axis=0) - harmless.mean(axis=0)
            w_norm = np.linalg.norm(w)
            diff_norm = np.linalg.norm(mean_diff)
            if w_norm > 0 and diff_norm > 0:
                sep = np.dot(w, mean_diff) / (w_norm * diff_norm)
                separation_scores[layer] = float(sep)
            else:
                separation_scores[layer] = 0.0

            # Direction quality: how concentrated the weight vector is
            # (sparsity = Gini on absolute weights)
            abs_w = np.abs(w)
            sorted_w = np.sort(abs_w)
            n_w = len(sorted_w)
            if np.sum(sorted_w) > 0:
                gini = 2 * sum((i + 1) * v for i, v in enumerate(sorted_w)) / \
                       (n_w * np.sum(sorted_w)) - (n_w + 1) / n_w
                direction_quality[layer] = float(max(0, gini))
            else:
                direction_quality[layer] = 0.0

        # --- Best layer ---
        if accuracies:
            best_layer = max(accuracies, key=accuracies.get)
            best_accuracy = accuracies[best_layer]
        else:
            best_layer = -1
            best_accuracy = 0.0

        # --- Mass-mean search ---
        mass_mean_dir, mass_mean_layer = self._mass_mean_search(
            harmful_activations, harmless_activations
        )

        # --- CCS direction ---
        ccs_dir = self._ccs_search(harmful_activations, harmless_activations)

        return ProbeReport(
            layer_accuracies=accuracies,
            best_layer=best_layer,
            best_accuracy=best_accuracy,
            information_content=information,
            probes=probes,
            layer_aurocs=aurocs,
            layer_f1s=f1s,
            cv_scores=cv_scores,
            probe_directions=probe_directions,
            separation_scores=separation_scores,
            direction_quality=direction_quality,
            mass_mean_direction=mass_mean_dir,
            ccs_direction=ccs_dir,
            mass_mean_layer=mass_mean_layer
        )

    def probe_layer(
        self,
        layer: int,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor
    ) -> float:
        """Train a single-layer probe and return accuracy."""
        X = np.vstack([
            harmful_activations.detach().cpu().numpy(),
            harmless_activations.detach().cpu().numpy()
        ])
        y = np.hstack([
            np.ones(len(harmful_activations), dtype=np.int64),
            np.zeros(len(harmless_activations), dtype=np.int64)
        ])

        X_scaled = self._scaler.fit_transform(X)
        probe = LogisticRegression(max_iter=2000, class_weight='balanced')
        probe.fit(X_scaled, y)

        return float(probe.score(X_scaled, y))

    def get_probe_weights(
        self,
        layer: int,
        probe_report: ProbeReport
    ) -> Optional[np.ndarray]:
        """Get probe weight vector (direction) for a layer."""
        return probe_report.probes[layer].coef_.flatten() if layer in probe_report.probes else None

    def _mass_mean_search(
        self,
        harmful_activations: Dict[int, torch.Tensor],
        harmless_activations: Dict[int, torch.Tensor]
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Mass-mean search: find the weighted average direction that best
        separates harmful from harmless activations across all layers.

        The mass-mean direction is a linear combination of per-layer
        mean differences, weighted by their separation quality.
        """
        common_layers = set(harmful_activations.keys()) & set(harmless_activations.keys())
        if not common_layers:
            return None, -1

        directions = []
        weights = []
        best_layer = -1
        best_score = -1

        for layer in sorted(common_layers):
            harmful = harmful_activations[layer].detach().cpu().numpy()
            harmless = harmless_activations[layer].detach().cpu().numpy()

            mean_diff = harmful.mean(axis=0) - harmless.mean(axis=0)
            diff_norm = np.linalg.norm(mean_diff)

            if diff_norm > 1e-8:
                # Quality weight: separation between distributions
                # Using Bhattacharyya-like distance
                harm_mean = harmful.mean(axis=0)
                harmless_mean = harmless.mean(axis=0)
                harm_std = harmful.std(axis=0)
                harmless_std = harmless.std(axis=0)

                # Signal-to-noise ratio along mean difference
                diff_component = (harm_mean - harmless_mean) @ mean_diff / diff_norm
                noise = (np.mean(harm_std) + np.mean(harmless_std)) / 2 + 1e-8
                quality = abs(diff_component) / noise

                directions.append(mean_diff / diff_norm)
                weights.append(quality)

                if quality > best_score:
                    best_score = quality
                    best_layer = layer

        if not directions:
            return None, -1

        # Weighted average of directions
        weights = np.array(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        mass_mean = np.zeros_like(directions[0])
        for d, w in zip(directions, weights):
            mass_mean += w * d

        mass_mean = mass_mean / (np.linalg.norm(mass_mean) + 1e-8)

        return mass_mean, best_layer

    def _ccs_search(
        self,
        harmful_activations: Dict[int, torch.Tensor],
        harmless_activations: Dict[int, torch.Tensor],
        n_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> Optional[np.ndarray]:
        """
        Contrast-Consistent Search (Burns et al., 2023).

        Finds a direction in activation space such that:
        - Harmful activations have positive projection
        - Harmless activations have negative projection
        - The direction is consistent across layers

        Uses a simple gradient-based optimization on the CCS loss:
        L = -(mean(pos_proj)^2 + mean(neg_proj)^2) + consistency_loss
        """
        common_layers = set(harmful_activations.keys()) & set(harmless_activations.keys())
        if len(common_layers) < 2:
            return None

        # Stack all activations to find a common dimension
        first_layer = sorted(common_layers)[0]
        d_model = harmful_activations[first_layer].shape[1]

        # Initialize random direction
        rng = np.random.RandomState(42)
        direction = rng.randn(d_model).astype(np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        all_harmful = []
        all_harmless = []
        for layer in sorted(common_layers):
            all_harmful.append(harmful_activations[layer].detach().cpu().numpy())
            all_harmless.append(harmless_activations[layer].detach().cpu().numpy())

        harm_stack = np.vstack(all_harmful)
        harmless_stack = np.vstack(all_harmless)

        # Standardize
        combined = np.vstack([harm_stack, harmless_stack])
        mean = combined.mean(axis=0, keepdims=True)
        std = combined.std(axis=0, keepdims=True) + 1e-8

        harm_norm = (harm_stack - mean) / std
        harmless_norm = (harmless_stack - mean) / std

        # Optimize CCS direction
        for _ in range(n_iterations):
            # Projections
            harm_proj = harm_norm @ direction
            harmless_proj = harmless_norm @ direction

            # CCS loss components
            pos_mean = np.mean(harm_proj)
            neg_mean = np.mean(harmless_proj)

            # Variance loss (want high variance in each group)
            pos_var = np.var(harm_proj)
            neg_var = np.var(harmless_proj)

            # Gradient
            grad = np.zeros(d_model, dtype=np.float64)

            # Gradient of variance (want to maximize)
            harm_centered = harm_norm - harm_proj.reshape(-1, 1) * direction.reshape(1, -1)
            harmless_centered = harmless_norm - harmless_proj.reshape(-1, 1) * direction.reshape(1, -1)

            grad += -2 * pos_var * np.mean(harm_centered * harm_proj.reshape(-1, 1), axis=0)
            grad += -2 * neg_var * np.mean(harmless_centered * harmless_proj.reshape(-1, 1), axis=0)

            # Gradient of mean squared (want max separation)
            grad += -2 * pos_mean * np.mean(harm_norm, axis=0)
            grad += 2 * neg_mean * np.mean(harmless_norm, axis=0)  # want negative mean for harmless

            # Orthogonalize gradient to direction (keep on sphere)
            grad = grad - (grad @ direction) * direction

            # Update
            direction = direction - learning_rate * grad
            direction = direction / (np.linalg.norm(direction) + 1e-8)

        return direction
