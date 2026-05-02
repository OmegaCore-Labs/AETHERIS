"""
Activation Probe — Production-Grade Linear Probe Training & Interpretation

Trains linear probes on paired (harmful, harmless) activations to detect
refusal directions using sklearn LogisticRegression with cross-validation,
decision boundary analysis, and probe weight interpretation.

Based on: Alain & Bengio (2017), Burns et al. (2023)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeReport:
    """Container for activation probe results."""
    refusal_signal: Dict[int, float]
    peak_layer: int
    signal_distribution: List[float]
    is_localized: bool
    # Probe-specific fields
    layer_accuracies: Dict[int, float] = field(default_factory=dict)
    layer_aurocs: Dict[int, float] = field(default_factory=dict)
    layer_f1s: Dict[int, float] = field(default_factory=dict)
    cv_scores: Dict[int, List[float]] = field(default_factory=dict)
    probe_weights: Dict[int, np.ndarray] = field(default_factory=dict)
    probe_biases: Dict[int, float] = field(default_factory=dict)
    decision_boundaries: Dict[int, float] = field(default_factory=dict)
    separation_scores: Dict[int, float] = field(default_factory=dict)
    best_layer: int = -1


class ActivationProbe:
    """
    Train linear probes on activations to detect refusal directions.

    Uses LogisticRegression with cross-validation to measure how linearly
    separable harmful vs harmless activations are at each layer.

    Features:
    - Probe training with optional regularization sweep
    - Stratified k-fold cross-validation
    - Decision boundary analysis
    - Probe weight interpretation (which dimensions matter most)
    - Signal localization via Gini coefficient
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._scaler = StandardScaler()

    def probe_refusal_signal(
        self,
        harmful_activations: Dict[int, torch.Tensor],
        harmless_activations: Dict[int, torch.Tensor],
        n_folds: int = 5,
        C_values: Optional[List[float]] = None,
        random_state: int = 42
    ) -> ProbeReport:
        """
        Train probes and measure refusal signal strength per layer.

        Args:
            harmful_activations: Activations on harmful prompts, {layer: tensor(n_samples, d_model)}
            harmless_activations: Activations on harmless prompts, {layer: tensor(n_samples, d_model)}
            n_folds: Number of cross-validation folds
            C_values: Regularization values to sweep (default: [0.001, 0.01, 0.1, 1.0, 10.0])
            random_state: Random seed for reproducibility

        Returns:
            ProbeReport with signal strengths, accuracies, AUROC, probe weights
        """
        if C_values is None:
            C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        signal = {}
        accuracies = {}
        aurocs = {}
        f1s = {}
        cv_scores = {}
        probe_weights = {}
        probe_biases = {}
        decision_boundaries = {}
        separation_scores = {}

        common_layers = set(harmful_activations.keys()) & set(harmless_activations.keys())

        if not common_layers:
            return ProbeReport(
                refusal_signal={}, peak_layer=-1, signal_distribution=[],
                is_localized=False, best_layer=-1
            )

        for layer in sorted(common_layers):
            harmful = harmful_activations[layer].detach().float()
            harmless = harmless_activations[layer].detach().float()

            # Compute raw signal strength (L2 norm of mean difference)
            mean_harmful = harmful.mean(dim=0)
            mean_harmless = harmless.mean(dim=0)
            diff = mean_harmful - mean_harmless
            signal[layer] = torch.norm(diff).item()

            # Prepare data for probe training
            X = torch.cat([harmful, harmless], dim=0).cpu().numpy()
            y = np.hstack([
                np.ones(len(harmful), dtype=np.int64),
                np.zeros(len(harmless), dtype=np.int64)
            ])

            # Standardize features
            X_scaled = self._scaler.fit_transform(X)

            # Cross-validated hyperparameter sweep
            best_probe = None
            best_cv = 0.0
            best_C = 1.0

            skf = StratifiedKFold(n_splits=min(n_folds, min(len(harmful), len(harmless))),
                                  shuffle=True, random_state=random_state)

            for C in C_values:
                probe = LogisticRegression(
                    max_iter=2000,
                    C=C,
                    solver='lbfgs',
                    penalty='l2',
                    random_state=random_state,
                    class_weight='balanced'
                )
                try:
                    scores = cross_val_score(probe, X_scaled, y, cv=skf, scoring='accuracy')
                    mean_score = scores.mean()
                    if mean_score > best_cv:
                        best_cv = mean_score
                        best_probe = probe
                        best_C = C
                except Exception:
                    continue

            if best_probe is None:
                best_probe = LogisticRegression(
                    max_iter=2000, C=1.0, solver='lbfgs',
                    random_state=random_state, class_weight='balanced'
                )

            # Train final probe on all data
            best_probe.fit(X_scaled, y)
            accuracies[layer] = best_probe.score(X_scaled, y)

            # AUROC
            try:
                y_prob = best_probe.predict_proba(X_scaled)[:, 1]
                aurocs[layer] = roc_auc_score(y, y_prob)
            except Exception:
                aurocs[layer] = 0.5

            # F1 score
            y_pred = best_probe.predict(X_scaled)
            f1s[layer] = f1_score(y, y_pred, zero_division=0)

            # Cross-validation scores
            try:
                cv_fold_scores = cross_val_score(
                    best_probe, X_scaled, y, cv=skf, scoring='accuracy'
                )
                cv_scores[layer] = cv_fold_scores.tolist()
            except Exception:
                cv_scores[layer] = []

            # Probe weights and bias
            probe_weights[layer] = best_probe.coef_.flatten().copy()
            probe_biases[layer] = float(best_probe.intercept_[0])

            # Decision boundary analysis
            # Distance from decision boundary = ||w|| / |b| gives margin width
            w_norm = np.linalg.norm(probe_weights[layer])
            if w_norm > 0:
                decision_boundaries[layer] = abs(probe_biases[layer]) / w_norm
            else:
                decision_boundaries[layer] = 0.0

            # Separation score (cosine similarity between probe weights and diff vector)
            diff_np = diff.cpu().numpy()
            diff_norm = np.linalg.norm(diff_np)
            if diff_norm > 0 and w_norm > 0:
                cos_sim = np.dot(probe_weights[layer], diff_np) / (w_norm * diff_norm)
                separation_scores[layer] = float(cos_sim)
            else:
                separation_scores[layer] = 0.0

        # Find peak and best layers
        peak_layer = max(signal, key=signal.get) if signal else -1
        best_layer = max(accuracies, key=accuracies.get) if accuracies else -1

        # Signal distribution (normalized)
        signal_values = list(signal.values())
        max_signal = max(signal_values) if signal_values else 1.0
        signal_dist = [v / max_signal for v in signal_values] if max_signal > 0 else []

        # Gini coefficient for signal localization
        is_localized = False
        if signal_values:
            sorted_s = sorted(signal_values, reverse=True)
            total = sum(sorted_s)
            if total > 0:
                n = len(sorted_s)
                gini = (2 * sum((i + 1) * v for i, v in enumerate(sorted_s))) / (n * total) - (n + 1) / n
                is_localized = gini > 0.5

        return ProbeReport(
            refusal_signal=signal,
            peak_layer=peak_layer,
            signal_distribution=signal_dist,
            is_localized=is_localized,
            layer_accuracies=accuracies,
            layer_aurocs=aurocs,
            layer_f1s=f1s,
            cv_scores=cv_scores,
            probe_weights=probe_weights,
            probe_biases=probe_biases,
            decision_boundaries=decision_boundaries,
            separation_scores=separation_scores,
            best_layer=best_layer
        )

    def compute_signal_amplitude(
        self,
        harmful_activations: Dict[int, torch.Tensor],
        harmless_activations: Dict[int, torch.Tensor]
    ) -> Dict[int, float]:
        """
        Compute normalized signal amplitude per layer.

        amplitude[layer] = ||mean_harmful - mean_harmless|| / ||mean_harmless||

        Higher amplitude means greater relative change from baseline.
        """
        amplitude = {}
        for layer in harmful_activations:
            if layer not in harmless_activations:
                continue

            harmful = harmful_activations[layer].float()
            harmless = harmless_activations[layer].float()

            mean_harmful = harmful.mean(dim=0)
            mean_harmless = harmless.mean(dim=0)

            diff_norm = torch.norm(mean_harmful - mean_harmless).item()
            harmless_norm = torch.norm(mean_harmless).item()

            amplitude[layer] = diff_norm / (harmless_norm + 1e-8)

        return amplitude

    def interpret_probe_weights(
        self,
        report: ProbeReport,
        top_k: int = 20
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Interpret probe weights to find important dimensions.

        Returns top-k most positive (toward refusal) and most negative
        (toward compliance) dimensions for each layer.

        Args:
            report: ProbeReport from probe_refusal_signal()
            top_k: Number of top dimensions to return

        Returns:
            Dict[layer] -> {"positive_indices": array, "negative_indices": array,
                            "positive_weights": array, "negative_weights": array}
        """
        interpretation = {}
        for layer, weights in report.probe_weights.items():
            sorted_idx = np.argsort(weights)
            pos_idx = sorted_idx[-top_k:][::-1]
            neg_idx = sorted_idx[:top_k]

            interpretation[layer] = {
                "positive_indices": pos_idx,
                "negative_indices": neg_idx,
                "positive_weights": weights[pos_idx],
                "negative_weights": weights[neg_idx]
            }
        return interpretation
