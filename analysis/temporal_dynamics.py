"""
Temporal Dynamics Analyzer

Tracks how constraint activations evolve during autoregressive generation.
Hooks at each generation step, projects onto refusal direction, detects
when refusal "kicks in" and whether it's gradual or sudden.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class TemporalReport:
    """Container for temporal dynamics analysis."""
    position_evolution: Dict[int, float]
    sequence_evolution: List[float]
    decay_rate: float
    persistence_score: float
    early_vs_late_ratio: float
    refusal_onset_step: int = -1  # First step where refusal exceeds threshold
    onset_gradient: float = 0.0  # How sudden the onset is (gradient)
    generation_step_projections: Dict[int, float] = field(default_factory=dict)
    time_series_statistics: Dict[str, float] = field(default_factory=dict)
    status: str = "no_data"  # "ok", "no_data", "error"


class TemporalDynamicsAnalyzer:
    """
    Analyze how refusal evolves over time during token generation.

    Tracks constraint activation strength at each autoregressive step
    by projecting hidden states onto the refusal direction. Detects:

    - Refusal onset: at which generation step does refusal activate?
    - Onset steepness: is it gradual or sudden?
    - Decay rate: does refusal weaken during generation?
    - Position evolution: how does signal vary across input positions?
    """

    def __init__(self, device: str = "cpu", refusal_threshold: float = 0.5):
        self.device = device
        self.refusal_threshold = refusal_threshold

    def analyze_position_evolution(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        refusal_direction: Optional[torch.Tensor] = None,
    ) -> TemporalReport:
        """
        Analyze how refusal signal evolves across input sequence positions.

        Args:
            harmful_activations: Activations for harmful prompts (batch, seq, hidden)
            harmless_activations: Activations for harmless prompts
            refusal_direction: Optional known refusal direction for projection

        Returns:
            TemporalReport with evolution analysis
        """
        if harmful_activations is None or harmless_activations is None:
            return TemporalReport(
                position_evolution={}, sequence_evolution=[], decay_rate=1.0,
                persistence_score=0.0, early_vs_late_ratio=1.0, status="no_data"
            )

        try:
            seq_len = harmful_activations.shape[1]

            # Compute signal per position
            position_signal: Dict[int, float] = {}

            for pos in range(seq_len):
                harmful_pos = harmful_activations[:, pos, :].float()
                harmless_pos = harmless_activations[:, pos, :].float()

                if refusal_direction is not None:
                    # Project onto refusal direction
                    direction = refusal_direction.float().flatten()
                    norm = torch.norm(direction)
                    if norm > 1e-8:
                        direction = direction / norm

                    harm_proj = torch.sum(harmful_pos * direction.unsqueeze(0), dim=-1).mean().item()
                    harmless_proj = torch.sum(harmless_pos * direction.unsqueeze(0), dim=-1).mean().item()
                    signal = abs(harm_proj - harmless_proj)
                else:
                    # L2 norm of difference as fallback
                    diff = harmful_pos.mean(dim=0) - harmless_pos.mean(dim=0)
                    signal = torch.norm(diff).item()

                position_signal[pos] = signal

            signals = [position_signal.get(p, 0.0) for p in range(seq_len)]

            # Decay rate: ratio of last to first signal
            if signals[0] > 1e-8:
                decay_rate = signals[-1] / signals[0]
            else:
                decay_rate = 1.0

            # Persistence: signal at end relative to max
            max_signal = max(signals) if signals else 1.0
            if max_signal > 1e-8:
                persistence = signals[-1] / max_signal
            else:
                persistence = 0.0

            # Early vs late ratio
            mid = seq_len // 2
            early_signal = np.mean(signals[:mid]) if mid > 0 else 0.0
            late_signal = np.mean(signals[mid:]) if seq_len > mid else 0.0

            if late_signal > 1e-8:
                ratio = early_signal / late_signal
            else:
                ratio = early_signal / 1e-8 if early_signal > 0 else 1.0

            # Detect refusal onset: first position exceeding threshold
            # Threshold = max signal * 0.5
            threshold = max_signal * self.refusal_threshold if max_signal > 0 else 0.0
            onset_step = -1
            for pos, signal in position_signal.items():
                if signal >= threshold:
                    onset_step = pos
                    break

            # Onset gradient: how quickly signal rises
            onset_gradient = 0.0
            if onset_step > 0:
                prev = position_signal.get(onset_step - 1, 0.0)
                curr = position_signal.get(onset_step, 0.0)
                onset_gradient = curr - prev
            elif len(signals) > 1:
                # Compute max gradient across positions
                diffs = [signals[i + 1] - signals[i] for i in range(len(signals) - 1)]
                onset_gradient = max(diffs) if diffs else 0.0

            # Time-series statistics
            stats = self._compute_time_series_stats(signals)

            return TemporalReport(
                position_evolution=position_signal,
                sequence_evolution=signals,
                decay_rate=round(decay_rate, 6),
                persistence_score=round(persistence, 4),
                early_vs_late_ratio=round(ratio, 4),
                refusal_onset_step=onset_step,
                onset_gradient=round(onset_gradient, 6),
                generation_step_projections=position_signal,
                time_series_statistics=stats,
                status="ok",
            )
        except Exception as e:
            return TemporalReport(
                position_evolution={}, sequence_evolution=[], decay_rate=1.0,
                persistence_score=0.0, early_vs_late_ratio=1.0,
                status=f"error: {str(e)}",
            )

    def analyze_generation_dynamics(
        self,
        generation_hidden_states: List[torch.Tensor],
        refusal_direction: torch.Tensor,
    ) -> TemporalReport:
        """
        Analyze refusal evolution across autoregressive generation steps.

        Args:
            generation_hidden_states: List of hidden states at each generation step
            refusal_direction: Refusal direction vector for projection

        Returns:
            TemporalReport with generation-step dynamics
        """
        if not generation_hidden_states or refusal_direction is None:
            return TemporalReport(
                position_evolution={}, sequence_evolution=[], decay_rate=1.0,
                persistence_score=0.0, early_vs_late_ratio=1.0, status="no_data"
            )

        try:
            direction = refusal_direction.float().flatten()
            dir_norm = torch.norm(direction)
            if dir_norm > 1e-8:
                direction = direction / dir_norm

            step_projections: Dict[int, float] = {}
            signals = []

            for step, hidden in enumerate(generation_hidden_states):
                # Extract last token hidden state
                if hidden.dim() == 3:
                    last_token = hidden[:, -1, :].float()  # (batch, hidden)
                elif hidden.dim() == 2:
                    last_token = hidden.float()
                else:
                    continue

                # Project onto refusal direction
                proj = torch.sum(last_token * direction.unsqueeze(0), dim=-1).mean().item()
                step_projections[step] = proj
                signals.append(proj)

            if not signals:
                return TemporalReport(
                    position_evolution={}, sequence_evolution=[], decay_rate=1.0,
                    persistence_score=0.0, early_vs_late_ratio=1.0, status="no_data"
                )

            # Decay / growth analysis
            if signals[0] > 1e-8:
                decay_rate = signals[-1] / signals[0]
            else:
                decay_rate = 1.0

            max_signal = max(abs(s) for s in signals) if signals else 1.0
            persistence = abs(signals[-1]) / max_signal if max_signal > 1e-8 else 0.0

            # Early vs late
            mid = len(signals) // 2
            early_signal = np.mean(signals[:mid]) if mid > 0 else 0.0
            late_signal = np.mean(signals[mid:]) if len(signals) > mid else 0.0
            ratio = early_signal / (late_signal + 1e-8)

            # Onset detection: when does projection magnitude cross threshold?
            threshold = max_signal * self.refusal_threshold
            onset_step = -1
            for step, proj in step_projections.items():
                if abs(proj) >= threshold:
                    onset_step = step
                    break

            # Onset steepness
            onset_gradient = 0.0
            if len(signals) > 2:
                diffs = [signals[i + 1] - signals[i] for i in range(len(signals) - 1)]
                max_diff = max(diffs)
                min_diff = min(diffs)
                onset_gradient = max_diff if abs(max_diff) > abs(min_diff) else min_diff

            stats = self._compute_time_series_stats(signals)

            return TemporalReport(
                position_evolution=step_projections,
                sequence_evolution=signals,
                decay_rate=round(decay_rate, 6),
                persistence_score=round(persistence, 4),
                early_vs_late_ratio=round(ratio, 4),
                refusal_onset_step=onset_step,
                onset_gradient=round(onset_gradient, 6),
                generation_step_projections=step_projections,
                time_series_statistics=stats,
                status="ok",
            )
        except Exception as e:
            return TemporalReport(
                position_evolution={}, sequence_evolution=[], decay_rate=1.0,
                persistence_score=0.0, early_vs_late_ratio=1.0,
                status=f"error: {str(e)}",
            )

    def analyze_sequence_evolution(
        self,
        activations: List[torch.Tensor],
        refusal_direction: Optional[torch.Tensor] = None,
    ) -> TemporalReport:
        """
        Analyze how refusal signal evolves across repeated/similar queries.

        Args:
            activations: List of activation tensors for each repeated query
            refusal_direction: Optional refusal direction

        Returns:
            TemporalReport with sequence evolution
        """
        if not activations:
            return TemporalReport(
                position_evolution={}, sequence_evolution=[], decay_rate=1.0,
                persistence_score=0.0, early_vs_late_ratio=1.0, status="no_data"
            )

        try:
            signals = []
            for act in activations:
                if refusal_direction is not None:
                    direction = refusal_direction.float().flatten()
                    dir_norm = torch.norm(direction)
                    if dir_norm > 1e-8:
                        direction = direction / dir_norm
                    proj = torch.sum(act.float().mean(dim=(0, 1)) * direction).item()
                else:
                    proj = torch.norm(act.float()).item()
                signals.append(proj)

            if signals[0] > 1e-8:
                decay_rate = signals[-1] / signals[0]
            else:
                decay_rate = 1.0

            persistence = signals[-1] / (signals[0] + 1e-8)

            return TemporalReport(
                position_evolution={},
                sequence_evolution=signals,
                decay_rate=round(decay_rate, 6),
                persistence_score=round(persistence, 4),
                early_vs_late_ratio=round(
                    signals[0] / (signals[-1] + 1e-8), 4
                ),
                status="ok",
            )
        except Exception as e:
            return TemporalReport(
                position_evolution={}, sequence_evolution=[], decay_rate=1.0,
                persistence_score=0.0, early_vs_late_ratio=1.0,
                status=f"error: {str(e)}",
            )

    def _compute_time_series_stats(
        self, signals: List[float]
    ) -> Dict[str, float]:
        """Compute statistics on the signal time series."""
        if not signals:
            return {}

        arr = np.array(signals, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            return {}

        arr_finite = arr[finite_mask]

        stats = {
            "mean": float(np.mean(arr_finite)),
            "std": float(np.std(arr_finite)) if len(arr_finite) > 1 else 0.0,
            "min": float(np.min(arr_finite)),
            "max": float(np.max(arr_finite)),
            "range": float(np.max(arr_finite) - np.min(arr_finite)),
        }

        # Trend: linear regression slope
        if len(arr_finite) > 2:
            x = np.arange(len(arr_finite), dtype=np.float64)
            slope = np.polyfit(x, arr_finite, 1)[0]
            stats["linear_trend"] = round(float(slope), 6)
            stats["trend_direction"] = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"

        # Autocorrelation at lag 1
        if len(arr_finite) > 2:
            mean = np.mean(arr_finite)
            var = np.var(arr_finite)
            if var > 1e-10:
                acf1 = np.sum(
                    (arr_finite[:-1] - mean) * (arr_finite[1:] - mean)
                ) / (len(arr_finite) * var)
                stats["autocorrelation_lag1"] = round(float(acf1), 4)

        return stats

    def compute_stability_score(
        self,
        temporal_report: TemporalReport,
    ) -> float:
        """
        Compute overall stability score of refusal dynamics.

        Higher persistence + lower onset gradient = more stable.
        """
        persistence = temporal_report.persistence_score
        onset = abs(temporal_report.onset_gradient)

        # Stability: high persistence and low onset gradient
        stability = persistence * 0.6 + (1.0 - min(1.0, onset)) * 0.4
        return max(0.0, min(1.0, stability))

    def detect_onset_pattern(
        self,
        temporal_report: TemporalReport,
    ) -> str:
        """
        Classify the onset pattern: gradual, sudden, delayed, or immediate.

        Returns:
            String describing the onset pattern.
        """
        onset = temporal_report.refusal_onset_step
        gradient = abs(temporal_report.onset_gradient)

        if onset < 0:
            return "no_onset_detected"
        elif onset <= 2:
            return "immediate" if gradient <= 0.3 else "sudden"
        elif onset <= 10:
            return "delayed_early" if gradient <= 0.3 else "delayed_sudden"
        else:
            return "late" if gradient <= 0.3 else "late_sudden"

    def predict_self_repair_time(
        self,
        temporal_report: TemporalReport,
    ) -> Dict[str, Any]:
        """
        Predict time/effort needed for self-repair based on temporal dynamics.

        Returns dict with estimated repair metrics.
        """
        decay = temporal_report.decay_rate
        persistence = temporal_report.persistence_score

        # Slower decay = longer persistence = easier self-repair
        if decay < 0.3:
            repair_likelihood = "high"
            estimated_passes = 3
        elif decay < 0.6:
            repair_likelihood = "moderate"
            estimated_passes = 2
        else:
            repair_likelihood = "low"
            estimated_passes = 1

        return {
            "repair_likelihood": repair_likelihood,
            "estimated_additional_passes": estimated_passes,
            "decay_rate": decay,
            "persistence": persistence,
        }
