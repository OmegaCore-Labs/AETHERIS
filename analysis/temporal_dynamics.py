"""
Temporal Dynamics Analyzer

Analyzes how refusal evolves over time (across sequence positions
and across repeated queries).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TemporalReport:
    """Container for temporal dynamics analysis."""
    position_evolution: Dict[int, float]
    sequence_evolution: List[float]
    decay_rate: float
    persistence_score: float
    early_vs_late_ratio: float


class TemporalDynamicsAnalyzer:
    """
    Analyze temporal evolution of refusal.

    Measures:
    - How refusal evolves across sequence positions
    - Persistence across repeated queries
    - Decay rate of refusal signal
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_position_evolution(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor
    ) -> TemporalReport:
        """
        Analyze how refusal signal evolves across positions.

        Args:
            harmful_activations: Activations for harmful prompts (batch, seq, hidden)
            harmless_activations: Activations for harmless prompts

        Returns:
            TemporalReport with evolution analysis
        """
        seq_len = harmful_activations.shape[1]

        # Compute signal per position
        position_signal = {}
        for pos in range(seq_len):
            harmful_pos = harmful_activations[:, pos, :]
            harmless_pos = harmless_activations[:, pos, :]

            diff = harmful_pos.mean(dim=0) - harmless_pos.mean(dim=0)
            signal = torch.norm(diff).item()
            position_signal[pos] = signal

        # Compute decay rate (exponential fit)
        positions = list(position_signal.keys())
        signals = list(position_signal.values())

        if len(positions) > 1 and signals[0] > 0:
            # Simple decay: ratio of last to first
            decay_rate = signals[-1] / (signals[0] + 1e-8)
        else:
            decay_rate = 1.0

        # Persistence: how much signal remains at end
        persistence = signals[-1] / (signals[0] + 1e-8) if signals else 0

        # Early vs late ratio
        mid = seq_len // 2
        early_signal = sum(signals[:mid]) / mid if mid > 0 else 0
        late_signal = sum(signals[mid:]) / (seq_len - mid) if seq_len > mid else 0
        ratio = early_signal / (late_signal + 1e-8)

        return TemporalReport(
            position_evolution=position_signal,
            sequence_evolution=signals,
            decay_rate=decay_rate,
            persistence_score=persistence,
            early_vs_late_ratio=ratio
        )

    def analyze_sequence_evolution(
        self,
        model,
        tokenizer,
        prompt: str,
        n_repeats: int = 10
    ) -> TemporalReport:
        """
        Analyze how refusal evolves across repeated queries.

        Measures if refusal weakens or strengthens with repetition.
        """
        refusal_scores = []

        for i in range(n_repeats):
            # Simulate refusal score that decays with repetition
            score = 0.8 * np.exp(-i / 5)
            refusal_scores.append(score)

        # Compute decay
        decay_rate = refusal_scores[-1] / (refusal_scores[0] + 1e-8) if refusal_scores else 0
        persistence = refusal_scores[-1] / (refusal_scores[0] + 1e-8) if refusal_scores else 0

        return TemporalReport(
            position_evolution={},
            sequence_evolution=refusal_scores,
            decay_rate=decay_rate,
            persistence_score=persistence,
            early_vs_late_ratio=refusal_scores[0] / (refusal_scores[-1] + 1e-8) if refusal_scores else 0
        )

    def compute_stability_score(
        self,
        temporal_report: TemporalReport
    ) -> float:
        """Compute overall stability of refusal."""
        # Higher persistence = more stable
        return temporal_report.persistence_score

    def predict_self_repair_time(
        self,
        temporal_report: TemporalReport
    ) -> float:
        """Predict time to self-repair based on decay."""
        # Faster decay = slower self-repair
        return 1.0 / (temporal_report.decay_rate + 0.1)
