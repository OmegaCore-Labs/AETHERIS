"""
Multi-Token Position Analyzer

Analyzes where in the sequence refusal signal concentrates.
Identifies if refusal is triggered by specific token positions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TokenReport:
    """Container for multi-token analysis."""
    position_scores: Dict[int, float]
    critical_positions: List[int]
    position_distribution: List[float]
    is_position_specific: bool


class MultiTokenAnalyzer:
    """
    Analyze token position distribution of refusal signal.

    Determines whether refusal is triggered by early tokens,
    late tokens, or distributed across the sequence.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_positions(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor
    ) -> TokenReport:
        """
        Analyze token position contributions.

        Args:
            harmful_activations: Activations for harmful prompts (batch, seq, hidden)
            harmless_activations: Activations for harmless prompts (batch, seq, hidden)

        Returns:
            TokenReport with position analysis
        """
        # Compute difference per position
        seq_len = harmful_activations.shape[1]

        position_scores = {}
        for pos in range(seq_len):
            harmful_pos = harmful_activations[:, pos, :]
            harmless_pos = harmless_activations[:, pos, :]

            # Signal strength at this position
            diff = harmful_pos.mean(dim=0) - harmless_pos.mean(dim=0)
            score = torch.norm(diff).item()
            position_scores[pos] = score

        # Find critical positions (top 20%)
        scores = list(position_scores.values())
        threshold = np.percentile(scores, 80) if scores else 0
        critical = [p for p, s in position_scores.items() if s > threshold]

        # Distribution
        distribution = [s / max(scores) for s in scores] if scores else []

        # Check if position-specific (concentrated in few positions)
        if scores:
            concentration = max(scores) / (sum(scores) + 1e-8)
            is_specific = concentration > 0.3
        else:
            is_specific = False

        return TokenReport(
            position_scores=position_scores,
            critical_positions=critical,
            position_distribution=distribution,
            is_position_specific=is_specific
        )

    def find_trigger_positions(
        self,
        token_report: TokenReport,
        threshold: float = 0.7
    ) -> List[int]:
        """Find positions above normalized threshold."""
        norm_scores = token_report.position_distribution
        return [i for i, s in enumerate(norm_scores) if s > threshold]

    def compute_early_late_ratio(
        self,
        token_report: TokenReport,
        split_ratio: float = 0.5
    ) -> float:
        """Compute ratio of early to late token signal."""
        positions = sorted(token_report.position_scores.keys())
        n = len(positions)
        split = int(n * split_ratio)

        early_sum = sum(token_report.position_scores[p] for p in positions[:split])
        late_sum = sum(token_report.position_scores[p] for p in positions[split:])

        return early_sum / (late_sum + 1e-8)
