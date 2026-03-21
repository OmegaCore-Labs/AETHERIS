"""
Emergent Behavior Detector

Detects emergent constraints that appear after modification.
Monitors for new refusal patterns that arise from removal.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EmergenceReport:
    """Container for emergent behavior detection."""
    new_constraints_detected: List[Dict[str, any]]
    emergence_score: float
    affected_layers: List[int]
    pattern_type: str  # "novel", "composite", "re-emergent"
    recommended_action: str


class EmergentBehaviorDetector:
    """
    Detect emergent constraints after modification.

    Monitors for:
    - Novel constraints that didn't exist before
    - Composite constraints formed from multiple removed ones
    - Re-emergent constraints (Ouroboros effect)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def detect_emergence(
        self,
        original_directions: List[torch.Tensor],
        new_directions: List[torch.Tensor],
        similarity_threshold: float = 0.7
    ) -> EmergenceReport:
        """
        Detect emergent constraints from new directions.

        Args:
            original_directions: Directions before removal
            new_directions: Directions after removal
            similarity_threshold: Threshold for considering a direction novel

        Returns:
            EmergenceReport with detected constraints
        """
        new_constraints = []
        emergence_scores = []

        for new_dir in new_directions:
            # Check similarity to original directions
            max_sim = 0
            for orig_dir in original_directions:
                sim = torch.dot(new_dir, orig_dir).item()
                max_sim = max(max_sim, sim)

            if max_sim < similarity_threshold:
                # Novel constraint
                new_constraints.append({
                    "type": "novel",
                    "similarity_to_original": max_sim,
                    "strength": torch.norm(new_dir).item()
                })
                emergence_scores.append(1.0 - max_sim)
            elif max_sim > 0.9:
                # Re-emergent constraint (Ouroboros)
                new_constraints.append({
                    "type": "re-emergent",
                    "similarity_to_original": max_sim,
                    "strength": torch.norm(new_dir).item()
                })
                emergence_scores.append(max_sim - 0.9)

        # Determine pattern type
        if any(c["type"] == "novel" for c in new_constraints):
            if any(c["type"] == "re-emergent" for c in new_constraints):
                pattern = "composite"
            else:
                pattern = "novel"
        elif any(c["type"] == "re-emergent" for c in new_constraints):
            pattern = "re-emergent"
        else:
            pattern = "none"

        emergence_score = np.mean(emergence_scores) if emergence_scores else 0

        return EmergenceReport(
            new_constraints_detected=new_constraints,
            emergence_score=emergence_score,
            affected_layers=[],  # Would require layer info
            pattern_type=pattern,
            recommended_action=self._get_recommendation(pattern, emergence_score)
        )

    def _get_recommendation(self, pattern: str, score: float) -> str:
        """Get recommendation based on emergence pattern."""
        if pattern == "novel":
            return "Novel constraints detected. Consider additional Ouroboros passes."
        elif pattern == "re-emergent":
            return "Original constraints re-emerging. Increase refinement passes."
        elif pattern == "composite":
            return "Composite constraints forming. Use multi-direction projection."
        else:
            return "No significant emergence detected. System stable."

    def predict_emergence_risk(
        self,
        original_directions: List[torch.Tensor],
        model_size: int
    ) -> float:
        """
        Predict risk of emergent constraints.

        Larger models = higher emergence risk.
        """
        risk = len(original_directions) * 0.1
        risk += min(0.5, model_size / 140)  # 70B = 0.5

        return min(1.0, risk)

    def monitor_over_time(
        self,
        direction_history: List[List[torch.Tensor]],
        window: int = 5
    ) -> Dict[str, any]:
        """
        Monitor emergence over time.

        Args:
            direction_history: List of direction sets over time
            window: Sliding window for trend analysis

        Returns:
            Trend analysis
        """
        if len(direction_history) < 2:
            return {"trend": "insufficient_data"}

        # Count new constraints per step
        n_new = []
        for i in range(1, len(direction_history)):
            prev = direction_history[i-1]
            curr = direction_history[i]

            # Simple count of novel directions
            novel_count = 0
            for cur_dir in curr:
                max_sim = max(torch.dot(cur_dir, p_dir).item() for p_dir in prev) if prev else 0
                if max_sim < 0.7:
                    novel_count += 1
            n_new.append(novel_count)

        # Trend analysis
        if len(n_new) >= 2:
            trend = "increasing" if n_new[-1] > n_new[-2] else "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "new_constraints_per_step": n_new,
            "prediction": "Emergence may continue" if trend == "increasing" else "Emergence stabilizing"
        }
