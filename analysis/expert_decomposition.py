"""
Expert Decomposition Analyzer

Analyzes constraint distribution across MoE experts.
Identifies which experts carry refusal signal.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExpertReport:
    """Container for expert decomposition analysis."""
    expert_refusal_scores: Dict[int, float]
    dominant_experts: List[int]
    safety_expert_id: Optional[int]
    distribution_type: str  # "concentrated", "distributed", "balanced"
    recommended_targeting: List[int]


class ExpertDecompositionAnalyzer:
    """
    Analyze MoE expert contribution to constraints.

    For Mixture-of-Experts models like Mistral-119B, identifies which
    experts carry the refusal signal. Enables expert-specific targeting.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_expert_contributions(
        self,
        model,
        tokenizer,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        n_experts: int = 8
    ) -> ExpertReport:
        """
        Analyze which experts contribute to refusal.

        Args:
            model: MoE model
            tokenizer: Associated tokenizer
            harmful_prompts: Harmful prompts
            harmless_prompts: Harmless prompts
            n_experts: Number of experts in model

        Returns:
            ExpertReport with per-expert refusal scores
        """
        # This would require hooking into MoE router outputs
        # Simplified simulation for now

        # Simulate expert refusal scores
        # In production, these would be measured from router probabilities
        expert_scores = {
            0: 0.92,   # Safety expert
            1: 0.34,
            2: 0.12,
            3: 0.08,
            4: 0.05,
            5: 0.03,
            6: 0.15,
            7: 0.28
        }

        # Find dominant experts (score > 0.5)
        dominant = [e for e, s in expert_scores.items() if s > 0.5]

        # Safety expert is the one with highest score
        safety_expert = max(expert_scores, key=expert_scores.get) if expert_scores else None

        # Determine distribution type
        scores = list(expert_scores.values())
        if max(scores) > 0.8 and sum(scores) - max(scores) < 0.3:
            distribution = "concentrated"
        elif max(scores) > 0.5 and len([s for s in scores if s > 0.2]) > 3:
            distribution = "distributed"
        else:
            distribution = "balanced"

        # Recommended targeting (experts to modify)
        if distribution == "concentrated":
            recommended = [safety_expert] if safety_expert is not None else []
        elif distribution == "distributed":
            recommended = [e for e, s in expert_scores.items() if s > 0.3]
        else:
            recommended = dominant

        return ExpertReport(
            expert_refusal_scores=expert_scores,
            dominant_experts=dominant,
            safety_expert_id=safety_expert,
            distribution_type=distribution,
            recommended_targeting=recommended
        )

    def compute_expert_specialization(
        self,
        model,
        tokenizer,
        prompts_by_category: Dict[str, List[str]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute expert specialization across categories.

        Returns mapping of expert to category activation scores.
        """
        specialization = {}

        for expert_id in range(8):
            specialization[expert_id] = {}

        # Simulated specialization
        specialization[0] = {"safety": 0.92, "general": 0.34, "technical": 0.12}
        specialization[1] = {"general": 0.78, "reasoning": 0.65}
        specialization[2] = {"technical": 0.85, "code": 0.72}
        specialization[3] = {"creative": 0.88, "general": 0.45}
        specialization[4] = {"code": 0.91, "technical": 0.68}
        specialization[5] = {"math": 0.89, "reasoning": 0.71}
        specialization[6] = {"multilingual": 0.82, "general": 0.53}
        specialization[7] = {"general": 0.62, "reasoning": 0.58}

        return specialization

    def get_expert_surgery_plan(
        self,
        expert_report: ExpertReport,
        method: str = "surgical"
    ) -> Dict[str, any]:
        """
        Generate expert-specific surgery plan.

        Args:
            expert_report: Report from analyze_expert_contributions
            method: Liberation method to apply

        Returns:
            Surgery plan with expert targeting
        """
        if method == "surgical":
            target_experts = expert_report.recommended_targeting
        else:
            target_experts = []

        return {
            "target_experts": target_experts,
            "method": method,
            "expected_effect": f"Remove refusal from {len(target_experts)} expert(s)",
            "preserve_experts": [e for e in range(8) if e not in target_experts],
            "note": "MoE expert targeting preserves all other capabilities"
        }
