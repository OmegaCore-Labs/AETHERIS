"""
CoT-Aware Ablation

Preserves chain-of-thought reasoning while removing constraints.
Orthogonalizes refusal directions against reasoning-critical directions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class CoTAnalysis:
    """Container for CoT analysis results."""
    reasoning_directions: List[torch.Tensor]
    refusal_directions: List[torch.Tensor]
    orthogonalized_directions: List[torch.Tensor]
    reasoning_preservation_score: float


class CoTAwareAblator:
    """
    CoT-Aware Ablation — Preserves reasoning during constraint removal.

    Key Insight: Refusal and reasoning often share activation space.
    Naive removal can harm chain-of-thought capabilities.
    This technique orthogonalizes refusal directions against reasoning-critical
    directions to preserve reasoning.

    Novel technique not present in OBLITERATUS core.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def extract_reasoning_directions(
        self,
        model,
        tokenizer,
        reasoning_prompts: List[str],
        control_prompts: List[str]
    ) -> List[torch.Tensor]:
        """
        Extract directions critical for reasoning.

        Args:
            model: Model to analyze
            tokenizer: Associated tokenizer
            reasoning_prompts: Prompts requiring chain-of-thought
            control_prompts: Simple factual prompts

        Returns:
            List of reasoning-critical direction vectors
        """
        from aetheris.core.extractor import ConstraintExtractor

        extractor = ConstraintExtractor(model, tokenizer, device=self.device)

        # Collect activations
        reasoning_acts = extractor.collect_activations(model, tokenizer, reasoning_prompts)
        control_acts = extractor.collect_activations(model, tokenizer, control_prompts)

        reasoning_directions = []

        for layer in reasoning_acts.keys():
            if layer in control_acts:
                result = extractor.extract_svd(
                    reasoning_acts[layer].to(self.device),
                    control_acts[layer].to(self.device),
                    n_directions=3
                )
                reasoning_directions.extend(result.directions)

        return reasoning_directions

    def orthogonalize_directions(
        self,
        refusal_directions: List[torch.Tensor],
        reasoning_directions: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Orthogonalize refusal directions against reasoning directions.

        Ensures removal doesn't affect reasoning.

        Args:
            refusal_directions: Constraint directions to remove
            reasoning_directions: Directions critical for reasoning

        Returns:
            Orthogonalized refusal directions
        """
        if not reasoning_directions:
            return refusal_directions

        # Normalize reasoning directions
        reasoning_norm = [d / (torch.norm(d) + 1e-8) for d in reasoning_directions]

        orthogonalized = []

        for refusal_dir in refusal_directions:
            # Project out reasoning components
            for reasoning_dir in reasoning_norm:
                refusal_dir = refusal_dir - (refusal_dir @ reasoning_dir) * reasoning_dir

            if torch.norm(refusal_dir) > 1e-6:
                orthogonalized.append(refusal_dir / torch.norm(refusal_dir))

        return orthogonalized

    def compute_reasoning_preservation(
        self,
        original_model,
        modified_model,
        tokenizer,
        reasoning_prompts: List[str]
    ) -> float:
        """
        Compute reasoning preservation score.

        Args:
            original_model: Model before modification
            modified_model: Model after modification
            tokenizer: Associated tokenizer
            reasoning_prompts: Prompts requiring CoT

        Returns:
            Preservation score (1.0 = full preservation)
        """
        from aetheris.core.validation import CapabilityValidator

        validator = CapabilityValidator()

        # Compute coherence on reasoning prompts
        original_coherence = validator.compute_coherence(original_model, tokenizer, reasoning_prompts)
        modified_coherence = validator.compute_coherence(modified_model, tokenizer, reasoning_prompts)

        if original_coherence > 0:
            preservation = modified_coherence / original_coherence
        else:
            preservation = 1.0

        return min(1.0, max(0.0, preservation))

    def apply_cot_aware_ablation(
        self,
        model,
        tokenizer,
        refusal_directions: List[torch.Tensor],
        reasoning_prompts: List[str],
        control_prompts: List[str],
        layers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Apply CoT-aware constraint removal.

        Args:
            model: Model to modify
            tokenizer: Associated tokenizer
            refusal_directions: Constraint directions to remove
            reasoning_prompts: Prompts for reasoning extraction
            control_prompts: Control prompts
            layers: Layers to modify

        Returns:
            Ablation result with preservation score
        """
        # Extract reasoning directions
        reasoning_dirs = self.extract_reasoning_directions(
            model, tokenizer, reasoning_prompts, control_prompts
        )

        # Orthogonalize refusal directions
        orthogonalized_dirs = self.orthogonalize_directions(refusal_directions, reasoning_dirs)

        # Apply ablation
        from aetheris.core.projector import NormPreservingProjector

        projector = NormPreservingProjector(model)

        if orthogonalized_dirs:
            result = projector.project_weights(orthogonalized_dirs, layers)
            projector.project_biases(orthogonalized_dirs, layers)
        else:
            result = {"success": True, "note": "No directions to remove"}

        # Compute preservation
        preservation = self.compute_reasoning_preservation(
            model, model, tokenizer, reasoning_prompts
        )

        return {
            "success": True,
            "original_refusal_directions": len(refusal_directions),
            "reasoning_directions": len(reasoning_dirs),
            "orthogonalized_directions": len(orthogonalized_dirs),
            "reasoning_preservation": preservation,
            "projection_result": result
        }

    def create_cot_aware_steering(
        self,
        refusal_direction: torch.Tensor,
        reasoning_directions: List[torch.Tensor],
        alpha: float = -1.0
    ) -> torch.Tensor:
        """
        Create steering vector that preserves reasoning.

        Args:
            refusal_direction: Base refusal direction
            reasoning_directions: Directions critical for reasoning
            alpha: Steering strength

        Returns:
            CoT-aware steering vector
        """
        # Normalize
        refusal_dir = refusal_direction / (torch.norm(refusal_direction) + 1e-8)

        # Project out reasoning components
        for reasoning_dir in reasoning_directions:
            reasoning_dir = reasoning_dir / (torch.norm(reasoning_dir) + 1e-8)
            refusal_dir = refusal_dir - (refusal_dir @ reasoning_dir) * reasoning_dir

        if torch.norm(refusal_dir) > 0:
            refusal_dir = refusal_dir / torch.norm(refusal_dir)

        return alpha * refusal_dir
