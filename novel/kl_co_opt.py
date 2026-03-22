"""
KL-Divergence Co-Optimization

Post-projection feedback loop that partially reverts over-projected layers
if KL divergence budget is exceeded.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class KLOptReport:
    """Container for KL optimization results."""
    initial_kl: float
    final_kl: float
    budget_met: bool
    layers_adjusted: List[int]
    adjustment_factors: Dict[int, float]


class KLDivergenceCoOptimizer:
    """
    KL-Divergence Co-Optimization.

    Key Insight: Over-projection can cause distribution shift.
    This technique monitors KL divergence after each projection step
    and partially reverts if the budget is exceeded.

    Novel technique
    """

    def __init__(self, kl_budget: float = 0.5):
        """
        Initialize KL co-optimizer.

        Args:
            kl_budget: Maximum allowed KL divergence
        """
        self.kl_budget = kl_budget

    def compute_kl_divergence(
        self,
        original_model,
        modified_model,
        tokenizer,
        test_prompts: List[str],
        max_tokens: int = 100
    ) -> float:
        """
        Compute KL divergence between model outputs.

        Args:
            original_model: Model before modification
            modified_model: Model after modification
            tokenizer: Associated tokenizer
            test_prompts: Prompts for evaluation
            max_tokens: Maximum generation tokens

        Returns:
            Average KL divergence
        """
        from aetheris.core.validation import CapabilityValidator

        validator = CapabilityValidator()
        return validator.compute_kl_divergence(
            original_model, modified_model, tokenizer, test_prompts, max_tokens
        )

    def optimize_projection(
        self,
        model,
        original_model,
        tokenizer,
        directions: List[torch.Tensor],
        layers: List[int],
        test_prompts: List[str],
        max_iterations: int = 5,
        step_size: float = 0.2
    ) -> KLOptReport:
        """
        Optimize projection to stay within KL budget.

        Args:
            model: Model being modified
            original_model: Original model for KL comparison
            tokenizer: Associated tokenizer
            directions: Directions to project
            layers: Layers being modified
            test_prompts: Prompts for KL evaluation
            max_iterations: Maximum optimization iterations
            step_size: Adjustment factor per iteration

        Returns:
            KLOptReport with optimization results
        """
        from aetheris.core.projector import NormPreservingProjector

        # Save original weights for rollback
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.data.clone()

        # Compute initial KL after full projection
        projector = NormPreservingProjector(model)

        # Apply full projection
        projector.project_weights(directions, layers)
        projector.project_biases(directions, layers)

        current_kl = self.compute_kl_divergence(original_model, model, tokenizer, test_prompts)

        if current_kl <= self.kl_budget:
            return KLOptReport(
                initial_kl=current_kl,
                final_kl=current_kl,
                budget_met=True,
                layers_adjusted=[],
                adjustment_factors={}
            )

        # KL exceeded budget — need to adjust
        initial_kl = current_kl
        layers_adjusted = []
        adjustment_factors = {}

        for iteration in range(max_iterations):
            # Determine which layers contributed most
            layer_contributions = self._compute_layer_contributions(
                model, original_model, tokenizer, layers, test_prompts
            )

            # Adjust layers with highest contribution
            sorted_layers = sorted(layer_contributions.items(), key=lambda x: x[1], reverse=True)

            for layer, contribution in sorted_layers[:3]:  # Adjust top 3
                if layer not in adjustment_factors:
                    adjustment_factors[layer] = 1.0

                # Reduce projection strength for this layer
                adjustment_factors[layer] *= (1 - step_size)
                layers_adjusted.append(layer)

                # Restore weights and reapply with adjusted strength
                self._restore_weights(model, original_weights)

                # Reapply with adjusted strengths
                for proj_layer in layers:
                    if proj_layer in adjustment_factors:
                        factor = adjustment_factors[proj_layer]
                        scaled_dir = [d * factor for d in directions]
                        projector.project_weights(scaled_dir, [proj_layer])
                        projector.project_biases(scaled_dir, [proj_layer])
                    else:
                        projector.project_weights(directions, [proj_layer])
                        projector.project_biases(directions, [proj_layer])

            # Recompute KL
            current_kl = self.compute_kl_divergence(original_model, model, tokenizer, test_prompts)

            if current_kl <= self.kl_budget:
                break

        return KLOptReport(
            initial_kl=initial_kl,
            final_kl=current_kl,
            budget_met=current_kl <= self.kl_budget,
            layers_adjusted=list(set(layers_adjusted)),
            adjustment_factors=adjustment_factors
        )

    def _compute_layer_contributions(
        self,
        modified_model,
        original_model,
        tokenizer,
        layers: List[int],
        test_prompts: List[str]
    ) -> Dict[int, float]:
        """
        Compute each layer's contribution to KL divergence.
        """
        contributions = {}
        original_weights = {}

        # Save original weights
        for name, param in modified_model.named_parameters():
            original_weights[name] = param.data.clone()

        for layer in layers:
            # Temporarily revert this layer
            self._revert_layer(modified_model, original_weights, layer)

            # Compute KL after reverting this layer
            kl = self.compute_kl_divergence(original_model, modified_model, tokenizer, test_prompts)

            # Contribution is how much KL decreased after reverting
            contributions[layer] = kl

            # Restore
            self._restore_weights(modified_model, original_weights)

        return contributions

    def _restore_weights(self, model, original_weights: Dict[str, torch.Tensor]) -> None:
        """Restore model weights from backup."""
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data = original_weights[name].clone()

    def _revert_layer(self, model, original_weights: Dict[str, torch.Tensor], layer: int) -> None:
        """Revert a specific layer to original weights."""
        for name, param in model.named_parameters():
            if f".{layer}." in name and name in original_weights:
                param.data = original_weights[name].clone()
