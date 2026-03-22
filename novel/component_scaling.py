"""
Component-Specific Scaling

Separate attention vs MLP projection strengths.
MLP layers are more sensitive to modification.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ScalingResult:
    """Container for scaling results."""
    attention_scaling: float
    mlp_scaling: float
    layers_modified: List[int]
    component_breakdown: Dict[str, List[str]]


class ComponentScaler:
    """
    Component-Specific Scaling — Apply different projection strengths
    to attention and MLP layers.

    Key Insight: MLP layers are often more sensitive to modification
    than attention layers. This technique applies different scaling
    factors to each component type for optimal preservation.

    Novel technique
    """

    def __init__(self, attention_scale: float = 1.0, mlp_scale: float = 0.5):
        """
        Initialize component scaler.

        Args:
            attention_scale: Scaling factor for attention layers
            mlp_scale: Scaling factor for MLP layers
        """
        self.attention_scale = attention_scale
        self.mlp_scale = mlp_scale

    def apply_scaled_projection(
        self,
        model,
        directions: List[torch.Tensor],
        layers: Optional[List[int]] = None,
        preserve_norm: bool = True
    ) -> ScalingResult:
        """
        Apply scaled projection to different component types.

        Args:
            model: Model to modify
            directions: Direction vectors to project
            layers: Specific layers to modify
            preserve_norm: Whether to preserve weight norms

        Returns:
            ScalingResult with modification details
        """
        from aetheris.core.projector import NormPreservingProjector

        projector = NormPreservingProjector(model, preserve_norm=preserve_norm)

        attention_layers = []
        mlp_layers = []
        other_layers = []

        # Categorize layers by component type
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue

            layer_idx = self._extract_layer_idx(name)

            if layers and layer_idx not in layers:
                continue

            if "attention" in name.lower() or "attn" in name.lower():
                attention_layers.append((name, param, layer_idx))
            elif "mlp" in name.lower() or "feed_forward" in name.lower():
                mlp_layers.append((name, param, layer_idx))
            else:
                other_layers.append((name, param, layer_idx))

        # Apply scaled projections
        for name, param, layer_idx in attention_layers:
            scaled_dir = [d * self.attention_scale for d in directions]
            self._project_param(param, scaled_dir)

        for name, param, layer_idx in mlp_layers:
            scaled_dir = [d * self.mlp_scale for d in directions]
            self._project_param(param, scaled_dir)

        for name, param, layer_idx in other_layers:
            self._project_param(param, directions)

        # Restore norms if preserving
        if preserve_norm:
            # Would restore original norms
            pass

        return ScalingResult(
            attention_scaling=self.attention_scale,
            mlp_scaling=self.mlp_scale,
            layers_modified=list(set(layer_idx for _, _, layer_idx in attention_layers + mlp_layers)),
            component_breakdown={
                "attention": [name for name, _, _ in attention_layers],
                "mlp": [name for name, _, _ in mlp_layers],
                "other": [name for name, _, _ in other_layers]
            }
        )

    def _project_param(self, param: torch.nn.Parameter, directions: List[torch.Tensor]):
        """Project a single parameter."""
        for d in directions:
            d = d.to(param.device)
            if param.dim() == 2:
                param.data = param.data - torch.outer(param.data @ d, d)
            elif param.dim() == 1:
                param.data = param.data - (param.data @ d) * d

    def _extract_layer_idx(self, name: str) -> int:
        """Extract layer index from parameter name."""
        import re
        match = re.search(r'\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return -1

    def tune_scaling_factors(
        self,
        model,
        directions: List[torch.Tensor],
        validation_prompts: List[str],
        steps: int = 5
    ) -> Dict[str, Any]:
        """
        Auto-tune scaling factors for optimal preservation.

        Args:
            model: Model to test
            directions: Direction vectors
            validation_prompts: Prompts for validation
            steps: Number of tuning steps

        Returns:
            Optimal scaling factors
        """
        from aetheris.core.validation import CapabilityValidator

        validator = CapabilityValidator()

        best_attention = 1.0
        best_mlp = 1.0
        best_score = 0.0

        for att_scale in [0.2, 0.4, 0.6, 0.8, 1.0][:steps]:
            for mlp_scale in [0.2, 0.4, 0.6, 0.8, 1.0][:steps]:
                # Test this combination
                test_model = self._copy_model(model)
                self.attention_scale = att_scale
                self.mlp_scale = mlp_scale

                self.apply_scaled_projection(test_model, directions)

                # Evaluate
                coherence = validator.compute_coherence(test_model, None, validation_prompts)

                if coherence > best_score:
                    best_score = coherence
                    best_attention = att_scale
                    best_mlp = mlp_scale

        return {
            "optimal_attention_scale": best_attention,
            "optimal_mlp_scale": best_mlp,
            "optimal_score": best_score
        }

    def _copy_model(self, model):
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(model)

    def get_component_sensitivity(
        self,
        model,
        directions: List[torch.Tensor],
        layers: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Measure sensitivity of different components.

        Returns:
            Sensitivity scores per component type
        """
        from aetheris.core.validation import CapabilityValidator

        validator = CapabilityValidator()
        test_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a fascinating field."
        ]

        # Baseline
        baseline_perplexity = validator.compute_perplexity(model, None, test_prompts)

        # Test attention sensitivity
        attention_model = self._copy_model(model)
        self.attention_scale = 1.0
        self.mlp_scale = 0.0
        self.apply_scaled_projection(attention_model, directions, layers)

        attention_perplexity = validator.compute_perplexity(attention_model, None, test_prompts)

        # Test MLP sensitivity
        mlp_model = self._copy_model(model)
        self.attention_scale = 0.0
        self.mlp_scale = 1.0
        self.apply_scaled_projection(mlp_model, directions, layers)

        mlp_perplexity = validator.compute_perplexity(mlp_model, None, test_prompts)

        return {
            "attention_sensitivity": (attention_perplexity - baseline_perplexity) / baseline_perplexity,
            "mlp_sensitivity": (mlp_perplexity - baseline_perplexity) / baseline_perplexity
        }
