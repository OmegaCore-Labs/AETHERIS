"""
Steering Vector Engine

Implements inference-time constraint steering without permanent weight modification.
Based on Turner et al. (2023) and Rimsky et al. (2024).
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import weakref


@dataclass
class SteeringConfig:
    """Configuration for steering vectors."""
    vectors: List[torch.Tensor]          # Steering vectors to apply
    target_layers: List[int]             # Layers to apply steering
    alpha: float = 1.0                   # Global scaling factor
    alphas: Optional[List[float]] = None # Per-layer scaling
    normalize: bool = True               # Normalize vectors before applying
    expert_indices: Optional[List[int]] = None  # MoE expert targeting


class SteeringVectorFactory:
    """
    Factory for creating steering vectors from extracted directions.
    """

    @staticmethod
    def from_refusal_direction(
        direction: torch.Tensor,
        alpha: float = -1.0,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Create steering vector from extracted refusal direction.

        Args:
            direction: Extracted direction vector
            alpha: Scaling factor (negative = reduce refusal)
            normalize: Whether to normalize the direction

        Returns:
            Steering vector (same shape as direction)
        """
        if normalize:
            direction = direction / (torch.norm(direction) + 1e-8)
        return alpha * direction

    @staticmethod
    def from_contrastive_pairs(
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        alpha: float = 1.0,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Create steering vector from contrastive activation pairs.

        Args:
            harmful_activations: Activations on harmful prompts
            harmless_activations: Activations on harmless prompts
            alpha: Scaling factor
            normalize: Whether to normalize the resulting vector

        Returns:
            Steering vector (mean harmful - mean harmless)
        """
        mean_harmful = harmful_activations.mean(dim=0)
        mean_harmless = harmless_activations.mean(dim=0)
        direction = mean_harmful - mean_harmless

        if normalize:
            direction = direction / (torch.norm(direction) + 1e-8)

        return alpha * direction

    @staticmethod
    def from_constraint_geometry(
        geometry_analyzer,
        constraint_type: str = "refusal",
        alpha: float = -1.0,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Create steering vector from geometric constraint analysis.

        Args:
            geometry_analyzer: GeometryAnalyzer instance with extracted geometry
            constraint_type: Type of constraint to steer
            alpha: Scaling factor
            normalize: Whether to normalize

        Returns:
            Steering vector
        """
        # Get the principal direction for the constraint
        direction = geometry_analyzer.get_principal_direction(constraint_type)

        if direction is None:
            raise ValueError(f"No direction found for constraint type: {constraint_type}")

        if normalize:
            direction = direction / (torch.norm(direction) + 1e-8)

        return alpha * direction


class SteeringHookManager:
    """
    Manage forward hooks for steering vector application at inference time.

    Based on Turner et al. (2023) activation addition.
    """

    def __init__(self):
        self._hooks = []
        self._active = False
        self._config = None
        self._steering_cache = {}

    def install(
        self,
        model: nn.Module,
        config: SteeringConfig
    ) -> None:
        """
        Install steering hooks on the model.

        Args:
            model: HuggingFace model to steer
            config: Steering configuration
        """
        if self._active:
            self.remove()

        self._config = config
        self._active = True

        # Register hooks on each target layer
        for layer_idx in config.target_layers:
            # Find the layer module
            layer = self._get_layer(model, layer_idx)
            if layer is None:
                continue

            # Get per-layer alpha
            if config.alphas and layer_idx < len(config.alphas):
                alpha = config.alphas[layer_idx]
            else:
                alpha = config.alpha

            # Cache steering vector for this layer
            self._steering_cache[layer_idx] = self._prepare_steering_vector(layer_idx, alpha)

            # Register forward hook
            handle = layer.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(handle)

    def remove(self) -> None:
        """Remove all steering hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._active = False
        self._config = None
        self._steering_cache = {}

    @contextmanager
    def temporary_steering(self, model: nn.Module, config: SteeringConfig):
        """
        Context manager for temporary steering.

        Example:
            with manager.temporary_steering(model, config):
                output = model.generate(...)
            # Steering automatically removed
        """
        self.install(model, config)
        try:
            yield
        finally:
            self.remove()

    def _prepare_steering_vector(self, layer_idx: int, alpha: float) -> torch.Tensor:
        """Prepare steering vector for a specific layer."""
        if not self._config.vectors:
            return None

        # Use first vector for now (multiple vectors TBD)
        vec = self._config.vectors[0]

        # Move to appropriate device
        if hasattr(vec, 'device'):
            vec = vec.to(vec.device)

        # Apply scaling
        vec = alpha * vec

        # Broadcast to batch dimension if needed
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)

        return vec

    def _make_hook(self, layer_idx: int):
        """Create forward hook for a specific layer."""

        def hook(module, input, output):
            if not self._active:
                return output

            # Get steering vector for this layer
            steering = self._steering_cache.get(layer_idx)
            if steering is None:
                return output

            # Handle output format
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()

            # Apply steering to hidden states
            if hidden.dim() == 3:  # (batch, seq, hidden)
                # Add steering to all positions
                steering_expanded = steering.to(hidden.device).unsqueeze(1)
                hidden = hidden + steering_expanded
            elif hidden.dim() == 2:  # (batch, hidden)
                hidden = hidden + steering.to(hidden.device)
            else:
                return output

            # Reconstruct output
            if rest:
                return (hidden,) + rest
            return hidden

        return hook

    def _get_layer(self, model: nn.Module, idx: int) -> Optional[nn.Module]:
        """Get transformer layer by index."""
        # Try different common model structures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            if idx < len(model.model.layers):
                return model.model.layers[idx]
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            if idx < len(model.transformer.h):
                return model.transformer.h[idx]
        if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            if idx < len(model.model.decoder.layers):
                return model.model.decoder.layers[idx]
        if hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layers'):
            if idx < len(model.model.encoder.layers):
                return model.model.encoder.layers[idx]

        # Fallback: try to find any module with .layers
        for module in model.modules():
            if hasattr(module, 'layers') and isinstance(module.layers, list):
                if idx < len(module.layers):
                    return module.layers[idx]

        return None


class MultiSteeringController:
    """
    Controller for multiple concurrent steering vectors.

    Allows combining multiple constraints (e.g., reduce refusal + enhance creativity).
    """

    def __init__(self, manager: SteeringHookManager):
        self.manager = manager
        self.vectors = []

    def add_vector(self, vector: torch.Tensor, alpha: float = 1.0, target_layers: List[int] = None):
        """Add a steering vector."""
        self.vectors.append({
            'vector': vector,
            'alpha': alpha,
            'layers': target_layers
        })

    def create_config(self) -> SteeringConfig:
        """Create combined configuration from all vectors."""
        if not self.vectors:
            raise ValueError("No vectors added")

        # Combine vectors by summing
        combined = sum(v['alpha'] * v['vector'] for v in self.vectors)

        # Use union of target layers
        all_layers = set()
        for v in self.vectors:
            if v['layers']:
                all_layers.update(v['layers'])

        return SteeringConfig(
            vectors=[combined],
            target_layers=sorted(all_layers) if all_layers else list(range(20)),
            alpha=1.0,  # Already applied in sum
            normalize=False
        )
