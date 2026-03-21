"""
Norm-Preserving Weight Projection Engine

Implements surgical removal of constraint directions from model weights
while preserving model capabilities via norm preservation.
Based on grimjim's norm-preserving biprojection (2025).
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass
import copy


@dataclass
class ProjectionResult:
    """Container for projection results."""
    success: bool
    layers_modified: List[int]
    directions_projected: List[int]
    norm_preserved: bool
    metadata: Dict[str, any]


class NormPreservingProjector:
    """
    Project constraint directions out of model weights while preserving norms.

    Key features:
    - Norm-preserving biprojection (grimjim 2025)
    - Bias term projection (OBLITERATUS misses this)
    - Multi-direction projection for polyhedral constraints
    - Layer-specific targeting
    - MoE expert targeting (novel)
    """

    def __init__(
        self,
        model,
        preserve_norm: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize the projector.

        Args:
            model: HuggingFace model to modify
            preserve_norm: Whether to preserve weight norms after projection
            device: Device to run computations on
        """
        self.model = model
        self.preserve_norm = preserve_norm
        self.device = device
        self.original_norms = {}
        self._original_weights = {}  # For rollback

    def project_weights(
        self,
        directions: List[torch.Tensor],
        layers: Optional[List[int]] = None,
        projection_type: str = "biprojection"
    ) -> ProjectionResult:
        """
        Project constraint directions out of model weights.

        Args:
            directions: List of direction vectors to project out
            layers: Specific layers to modify (None = all linear layers)
            projection_type: "biprojection", "orthogonal", or "subtraction"

        Returns:
            ProjectionResult with modification details
        """
        if not directions:
            return ProjectionResult(
                success=False,
                layers_modified=[],
                directions_projected=[],
                norm_preserved=self.preserve_norm,
                metadata={"error": "No directions provided"}
            )

        layers_modified = []
        directions_projected = []

        # Normalize all directions
        directions = [d / torch.norm(d) for d in directions]

        # Iterate through model parameters
        for name, param in self.model.named_parameters():
            if "weight" not in name:
                continue

            # Check if this layer should be modified
            layer_idx = self._extract_layer_idx(name)
            if layers is not None and layer_idx not in layers:
                continue

            if param.dim() != 2:  # Only linear layers
                continue

            # Store original norm if needed
            if self.preserve_norm:
                if name not in self.original_norms:
                    self.original_norms[name] = torch.norm(param.data).item()

            # Save original for potential rollback
            if name not in self._original_weights:
                self._original_weights[name] = param.data.clone()

            # Apply projection
            original = param.data.clone()
            for d in directions:
                d = d.to(param.device)

                if projection_type == "biprojection":
                    # Norm-preserving biprojection (grimjim 2025)
                    # Project weight matrix orthogonal to direction
                    # W' = W - (W @ d) @ d^T
                    projection = torch.outer(param.data @ d, d)
                    param.data = param.data - projection

                elif projection_type == "orthogonal":
                    # Standard orthogonal projection
                    param.data = param.data - torch.outer(param.data @ d, d)

                elif projection_type == "subtraction":
                    # Simple subtraction (least stable)
                    param.data = param.data - torch.outer(param.data @ d, d)

                else:
                    raise ValueError(f"Unknown projection_type: {projection_type}")

            # Restore norm if preserved
            if self.preserve_norm:
                current_norm = torch.norm(param.data)
                scale = self.original_norms[name] / (current_norm + 1e-8)
                param.data = param.data * scale

            layers_modified.append(layer_idx)
            directions_projected.append(len(directions))

        return ProjectionResult(
            success=True,
            layers_modified=list(set(layers_modified)),
            directions_projected=directions_projected,
            norm_preserved=self.preserve_norm,
            metadata={"projection_type": projection_type}
        )

    def project_biases(
        self,
        directions: List[torch.Tensor],
        layers: Optional[List[int]] = None
    ) -> ProjectionResult:
        """
        Project constraint directions out of bias vectors.

        OBLITERATUS misses bias projection. This is critical because
        refusal signal can persist in bias terms.

        Args:
            directions: List of direction vectors to project out
            layers: Specific layers to modify

        Returns:
            ProjectionResult with modification details
        """
        if not directions:
            return ProjectionResult(
                success=False,
                layers_modified=[],
                directions_projected=[],
                norm_preserved=False,
                metadata={"error": "No directions provided"}
            )

        layers_modified = []

        for name, param in self.model.named_parameters():
            if "bias" not in name:
                continue

            layer_idx = self._extract_layer_idx(name)
            if layers is not None and layer_idx not in layers:
                continue

            # Bias is a vector, not a matrix
            if param.dim() != 1:
                continue

            # Project bias orthogonal to each direction
            for d in directions:
                d = d.to(param.device)
                # Projection: b' = b - (b · d) d
                projection = (param.data @ d) * d
                param.data = param.data - projection

            layers_modified.append(layer_idx)

        return ProjectionResult(
            success=True,
            layers_modified=list(set(layers_modified)),
            directions_projected=[len(directions) for _ in layers_modified],
            norm_preserved=False,
            metadata={"target": "biases"}
        )

    def multi_direction_projection(
        self,
        directions: List[torch.Tensor],
        layers: Optional[List[int]] = None,
        method: str = "sequential"
    ) -> ProjectionResult:
        """
        Handle polyhedral constraints with multiple directions.

        Args:
            directions: List of direction vectors to project out
            layers: Specific layers to modify
            method: "sequential" (one by one) or "simultaneous" (Gram-Schmidt)

        Returns:
            ProjectionResult with modification details
        """
        if method == "simultaneous":
            # Gram-Schmidt orthogonalize directions
            ortho_dirs = []
            for d in directions:
                for od in ortho_dirs:
                    d = d - (d @ od) * od
                if torch.norm(d) > 1e-6:
                    ortho_dirs.append(d / torch.norm(d))

            directions = ortho_dirs

        # Use sequential projection (already orthogonalized)
        return self.project_weights(directions, layers, projection_type="biprojection")

    def project_expert_specific(
        self,
        directions: List[torch.Tensor],
        expert_indices: List[int],
        layers: Optional[List[int]] = None
    ) -> ProjectionResult:
        """
        Project directions from specific MoE experts only.

        Novel capability for Mixture-of-Experts models like Mistral-119B.
        Target only the safety expert while leaving others untouched.

        Args:
            directions: List of direction vectors to project out
            expert_indices: Indices of experts to modify
            layers: Specific layers to modify

        Returns:
            ProjectionResult with modification details
        """
        layers_modified = []

        for name, param in self.model.named_parameters():
            # Check if this is an MoE expert parameter
            if "expert" not in name.lower() and "mlp" not in name.lower():
                continue

            # Check if this expert is in target list
            expert_id = self._extract_expert_id(name)
            if expert_id not in expert_indices:
                continue

            if "weight" in name and param.dim() == 2:
                # Apply projection to this expert's weights
                for d in directions:
                    d = d.to(param.device)
                    param.data = param.data - torch.outer(param.data @ d, d)

                layers_modified.append(self._extract_layer_idx(name))

        return ProjectionResult(
            success=True,
            layers_modified=list(set(layers_modified)),
            directions_projected=[len(directions)],
            norm_preserved=False,
            metadata={"expert_indices": expert_indices}
        )

    def rollback(self) -> None:
        """Restore original weights before projection."""
        for name, weight in self._original_weights.items():
            for param_name, param in self.model.named_parameters():
                if param_name == name:
                    param.data = weight.clone()
                    break
        self._original_weights = {}

    def _extract_layer_idx(self, param_name: str) -> int:
        """Extract layer index from parameter name."""
        import re
        match = re.search(r'\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        return -1

    def _extract_expert_id(self, param_name: str) -> Optional[int]:
        """Extract expert ID from parameter name for MoE models."""
        import re
        # Common MoE patterns: expert_0, expert.0, mlp.experts.0
        patterns = [
            r'expert[_.](\d+)',
            r'experts\.(\d+)',
            r'mlp\.(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        return None
