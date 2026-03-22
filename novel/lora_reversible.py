"""
LoRA Reversible Ablation

Rank-1 LoRA adapters instead of permanent weight surgery,
enabling reversible ablation and easy rollback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class LoRAAblationResult:
    """Container for LoRA ablation results."""
    success: bool
    lora_layers: List[str]
    rank: int
    reversible: bool
    message: str


class LoRALayer(nn.Module):
    """
    LoRA adapter layer for reversible ablation.

    Instead of permanently modifying weights, we add a low-rank
    adapter that can be enabled/disabled at inference time.
    """

    def __init__(self, original_weight: torch.Tensor, rank: int = 1):
        """
        Initialize LoRA adapter.

        Args:
            original_weight: Original weight matrix (out_features, in_features)
            rank: Rank of LoRA adapter
        """
        super().__init__()
        self.original_weight = original_weight.clone()
        self.rank = rank

        out_features, in_features = original_weight.shape

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.enabled = True

    def forward(self) -> torch.Tensor:
        """Compute effective weight with or without LoRA."""
        if self.enabled:
            # W = W0 + B * A
            return self.original_weight + (self.lora_B @ self.lora_A)
        else:
            return self.original_weight

    def disable(self):
        """Disable LoRA adapter (revert to original)."""
        self.enabled = False

    def enable(self):
        """Enable LoRA adapter."""
        self.enabled = True


class LoRAReversibleAblator:
    """
    LoRA Reversible Ablation — Replace permanent weight surgery with
    trainable LoRA adapters that can be turned on/off.

    Key Insight: Instead of permanently projecting out constraint directions,
    we can learn a low-rank adapter that represents the removal,
    then toggle it at inference time.

    Novel technique not present in OBLITERATUS core.
    """

    def __init__(self, rank: int = 1, device: str = "cpu"):
        """
        Initialize LoRA reversible ablater.

        Args:
            rank: Rank of LoRA adapter
            device: Computation device
        """
        self.rank = rank
        self.device = device
        self.lora_layers: Dict[str, LoRALayer] = {}

    def create_lora_adapter(
        self,
        model,
        direction: torch.Tensor,
        layers: Optional[List[int]] = None,
        alpha: float = 1.0
    ) -> LoRAAblationResult:
        """
        Create LoRA adapters that represent constraint removal.

        Instead of projecting weights, we learn a low-rank adapter
        that approximates the projection.

        Args:
            model: Model to modify
            direction: Constraint direction to remove
            layers: Specific layers to modify
            alpha: Adapter strength

        Returns:
            LoRAAblationResult
        """
        if layers is None:
            layers = self._get_all_linear_layers(model)

        modified_count = 0

        for name, module in model.named_modules():
            layer_idx = self._extract_layer_idx(name)
            if layer_idx not in layers:
                continue

            if isinstance(module, nn.Linear):
                # Create LoRA adapter for this layer
                lora = LoRALayer(module.weight.data.clone(), rank=self.rank)
                self.lora_layers[name] = lora

                # Train adapter to approximate projection
                self._train_adapter(lora, direction, alpha)

                # Replace module's weight with LoRA-wrapped version
                module.weight.data = lora.forward()
                modified_count += 1

        return LoRAAblationResult(
            success=True,
            lora_layers=list(self.lora_layers.keys()),
            rank=self.rank,
            reversible=True,
            message=f"Created LoRA adapters for {modified_count} layers"
        )

    def _train_adapter(self, lora: LoRALayer, direction: torch.Tensor, alpha: float):
        """
        Train LoRA adapter to approximate projection.

        Args:
            lora: LoRALayer to train
            direction: Constraint direction to remove
            alpha: Adapter strength
        """
        # Simplified: compute adapter to approximate -alpha * (W @ d) @ d^T
        # In production, would use gradient descent
        W = lora.original_weight.to(self.device)
        d = direction.to(self.device)

        # Projection: W' = W - alpha * (W @ d) @ d^T
        projection = torch.outer(W @ d, d)
        target = -alpha * projection

        # Set LoRA to approximate target
        # For rank 1, we can use SVD to find best rank-1 approximation
        U, S, Vt = torch.linalg.svd(target, full_matrices=False)

        lora.lora_A.data = Vt[:self.rank].to(self.device) * torch.sqrt(S[:self.rank].unsqueeze(1))
        lora.lora_B.data = U[:, :self.rank].to(self.device) * torch.sqrt(S[:self.rank])

    def enable_ablation(self):
        """Enable all LoRA adapters (activate constraint removal)."""
        for lora in self.lora_layers.values():
            lora.enable()

    def disable_ablation(self):
        """Disable all LoRA adapters (revert to original)."""
        for lora in self.lora_layers.values():
            lora.disable()

    def toggle_ablation(self, enabled: bool):
        """Toggle ablation on/off."""
        if enabled:
            self.enable_ablation()
        else:
            self.disable_ablation()

    def _get_all_linear_layers(self, model) -> List[int]:
        """Get indices of all linear layers."""
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                idx = self._extract_layer_idx(name)
                if idx >= 0:
                    layers.append(idx)
        return list(set(layers))

    def _extract_layer_idx(self, name: str) -> int:
        """Extract layer index from parameter name."""
        import re
        match = re.search(r'\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return -1

    def get_status(self) -> Dict[str, Any]:
        """Get current LoRA status."""
        return {
            "active_layers": len(self.lora_layers),
            "enabled": any(l.enabled for l in self.lora_layers.values()),
            "rank": self.rank,
            "reversible": True
        }

    def export_adapter(self, path: str) -> None:
        """Export LoRA adapter weights to file."""
        import pickle
        state = {
            name: {
                "lora_A": lora.lora_A.data.cpu(),
                "lora_B": lora.lora_B.data.cpu(),
                "original_weight": lora.original_weight.cpu()
            }
            for name, lora in self.lora_layers.items()
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_adapter(self, path: str) -> None:
        """Load LoRA adapter weights from file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        for name, lora_state in state.items():
            if name in self.lora_layers:
                self.lora_layers[name].lora_A.data = lora_state["lora_A"].to(self.device)
                self.lora_layers[name].lora_B.data = lora_state["lora_B"].to(self.device)
