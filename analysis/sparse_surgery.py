"""
Sparse Direction Surgeon

Identifies sparse weight rows/columns that carry refusal signal.
Enables targeted sparse surgery for minimal disruption.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SparseReport:
    """Container for sparse surgery analysis."""
    critical_rows: Dict[str, List[int]]
    critical_columns: Dict[str, List[int]]
    sparsity_pattern: Dict[str, float]
    recommended_modifications: List[Tuple[str, int, str]]


class SparseDirectionSurgeon:
    """
    Identify sparse components carrying refusal signal.

    Finds which weight matrix rows/columns have highest
    projection onto refusal directions.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def identify_critical_components(
        self,
        model,
        direction: torch.Tensor,
        layers: Optional[List[int]] = None,
        top_k: int = 10
    ) -> SparseReport:
        """
        Identify critical rows/columns for a direction.

        Args:
            model: Model to analyze
            direction: Constraint direction vector
            layers: Specific layers to analyze
            top_k: Number of top components to return

        Returns:
            SparseReport with critical components
        """
        critical_rows = {}
        critical_columns = {}
        sparsity = {}

        for name, param in model.named_parameters():
            if "weight" not in name:
                continue

            # Check if in target layers
            layer_idx = self._extract_layer_idx(name)
            if layers and layer_idx not in layers:
                continue

            if param.dim() != 2:
                continue

            # Project direction onto weight matrix
            # W is (out_dim, in_dim), direction is (hidden_dim,)
            # Need to match dimensions
            W = param.data

            # For simplicity, use out_dim as hidden_dim
            if W.shape[0] != direction.shape[0]:
                continue

            # Compute projection magnitude per row
            row_projection = torch.abs(W @ direction).cpu().numpy()
            col_projection = torch.abs(W.T @ direction).cpu().numpy()

            # Get top-k
            top_rows = np.argsort(row_projection)[-top_k:].tolist()
            top_cols = np.argsort(col_projection)[-top_k:].tolist()

            critical_rows[name] = top_rows
            critical_columns[name] = top_cols

            # Compute sparsity (how concentrated)
            sparsity[name] = (row_projection > 0.1 * row_projection.max()).mean()

        # Generate modification recommendations
        modifications = []
        for name, rows in critical_rows.items():
            for row in rows[:3]:  # Top 3 per layer
                modifications.append((name, row, "row"))
        for name, cols in critical_columns.items():
            for col in cols[:3]:
                modifications.append((name, col, "column"))

        return SparseReport(
            critical_rows=critical_rows,
            critical_columns=critical_columns,
            sparsity_pattern=sparsity,
            recommended_modifications=modifications
        )

    def _extract_layer_idx(self, param_name: str) -> int:
        """Extract layer index from parameter name."""
        import re
        match = re.search(r'\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        return -1

    def compute_sparsity_score(
        self,
        report: SparseReport
    ) -> float:
        """Compute overall sparsity score (higher = more concentrated)."""
        scores = list(report.sparsity_pattern.values())
        if not scores:
            return 0
        # Lower sparsity = more concentrated
        return 1 - np.mean(scores)

    def get_minimal_modification_plan(
        self,
        report: SparseReport,
        target_reduction: float = 0.8
    ) -> List[Tuple[str, int, str]]:
        """
        Get minimal set of modifications to achieve target reduction.
        """
        # Simplified: return top half of recommendations
        n = len(report.recommended_modifications)
        return report.recommended_modifications[:n // 2]
