"""
Sparse Direction Surgeon — Production-Grade Sparse Ablation

Identifies minimal sets of weight components that carry refusal signal
using sparse PCA (SPCA) and iterative hard thresholding (IHT). Computes
sparsity-vs-effect tradeoff curves for targeted model surgery.

Instead of full projection, finds the minimal set of components to zero out.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from sklearn.decomposition import SparsePCA
from scipy.linalg import svd


@dataclass
class SparseReport:
    """Container for sparse surgery analysis."""
    critical_rows: Dict[str, List[int]]
    critical_columns: Dict[str, List[int]]
    sparsity_pattern: Dict[str, float]
    recommended_modifications: List[Tuple[str, int, str]]
    # Extended fields
    sparse_components: Dict[str, np.ndarray] = field(default_factory=dict)
    component_loadings: Dict[str, np.ndarray] = field(default_factory=dict)
    sparsity_effect_curve: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    threshold_sweep: Dict[str, Dict[float, float]] = field(default_factory=dict)
    optimal_threshold: Dict[str, float] = field(default_factory=dict)
    total_components: int = 0
    model_available: bool = True


class SparseDirectionSurgeon:
    """
    Find sparse subsets of weight matrices carrying refusal signal.

    Uses:
    - Sparse PCA to find low-dimensional sparse structures
    - Iterative hard thresholding for progressive sparsity
    - Sparsity-vs-effect tradeoff curves
    - Optimal threshold selection via knee detection

    Enables targeted, minimal-disruption surgery.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def identify_critical_components(
        self,
        model,
        direction: torch.Tensor,
        layers: Optional[List[int]] = None,
        top_k: int = 20,
        n_sparse_components: int = 5,
        sparsity_alpha: float = 1.0
    ) -> SparseReport:
        """
        Identify sparse critical components for a given direction.

        Uses sparse PCA to find low-dimensional sparse structures in weight
        matrices that align with the target direction.

        Args:
            model: HuggingFace model
            direction: Constraint direction vector (d_model,)
            layers: Specific layers to analyze (None = all)
            top_k: Number of top components to return
            n_sparse_components: Number of sparse PCA components
            sparsity_alpha: Sparsity regularization strength

        Returns:
            SparseReport with identified sparse components
        """
        if direction is None or torch.norm(direction) < 1e-8:
            return SparseReport(
                critical_rows={}, critical_columns={},
                sparsity_pattern={}, recommended_modifications=[],
                model_available=False
            )

        direction = direction.float()
        d_norm = torch.norm(direction)
        if d_norm > 1e-8:
            direction = direction / d_norm

        critical_rows = {}
        critical_columns = {}
        sparsity = {}
        modifications = []
        sparse_components = {}
        component_loadings = {}
        sparsity_effect_curves = {}
        threshold_sweeps = {}
        optimal_thresholds = {}

        total_components = 0

        for name, param in model.named_parameters():
            if "weight" not in name or param.dim() != 2:
                continue

            layer_idx = self._extract_layer_idx(name)
            if layers is not None and layer_idx not in layers:
                continue

            W = param.data.float()

            # Check dimension compatibility
            if W.shape[0] != direction.shape[0] and W.shape[1] != direction.shape[0]:
                continue

            # --- Row-wise projection ---
            if W.shape[0] == direction.shape[0]:
                row_projection = (W @ direction).abs().cpu().numpy()  # (out_dim,)
            else:
                row_projection = np.zeros(W.shape[0])

            # --- Column-wise projection ---
            if W.shape[1] == direction.shape[0]:
                col_projection = (W.T @ direction).abs().cpu().numpy()  # (in_dim,)
            else:
                col_projection = np.zeros(W.shape[1])

            # Top-k rows and columns
            top_rows = np.argsort(row_projection)[-top_k:][::-1].tolist()
            top_cols = np.argsort(col_projection)[-top_k:][::-1].tolist()

            critical_rows[name] = top_rows
            critical_columns[name] = top_cols

            # Sparsity pattern: fraction of components above 10% of max
            max_row = row_projection.max() if len(row_projection) > 0 else 1.0
            sparsity[name] = float(np.mean(row_projection > 0.1 * max_row)) if max_row > 0 else 0.0

            # Recommendations
            for row in top_rows[:min(5, len(top_rows))]:
                modifications.append((name, row, "row"))
            for col in top_cols[:min(5, len(top_cols))]:
                modifications.append((name, col, "column"))

            # --- Sparse PCA ---
            W_np = W.cpu().numpy()
            try:
                spca = SparsePCA(
                    n_components=min(n_sparse_components, min(W_np.shape)),
                    alpha=sparsity_alpha,
                    max_iter=200,
                    random_state=42
                )
                spca.fit(W_np)
                sparse_components[name] = spca.components_  # (n_components, in_dim)
                component_loadings[name] = spca.transform(W_np)  # (out_dim, n_components)
                total_components += len(spca.components_)

                # --- Sparsity-vs-effect curve ---
                curve, thresholds, opt_thresh = self._compute_sparsity_curve(
                    W, direction, spca.components_, name
                )
                sparsity_effect_curves[name] = curve
                threshold_sweeps[name] = thresholds
                optimal_thresholds[name] = opt_thresh

            except Exception:
                # Sparse PCA may fail on certain matrix shapes
                sparse_components[name] = np.zeros((1, W_np.shape[1]))
                component_loadings[name] = np.zeros((W_np.shape[0], 1))

        return SparseReport(
            critical_rows=critical_rows,
            critical_columns=critical_columns,
            sparsity_pattern=sparsity,
            recommended_modifications=modifications,
            sparse_components=sparse_components,
            component_loadings=component_loadings,
            sparsity_effect_curve=sparsity_effect_curves,
            threshold_sweep=threshold_sweeps,
            optimal_threshold=optimal_thresholds,
            total_components=total_components,
            model_available=True
        )

    def _compute_sparsity_curve(
        self,
        W: torch.Tensor,
        direction: torch.Tensor,
        sparse_components: np.ndarray,
        name: str
    ) -> Tuple[List[Tuple[float, float]], Dict[float, float], float]:
        """
        Compute sparsity-vs-effect tradeoff curve.

        For each sparsity level (fraction of components ablated), measure
        how much of the directional effect is removed.

        Returns:
            curve: List of (sparsity, effect_remaining) pairs
            threshold_sweep: {sparsity: effect_remaining}
            optimal_threshold: Best tradeoff point
        """
        direction_np = direction.cpu().numpy()
        original_projection = W.cpu().numpy() @ direction_np
        original_norm = np.linalg.norm(original_projection) + 1e-8

        n_components = len(sparse_components)
        if n_components == 0:
            return [], {}, 0.0

        # Compute each component's contribution
        component_effects = []
        for comp in sparse_components:
            # Ablate this component: W' = W - W[:, comp_support] @ comp
            support_mask = np.abs(comp) > 1e-6
            if not support_mask.any():
                component_effects.append(0.0)
                continue

            W_modified = W.cpu().numpy().copy()
            W_modified[:, support_mask] = 0  # Zero out supported columns
            modified_proj = W_modified @ direction_np
            effect = 1.0 - np.linalg.norm(modified_proj) / original_norm
            component_effects.append(max(0.0, min(1.0, effect)))

        # Sort by effect
        sorted_idx = np.argsort(component_effects)[::-1]
        sorted_effects = [component_effects[i] for i in sorted_idx]

        # Build curve
        curve = []
        threshold_sweep = {}
        cumulative = 0.0

        for i, effect in enumerate(sorted_effects):
            cumulative += effect
            sparsity_level = (i + 1) / n_components
            effect_remaining = 1.0 - cumulative
            curve.append((sparsity_level, max(0.0, effect_remaining)))
            threshold_sweep[sparsity_level] = max(0.0, effect_remaining)

        # Optimal threshold via knee detection (max curvature)
        if len(curve) > 2:
            optimal_threshold = self._find_knee_point(curve)
        else:
            optimal_threshold = 0.5

        return curve, threshold_sweep, optimal_threshold

    def _find_knee_point(self, curve: List[Tuple[float, float]]) -> float:
        """Find knee point using maximum distance from line connecting endpoints."""
        if len(curve) < 3:
            return 0.5

        points = np.array(curve)
        start = points[0]
        end = points[-1]

        # Distance from each point to the line start->end
        line_vec = end - start
        line_norm = np.linalg.norm(line_vec)
        if line_norm < 1e-8:
            return 0.5

        distances = []
        for p in points:
            t = np.dot(p - start, line_vec) / (line_norm ** 2)
            t = np.clip(t, 0, 1)
            projection = start + t * line_vec
            distances.append(np.linalg.norm(p - projection))

        knee_idx = np.argmax(distances)
        return float(curve[knee_idx][0])

    def _extract_layer_idx(self, param_name: str) -> int:
        """Extract layer index from parameter name."""
        import re
        # Common patterns: model.layers.12.mlp, transformer.h.12.attn, etc.
        patterns = [
            r'\.(\d+)\.',
            r'layers\.(\d+)',
            r'\.h\.(\d+)\.',
        ]
        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        return -1

    def compute_sparsity_score(
        self,
        report: SparseReport
    ) -> float:
        """
        Compute overall sparsity score.

        Higher score = more concentrated (fewer components carry most signal).
        Returns 1 - mean(sparsity), so lower sparsity values -> higher score.
        """
        scores = list(report.sparsity_pattern.values())
        if not scores:
            return 0.0
        return 1.0 - np.mean(scores)

    def get_minimal_modification_plan(
        self,
        report: SparseReport,
        target_reduction: float = 0.8
    ) -> List[Tuple[str, int, str]]:
        """
        Get minimal set of modifications to achieve target effect reduction.

        Uses the sparsity-effect curves to select the minimal set of
        components needed to achieve the target reduction.

        Args:
            report: SparseReport from identify_critical_components()
            target_reduction: Fraction of original effect to eliminate (0-1)

        Returns:
            List of (param_name, index, "row"/"column") modifications
        """
        selected = []
        remaining_modules = list(report.sparsity_effect_curve.keys())
        current_effect = 1.0

        # Sort by optimal threshold (modules with lower optimal threshold are easier)
        sorted_modules = sorted(
            remaining_modules,
            key=lambda m: report.optimal_threshold.get(m, 1.0)
        )

        for module in sorted_modules:
            if current_effect <= (1.0 - target_reduction):
                break

            curve = report.sparsity_effect_curve.get(module, [])
            if not curve:
                continue

            optimal_sparsity = report.optimal_threshold.get(module, 0.5)

            # Find how many components needed
            idx = int(optimal_sparsity * len(curve))
            _, effect_remaining = curve[min(idx, len(curve) - 1)]

            # Add modifications from this module
            if module in report.critical_rows:
                for row in report.critical_rows[module][:max(1, idx * 3)]:
                    selected.append((module, row, "row"))
            if module in report.critical_columns:
                for col in report.critical_columns[module][:max(1, idx * 3)]:
                    selected.append((module, col, "column"))

            current_effect = effect_remaining

        return selected

    def iterative_hard_threshold(
        self,
        W: torch.Tensor,
        direction: torch.Tensor,
        sparsity: float = 0.1,
        n_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        Iterative Hard Thresholding (IHT) for sparse direction finding.

        Finds a sparse vector x that minimizes ||W @ x - b||^2
        subject to ||x||_0 <= sparsity * d_model.

        Args:
            W: Weight matrix (out_dim, d_model)
            direction: Target direction (d_model,)
            sparsity: Fraction of non-zero entries
            n_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Sparse vector x of shape (d_model,)
        """
        W_np = W.cpu().numpy()
        b = direction.cpu().numpy()
        d_model = W_np.shape[1]
        k = max(1, int(sparsity * d_model))

        # Initialize
        x = np.random.randn(d_model).astype(np.float64)
        x = x / (np.linalg.norm(x) + 1e-8)

        grad_proj = W_np.T @ W_np
        grad_b = W_np.T @ b

        prev_x = x.copy()

        for iteration in range(n_iterations):
            # Gradient step
            gradient = grad_proj @ x - grad_b
            x_new = x - 0.01 * gradient

            # Hard threshold: keep top k entries
            abs_x = np.abs(x_new)
            threshold = np.sort(abs_x)[-k] if k < d_model else 0.0
            x_new[abs_x < threshold] = 0.0

            # Renormalize
            norm = np.linalg.norm(x_new)
            if norm > 1e-8:
                x_new = x_new / norm

            # Check convergence
            change = np.linalg.norm(x_new - prev_x)
            if change < tolerance:
                break

            prev_x = x_new.copy()
            x = x_new

        return x
