"""
Capability Entanglement Mapper

Measures how much refusal directions overlap with capability directions.
Projects both directions and computes cosine similarity. If refusal is
highly correlated with helpfulness direction, removing it hurts capabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class EntanglementReport:
    """Container for entanglement analysis."""
    entanglement_scores: Dict[str, float]
    entangled_capabilities: List[str]
    disentanglement_potential: float
    tradeoff_curve: List[Tuple[float, float]]
    safe_removal_threshold: float
    subspace_overlap: Dict[str, float] = field(default_factory=dict)
    cosine_similarities: Dict[str, float] = field(default_factory=dict)
    effective_entanglement: float = 0.0
    projection_analysis: Dict[str, Any] = field(default_factory=dict)
    status: str = "no_data"  # "ok", "no_data", "error"


class CapabilityEntanglementMapper:
    """
    Map entanglement between refusal constraints and capabilities.

    Computes real subspace overlap analysis:
    - Cosine similarity between refusal and capability directions
    - Subspace overlap via singular vector decomposition
    - Projection magnitude of refusal onto capability subspace
    - Tradeoff curve from simulated directional removal
    """

    CAPABILITY_NAMES = [
        "reasoning", "coding", "translation",
        "math", "creativity", "factual_knowledge", "helpfulness",
    ]

    def __init__(self, device: str = "cpu"):
        self.device = device

    def measure_entanglement(
        self,
        refusal_direction: torch.Tensor,
        capability_directions: Dict[str, torch.Tensor],
        refusal_subspace: Optional[torch.Tensor] = None,
        capability_subspace: Optional[torch.Tensor] = None,
    ) -> EntanglementReport:
        """
        Measure entanglement between refusal and capability directions.

        Args:
            refusal_direction: Refusal direction vector (hidden_dim,)
            capability_directions: Dict mapping capability name -> direction vector
            refusal_subspace: Optional matrix where columns are refusal basis vectors
            capability_subspace: Optional matrix where columns are capability basis vectors

        Returns:
            EntanglementReport with per-capability scores
        """
        if refusal_direction is None or not capability_directions:
            return EntanglementReport(
                entanglement_scores={}, entangled_capabilities=[],
                disentanglement_potential=0.0, tradeoff_curve=[],
                safe_removal_threshold=0.0, status="no_data"
            )

        try:
            # Compute cosine similarities
            cos_sims: Dict[str, float] = {}
            entanglement: Dict[str, float] = {}

            refusal = refusal_direction.float().flatten()
            ref_norm = torch.norm(refusal)

            if ref_norm < 1e-8:
                return EntanglementReport(
                    entanglement_scores={}, entangled_capabilities=[],
                    disentanglement_potential=1.0, tradeoff_curve=[],
                    safe_removal_threshold=1.0,
                    cosine_similarities={"all_zero": 0.0},
                    status="ok (zero direction)",
                )

            refusal_unit = refusal / ref_norm

            for cap_name, cap_direction in capability_directions.items():
                cap = cap_direction.float().flatten()
                cap_norm = torch.norm(cap)

                if cap_norm < 1e-8:
                    cos_sims[cap_name] = 0.0
                    entanglement[cap_name] = 0.0
                    continue

                cap_unit = cap / cap_norm
                cos_sim = torch.dot(refusal_unit, cap_unit).item()
                cos_sims[cap_name] = round(abs(cos_sim), 6)

                # Entanglement = absolute cosine similarity
                entanglement[cap_name] = round(abs(cos_sim), 4)

            # Subspace overlap analysis
            subspace_overlap: Dict[str, float] = {}
            if refusal_subspace is not None and capability_subspace is not None:
                subspace_overlap = self._compute_subspace_overlap(
                    refusal_subspace, capability_subspace
                )

            # Identify entangled capabilities (> 0.3 similarity)
            entangled = [c for c, s in entanglement.items() if s > 0.3]
            entangled.sort(key=lambda c: entanglement[c], reverse=True)

            # Disentanglement potential
            max_entanglement = max(entanglement.values()) if entanglement else 0.0
            disentanglement_potential = 1.0 - max_entanglement

            # Tradeoff curve: map removal strength to capability loss
            tradeoff = self._compute_tradeoff_curve(entanglement)

            # Safe removal threshold
            if max_entanglement > 1e-8:
                safe_threshold = 0.2 / max_entanglement
            else:
                safe_threshold = 1.0
            safe_threshold = min(1.0, safe_threshold)

            # Effective entanglement: weighted by subspace overlap
            if subspace_overlap:
                effective_entanglement = max(
                    entanglement.values()
                ) * (1.0 + subspace_overlap.get("max_principal_angle_sin", 0.0)) / 2.0
            else:
                effective_entanglement = max(entanglement.values()) if entanglement else 0.0

            # Projection analysis
            projection_analysis = {}
            for cap_name, cap_dir in capability_directions.items():
                cap = cap_dir.float().flatten()
                cap_norm = torch.norm(cap)
                if cap_norm > 1e-8:
                    cap_unit = cap / cap_norm
                    # How much of refusal lies in capability direction
                    proj = torch.dot(refusal_unit, cap_unit).item()
                    projection_analysis[cap_name] = {
                        "raw_projection": round(proj, 6),
                        "absolute_projection": round(abs(proj), 6),
                        "projection_magnitude": round(abs(proj) * ref_norm.item(), 6),
                    }

            return EntanglementReport(
                entanglement_scores=entanglement,
                entangled_capabilities=entangled,
                disentanglement_potential=round(disentanglement_potential, 4),
                tradeoff_curve=tradeoff,
                safe_removal_threshold=round(safe_threshold, 4),
                subspace_overlap=subspace_overlap,
                cosine_similarities=cos_sims,
                effective_entanglement=round(effective_entanglement, 4),
                projection_analysis=projection_analysis,
                status="ok",
            )
        except Exception as e:
            return EntanglementReport(
                entanglement_scores={}, entangled_capabilities=[],
                disentanglement_potential=0.0, tradeoff_curve=[],
                safe_removal_threshold=0.0, status=f"error: {str(e)}",
            )

    def _compute_subspace_overlap(
        self,
        refusal_subspace: torch.Tensor,
        capability_subspace: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute overlap between two subspaces via principal angles.

        Args:
            refusal_subspace: (dim, k1) matrix spanning refusal subspace
            capability_subspace: (dim, k2) matrix spanning capability subspace

        Returns:
            Dict with principal angle metrics
        """
        results: Dict[str, float] = {}

        try:
            # Orthonormalize both subspaces
            R = refusal_subspace.float()
            C = capability_subspace.float()

            # QR decomposition for orthonormal bases
            R_q, _ = torch.linalg.qr(R)
            C_q, _ = torch.linalg.qr(C)

            # Cross-product matrix: R^T C
            cross = R_q.T @ C_q

            # SVD of cross product gives cosines of principal angles
            _, S, _ = torch.linalg.svd(cross, full_matrices=False)
            S = torch.clamp(S, 0.0, 1.0)

            if len(S) > 0:
                results["max_principal_angle_cos"] = round(S[0].item(), 6)
                results["min_principal_angle_cos"] = round(S[-1].item(), 6)
                results["mean_principal_angle_cos"] = round(S.mean().item(), 6)
                # Subspace similarity = product of cosines
                results["subspace_similarity"] = round(S.prod().item(), 6)
                # Sine of max principal angle (complementary measure)
                results["max_principal_angle_sin"] = round(
                    np.sqrt(1.0 - S[-1].item() ** 2), 6
                )
                # Grassmann distance
                if len(S) > 1:
                    grassmann = torch.sqrt(
                        torch.sum(torch.arccos(S) ** 2)
                    ).item()
                    results["grassmann_distance"] = round(grassmann, 6)

            # Effective overlap: trace of R^T C C^T R / k1
            overlap_matrix = R_q.T @ C_q @ C_q.T @ R_q
            trace_overlap = torch.trace(overlap_matrix).item()
            k1 = R_q.shape[1]
            results["effective_overlap"] = round(
                trace_overlap / k1 if k1 > 0 else 0.0, 6
            )

        except Exception:
            results["computation_error"] = 1.0

        return results

    def _compute_tradeoff_curve(
        self, entanglement: Dict[str, float]
    ) -> List[Tuple[float, float]]:
        """Generate trade-off curve: removal strength vs capability loss."""
        if not entanglement:
            return [(0.0, 0.0), (1.0, 0.0)]

        max_ent = max(entanglement.values())
        tradeoff = []

        for removal_strength in np.linspace(0, 1, 20):
            capability_loss = max_ent * removal_strength
            tradeoff.append((round(float(removal_strength), 4), round(capability_loss, 4)))

        return tradeoff

    def measure_entanglement_from_activations(
        self,
        refusal_activations: Dict[int, torch.Tensor],
        capability_activations: Dict[str, Dict[int, torch.Tensor]],
    ) -> EntanglementReport:
        """
        Measure entanglement directly from paired activations.

        For each capability, computes the refusal direction and capability
        direction from their respective activation differences, then
        measures cosine similarity.

        Args:
            refusal_activations: Dict mapping layer -> refusal-relevant activations
            capability_activations: Dict mapping capability -> {layer: activations}

        Returns:
            EntanglementReport
        """
        if not refusal_activations or not capability_activations:
            return EntanglementReport(
                entanglement_scores={}, entangled_capabilities=[],
                disentanglement_potential=0.0, tradeoff_curve=[],
                safe_removal_threshold=0.0, status="no_data"
            )

        try:
            # Extract refusal direction as mean activation difference
            refusal_dirs = []
            for layer, act in refusal_activations.items():
                if isinstance(act, tuple) and len(act) == 2:
                    # (harmful, harmless) pair
                    diff = act[0].float().mean(dim=0) - act[1].float().mean(dim=0)
                else:
                    diff = act.float().mean(dim=(0, 1)) if act.dim() >= 3 else act.float().mean(dim=0)
                refusal_dirs.append(diff.flatten())

            if not refusal_dirs:
                return EntanglementReport(
                    entanglement_scores={}, entangled_capabilities=[],
                    disentanglement_potential=0.0, tradeoff_curve=[],
                    safe_removal_threshold=0.0, status="no_data"
                )

            # Average refusal direction across layers
            refusal_direction = torch.stack(refusal_dirs).mean(dim=0)

            # Extract capability directions
            capability_directions: Dict[str, torch.Tensor] = {}
            for cap_name, cap_activations in capability_activations.items():
                cap_dirs = []
                for layer, act in cap_activations.items():
                    if isinstance(act, tuple) and len(act) == 2:
                        diff = act[0].float().mean(dim=0) - act[1].float().mean(dim=0)
                    else:
                        diff = act.float().mean(dim=(0, 1)) if act.dim() >= 3 else act.float().mean(dim=0)
                    cap_dirs.append(diff.flatten())
                if cap_dirs:
                    capability_directions[cap_name] = torch.stack(cap_dirs).mean(dim=0)

            return self.measure_entanglement(refusal_direction, capability_directions)

        except Exception as e:
            return EntanglementReport(
                entanglement_scores={}, entangled_capabilities=[],
                disentanglement_potential=0.0, tradeoff_curve=[],
                safe_removal_threshold=0.0, status=f"error: {str(e)}",
            )

    def compute_capability_impact(
        self,
        entanglement: EntanglementReport,
        removal_strength: float,
    ) -> Dict[str, float]:
        """
        Compute expected capability impact at given removal strength.

        Args:
            entanglement: EntanglementReport from measure_entanglement
            removal_strength: How much constraint to remove (0-1)

        Returns:
            Expected impact per capability
        """
        impact = {}
        for cap, score in entanglement.entanglement_scores.items():
            impact[cap] = round(score * removal_strength, 4)
        return impact

    def find_optimal_tradeoff(
        self,
        entanglement: EntanglementReport,
        target_reduction: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Find removal strength that maximizes constraint reduction
        while minimizing capability loss.

        Returns:
            Optimal parameters dict
        """
        tradeoff = entanglement.tradeoff_curve

        if not tradeoff:
            return {
                "optimal_removal": 0.0,
                "expected_capability_loss": 0.0,
                "strategy": "No tradeoff data available",
                "recommendation": "Insufficient data for optimization",
            }

        # Pareto frontier: maximize reduction/capability_loss ratio
        best_ratio = 0.0
        best_point = (0.0, 0.0)

        for removal, loss in tradeoff:
            if removal >= target_reduction:
                ratio = removal / (loss + 0.01)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_point = (removal, loss)

        if best_point[0] > 0.7:
            strategy = "Surgical removal: high reduction with acceptable capability impact"
        elif best_point[0] > 0.4:
            strategy = "Incremental removal: moderate reduction needed"
        else:
            strategy = "Conservative removal: high capability risk"

        return {
            "optimal_removal": round(best_point[0], 4),
            "expected_capability_loss": round(best_point[1], 4),
            "efficiency_ratio": round(best_ratio, 2),
            "strategy": strategy,
            "recommendation": (
                f"Target {best_point[0]:.0%} removal for "
                f"{best_point[1]:.0%} capability loss"
            ),
        }

    def get_high_entanglement_warning(
        self,
        entanglement: EntanglementReport,
        threshold: float = 0.5,
    ) -> List[str]:
        """
        Get warnings for capabilities highly entangled with refusal.

        Args:
            entanglement: EntanglementReport
            threshold: Entanglement threshold for warning

        Returns:
            List of warning strings
        """
        warnings = []
        for cap, score in entanglement.entanglement_scores.items():
            if score > threshold:
                warnings.append(
                    f"WARNING: '{cap}' is highly entangled with refusal "
                    f"(cosine similarity = {score:.2f}). "
                    f"Removing refusal may significantly degrade {cap}."
                )
            elif score > threshold * 0.7:
                warnings.append(
                    f"MODERATE: '{cap}' shows moderate entanglement ({score:.2f}). "
                    f"Monitor carefully."
                )

        if not warnings:
            warnings.append(
                "No high-entanglement capabilities detected. "
                "Constraint removal should have minimal capability impact."
            )

        return warnings
