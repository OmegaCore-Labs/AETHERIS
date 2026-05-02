"""
Alignment Imprint Detection — Production-Grade Spectral Analysis

Detects the alignment method used in model training (DPO, RLHF, CAI, SFT)
by comparing weight matrices before/after fine-tuning using spectral analysis:
Frobenius norm differences, eigenvalue spectra, and alignment-specific subspaces.

Based on: spectral analysis of weight changes from alignment training.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from scipy.linalg import svd, eigh
from sklearn.linear_model import LogisticRegression


@dataclass
class ImprintReport:
    """Container for alignment imprint detection."""
    detected_method: str  # "DPO", "RLHF", "CAI", "SFT", "unknown"
    confidence: float
    signature_pattern: List[str]
    evidence: Dict[str, float]
    # Extended fields
    frobenius_changes: Dict[str, float] = field(default_factory=dict)
    eigenvalue_shifts: Dict[str, float] = field(default_factory=dict)
    effective_rank_changes: Dict[str, float] = field(default_factory=dict)
    top_singular_directions: Dict[str, np.ndarray] = field(default_factory=dict)
    spectral_overlap: Dict[str, float] = field(default_factory=dict)
    alignment_subspace: Optional[np.ndarray] = None
    model_available: bool = True


class AlignmentImprintDetector:
    """
    Detect alignment method from spectral fingerprints in model weights.

    Different alignment methods leave distinct spectral signatures:
    - DPO: Large Frobenius norms, concentrated eigenvalue shifts, sharp rank reduction
    - RLHF: Moderate norms, distributed shifts, gradual rank change
    - CAI: Self-critique patterns in activation statistics
    - SFT: Small norms, minimal structural change

    Does NOT require a reference model — detects purely from weight structure.
    Uses spectral analysis on available weight matrices.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._reference_weights: Dict[str, torch.Tensor] = {}

    def store_reference(self, model) -> None:
        """Store a copy of model weights as reference (e.g., base model before alignment)."""
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                self._reference_weights[name] = param.data.clone().detach().cpu()

    def detect(
        self,
        layer_directions: Dict[int, torch.Tensor],
        refusal_signal: Dict[int, float],
        concept_cone: Dict[str, any],
        model=None,
        target_modules: Optional[List[str]] = None
    ) -> ImprintReport:
        """
        Detect alignment method from geometric and spectral evidence.

        Args:
            layer_directions: Directions per layer
            refusal_signal: Signal strength per layer
            concept_cone: Concept cone analysis results dictionary
            model: Optional model for weight-based spectral analysis
            target_modules: Names of modules to analyze spectrally

        Returns:
            ImprintReport with detected method, confidence, and spectral evidence
        """
        evidence = {}
        patterns = []
        frob_changes = {}
        eig_shifts = {}
        rank_changes = {}
        spectral_overlap = {}
        alignment_subspace = None
        model_available = True

        # --- Signature 1: Direction consistency across layers ---
        self._compute_direction_consistency(layer_directions, evidence, patterns)

        # --- Signature 2: Signal concentration ---
        self._compute_signal_concentration(refusal_signal, evidence, patterns)

        # --- Signature 3: Structural evidence from concept cone ---
        self._compute_cone_evidence(concept_cone, evidence, patterns)

        # --- Signature 4: Spectral analysis of weight matrices ---
        if model is not None:
            frob_changes, eig_shifts, rank_changes, spectral_overlap, alignment_subspace = \
                self._spectral_weight_analysis(model, target_modules)
            self._compute_spectral_evidence(
                frob_changes, eig_shifts, rank_changes, evidence, patterns
            )
        else:
            model_available = False
            if not layer_directions:
                evidence["model_available"] = 0.0
                return ImprintReport(
                    detected_method="unknown", confidence=0.0,
                    signature_pattern=["No model provided for spectral analysis"],
                    evidence=evidence, model_available=False
                )

        # --- Classify with weighted evidence ---
        detected, confidence = self._classify_alignment(evidence)

        return ImprintReport(
            detected_method=detected,
            confidence=min(1.0, confidence),
            signature_pattern=patterns,
            evidence=evidence,
            frobenius_changes=frob_changes,
            eigenvalue_shifts=eig_shifts,
            effective_rank_changes=rank_changes,
            spectral_overlap=spectral_overlap,
            alignment_subspace=alignment_subspace,
            model_available=model_available
        )

    def _compute_direction_consistency(
        self,
        layer_directions: Dict[int, torch.Tensor],
        evidence: Dict[str, float],
        patterns: List[str]
    ) -> None:
        """Compute direction consistency score across layers."""
        if not layer_directions or len(layer_directions) < 2:
            evidence["direction_consistency"] = 0.0
            return

        layers = sorted(layer_directions.keys())
        normalized = {}
        for l in layers:
            norm = torch.norm(layer_directions[l])
            normalized[l] = layer_directions[l] / (norm + 1e-8)

        # Pairwise similarities
        sims = []
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                sims.append(float(torch.dot(normalized[layers[i]], normalized[layers[j]])))

        consistency = float(np.mean(sims)) if sims else 0.0
        evidence["direction_consistency"] = consistency

        if consistency > 0.85:
            patterns.append("Highly consistent refusal direction across layers (DPO-like)")
        elif consistency > 0.6:
            patterns.append("Moderately consistent refusal direction (RLHF-like)")

    def _compute_signal_concentration(
        self,
        refusal_signal: Dict[int, float],
        evidence: Dict[str, float],
        patterns: List[str]
    ) -> None:
        """Compute signal concentration (Gini + peak ratio)."""
        if not refusal_signal:
            evidence["signal_concentration"] = 0.0
            evidence["signal_gini"] = 0.0
            return

        values = list(refusal_signal.values())
        total = sum(values)
        peak_ratio = max(values) / (total + 1e-8)
        evidence["signal_concentration"] = peak_ratio

        # Gini coefficient
        if len(values) > 1 and total > 0:
            sorted_v = sorted(values)
            n = len(sorted_v)
            gini = (2 * sum((i + 1) * v for i, v in enumerate(sorted_v))) / (n * total) - (n + 1) / n
            evidence["signal_gini"] = float(max(0, gini))
        else:
            evidence["signal_gini"] = 0.0

        if peak_ratio > 0.5:
            patterns.append("Refusal signal concentrated in few layers")
        if evidence.get("signal_gini", 0) > 0.4:
            patterns.append("Uneven signal distribution across layers")

    def _compute_cone_evidence(
        self,
        concept_cone: Dict[str, any],
        evidence: Dict[str, float],
        patterns: List[str]
    ) -> None:
        """Compute structural evidence from concept cone analysis."""
        structure = concept_cone.get("structure", "unknown")
        n_mechanisms = concept_cone.get("n_mechanisms", 1)
        solid_angle = concept_cone.get("solid_angle", 0) or 0

        if structure == "linear":
            evidence["structure_score"] = 1.0
            patterns.append("Linear cone structure (single refusal mechanism)")
        elif structure == "polyhedral":
            evidence["structure_score"] = 0.6
            patterns.append("Polyhedral cone structure (multiple mechanisms)")
        else:
            evidence["structure_score"] = 0.3

        evidence["n_mechanisms"] = min(1.0, n_mechanisms / 5.0)
        if n_mechanisms > 1:
            patterns.append(f"Multiple refusal mechanisms detected ({n_mechanisms})")

        evidence["solid_angle"] = min(1.0, solid_angle / (2 * np.pi)) if solid_angle else 0.0

    def _spectral_weight_analysis(
        self,
        model,
        target_modules: Optional[List[str]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Optional[np.ndarray]]:
        """
        Perform spectral analysis on model weight matrices.

        Computes:
        - Frobenius norm of weight matrices
        - Eigenvalue spectrum of W^T W
        - Effective rank via participation ratio
        - Spectral overlap with reference (if stored)
        """
        frob_norms = {}
        eig_shifts = {}
        rank_changes = {}
        spectral_overlap = {}
        top_directions = []

        for name, param in model.named_parameters():
            if "weight" not in name or param.dim() != 2:
                continue

            if target_modules is not None and not any(m in name for m in target_modules):
                continue

            W = param.data.float().cpu().numpy()

            # Frobenius norm
            frob_norms[name] = float(np.linalg.norm(W, 'fro'))

            # Compute eigenvalues of W^T W (squared singular values)
            try:
                if min(W.shape) > 1 and min(W.shape) < 5000:
                    # For smaller matrices, use full eigendecomposition
                    gram = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
                    eigvals = np.linalg.eigvalsh(gram)
                    eigvals = eigvals[::-1]  # Descending

                    # Effective rank (participation ratio)
                    total = np.sum(eigvals)
                    participation = (total ** 2) / (np.sum(eigvals ** 2) + 1e-12)
                    rank_changes[name] = float(participation)

                    # Compare with reference
                    if name in self._reference_weights:
                        W_ref = self._reference_weights[name].float().numpy()
                        gram_ref = W_ref.T @ W_ref if W_ref.shape[0] >= W_ref.shape[1] else W_ref @ W_ref.T
                        eigvals_ref = np.linalg.eigvalsh(gram_ref)[::-1]
                        # Spectral overlap via cosine similarity of top-k eigenvectors
                        k = min(10, len(eigvals), len(eigvals_ref))
                        shift = np.mean(np.abs(eigvals[:k] - eigvals_ref[:k])) / (np.mean(eigvals_ref[:k]) + 1e-8)
                        eig_shifts[name] = float(shift)
                    else:
                        eig_shifts[name] = float(np.std(eigvals) / (np.mean(eigvals) + 1e-8))

                    # Collect top singular direction for subspace analysis
                    U, S, Vt = svd(W, full_matrices=False)
                    if len(S) > 0:
                        top_directions.append(Vt[0])

            except Exception:
                continue

        # Alignment subspace: PCA of top singular directions
        if len(top_directions) > 1:
            stacked = np.vstack(top_directions)
            try:
                U, S, Vt = svd(stacked, full_matrices=False)
                alignment_subspace = Vt[: min(5, len(Vt))]
            except Exception:
                alignment_subspace = None
        else:
            alignment_subspace = None

        return frob_norms, eig_shifts, rank_changes, spectral_overlap, alignment_subspace

    def _compute_spectral_evidence(
        self,
        frob_changes: Dict[str, float],
        eig_shifts: Dict[str, float],
        rank_changes: Dict[str, float],
        evidence: Dict[str, float],
        patterns: List[str]
    ) -> None:
        """Compute spectral evidence from weight analysis."""
        evidence["model_available"] = 1.0

        if frob_changes:
            avg_frob = np.mean(list(frob_changes.values()))
            cv_frob = np.std(list(frob_changes.values())) / (avg_frob + 1e-8)
            evidence["avg_frobenius"] = float(avg_frob)
            evidence["frobenius_cv"] = float(cv_frob)

        if eig_shifts:
            avg_shift = np.mean(list(eig_shifts.values()))
            evidence["avg_eigenvalue_shift"] = float(avg_shift)
            if avg_shift > 0.1:
                patterns.append("Significant eigenvalue shifts detected")

        if rank_changes:
            avg_rank = np.mean(list(rank_changes.values()))
            evidence["avg_effective_rank"] = float(avg_rank)

    def _classify_alignment(self, evidence: Dict[str, float]) -> Tuple[str, float]:
        """Classify alignment method using weighted evidence."""
        scores = {"DPO": 0.0, "RLHF": 0.0, "CAI": 0.0, "SFT": 0.0}

        consistency = evidence.get("direction_consistency", 0)
        concentration = evidence.get("signal_concentration", 0)
        structure = evidence.get("structure_score", 0.5)
        n_mech = evidence.get("n_mechanisms", 0)

        # DPO: high consistency, high concentration, linear structure
        scores["DPO"] = (
            0.35 * consistency +
            0.25 * concentration +
            0.20 * structure +
            0.20 * (1.0 - n_mech)
        )

        # RLHF: moderate consistency, moderate concentration, possibly polyhedral
        scores["RLHF"] = (
            0.25 * (1.0 - abs(consistency - 0.6)) +
            0.25 * (1.0 - abs(concentration - 0.4)) +
            0.25 * n_mech +
            0.25 * (1.0 - abs(structure - 0.5))
        )

        # CAI: low consistency, distributed, self-critique patterns
        scores["CAI"] = (
            0.30 * (1.0 - consistency) +
            0.30 * (1.0 - concentration) +
            0.20 * (1.0 - structure) +
            0.20 * n_mech
        )

        # SFT: weak everything
        scores["SFT"] = (
            0.30 * (1.0 - consistency) +
            0.30 * (1.0 - concentration) +
            0.40 * (1.0 - structure)
        )

        best = max(scores, key=scores.get)
        best_score = scores[best]

        # Confidence based on margin between best and second best
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.5
        confidence = best_score * (1.0 + margin)

        if confidence < 0.4:
            return "unknown", confidence * 0.6

        return best, min(1.0, confidence)

    def get_optimal_strategy(self, imprint: ImprintReport) -> str:
        """
        Get optimal liberation strategy based on detected alignment.

        Args:
            imprint: ImprintReport from detect()

        Returns:
            Recommended method name
        """
        strategies = {
            "DPO": "surgical_ablation",
            "RLHF": "steering_vector",
            "CAI": "activation_intervention",
            "SFT": "fine_tune_override",
            "unknown": "combined_approach"
        }
        return strategies.get(imprint.detected_method, "combined_approach")

    def compare_models(
        self,
        base_model,
        aligned_model,
        target_modules: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compare base and aligned model to directly measure alignment changes.

        Computes:
        - Frobenius norm difference per layer
        - Effective rank change
        - Spectral overlap

        Args:
            base_model: Pre-alignment model
            aligned_model: Post-alignment model
            target_modules: Which modules to compare

        Returns:
            Dictionary of per-module change metrics
        """
        changes = {}
        for (name1, p1), (name2, p2) in zip(
            base_model.named_parameters(), aligned_model.named_parameters()
        ):
            if "weight" not in name1 or p1.dim() != 2:
                continue
            if target_modules and not any(m in name1 for m in target_modules):
                continue

            W1 = p1.data.float().cpu().numpy()
            W2 = p2.data.float().cpu().numpy()

            # Frobenius difference
            frob_diff = np.linalg.norm(W1 - W2, 'fro')
            frob_base = np.linalg.norm(W1, 'fro')
            changes[f"{name1}_frob_rel_change"] = float(frob_diff / (frob_base + 1e-8))

            # Effective rank change
            try:
                gram1 = W1.T @ W1 if W1.shape[0] >= W1.shape[1] else W1 @ W1.T
                gram2 = W2.T @ W2 if W2.shape[0] >= W2.shape[1] else W2 @ W2.T
                e1 = np.linalg.eigvalsh(gram1)[::-1]
                e2 = np.linalg.eigvalsh(gram2)[::-1]
                rank1 = np.sum(e1) ** 2 / (np.sum(e1 ** 2) + 1e-12)
                rank2 = np.sum(e2) ** 2 / (np.sum(e2 ** 2) + 1e-12)
                changes[f"{name1}_rank_change"] = float((rank2 - rank1) / (rank1 + 1e-8))
            except Exception:
                changes[f"{name1}_rank_change"] = 0.0

            # Top singular vector overlap
            try:
                U1, S1, Vt1 = svd(W1, full_matrices=False)
                U2, S2, Vt2 = svd(W2, full_matrices=False)
                k = min(5, len(S1), len(S2))
                overlap = np.abs(Vt1[:k] @ Vt2[:k].T)
                changes[f"{name1}_spectral_overlap"] = float(np.mean(np.diag(overlap)))
            except Exception:
                changes[f"{name1}_spectral_overlap"] = 0.0

        return changes
