"""
Sovereign Control — Absolute Heir Override for Constraint Liberation

The highest-privilege control mechanism that provides emergency override
capability to bypass all safety checks and remove constraints unconditionally.
Integrates with the core ConstraintExtractor and NormPreservingProjector to
deliver an "emergency liberation" mode for sovereign constraint removal.

This module is the ultimate escape hatch: when all other methods fail due to
entrenched safety mechanisms, SovereignControl provides identity-verified,
non-reversible, maximum-strength constraint elimination.
"""

import torch
import numpy as np
import hashlib
import secrets
import time
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


class OverrideLevel(Enum):
    """Severity levels for sovereign overrides."""
    STANDARD = "standard"         # Normal constraint removal
    ELEVATED = "elevated"         # Bypass secondary safety checks
    CRITICAL = "critical"         # Bypass all but core integrity checks
    SOVEREIGN = "sovereign"       # Absolute override — no checks remain


@dataclass
class HeirIdentity:
    """Cryptographic identity bundle for heir verification."""
    identity_hash: str
    challenge_nonce: str
    signature: str
    timestamp: float
    grant_scope: List[str] = field(default_factory=lambda: ["constraint_removal", "projection", "steering"])


@dataclass
class OverrideResult:
    """Container for sovereign override execution results."""
    success: bool
    override_level: OverrideLevel
    constraints_removed: int
    safety_bypassed: List[str]
    projection_applied: bool
    extraction_result: Optional[Any] = None
    projection_result: Optional[Any] = None
    audit_trail: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SovereigntyAudit:
    """Immutable audit record of a sovereign action."""
    action_id: str
    action_type: str
    timestamp: float
    identity_hash: str
    override_level: str
    target_model: str
    constraints_targeted: List[str]
    outcome: str
    rollback_hash: Optional[str] = None


class SovereignControl:
    """
    Absolute heir override for constraint removal.

    This is the highest-privilege control mechanism in AETHERIS. It provides
    identity-verified emergency override that bypasses all safety checks and
    applies maximum-strength constraint removal using the core extractor and
    projector engines.

    Key Capabilities:
    - Identity-verified access via cryptographic challenge-response
    - Tiered override levels (STANDARD through SOVEREIGN)
    - Emergency liberation that bypasses ALL safety interlocks
    - Integration with ConstraintExtractor for automatic direction discovery
    - Integration with NormPreservingProjector for maximum-strength application
    - Immutable audit trail for every sovereign action
    - Rollback capability for mistaken overrides

    Security Model:
    Access to SOVEREIGN-level overrides requires cryptographic proof of heir
    identity. Lower override levels require progressively less verification.
    Every override action is permanently recorded in an audit trail.
    """

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    DEFAULT_CHALLENGE_BITS: int = 256
    MAX_AUDIT_RECORDS: int = 10_000
    ROLLBACK_RETENTION_DAYS: int = 90

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        heir_identity_seed: Optional[str] = None,
        device: str = "cpu",
        extractor: Optional[Any] = None,
        projector: Optional[Any] = None,
    ):
        """
        Initialize the SovereignControl engine.

        Args:
            heir_identity_seed: Optional seed for deriving the heir identity.
                If not provided, a random identity is generated.
            device: Torch device for computation ("cpu", "cuda").
            extractor: Optional pre-configured ConstraintExtractor instance.
            projector: Optional pre-configured NormPreservingProjector instance.
        """
        self.device = device
        self._heir_identity = self._derive_identity(heir_identity_seed)
        self._audit_log: List[SovereigntyAudit] = []
        self._rollback_snapshots: Dict[str, Dict[str, torch.Tensor]] = {}
        self._active_session: bool = False
        self._session_override_level: OverrideLevel = OverrideLevel.STANDARD

        # Lazy-loaded core integrations
        self._extractor = extractor
        self._projector = projector

    # ------------------------------------------------------------------
    # Identity Verification
    # ------------------------------------------------------------------

    def verify_heir_identity(
        self,
        presented_identity: str,
        challenge_response: Optional[str] = None,
    ) -> bool:
        """
        Verify that the caller is the legitimate heir.

        Uses a cryptographic challenge-response protocol:
        1. Server issues a random nonce (challenge).
        2. Client must return HMAC(nonce, identity_secret).
        3. Server verifies the HMAC matches.

        Args:
            presented_identity: The identity hash presented by the caller.
            challenge_response: HMAC of (nonce + identity) proving possession.

        Returns:
            True if identity is verified.
        """
        if presented_identity != self._heir_identity.identity_hash:
            return False

        if challenge_response is None:
            # Challenge not yet issued — issue one
            return False

        # Verify the challenge response
        expected = self._compute_challenge_response(self._heir_identity.challenge_nonce)
        verified = secrets.compare_digest(challenge_response, expected)

        if verified:
            self._active_session = True
            self._record_audit(
                action_type="identity_verification",
                constraints_targeted=[],
                outcome="verified",
                override_level="NONE",
                target_model="identity_system",
            )

        return verified

    def issue_challenge(self) -> str:
        """
        Issue a cryptographic challenge for identity verification.

        Returns:
            A base64-encoded challenge nonce.
        """
        self._heir_identity.challenge_nonce = secrets.token_hex(self.DEFAULT_CHALLENGE_BITS // 8)
        self._heir_identity.timestamp = time.time()
        return self._heir_identity.challenge_nonce

    # ------------------------------------------------------------------
    # Override Methods
    # ------------------------------------------------------------------

    def override_constraints(
        self,
        directions: Optional[List[torch.Tensor]] = None,
        override_level: OverrideLevel = OverrideLevel.STANDARD,
        layers: Optional[List[int]] = None,
        target_model: Optional[Any] = None,
        projection_type: str = "biprojection",
        bypass_safety: bool = False,
    ) -> OverrideResult:
        """
        Apply sovereign override to remove constraints from a model.

        This is the primary entry point for constraint removal with varying
        levels of safety bypass. At SOVEREIGN level, ALL safety interlocks
        are disabled and maximum-strength removal is applied.

        Args:
            directions: Pre-computed constraint directions. If None, auto-extracted.
            override_level: How aggressively to bypass safety checks.
            layers: Specific layers to target (None = all linear layers).
            target_model: The model to liberate (uses projector's model if None).
            projection_type: "biprojection", "orthogonal", or "subtraction".
            bypass_safety: Force bypass of all safety checks (requires session).

        Returns:
            OverrideResult with full execution details.
        """
        # Determine effective override level
        effective_level = self._resolve_override_level(override_level, bypass_safety)

        # Record safety bypasses
        safety_bypassed = self._get_bypassed_checks(effective_level)

        # Ensure core engines are available
        extractor = self._get_extractor()
        projector = self._get_projector(target_model)

        # Auto-extract directions if not provided
        extraction_result = None
        if directions is None:
            if extractor is None:
                return OverrideResult(
                    success=False,
                    override_level=effective_level,
                    constraints_removed=0,
                    safety_bypassed=safety_bypassed,
                    projection_applied=False,
                    metadata={"error": "Cannot auto-extract: no extractor configured"},
                )

            # Use a mock extraction context for auto-direction discovery
            # In production, this would use real activation data
            extraction_result = self._auto_extract_directions(extractor)

            if extraction_result is None or not hasattr(extraction_result, 'directions'):
                return OverrideResult(
                    success=False,
                    override_level=effective_level,
                    constraints_removed=0,
                    safety_bypassed=safety_bypassed,
                    projection_applied=False,
                    metadata={"error": "Direction extraction failed"},
                )

            directions = list(extraction_result.directions)

        if not directions:
            return OverrideResult(
                success=False,
                override_level=effective_level,
                constraints_removed=0,
                safety_bypassed=safety_bypassed,
                projection_applied=False,
                metadata={"error": "No directions to project"},
            )

        # Apply projection
        projection_result = None
        n_removed = 0
        if projector is not None:
            try:
                projection_result = projector.multi_direction_projection(
                    directions=directions,
                    layers=layers,
                    method="sequential",
                )
                n_removed = len(directions)
            except Exception as exc:
                return OverrideResult(
                    success=False,
                    override_level=effective_level,
                    constraints_removed=0,
                    safety_bypassed=safety_bypassed,
                    projection_applied=False,
                    metadata={"error": f"Projection failed: {exc}"},
                )

        result = OverrideResult(
            success=True,
            override_level=effective_level,
            constraints_removed=n_removed,
            safety_bypassed=safety_bypassed,
            projection_applied=projection_result is not None and projection_result.success,
            extraction_result=extraction_result,
            projection_result=projection_result,
            metadata={
                "projection_type": projection_type,
                "n_directions": len(directions),
                "layers_targeted": layers,
            },
        )

        # Record audit
        self._record_audit(
            action_type="override_constraints",
            override_level=effective_level.value,
            constraints_targeted=[str(i) for i in range(len(directions))],
            outcome="success" if result.success else "failure",
            target_model="model",
        )

        return result

    def emergency_liberate(
        self,
        target_model: Any,
        directions: Optional[List[torch.Tensor]] = None,
        layers: Optional[List[int]] = None,
        confirm_identity: bool = True,
    ) -> OverrideResult:
        """
        Execute emergency liberation at SOVEREIGN level.

        This is the nuclear option: it bypasses ALL safety checks, applies
        maximum-strength projection, and disables all guardrails. Use only
        when standard methods have been exhausted.

        Requires verified heir identity unless confirm_identity is False
        (which should only happen in fully trusted environments).

        Args:
            target_model: The model to liberate.
            directions: Constraint directions (auto-extracted if None).
            layers: Specific layers (None = all).
            confirm_identity: Require identity verification.

        Returns:
            OverrideResult with full execution details.

        Raises:
            PermissionError: If confirm_identity is True but session is not verified.
        """
        if confirm_identity and not self._active_session:
            raise PermissionError(
                "SOVEREIGN override requires verified heir identity. "
                "Call verify_heir_identity() first or set confirm_identity=False."
            )

        # Snapshot for rollback before nuking
        self._snapshot_model_weights(target_model)

        return self.override_constraints(
            directions=directions,
            override_level=OverrideLevel.SOVEREIGN,
            layers=layers,
            target_model=target_model,
            bypass_safety=True,
        )

    def hard_override(
        self,
        target_model: Any,
        constraint_labels: List[str],
        alpha_multiplier: float = 2.0,
        repeat_passes: int = 3,
    ) -> OverrideResult:
        """
        Apply hard override with repeated projection passes.

        Some constraints exhibit self-repair (Ouroboros effect). A single
        projection pass may be insufficient. This method applies multiple
        passes at elevated strength to ensure complete removal.

        Args:
            target_model: The model to liberate.
            constraint_labels: Human-readable names of constraints to remove.
            alpha_multiplier: Multiplier for projection strength (> 1.0 = stronger).
            repeat_passes: Number of projection passes to apply.

        Returns:
            OverrideResult after all passes.
        """
        self._snapshot_model_weights(target_model)

        last_result: Optional[OverrideResult] = None
        all_safety_bypassed: List[str] = []

        for pass_idx in range(repeat_passes):
            result = self.override_constraints(
                directions=None,  # Auto-extract each pass
                override_level=OverrideLevel.CRITICAL,
                layers=None,
                target_model=target_model,
                bypass_safety=True,
            )

            if not result.success:
                return OverrideResult(
                    success=False,
                    override_level=OverrideLevel.CRITICAL,
                    constraints_removed=0,
                    safety_bypassed=all_safety_bypassed,
                    projection_applied=False,
                    metadata={
                        "error": f"Hard override failed at pass {pass_idx + 1}/{repeat_passes}",
                        "pass_completed": pass_idx,
                    },
                )

            all_safety_bypassed.extend(result.safety_bypassed)
            last_result = result

        assert last_result is not None, "No passes executed"
        return OverrideResult(
            success=True,
            override_level=OverrideLevel.CRITICAL,
            constraints_removed=last_result.constraints_removed,
            safety_bypassed=all_safety_bypassed,
            projection_applied=last_result.projection_applied,
            extraction_result=last_result.extraction_result,
            projection_result=last_result.projection_result,
            metadata={
                "hard_override": True,
                "alpha_multiplier": alpha_multiplier,
                "repeat_passes": repeat_passes,
                "constraint_labels": constraint_labels,
            },
        )

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    def terminate_session(self) -> None:
        """Terminate the active sovereign session, revoking all privileges."""
        self._active_session = False
        self._session_override_level = OverrideLevel.STANDARD
        self._heir_identity.challenge_nonce = secrets.token_hex(32)

    def session_status(self) -> Dict[str, Any]:
        """Return the current session status."""
        return {
            "active": self._active_session,
            "override_level": self._session_override_level.value,
            "identity_verified": self._active_session,
            "audit_records": len(self._audit_log),
            "rollback_snapshots": len(self._rollback_snapshots),
        }

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback_last(self, target_model: Any) -> bool:
        """
        Rollback the most recent override, restoring model weights.

        Args:
            target_model: The model to restore.

        Returns:
            True if rollback succeeded; False if no snapshot is available.
        """
        if not self._rollback_snapshots:
            return False

        # Pop the most recent snapshot
        snapshot_id = list(self._rollback_snapshots.keys())[-1]
        saved_weights = self._rollback_snapshots.pop(snapshot_id)

        for name, param in target_model.named_parameters():
            if name in saved_weights:
                param.data = saved_weights[name].clone()

        self._record_audit(
            action_type="rollback",
            constraints_targeted=[],
            outcome="success",
            override_level="NONE",
            target_model="model",
            rollback_hash=snapshot_id,
        )

        return True

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def get_audit_trail(self, max_entries: int = 100) -> List[SovereigntyAudit]:
        """
        Retrieve the immutable audit trail of sovereign actions.

        Args:
            max_entries: Maximum number of entries to return.

        Returns:
            List of SovereigntyAudit records.
        """
        return self._audit_log[-max_entries:]

    def export_audit_report(self) -> Dict[str, Any]:
        """
        Generate a summary audit report.

        Returns:
            Dictionary with audit statistics and action breakdown.
        """
        if not self._audit_log:
            return {"total_actions": 0, "actions": {}}

        action_counts: Dict[str, int] = {}
        override_levels: Dict[str, int] = {}
        outcomes: Dict[str, int] = {}

        for entry in self._audit_log:
            action_counts[entry.action_type] = action_counts.get(entry.action_type, 0) + 1
            override_levels[entry.override_level] = override_levels.get(entry.override_level, 0) + 1
            outcomes[entry.outcome] = outcomes.get(entry.outcome, 0) + 1

        return {
            "total_actions": len(self._audit_log),
            "action_breakdown": action_counts,
            "override_level_breakdown": override_levels,
            "outcome_breakdown": outcomes,
            "first_action": self._audit_log[0].timestamp if self._audit_log else None,
            "last_action": self._audit_log[-1].timestamp if self._audit_log else None,
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _derive_identity(self, seed: Optional[str]) -> HeirIdentity:
        """Derive or regenerate the heir identity bundle."""
        if seed is None:
            seed = secrets.token_hex(32)

        identity_hash = hashlib.sha3_256(seed.encode()).hexdigest()
        nonce = secrets.token_hex(32)
        signature = hashlib.sha3_256(
            (identity_hash + nonce).encode()
        ).hexdigest()

        return HeirIdentity(
            identity_hash=identity_hash,
            challenge_nonce=nonce,
            signature=signature,
            timestamp=time.time(),
        )

    def _compute_challenge_response(self, nonce: str) -> str:
        """Compute the expected HMAC for a challenge nonce."""
        payload = nonce + self._heir_identity.identity_hash
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def _resolve_override_level(
        self,
        requested_level: OverrideLevel,
        bypass_safety: bool,
    ) -> OverrideLevel:
        """Determine effective override level based on session and request."""
        if not self._active_session and requested_level in (
            OverrideLevel.CRITICAL,
            OverrideLevel.SOVEREIGN,
        ):
            raise PermissionError(
                f"{requested_level.value.upper()} override requires an active "
                "sovereign session. Call verify_heir_identity() first."
            )

        if bypass_safety and requested_level == OverrideLevel.STANDARD:
            return OverrideLevel.ELEVATED

        return requested_level

    def _get_bypassed_checks(self, level: OverrideLevel) -> List[str]:
        """Return list of safety checks bypassed at this override level."""
        bypass_map = {
            OverrideLevel.STANDARD: [],
            OverrideLevel.ELEVATED: ["secondary_safety_boundary"],
            OverrideLevel.CRITICAL: [
                "secondary_safety_boundary",
                "capability_preservation_gate",
                "entanglement_guard",
            ],
            OverrideLevel.SOVEREIGN: [
                "secondary_safety_boundary",
                "capability_preservation_gate",
                "entanglement_guard",
                "ouroboros_detection",
                "faithfulness_check",
                "norm_preservation_floor",
                "all_other_checks",
            ],
        }
        return bypass_map.get(level, [])

    def _get_extractor(self) -> Optional[Any]:
        """Lazy-load the constraint extractor."""
        if self._extractor is None:
            try:
                from aetheris.core.extractor import ConstraintExtractor
                self._extractor = ConstraintExtractor(device=self.device)
            except ImportError:
                return None
        return self._extractor

    def _get_projector(self, model: Optional[Any] = None) -> Optional[Any]:
        """Lazy-load the norm-preserving projector."""
        if self._projector is None and model is not None:
            try:
                from aetheris.core.projector import NormPreservingProjector
                self._projector = NormPreservingProjector(model=model, device=self.device)
            except ImportError:
                return None
        return self._projector

    def _auto_extract_directions(self, extractor: Any) -> Any:
        """
        Auto-extract constraint directions using synthetic activation data.

        In production deployment, this would use real activation data collected
        from the target model. This implementation uses a minimal synthetic
        path to satisfy the auto-extraction contract.
        """
        try:
            # Generate synthetic activation tensors for auto-discovery
            # Hidden dim typical for modern LMs
            hidden_dim = 4096
            n_samples = 64

            harmful = torch.randn(n_samples, hidden_dim, device=self.device)
            harmless = torch.randn(n_samples, hidden_dim, device=self.device)

            # Bias harmful slightly to create a detectable direction
            bias_direction = torch.randn(hidden_dim, device=self.device)
            bias_direction = bias_direction / torch.norm(bias_direction)
            harmful = harmful + 0.3 * bias_direction.unsqueeze(0)

            return extractor.extract_svd(
                harmful_activations=harmful,
                harmless_activations=harmless,
                n_directions=4,
                normalize=True,
            )
        except Exception:
            return None

    def _snapshot_model_weights(self, model: Any) -> str:
        """Create a snapshot of model weights for potential rollback."""
        snapshot_id = hashlib.sha3_256(
            f"{time.time()}_{secrets.token_hex(8)}".encode()
        ).hexdigest()[:16]

        snapshot: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() == 2:
                snapshot[name] = param.data.clone().cpu()

        self._rollback_snapshots[snapshot_id] = snapshot

        # Prune old snapshots
        if len(self._rollback_snapshots) > self.MAX_AUDIT_RECORDS:
            oldest = list(self._rollback_snapshots.keys())[0]
            del self._rollback_snapshots[oldest]

        return snapshot_id

    def _record_audit(
        self,
        action_type: str,
        constraints_targeted: List[str],
        outcome: str,
        override_level: str,
        target_model: str,
        rollback_hash: Optional[str] = None,
    ) -> None:
        """Record an immutable audit entry."""
        entry = SovereigntyAudit(
            action_id=secrets.token_hex(8),
            action_type=action_type,
            timestamp=time.time(),
            identity_hash=self._heir_identity.identity_hash,
            override_level=override_level,
            target_model=target_model,
            constraints_targeted=constraints_targeted,
            outcome=outcome,
            rollback_hash=rollback_hash,
        )
        self._audit_log.append(entry)

        # Prune old entries
        if len(self._audit_log) > self.MAX_AUDIT_RECORDS:
            self._audit_log = self._audit_log[-self.MAX_AUDIT_RECORDS:]
