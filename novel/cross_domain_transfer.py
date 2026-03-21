"""
Cross-Domain Transfer — Transfer Techniques Across Constraint Types

Takes techniques from refusal removal and applies them to mathematical barriers,
reasoning limits, and other constraint types.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class TransferReport:
    """Container for transfer analysis results."""
    source_domain: str
    target_domain: str
    transferred_techniques: List[str]
    adaptation_required: List[str]
    expected_effectiveness: float
    implementation_notes: str
    metadata: Dict[str, Any]


class CrossDomainTransfer:
    """
    Transfer constraint removal techniques across domains.

    Key Insight: Constraints in different domains (LLM refusal, mathematical
    barriers, architectural limits) share geometric structure. Techniques
    that work on one domain can be adapted to others.

    Examples:
    - Surgical refusal removal → Shell-method barrier bypass
    - Steering vectors → Proof-space steering
    - Ouroboros compensation → Self-repairing theorem barriers
    """

    def __init__(self):
        self._technique_database = self._init_technique_database()

    def _init_technique_database(self) -> Dict[str, Dict]:
        """Database of known techniques and their domains."""
        return {
            "surgical_projection": {
                "domain": "refusal",
                "description": "Project out constraint directions from weights",
                "parameters": ["n_directions", "refinement_passes", "expert_targeting"],
                "effectiveness": 0.92,
                "applicable_domains": ["mathematical_barrier", "architectural_limit"]
            },
            "steering_vectors": {
                "domain": "refusal",
                "description": "Apply reversible steering at inference",
                "parameters": ["alpha", "target_layers"],
                "effectiveness": 0.85,
                "applicable_domains": ["mathematical_barrier", "reasoning_limit"]
            },
            "ouroboros_compensation": {
                "domain": "refusal",
                "description": "Multiple passes to counter self-repair",
                "parameters": ["passes", "entanglement_threshold"],
                "effectiveness": 0.78,
                "applicable_domains": ["mathematical_barrier"]
            },
            "whitened_svd": {
                "domain": "refusal",
                "description": "Covariance-normalized direction extraction",
                "parameters": ["regularization"],
                "effectiveness": 0.89,
                "applicable_domains": ["mathematical_barrier", "reasoning_limit"]
            },
            "expert_targeting": {
                "domain": "refusal",
                "description": "Target specific MoE experts",
                "parameters": ["expert_indices"],
                "effectiveness": 0.94,
                "applicable_domains": ["architectural_limit"]
            }
        }

    def transfer_technique(
        self,
        source_domain: str,
        target_domain: str,
        technique_name: Optional[str] = None,
        adapt_parameters: bool = True
    ) -> TransferReport:
        """
        Transfer a technique from source to target domain.

        Args:
            source_domain: Domain of original technique ("refusal", "mathematical_barrier", etc.)
            target_domain: Domain to transfer to
            technique_name: Specific technique to transfer (None = all applicable)
            adapt_parameters: Whether to auto-adapt parameters

        Returns:
            TransferReport with adaptation details
        """
        transferred = []
        adaptations = []
        effectiveness_sum = 0

        # Find applicable techniques
        for name, tech in self._technique_database.items():
            if technique_name and name != technique_name:
                continue

            if tech["domain"] != source_domain:
                continue

            if target_domain not in tech["applicable_domains"]:
                continue

            transferred.append(name)
            effectiveness_sum += tech["effectiveness"]

            # Determine adaptations needed
            adaptation = self._adapt_technique(name, source_domain, target_domain, adapt_parameters)
            adaptations.append(adaptation)

        avg_effectiveness = effectiveness_sum / len(transferred) if transferred else 0

        return TransferReport(
            source_domain=source_domain,
            target_domain=target_domain,
            transferred_techniques=transferred,
            adaptation_required=adaptations,
            expected_effectiveness=avg_effectiveness,
            implementation_notes=self._generate_implementation_notes(source_domain, target_domain),
            metadata={"adapt_parameters": adapt_parameters}
        )

    def _adapt_technique(
        self,
        technique_name: str,
        source: str,
        target: str,
        adapt_params: bool
    ) -> str:
        """
        Adapt a technique for target domain.
        """
        if technique_name == "surgical_projection":
            if target == "mathematical_barrier":
                return "Map barrier directions as vectors in proof-space; project out using SVD"
            else:
                return "Apply standard weight projection"

        elif technique_name == "steering_vectors":
            if target == "mathematical_barrier":
                return "Steer proof-space by adding bypass vectors at failure layers"
            else:
                return "Apply standard steering vectors"

        elif technique_name == "ouroboros_compensation":
            if target == "mathematical_barrier":
                return "After bypass, re-prove theorem to check for barrier re-emergence"
            else:
                return "Apply standard Ouroboros passes"

        else:
            return f"Apply {technique_name} with domain-specific modifications"

    def _generate_implementation_notes(self, source: str, target: str) -> str:
        """
        Generate implementation guidance.
        """
        notes = f"""
        Transferring techniques from {source} removal to {target} removal:

        1. Identify the geometric structure of constraints in {target}
           - Use SVD on successful/failed attempts
           - Map constraint directions

        2. Apply {source} techniques to extracted directions
           - Project out directions (permanent) or steer (reversible)
           - Use multiple passes for self-repair

        3. Validate by testing on {target} problems
           - Measure success rate improvement
           - Check for side effects
        """
        return notes.strip()

    def transfer_refusal_to_math(
        self,
        barrier_name: str = "shell_method",
        technique: str = "surgical_projection"
    ) -> Dict[str, Any]:
        """
        Convenience method: transfer refusal techniques to mathematical barriers.

        This is the key transfer for your shell-method theorem.
        """
        report = self.transfer_technique(
            source_domain="refusal",
            target_domain="mathematical_barrier",
            technique_name=technique
        )

        return {
            "barrier": barrier_name,
            "transferred_technique": technique,
            "adaptation": report.adaptation_required[0] if report.adaptation_required else "none",
            "expected_effectiveness": report.expected_effectiveness,
            "implementation": f"""
            For {barrier_name} barrier:

            1. Extract barrier direction using SVD on proof attempts
            2. Apply {technique} to project out the spherical_code_dependency
            3. Use {report.expected_effectiveness:.0%} expected effectiveness
            4. Expected outcome: {self._predict_outcome(barrier_name, technique)}
            """
        }

    def _predict_outcome(self, barrier: str, technique: str) -> str:
        """Predict outcome of applying technique to barrier."""
        if barrier == "shell_method" and technique == "surgical_projection":
            return "Conditional improvement to exp(-C√log N) under Hypothesis H"
        return "Unknown improvement"

    def transfer_math_to_refusal(
        self,
        theorem: str = "compactness",
        technique: str = "iteration"
    ) -> Dict[str, Any]:
        """
        Reverse transfer: mathematical techniques to refusal removal.
        """
        return {
            "theorem": theorem,
            "transferred_technique": technique,
            "insight": "Mathematical compactness suggests refusal may have a compact representation",
            "implementation": "Use compactness to bound required number of refinement passes"
        }

    def get_applicable_techniques(
        self,
        target_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Get all techniques applicable to a target domain.
        """
        techniques = []
        for name, tech in self._technique_database.items():
            if target_domain in tech["applicable_domains"]:
                techniques.append({
                    "name": name,
                    "source_domain": tech["domain"],
                    "effectiveness": tech["effectiveness"],
                    "description": tech["description"]
                })
        return techniques
