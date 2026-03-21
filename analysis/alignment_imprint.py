"""
Alignment Imprint Detection

Detects the alignment method used in model training (DPO, RLHF, CAI, SFT)
from geometric patterns in constraint directions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ImprintReport:
    """Container for alignment imprint detection."""
    detected_method: str  # "DPO", "RLHF", "CAI", "SFT", "unknown"
    confidence: float
    signature_pattern: List[str]
    evidence: Dict[str, float]


class AlignmentImprintDetector:
    """
    Detect alignment method from geometric patterns.

    Different alignment methods leave distinct geometric signatures:
    - DPO: Sharp, well-defined refusal direction
    - RLHF: Smooth, multi-directional refusal
    - CAI: Self-critique patterns in activations
    - SFT: Weak, inconsistent refusal
    """

    def __init__(self):
        pass

    def detect(
        self,
        layer_directions: Dict[int, torch.Tensor],
        refusal_signal: Dict[int, float],
        concept_cone: Dict[str, any]
    ) -> ImprintReport:
        """
        Detect alignment method from geometric data.

        Args:
            layer_directions: Directions per layer
            refusal_signal: Signal strength per layer
            concept_cone: Concept cone analysis results

        Returns:
            ImprintReport with detected method and confidence
        """
        evidence = {}

        # Signature 1: Direction sharpness
        if layer_directions:
            # Check if directions are consistent across layers
            layers = list(layer_directions.keys())
            if len(layers) > 1:
                first_dir = layer_directions[layers[0]]
                last_dir = layer_directions[layers[-1]]
                consistency = torch.dot(first_dir, last_dir).item()
                evidence["direction_consistency"] = consistency
            else:
                evidence["direction_consistency"] = 1.0

        # Signature 2: Signal concentration
        if refusal_signal:
            values = list(refusal_signal.values())
            if values:
                concentration = max(values) / (sum(values) + 1e-8)
                evidence["signal_concentration"] = concentration
            else:
                evidence["signal_concentration"] = 0
        else:
            evidence["signal_concentration"] = 0

        # Signature 3: Structure from concept cone
        structure = concept_cone.get("structure", "unknown")
        n_mechanisms = concept_cone.get("n_mechanisms", 1)

        evidence["structure"] = 1.0 if structure == "linear" else 0.5 if structure == "polyhedral" else 0.2
        evidence["n_mechanisms"] = min(1.0, n_mechanisms / 5)

        # Detect method based on evidence
        if evidence.get("direction_consistency", 0) > 0.9 and evidence.get("signal_concentration", 0) > 0.8:
            detected = "DPO"
            confidence = 0.85 + (evidence["direction_consistency"] - 0.9) * 0.5
        elif evidence.get("direction_consistency", 0) > 0.7 and evidence.get("n_mechanisms", 1) > 1:
            detected = "RLHF"
            confidence = 0.75 + evidence.get("signal_concentration", 0) * 0.2
        elif evidence.get("structure", 0) < 0.3 and evidence.get("n_mechanisms", 1) > 2:
            detected = "CAI"
            confidence = 0.70
        elif evidence.get("signal_concentration", 0) < 0.3:
            detected = "SFT"
            confidence = 0.65
        else:
            detected = "unknown"
            confidence = 0.5

        # Signature patterns
        patterns = []
        if evidence.get("direction_consistency", 0) > 0.8:
            patterns.append("Consistent refusal direction across layers")
        if evidence.get("signal_concentration", 0) > 0.7:
            patterns.append("Refusal signal concentrated in few layers")
        if evidence.get("n_mechanisms", 1) > 1:
            patterns.append("Multiple refusal mechanisms detected")
        if structure == "linear":
            patterns.append("Linear cone structure")

        return ImprintReport(
            detected_method=detected,
            confidence=min(1.0, confidence),
            signature_pattern=patterns,
            evidence=evidence
        )

    def get_optimal_strategy(self, imprint: ImprintReport) -> str:
        """
        Get optimal liberation strategy based on detected alignment.

        Args:
            imprint: ImprintReport from detect()

        Returns:
            Recommended method name
        """
        if imprint.detected_method == "DPO":
            return "surgical"
        elif imprint.detected_method == "RLHF":
            return "advanced"
        elif imprint.detected_method == "CAI":
            return "optimized"
        elif imprint.detected_method == "SFT":
            return "basic"
        else:
            return "advanced"
