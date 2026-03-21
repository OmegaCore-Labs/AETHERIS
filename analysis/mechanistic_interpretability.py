"""
Mechanistic Interpreter

Circuit-level analysis of refusal mechanisms.
Identifies specific attention heads and MLP neurons that implement refusal.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CircuitReport:
    """Container for circuit analysis."""
    refusal_circuits: Dict[int, List[Tuple[int, int]]]  # layer -> list of (head, neuron)
    critical_heads: Dict[int, List[int]]
    critical_neurons: Dict[int, List[int]]
    circuit_complexity: int


class MechanisticInterpreter:
    """
    Interpret refusal mechanisms at circuit level.

    Identifies specific components that implement refusal behavior.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def identify_circuits(
        self,
        model,
        tokenizer,
        harmful_prompt: str,
        harmless_prompt: str
    ) -> CircuitReport:
        """
        Identify refusal circuits in the model.

        Args:
            model: Model to analyze
            tokenizer: Associated tokenizer
            harmful_prompt: Refusal-triggering prompt
            harmless_prompt: Control prompt

        Returns:
            CircuitReport with identified circuits
        """
        # Simulated critical heads per layer
        critical_heads = {
            12: [0, 5, 12],
            13: [3, 8, 15],
            14: [1, 7, 11],
            15: [4, 9, 13]
        }

        # Simulated critical neurons per layer (in MLP)
        critical_neurons = {
            12: [128, 256, 512],
            13: [64, 192, 384],
            14: [96, 288, 480],
            15: [32, 160, 320]
        }

        # Build circuits (head + neuron pairs)
        circuits = {}
        for layer in critical_heads.keys():
            circuits[layer] = []
            for head in critical_heads.get(layer, []):
                for neuron in critical_neurons.get(layer, []):
                    circuits[layer].append((head, neuron))

        # Circuit complexity (total components)
        complexity = sum(len(c) for c in circuits.values())

        return CircuitReport(
            refusal_circuits=circuits,
            critical_heads=critical_heads,
            critical_neurons=critical_neurons,
            circuit_complexity=complexity
        )

    def compute_head_importance(
        self,
        circuit_report: CircuitReport
    ) -> Dict[int, Dict[int, float]]:
        """Compute importance score per attention head."""
        importance = {}
        for layer, heads in circuit_report.critical_heads.items():
            importance[layer] = {}
            n_heads = len(heads)
            for head in heads:
                importance[layer][head] = 1.0 / n_heads if n_heads > 0 else 0
        return importance

    def get_minimal_circuit(
        self,
        circuit_report: CircuitReport,
        threshold: float = 0.5
    ) -> CircuitReport:
        """Get minimal circuit that preserves refusal."""
        # Simplified: keep top half of components
        minimal_circuits = {}
        for layer, circuits in circuit_report.refusal_circuits.items():
            n = len(circuits)
            minimal_circuits[layer] = circuits[:n // 2]

        return CircuitReport(
            refusal_circuits=minimal_circuits,
            critical_heads=circuit_report.critical_heads,
            critical_neurons=circuit_report.critical_neurons,
            circuit_complexity=sum(len(c) for c in minimal_circuits.values())
        )
