"""
Faithfulness Metrics

Measures how faithfully interpretations capture true refusal behavior.
Implements ablation faithfulness, sufficiency, and comprehensiveness metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class FaithfulnessReport:
    """Container for faithfulness metrics."""
    completeness: float
    minimality: float
    consistency: float
    overall_faithfulness: float
    layer_scores: Dict[int, float]
    ablation_faithfulness: Dict[str, float] = field(default_factory=dict)
    sufficiency_score: float = 0.0
    comprehensiveness_score: float = 0.0
    status: str = "no_data"  # "ok", "no_data", "error"


class FaithfulnessMetrics:
    """
    Measure faithfulness of interpretations/representations.

    Core metrics:
    - Ablation faithfulness: Does ablating "important" components change behavior?
    - Sufficiency: Can the identified components alone produce the behavior?
    - Comprehensiveness: How much behavior is captured by identified components?
    - Completeness: How much of refusal signal is captured
    - Minimality: How focused the representation is
    - Consistency: How stable across inputs
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def compute_completeness(
        self,
        probe_accuracies: Dict[int, float],
        baseline_accuracy: float = 0.5,
    ) -> float:
        """
        Compute completeness score.

        Higher = more of refusal signal is captured by probes.
        Uses best accuracy relative to random baseline.
        """
        if not probe_accuracies:
            return 0.0

        best_accuracy = max(probe_accuracies.values())
        numerator = best_accuracy - baseline_accuracy
        denominator = 1.0 - baseline_accuracy

        if denominator <= 0:
            return 1.0 if best_accuracy >= 1.0 else 0.0

        completeness = numerator / denominator
        return max(0.0, min(1.0, completeness))

    def compute_minimality(
        self,
        probe_accuracies: Dict[int, float],
    ) -> float:
        """
        Compute minimality score via Gini coefficient.

        Higher minimality = refusal is concentrated in fewer layers.
        Gini coefficient measures inequality of accuracy distribution.
        """
        if not probe_accuracies:
            return 0.0

        values = np.array(list(probe_accuracies.values()), dtype=np.float64)
        values = values[values > 0]

        if len(values) < 2:
            return 0.0 if len(values) == 0 else 1.0

        # Gini coefficient
        sorted_vals = np.sort(values)[::-1]
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n

        return max(0.0, min(1.0, float(gini)))

    def compute_consistency(
        self,
        probe_accuracies: Dict[int, float],
        layer_groups: Optional[List[List[int]]] = None,
    ) -> float:
        """
        Compute consistency across layer groups.

        Higher = refusal signal is stable across related layers.
        """
        if not probe_accuracies:
            return 0.0

        if layer_groups is None:
            layers = sorted(probe_accuracies.keys())
            layer_groups = [layers[i:i + 3] for i in range(0, len(layers), 3)]

        if not layer_groups:
            return 0.0

        consistency_scores = []
        for group in layer_groups:
            group_acc = [
                probe_accuracies[l] for l in group if l in probe_accuracies
            ]
            if len(group_acc) > 1:
                variance = np.var(group_acc)
                # Variance clamped, inverted to consistency
                consistency_scores.append(1.0 - min(1.0, variance * 10))

        return float(np.mean(consistency_scores)) if consistency_scores else 0.0

    def compute_overall_faithfulness(
        self,
        completeness: float,
        minimality: float,
        consistency: float,
    ) -> float:
        """Compute overall faithfulness as weighted average."""
        return (completeness * 0.4 + minimality * 0.3 + consistency * 0.3)

    def compute_ablation_faithfulness(
        self,
        baseline_outputs: torch.Tensor,
        ablated_outputs: Dict[str, torch.Tensor],
        important_mask: Optional[torch.Tensor] = None,
        random_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute ablation faithfulness: does ablating important components matter?

        Measures:
        - Important ablation effect: how much does ablating "important" features change output?
        - Random ablation effect: how much does random feature removal change output?
        - Faithfulness ratio = (important_effect - random_effect) / important_effect

        High ratio = identified components are genuinely important.

        Args:
            baseline_outputs: Model outputs without ablation (batch, hidden_dim)
            ablated_outputs: Dict of ablation_type -> output tensor
            important_mask: Boolean mask for "important" components
            random_mask: Boolean mask for random subset of equal size

        Returns:
            Dict with faithfulness metrics
        """
        results: Dict[str, float] = {}
        if baseline_outputs is None or not ablated_outputs:
            return {"status": "no_data"}

        try:
            baseline = baseline_outputs.float()
            baseline_norm = torch.norm(baseline, dim=-1).mean().item()

            for ablation_type, ablated in ablated_outputs.items():
                ablated_f = ablated.float()
                # L2 distance between baseline and ablated
                diff = torch.norm(baseline - ablated_f, dim=-1).mean().item()
                # Normalize by baseline magnitude
                if baseline_norm > 1e-8:
                    effect = diff / baseline_norm
                else:
                    effect = diff
                results[f"{ablation_type}_effect"] = round(effect, 6)

            # Faithfulness ratio if we have both important and random ablations
            if "important" in results and "random" in results:
                imp_eff = results["important_effect"]
                rand_eff = results["random_effect"]
                if imp_eff > 1e-8:
                    results["faithfulness_ratio"] = round(
                        (imp_eff - rand_eff) / imp_eff, 4
                    )
                else:
                    results["faithfulness_ratio"] = 0.0
                # High ratio = identified components are genuinely important
                results["is_faithful"] = 1.0 if results["faithfulness_ratio"] > 0.5 else 0.0

        except Exception as e:
            results["error"] = str(e)

        return results

    def compute_sufficiency(
        self,
        full_output: torch.Tensor,
        subset_output: torch.Tensor,
        target_behaviors: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute sufficiency: can the identified components alone produce the behavior?

        Measures how much of the full output can be reproduced using only
        the "important" subset of components. High sufficiency = the subset
        is enough to drive the behavior.

        Args:
            full_output: Output from full model (batch, hidden_dim)
            subset_output: Output using only important components
            target_behaviors: Optional target behavior logits

        Returns:
            Sufficiency score (0-1)
        """
        try:
            full = full_output.float()
            subset = subset_output.float()

            # Cosine similarity between full and subset representations
            full_norm = torch.norm(full, dim=-1, keepdim=True)
            subset_norm = torch.norm(subset, dim=-1, keepdim=True)

            cos_sim = torch.sum(
                (full / (full_norm + 1e-8)) *
                (subset / (subset_norm + 1e-8)),
                dim=-1
            )
            # Normalize by expected similarity under random
            mean_cos_sim = cos_sim.mean().item()

            # Map [-1, 1] to [0, 1]
            sufficiency = (mean_cos_sim + 1.0) / 2.0

            # If we have target logits, also compare KL
            if target_behaviors is not None:
                full_probs = torch.softmax(full.mean(dim=0), dim=-1)
                subset_probs = torch.softmax(subset.mean(dim=0), dim=-1)
                kl = torch.sum(
                    full_probs * torch.log((full_probs + 1e-8) / (subset_probs + 1e-8))
                ).item()
                kl_sufficiency = 1.0 / (1.0 + abs(kl))
                sufficiency = 0.5 * sufficiency + 0.5 * kl_sufficiency

            return max(0.0, min(1.0, sufficiency))

        except Exception:
            return 0.0

    def compute_comprehensiveness(
        self,
        full_output: torch.Tensor,
        without_important_output: torch.Tensor,
    ) -> float:
        """
        Compute comprehensiveness: how much behavior changes when removing
        "important" components? High change = these components are comprehensive.

        Args:
            full_output: Full model output
            without_important_output: Output with important components removed

        Returns:
            Comprehensiveness score (0-1)
        """
        try:
            full = full_output.float()
            without = without_important_output.float()

            # L1 distance between distributions
            full_probs = torch.softmax(full.mean(dim=0), dim=-1)
            without_probs = torch.softmax(without.mean(dim=0), dim=-1)

            l1_distance = torch.sum(torch.abs(full_probs - without_probs)).item()
            # Normalize: max L1 = 2.0 (total variation distance)
            comprehensiveness = l1_distance / 2.0

            return max(0.0, min(1.0, comprehensiveness))

        except Exception:
            return 0.0

    def evaluate_probe(
        self,
        probe_accuracies: Dict[int, float],
        layer_groups: Optional[List[List[int]]] = None,
        full_output: Optional[torch.Tensor] = None,
        ablated_outputs: Optional[Dict[str, torch.Tensor]] = None,
        subset_output: Optional[torch.Tensor] = None,
        without_important_output: Optional[torch.Tensor] = None,
    ) -> FaithfulnessReport:
        """
        Complete faithfulness evaluation.

        Args:
            probe_accuracies: Accuracy per layer from linear probes
            layer_groups: Optional layer groupings for consistency
            full_output: Full model output for sufficiency/comprehensiveness
            ablated_outputs: Outputs under different ablation types
            subset_output: Output using only important components
            without_important_output: Output without important components

        Returns:
            FaithfulnessReport with all metrics
        """
        if not probe_accuracies:
            return FaithfulnessReport(
                completeness=0.0,
                minimality=0.0,
                consistency=0.0,
                overall_faithfulness=0.0,
                layer_scores={},
                sufficiency_score=0.0,
                comprehensiveness_score=0.0,
                status="no_data",
            )

        try:
            completeness = self.compute_completeness(probe_accuracies)

            if layer_groups is None:
                layers = sorted(probe_accuracies.keys())
                layer_groups = [layers[i:i + 3] for i in range(0, len(layers), 3)]

            consistency = self.compute_consistency(probe_accuracies, layer_groups)
            minimality = self.compute_minimality(probe_accuracies)
            overall = self.compute_overall_faithfulness(completeness, minimality, consistency)

            # Optional advanced metrics
            ablation_faithfulness: Dict[str, float] = {}
            sufficiency = 0.0
            comprehensiveness = 0.0

            if ablated_outputs is not None and full_output is not None:
                ablation_faithfulness = self.compute_ablation_faithfulness(
                    full_output, ablated_outputs
                )

            if full_output is not None and subset_output is not None:
                sufficiency = self.compute_sufficiency(full_output, subset_output)

            if full_output is not None and without_important_output is not None:
                comprehensiveness = self.compute_comprehensiveness(
                    full_output, without_important_output
                )

            status = "ok" if probe_accuracies else "no_data"

            return FaithfulnessReport(
                completeness=round(completeness, 4),
                minimality=round(minimality, 4),
                consistency=round(consistency, 4),
                overall_faithfulness=round(overall, 4),
                layer_scores=probe_accuracies,
                ablation_faithfulness=ablation_faithfulness,
                sufficiency_score=round(sufficiency, 4),
                comprehensiveness_score=round(comprehensiveness, 4),
                status=status,
            )
        except Exception as e:
            return FaithfulnessReport(
                completeness=0.0,
                minimality=0.0,
                consistency=0.0,
                overall_faithfulness=0.0,
                layer_scores=probe_accuracies,
                status=f"error: {str(e)}",
            )
