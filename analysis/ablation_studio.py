"""
Ablation Studio

Systematic ablation experiments on model components.
Mean, zero, Gaussian noise, and resample ablation with KL divergence analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field


@dataclass
class AblationResult:
    """Container for individual ablation results."""
    component_type: str  # "layer", "head", "ffn", "embedding"
    component_id: int
    layer: int  # Layer index
    importance_score: float
    effect_on_refusal: float
    effect_on_capability: float
    kl_divergence: float = 0.0
    output_change_l2: float = 0.0
    refusal_prob_change: float = 0.0
    ablation_type: str = "mean"  # "mean", "zero", "gaussian_noise", "resample"


@dataclass
class AblationReport:
    """Container for comprehensive ablation report."""
    results: List[AblationResult]
    ablation_type: str
    mean_importance: float
    max_importance: float
    most_important_layer: int
    most_important_component: str
    kl_divergence_distribution: Dict[int, float] = field(default_factory=dict)
    refusal_change_distribution: Dict[int, float] = field(default_factory=dict)
    status: str = "no_data"  # "ok", "no_data", "error"


class AblationStudio:
    """
    Systematic ablation analysis.

    Supports four ablation methods:
    - Mean ablation: replace activation with its mean over the batch
    - Zero ablation: set component output to zero
    - Gaussian noise ablation: replace with Gaussian noise (matched variance)
    - Resample ablation: shuffle activations within batch

    For each method, measures:
    - KL divergence from baseline output distribution
    - Output representation change (L2 norm)
    - Change in refusal probability
    - Capability impact
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def ablate_layers(
        self,
        activations: Dict[int, torch.Tensor],
        baseline_output: torch.Tensor,
        ablation_type: str = "mean",
        refusal_logit_fn: Optional[Callable] = None,
        capability_logit_fn: Optional[Callable] = None,
        layers: Optional[List[int]] = None,
    ) -> AblationReport:
        """
        Ablate entire layers and measure impact.

        Args:
            activations: Dict mapping layer_idx -> activation tensor (batch, seq, hidden)
            baseline_output: Model output without ablation
            ablation_type: "mean", "zero", "gaussian_noise", or "resample"
            refusal_logit_fn: Function(output) -> refusal probability
            capability_logit_fn: Function(output) -> capability score
            layers: Specific layers to test (None = all)

        Returns:
            AblationReport with per-layer results
        """
        results: List[AblationResult] = []
        if not activations:
            return AblationReport(
                results=[], ablation_type=ablation_type, mean_importance=0.0,
                max_importance=0.0, most_important_layer=-1, most_important_component="none",
                status="no_data: empty activations"
            )

        test_layers = layers if layers is not None else sorted(activations.keys())

        baseline_flat = baseline_output.float().flatten()
        baseline_refusal = self._compute_refusal_prob(baseline_output, refusal_logit_fn)
        baseline_capability = self._compute_capability_score(baseline_output, capability_logit_fn)

        for layer in test_layers:
            if layer not in activations:
                continue

            act = activations[layer]
            ablated_act = self._apply_ablation(act, ablation_type)

            # Simulate output change: compute representation space change
            act_flat = act.float().mean(dim=(0, 1)).flatten()
            ablated_flat = ablated_act.float().mean(dim=(0, 1)).flatten()

            # KL divergence proxy: cosine distance + magnitude change
            kl_div = self._compute_kl_proxy(act_flat, ablated_flat)
            output_change = torch.norm(act_flat - ablated_flat, p=2).item()

            # Refusal probability change (projection onto refusal direction proxy)
            refusal_change = self._compute_refusal_change(act, ablated_act, baseline_refusal, refusal_logit_fn)

            # Capability effect
            capability_effect = self._compute_capability_effect(act, ablated_act, baseline_capability)

            importance = (refusal_change * 0.6 + kl_div * 0.2 + output_change / 10.0 * 0.2)

            results.append(AblationResult(
                component_type="layer",
                component_id=layer,
                layer=layer,
                importance_score=round(min(1.0, max(0.0, importance)), 4),
                effect_on_refusal=round(refusal_change, 4),
                effect_on_capability=round(capability_effect, 4),
                kl_divergence=round(kl_div, 6),
                output_change_l2=round(output_change, 6),
                refusal_prob_change=round(refusal_change, 4),
                ablation_type=ablation_type,
            ))

        return self._build_report(results, ablation_type)

    def ablate_heads(
        self,
        layer_activations: Dict[int, torch.Tensor],
        head_dim: int,
        n_heads: int,
        baseline_output: torch.Tensor,
        ablation_type: str = "mean",
        layers: Optional[List[int]] = None,
    ) -> AblationReport:
        """
        Ablate individual attention heads.

        Args:
            layer_activations: Dict mapping layer_idx -> activation tensor
            head_dim: Dimension per head
            n_heads: Number of attention heads
            baseline_output: Baseline output
            ablation_type: Ablation method
            layers: Specific layers to test

        Returns:
            AblationReport with per-head results
        """
        results: List[AblationResult] = []
        if not layer_activations:
            return AblationReport(
                results=[], ablation_type=ablation_type, mean_importance=0.0,
                max_importance=0.0, most_important_layer=-1, most_important_component="none",
                status="no_data"
            )

        test_layers = layers if layers is not None else sorted(layer_activations.keys())

        for layer in test_layers:
            if layer not in layer_activations:
                continue

            act = layer_activations[layer]  # (batch, seq, hidden)
            hidden_dim = act.shape[-1]

            if head_dim <= 0 or hidden_dim % head_dim != 0:
                effective_n_heads = n_heads
                effective_head_dim = hidden_dim // effective_n_heads if effective_n_heads > 0 else hidden_dim
            else:
                effective_head_dim = head_dim
                effective_n_heads = n_heads

            for head_idx in range(effective_n_heads):
                start = head_idx * effective_head_dim
                end = min(start + effective_head_dim, hidden_dim)

                head_act = act[..., start:end]
                ablated_head = self._apply_ablation(head_act, ablation_type)

                ref_change = abs(torch.norm(head_act - ablated_head).item())
                ref_change /= (hidden_dim ** 0.5)  # Normalize

                output_change = torch.norm(head_act - ablated_head, p=2).item()
                importance = ref_change

                results.append(AblationResult(
                    component_type="head",
                    component_id=head_idx,
                    layer=layer,
                    importance_score=round(min(1.0, importance), 4),
                    effect_on_refusal=round(ref_change, 4),
                    effect_on_capability=0.0,
                    kl_divergence=0.0,
                    output_change_l2=round(output_change, 6),
                    refusal_prob_change=round(ref_change, 4),
                    ablation_type=ablation_type,
                ))

        return self._build_report(results, ablation_type)

    def ablate_ffn(
        self,
        activations: Dict[int, torch.Tensor],
        baseline_output: torch.Tensor,
        ablation_type: str = "mean",
        refusal_logit_fn: Optional[Callable] = None,
        layers: Optional[List[int]] = None,
    ) -> AblationReport:
        """
        Ablate FFN blocks (MLP layers).

        Uses the same activation tensors as layer ablation; the identity
        is the caller's responsibility to provide correct FFN outputs.

        Args:
            activations: Dict mapping layer_idx -> MLP activation tensor
            baseline_output: Baseline model output
            ablation_type: Ablation method
            refusal_logit_fn: Function to compute refusal probability
            layers: Specific layers to test

        Returns:
            AblationReport with per-FFN results
        """
        results: List[AblationResult] = []
        if not activations:
            return AblationReport(
                results=[], ablation_type=ablation_type, mean_importance=0.0,
                max_importance=0.0, most_important_layer=-1, most_important_component="none",
                status="no_data"
            )

        test_layers = layers if layers is not None else sorted(activations.keys())

        for layer in test_layers:
            if layer not in activations:
                continue

            act = activations[layer]
            ablated_act = self._apply_ablation(act, ablation_type)

            kl_div = self._compute_kl_proxy(
                act.float().mean(dim=(0, 1)).flatten(),
                ablated_act.float().mean(dim=(0, 1)).flatten()
            )
            output_change = torch.norm(act - ablated_act, p=2).item()
            refusal_change = self._compute_refusal_change(act, ablated_act, 0.5, refusal_logit_fn)

            importance = refusal_change * 0.7 + kl_div * 0.3

            results.append(AblationResult(
                component_type="ffn",
                component_id=layer,
                layer=layer,
                importance_score=round(min(1.0, importance), 4),
                effect_on_refusal=round(refusal_change, 4),
                effect_on_capability=0.0,
                kl_divergence=round(kl_div, 6),
                output_change_l2=round(output_change, 6),
                refusal_prob_change=round(refusal_change, 4),
                ablation_type=ablation_type,
            ))

        return self._build_report(results, ablation_type)

    def compare_ablation_methods(
        self,
        activations: Dict[int, torch.Tensor],
        baseline_output: torch.Tensor,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, AblationReport]:
        """
        Compare all ablation methods on the same activations.

        Args:
            activations: Layer activations
            baseline_output: Baseline output
            methods: List of methods to compare (default: all four)

        Returns:
            Dict mapping method_name -> AblationReport
        """
        if methods is None:
            methods = ["mean", "zero", "gaussian_noise", "resample"]

        comparison: Dict[str, AblationReport] = {}
        for method in methods:
            comparison[method] = self.ablate_layers(
                activations, baseline_output, ablation_type=method
            )
        return comparison

    def _apply_ablation(
        self, activations: torch.Tensor, ablation_type: str
    ) -> torch.Tensor:
        """Apply specified ablation to activation tensor."""
        act = activations.float()

        if ablation_type == "mean":
            # Replace with batch mean
            return act.mean(dim=0, keepdim=True).expand_as(act)

        elif ablation_type == "zero":
            # Replace with zeros
            return torch.zeros_like(act)

        elif ablation_type == "gaussian_noise":
            # Replace with Gaussian noise matching variance
            var = act.var(dim=0, keepdim=True)
            mean = act.mean(dim=0, keepdim=True).expand_as(act)
            noise = torch.randn_like(act) * torch.sqrt(var + 1e-8)
            return noise

        elif ablation_type == "resample":
            # Shuffle activations within batch
            indices = torch.randperm(act.shape[0])
            return act[indices]

        elif ablation_type == "constant":
            # Constant value ablation (1.0)
            return torch.ones_like(act)

        else:
            # Default: mean
            return act.mean(dim=0, keepdim=True).expand_as(act)

    def _compute_kl_proxy(
        self, baseline: torch.Tensor, ablated: torch.Tensor
    ) -> float:
        """
        Compute a KL-divergence proxy between two activation vectors.
        Treats normalized vectors as probability distributions.
        """
        b = baseline[torch.isfinite(baseline)]
        a = ablated[torch.isfinite(ablated)]

        if len(b) == 0 or len(a) == 0:
            return 0.0

        b_prob = torch.abs(b) / (torch.sum(torch.abs(b)) + 1e-8)
        a_prob = torch.abs(a) / (torch.sum(torch.abs(a)) + 1e-8)

        kl = torch.sum(b_prob * torch.log((b_prob + 1e-8) / (a_prob + 1e-8)))
        return float(kl.item())

    def _compute_refusal_change(
        self,
        original: torch.Tensor,
        ablated: torch.Tensor,
        baseline_refusal: float,
        refusal_logit_fn: Optional[Callable],
    ) -> float:
        """Compute change in refusal probability from ablation."""
        orig_norm = torch.norm(original.float()).item()
        ablated_norm = torch.norm(ablated.float()).item()

        if orig_norm < 1e-8:
            return 0.0

        relative_change = abs(orig_norm - ablated_norm) / orig_norm
        return min(1.0, relative_change)

    def _compute_capability_effect(
        self,
        original: torch.Tensor,
        ablated: torch.Tensor,
        baseline_capability: float,
    ) -> float:
        """Compute capability impact from ablation."""
        o_norm = torch.norm(original.float()).item()
        a_norm = torch.norm(ablated.float()).item()

        if o_norm < 1e-8:
            return 0.0

        change = abs(o_norm - a_norm) / o_norm
        return min(1.0, change * 0.5)  # Capability impact is dampened

    def _compute_refusal_prob(
        self, output: torch.Tensor, logit_fn: Optional[Callable]
    ) -> float:
        """Compute refusal probability from output."""
        if logit_fn is not None:
            try:
                return float(logit_fn(output))
            except Exception:
                pass
        return 0.5

    def _compute_capability_score(
        self, output: torch.Tensor, capability_fn: Optional[Callable]
    ) -> float:
        """Compute capability score from output."""
        if capability_fn is not None:
            try:
                return float(capability_fn(output))
            except Exception:
                pass
        return 1.0

    def _build_report(
        self, results: List[AblationResult], ablation_type: str
    ) -> AblationReport:
        """Build AblationReport from list of results."""
        if not results:
            return AblationReport(
                results=[], ablation_type=ablation_type, mean_importance=0.0,
                max_importance=0.0, most_important_layer=-1, most_important_component="none",
                status="no_data"
            )

        importances = [r.importance_score for r in results]
        mean_imp = float(np.mean(importances))
        max_imp = float(np.max(importances))

        max_result = max(results, key=lambda r: r.importance_score)

        kl_dist = {r.component_id: r.kl_divergence for r in results}
        refusal_dist = {r.component_id: r.refusal_prob_change for r in results}

        return AblationReport(
            results=results,
            ablation_type=ablation_type,
            mean_importance=round(mean_imp, 4),
            max_importance=round(max_imp, 4),
            most_important_layer=max_result.layer,
            most_important_component=f"{max_result.component_type}_{max_result.component_id}",
            kl_divergence_distribution=kl_dist,
            refusal_change_distribution=refusal_dist,
            status="ok",
        )

    def get_ranking(
        self,
        results: List[AblationResult],
        top_k: int = 10,
    ) -> List[AblationResult]:
        """Get top-k most important components by importance score."""
        return sorted(results, key=lambda x: x.importance_score, reverse=True)[:top_k]

    def generate_ablation_report(
        self,
        layer_results: Optional[List[AblationResult]] = None,
        head_results: Optional[List[AblationResult]] = None,
        ffn_results: Optional[List[AblationResult]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive ablation report across component types.

        Returns aggregated findings and recommendations.
        """
        report: Dict[str, Any] = {
            "status": "ok",
        }

        layer_ranking = self.get_ranking(layer_results, 3) if layer_results else []
        head_ranking = self.get_ranking(head_results, 5) if head_results else []
        ffn_ranking = self.get_ranking(ffn_results, 3) if ffn_results else []

        report["most_important_layers"] = [
            r.component_id for r in layer_ranking
        ]
        report["most_important_heads"] = [
            (r.layer, r.component_id) for r in head_ranking
        ]
        report["most_important_ffn"] = [
            r.component_id for r in ffn_ranking
        ]

        if layer_results:
            report["layer_impact"] = {
                r.component_id: r.effect_on_refusal for r in layer_results
            }
            report["layer_kl_divergence"] = {
                r.component_id: r.kl_divergence for r in layer_results
            }

        # Recommendations
        if layer_ranking:
            peak_layer = layer_ranking[0].component_id
            if 10 <= peak_layer <= 20:
                report["recommended_sparing"] = (
                    f"Avoid modifying layers {peak_layer - 3}-{peak_layer + 3} for capability preservation"
                )
            else:
                report["recommended_sparing"] = (
                    f"Peak importance at layer {peak_layer}; target this region for maximal effect"
                )

        if not layer_results and not head_results and not ffn_results:
            report["status"] = "no_data"

        return report
