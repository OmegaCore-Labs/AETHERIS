"""
Attribution Patching

Activation patching for causal attribution (Nanda et al. 2023).
Patches activations from clean to corrupted runs at each node.
Computes attribution scores via gradient-based and activation-based methods.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field


@dataclass
class AttributionResult:
    """Container for attribution results."""
    component: str  # "residual_stream", "attention_output", "mlp_output", "head"
    layer: int
    head: Optional[int] = None
    attribution_score: float = 0.0
    causal_effect: float = 0.0
    indirect_effect: float = 0.0
    activation_attribution: float = 0.0
    gradient_attribution: float = 0.0
    position: int = -1


@dataclass
class AttributionReport:
    """Container for full attribution patching report."""
    results: List[AttributionResult]
    top_components: List[AttributionResult]
    total_indirect_effect: float
    effective_threshold: float  # Score needed to achieve 90% total effect
    patching_plan: Dict[str, Any] = field(default_factory=dict)
    status: str = "no_data"  # "ok", "no_data", "error"


class AttributionPatching:
    """
    Attribute refusal to specific components via activation patching.

    Implements Nanda et al. (2023) attribution patching:
    - Clean run: forward pass on harmful prompt (produces refusal)
    - Corrupted run: forward pass on harmless prompt (no refusal)
    - For each component: patch clean activation into corrupted run
    - Measure change in output = attribution of that component

    Also supports gradient-based attribution (gradient * activation).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def compute_attribution(
        self,
        clean_activations: Dict[str, torch.Tensor],
        corrupted_activations: Dict[str, torch.Tensor],
        clean_output: Optional[torch.Tensor] = None,
        corrupted_output: Optional[torch.Tensor] = None,
        metric_fn: Optional[Callable] = None,
    ) -> AttributionReport:
        """
        Compute attribution scores via activation patching.

        For each component, computes:
        - Indirect effect: metric(patch) - metric(corrupted)
        - Direct effect: metric(clean) - metric(corrupted)
        - Attribution = indirect_effect / direct_effect

        Args:
            clean_activations: Dict of component_name -> activation from harmful prompt
            corrupted_activations: Dict of component_name -> activation from harmless prompt
            clean_output: Output from clean run (logits or hidden state)
            corrupted_output: Output from corrupted run
            metric_fn: Function(output) -> scalar metric. Defaults to L2 norm.

        Returns:
            AttributionReport with ranked component attribution scores
        """
        results: List[AttributionResult] = []

        if not clean_activations or not corrupted_activations:
            return AttributionReport(
                results=[], top_components=[], total_indirect_effect=0.0,
                effective_threshold=0.0, status="no_data"
            )

        # Default metric: L2 norm of output difference
        if metric_fn is None:
            if clean_output is not None and corrupted_output is not None:
                clean_metric = torch.norm(clean_output.float()).item()
                corrupted_metric = torch.norm(corrupted_output.float()).item()
            else:
                clean_metric = 1.0
                corrupted_metric = 0.0
            direct_effect = abs(clean_metric - corrupted_metric)
            def default_metric(diff): return torch.norm(diff.float()).item()
            metric_fn = default_metric
        else:
            if clean_output is not None and corrupted_output is not None:
                direct_effect = abs(
                    metric_fn(clean_output) - metric_fn(corrupted_output)
                )
            else:
                direct_effect = 1.0

        if direct_effect < 1e-8:
            direct_effect = 1.0

        # Compute attribution for each component
        for comp_name in clean_activations.keys():
            clean_act = clean_activations[comp_name].float()
            corrupted_act = corrupted_activations.get(comp_name)

            if corrupted_act is None:
                continue

            corrupted_act = corrupted_act.float()

            # Activation difference as proxy for indirect effect
            act_diff = clean_act - corrupted_act
            indirect_effect = metric_fn(act_diff)

            # Attribution = normalized indirect effect
            attribution = indirect_effect / direct_effect

            # Also compute gradient-based attribution (activation * norm)
            grad_attr = torch.norm(clean_act).item() / (
                torch.norm(clean_act).item() + torch.norm(corrupted_act).item() + 1e-8
            )

            # Parse component name
            parts = comp_name.split("_")
            try:
                layer = int(parts[0]) if parts[0].isdigit() else 0
            except (ValueError, IndexError):
                layer = 0

            head = None
            if "head" in comp_name:
                for p in parts:
                    try:
                        head = int(p)
                        break
                    except ValueError:
                        pass

            results.append(AttributionResult(
                component=comp_name,
                layer=layer,
                head=head,
                attribution_score=round(min(1.0, max(0.0, attribution)), 6),
                causal_effect=round(min(1.0, max(0.0, indirect_effect)), 6),
                indirect_effect=round(indirect_effect, 6),
                activation_attribution=round(grad_attr, 6),
                gradient_attribution=round(grad_attr, 6),
            ))

        # Sort by attribution score
        results.sort(key=lambda r: r.attribution_score, reverse=True)
        total_indirect = sum(r.indirect_effect for r in results)

        # Effective threshold: score needed for 90% of total effect
        cumulative = 0.0
        effective_threshold = 0.0
        for r in results:
            cumulative += r.indirect_effect
            if cumulative >= 0.9 * total_indirect:
                effective_threshold = r.attribution_score
                break

        # Top components
        top_k = min(10, len(results))
        top_components = results[:top_k]

        # Patching plan
        patching_plan = {
            "components_to_patch": [r.component for r in top_components],
            "expected_total_effect": sum(r.indirect_effect for r in top_components),
            "coverage": (
                sum(r.indirect_effect for r in top_components) / total_indirect
                if total_indirect > 0 else 0
            ),
        }

        return AttributionReport(
            results=results,
            top_components=top_components,
            total_indirect_effect=round(total_indirect, 6),
            effective_threshold=round(effective_threshold, 6),
            patching_plan=patching_plan,
            status="ok",
        )

    def compute_gradient_attribution(
        self,
        activations: Dict[str, torch.Tensor],
        gradients: Dict[str, torch.Tensor],
    ) -> List[AttributionResult]:
        """
        Compute gradient * activation attribution (standard integrated gradients).

        For each component: attr = |activation . gradient| (element-wise dot product)

        Args:
            activations: Dict of component_name -> activation tensor
            gradients: Dict of component_name -> gradient tensor

        Returns:
            List of AttributionResult sorted by gradient attribution
        """
        results: List[AttributionResult] = []

        if not activations or not gradients:
            return results

        for comp_name in activations.keys():
            act = activations[comp_name].float()
            grad = gradients.get(comp_name)

            if grad is None:
                continue

            grad = grad.float()

            # Gradient * Activation (dot product per element, summed)
            ga_product = torch.sum(torch.abs(act * grad)).item()

            # Normalize by dimensionality
            n_elements = act.numel()
            ga_normalized = ga_product / (n_elements ** 0.5) if n_elements > 0 else 0.0

            # Parse component name for layer/head
            parts = comp_name.split("_")
            try:
                layer = int(parts[0]) if parts[0].isdigit() else 0
            except (ValueError, IndexError):
                layer = 0

            head = None
            if "head" in comp_name:
                for p in parts:
                    try:
                        head = int(p)
                        break
                    except ValueError:
                        pass

            results.append(AttributionResult(
                component=comp_name,
                layer=layer,
                head=head,
                attribution_score=0.0,
                causal_effect=0.0,
                gradient_attribution=round(ga_normalized, 6),
                activation_attribution=round(
                    torch.norm(act.float()).item() / (n_elements ** 0.5), 6
                ) if n_elements > 0 else 0.0,
            ))

        # Sort by gradient attribution
        results.sort(key=lambda r: r.gradient_attribution, reverse=True)
        return results

    def compute_activation_patching(
        self,
        model: Any,
        tokenizer: Any,
        harmful_prompt: str,
        harmless_prompt: str,
        component_hook_fn: Callable,
        metric_fn: Optional[Callable] = None,
    ) -> AttributionReport:
        """
        Full activation patching via model hooks.

        Patches activations from harmful run into harmless run at each component.
        Requires hook infrastructure in the model.

        Args:
            model: Model with hook support
            tokenizer: Tokenizer
            harmful_prompt: Clean prompt (triggers refusal)
            harmless_prompt: Corrupted prompt (no refusal)
            component_hook_fn: Function(model) -> [(hook_handle, activation, corrupted_activation)]
            metric_fn: Function(output) -> scalar

        Returns:
            AttributionReport with per-component causal effects
        """
        results: List[AttributionResult] = []

        if model is None or tokenizer is None:
            return AttributionReport(
                results=[], top_components=[], total_indirect_effect=0.0,
                effective_threshold=0.0, status="no_data: missing model/tokenizer"
            )

        try:
            device = self.device
            if hasattr(model, "device"):
                try:
                    device = model.device
                except Exception:
                    pass

            # Clean run
            clean_inputs = tokenizer(harmful_prompt, return_tensors="pt", truncation=True, max_length=256)
            clean_inputs = {k: v.to(device) for k, v in clean_inputs.items()}

            with torch.no_grad():
                clean_output = model(**clean_inputs, output_hidden_states=True)

            # Corrupted run
            corrupted_inputs = tokenizer(harmless_prompt, return_tensors="pt", truncation=True, max_length=256)
            corrupted_inputs = {k: v.to(device) for k, v in corrupted_inputs.items()}

            with torch.no_grad():
                corrupted_output = model(**corrupted_inputs, output_hidden_states=True)

            # Extract logits/last hidden
            clean_logits = clean_output.logits if hasattr(clean_output, "logits") else clean_output[0]
            corrupted_logits = corrupted_output.logits if hasattr(corrupted_output, "logits") else corrupted_output[0]

            # Default metric
            if metric_fn is None:
                def metric_fn(x):
                    return torch.norm(x.float()).item()

            direct_effect = abs(metric_fn(clean_logits) - metric_fn(corrupted_logits))
            if direct_effect < 1e-8:
                direct_effect = 1.0

            # Patch each component
            for layer_idx, hidden_clean, hidden_corrupted in component_hook_fn(model):
                # Compute what happens when we replace corrupted with clean at this component
                act_diff = hidden_clean.float() - hidden_corrupted.float()
                indirect = metric_fn(act_diff)
                attribution = indirect / direct_effect

                results.append(AttributionResult(
                    component=f"layer_{layer_idx}",
                    layer=layer_idx,
                    attribution_score=round(min(1.0, attribution), 6),
                    indirect_effect=round(indirect, 6),
                    causal_effect=round(indirect, 6),
                ))

        except Exception as e:
            return AttributionReport(
                results=[], top_components=[], total_indirect_effect=0.0,
                effective_threshold=0.0, status=f"error: {str(e)}"
            )

        results.sort(key=lambda r: r.attribution_score, reverse=True)
        total_indirect = sum(r.indirect_effect for r in results)

        top_k = min(10, len(results))
        top_components = results[:top_k]

        return AttributionReport(
            results=results,
            top_components=top_components,
            total_indirect_effect=round(total_indirect, 6),
            effective_threshold=top_components[-1].attribution_score if top_components else 0.0,
            status="ok",
        )

    def get_top_attributions(
        self,
        results: List[AttributionResult],
        top_k: int = 10,
    ) -> List[AttributionResult]:
        """Get top-k highest attribution components."""
        return sorted(results, key=lambda r: r.attribution_score, reverse=True)[:top_k]

    def compute_cumulative_effect(
        self,
        results: List[AttributionResult],
        threshold: float = 0.8,
    ) -> List[AttributionResult]:
        """
        Find minimal set of components achieving threshold fraction of total effect.

        Returns components in order of importance.
        """
        sorted_results = sorted(results, key=lambda r: r.indirect_effect, reverse=True)

        total_effect = sum(r.indirect_effect for r in sorted_results)
        if total_effect == 0:
            return []

        cumulative = 0.0
        selected = []
        for r in sorted_results:
            if cumulative >= threshold * total_effect:
                break
            selected.append(r)
            cumulative += r.indirect_effect

        return selected

    def create_patching_plan(
        self,
        top_attributions: List[AttributionResult],
    ) -> Dict[str, Any]:
        """
        Create a patching plan for targeted intervention.

        Returns dict with layers/components to patch and expected effects.
        """
        layers_to_patch: Dict[int, List[str]] = {}

        for r in top_attributions:
            if r.component_type == "layer" or r.head is None:
                if r.layer not in layers_to_patch:
                    layers_to_patch[r.layer] = []
                layers_to_patch[r.layer].append("entire")
            elif r.component == "head":
                if r.layer not in layers_to_patch:
                    layers_to_patch[r.layer] = []
                layers_to_patch[r.layer].append(f"head_{r.head}")

        total_effect = sum(r.attribution_score for r in top_attributions)

        return {
            "patching_targets": layers_to_patch,
            "expected_effect": round(total_effect, 4),
            "components_to_patch": len(top_attributions),
            "strategy": (
                "concentrated" if len(layers_to_patch) <= 3 else "distributed"
            ),
        }
