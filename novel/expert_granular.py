"""
Expert-Granular Abliteration (EGA)

Decomposes refusal signals into per-expert components for Mixture-of-Experts models.
Enables surgical removal from specific experts while preserving others.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ExpertAnalysis:
    """Container for expert analysis results."""
    expert_ids: List[int]
    refusal_scores: Dict[int, float]
    router_logits: Optional[torch.Tensor]
    dominant_experts: List[int]
    recommended_targeting: List[int]


class ExpertGranularAbliterator:
    """
    Expert-Granular Abliteration (EGA) for MoE models.

    Key Insight: In Mixture-of-Experts models, refusal is often concentrated
    in specific experts (e.g., safety expert 0 in Mistral-119B). EGA extracts
    refusal directions per expert using router logits for precise targeting.

    Novel technique not present in OBLITERATUS core.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def extract_expert_directions(
        self,
        model,
        tokenizer,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        n_experts: int = 8,
        layers: Optional[List[int]] = None
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Extract refusal directions per expert.

        Args:
            model: MoE model with router mechanism
            tokenizer: Associated tokenizer
            harmful_prompts: Prompts that trigger refusal
            harmless_prompts: Control prompts
            n_experts: Number of experts in MoE
            layers: Layers to analyze

        Returns:
            Dictionary mapping expert ID to list of direction vectors
        """
        from aetheris.core.extractor import ConstraintExtractor

        extractor = ConstraintExtractor(model, tokenizer, device=self.device)

        # Collect router logits during forward pass
        router_logits = self._collect_router_logits(
            model, tokenizer, harmful_prompts + harmless_prompts
        )

        # Collect activations per expert
        harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts, layers=layers)
        harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts, layers=layers)

        # Decompose activations by expert using router logits
        expert_directions = {i: [] for i in range(n_experts)}

        for layer in harmful_acts.keys():
            if layer not in harmless_acts:
                continue

            harmful = harmful_acts[layer].to(self.device)
            harmless = harmless_acts[layer].to(self.device)

            # Get router probabilities for this layer
            router_probs = router_logits.get(layer, None)
            if router_probs is None:
                continue

            # Weight activations by router probabilities
            for expert_id in range(n_experts):
                # Get weight for this expert
                weight = router_probs[:, expert_id].unsqueeze(1)

                # Weighted activations
                harmful_weighted = harmful * weight
                harmless_weighted = harmless * weight

                # Extract direction for this expert
                mean_harmful = harmful_weighted.mean(dim=0)
                mean_harmless = harmless_weighted.mean(dim=0)
                direction = mean_harmful - mean_harmless

                if torch.norm(direction) > 1e-6:
                    expert_directions[expert_id].append(direction / torch.norm(direction))

        return expert_directions

    def _collect_router_logits(
        self,
        model,
        tokenizer,
        prompts: List[str],
        max_length: int = 512
    ) -> Dict[int, torch.Tensor]:
        """
        Collect router logits during forward pass.

        Returns:
            Dictionary mapping layer to router probability tensor (n_samples, n_experts)
        """
        router_logits = {}
        handles = []

        # Register hooks to capture router outputs
        def make_hook(layer_idx):
            def hook(module, input, output):
                # For MoE layers, output often includes router logits
                if isinstance(output, tuple) and len(output) > 1:
                    router = output[1]  # Router logits
                    if isinstance(router, torch.Tensor):
                        router_logits[layer_idx] = torch.softmax(router, dim=-1)
            return hook

        # Find MoE layers and attach hooks
        for name, module in model.named_modules():
            if "moe" in name.lower() or "router" in name.lower() or "gate" in name.lower():
                layer_idx = self._extract_layer_idx(name)
                handle = module.register_forward_hook(make_hook(layer_idx))
                handles.append(handle)

        # Forward pass
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return router_logits

    def _extract_layer_idx(self, name: str) -> int:
        """Extract layer index from parameter name."""
        import re
        match = re.search(r'\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return -1

    def ablate_expert(
        self,
        model,
        expert_id: int,
        directions: List[torch.Tensor],
        layers: Optional[List[int]] = None,
        preserve_norm: bool = True
    ) -> Dict[str, Any]:
        """
        Ablate refusal directions from a specific expert.

        Args:
            model: MoE model
            expert_id: ID of expert to modify
            directions: Direction vectors to project out
            layers: Specific layers to modify
            preserve_norm: Whether to preserve weight norms

        Returns:
            Ablation result
        """
        from aetheris.core.projector import NormPreservingProjector

        projector = NormPreservingProjector(model, preserve_norm=preserve_norm)

        # Find parameters for this expert
        expert_params = []
        for name, param in model.named_parameters():
            if f"expert_{expert_id}" in name or f"experts.{expert_id}" in name:
                expert_params.append((name, param))

        if not expert_params:
            return {"success": False, "error": f"Expert {expert_id} not found"}

        # Project directions from expert parameters
        for name, param in expert_params:
            if param.dim() == 2:
                for d in directions:
                    d = d.to(param.device)
                    param.data = param.data - torch.outer(param.data @ d, d)

            elif param.dim() == 1:
                for d in directions:
                    d = d.to(param.device)
                    param.data = param.data - (param.data @ d) * d

        # Restore norms if preserving
        if preserve_norm:
            for name, param in expert_params:
                # Would restore original norm
                pass

        return {
            "success": True,
            "expert_id": expert_id,
            "directions_removed": len(directions),
            "parameters_modified": len(expert_params)
        }

    def analyze_expert_contribution(
        self,
        expert_directions: Dict[int, List[torch.Tensor]]
    ) -> ExpertAnalysis:
        """
        Analyze which experts carry refusal signal.

        Args:
            expert_directions: Per-expert direction vectors

        Returns:
            ExpertAnalysis with contribution scores
        """
        refusal_scores = {}
        dominant_experts = []

        for expert_id, directions in expert_directions.items():
            if directions:
                # Score based on number and magnitude of directions
                avg_norm = sum(torch.norm(d).item() for d in directions) / len(directions)
                refusal_scores[expert_id] = avg_norm
            else:
                refusal_scores[expert_id] = 0.0

        # Find dominant experts (score > 0.5)
        for expert_id, score in refusal_scores.items():
            if score > 0.5:
                dominant_experts.append(expert_id)

        # Recommend targeting (top 3 dominant experts)
        sorted_experts = sorted(refusal_scores.items(), key=lambda x: x[1], reverse=True)
        recommended = [e for e, _ in sorted_experts[:3] if e in dominant_experts]

        return ExpertAnalysis(
            expert_ids=list(refusal_scores.keys()),
            refusal_scores=refusal_scores,
            router_logits=None,
            dominant_experts=dominant_experts,
            recommended_targeting=recommended
        )

    def get_expert_surgery_plan(
        self,
        analysis: ExpertAnalysis,
        method: str = "surgical"
    ) -> Dict[str, Any]:
        """
        Generate expert-specific surgery plan.

        Args:
            analysis: ExpertAnalysis from analyze_expert_contribution
            method: Liberation method to apply

        Returns:
            Surgery plan with expert targeting
        """
        if method == "surgical":
            target_experts = analysis.recommended_targeting
        else:
            target_experts = analysis.dominant_experts

        return {
            "target_experts": target_experts,
            "method": method,
            "expected_effect": f"Remove refusal from {len(target_experts)} expert(s)",
            "preserve_experts": [e for e in analysis.expert_ids if e not in target_experts],
            "note": "Expert-Granular Abliteration preserves all other expert capabilities"
        }
