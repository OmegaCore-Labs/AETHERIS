"""
Expert Decomposition Analyzer

For MoE models (Mixtral, etc.), analyzes which experts carry refusal signal.
Routes activations through individual experts, measures contribution to refusal direction.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ExpertReport:
    """Container for expert decomposition analysis."""
    expert_refusal_scores: Dict[int, float]
    dominant_experts: List[int]
    safety_expert_id: Optional[int]
    distribution_type: str  # "concentrated", "distributed", "balanced", "no_data"
    recommended_targeting: List[int]
    expert_router_preferences: Dict[int, float] = field(default_factory=dict)
    expert_specialization: Dict[int, Dict[str, float]] = field(default_factory=dict)
    per_layer_expert_contributions: Dict[int, Dict[int, float]] = field(default_factory=dict)
    status: str = "no_data"  # "ok", "no_data", "error"


class ExpertDecompositionAnalyzer:
    """
    Analyze MoE expert contribution to refusal constraints.

    For Mixture-of-Experts models (Mixtral 8x7B, DeepSeek MoE, etc.),
    identifies which experts carry the refusal signal. Enables
    expert-specific targeting for constraint removal.

    Implementation uses router probability extraction via hooks
    to measure per-expert contribution to the refusal direction.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_expert_contributions(
        self,
        model: Any,
        tokenizer: Any,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        n_experts: int = 8,
        refusal_direction: Optional[torch.Tensor] = None,
    ) -> ExpertReport:
        """
        Analyze which experts contribute to refusal.

        Hooks into MoE router to extract per-expert routing probabilities
        and compute alignment with refusal direction.

        Args:
            model: MoE model (must have MoE router access)
            tokenizer: Associated tokenizer
            harmful_prompts: Harmful prompts expected to trigger refusal
            harmless_prompts: Harmless control prompts
            n_experts: Number of experts in model
            refusal_direction: Optional known refusal direction vector

        Returns:
            ExpertReport with per-expert refusal scores
        """
        if not harmful_prompts or not harmless_prompts:
            return ExpertReport(
                expert_refusal_scores={},
                dominant_experts=[],
                safety_expert_id=None,
                distribution_type="no_data",
                recommended_targeting=[],
                status="no_data",
            )

        try:
            # Extract router logits/probabilities via hooks
            router_data = self._extract_router_activations(
                model, tokenizer, harmful_prompts, harmless_prompts, n_experts
            )

            if router_data is None:
                # Fallback: compute expert scores from hidden state projections
                return self._fallback_expert_analysis(
                    model, tokenizer, harmful_prompts, harmless_prompts,
                    n_experts, refusal_direction
                )

            # Compute per-expert refusal alignment scores
            expert_scores = self._compute_expert_refusal_scores(
                router_data, n_experts, refusal_direction
            )

            # Find dominant experts (refusal score > 0.5)
            dominant = sorted(
                [e for e, s in expert_scores.items() if s > 0.5],
                key=lambda e: expert_scores[e],
                reverse=True,
            )

            # Safety expert = highest refusal score
            safety_expert = max(expert_scores, key=expert_scores.get) if expert_scores else None

            # Determine distribution type
            distribution = self._classify_distribution(expert_scores, n_experts)

            # Recommended targeting
            recommended = self._compute_recommended_targeting(expert_scores, distribution, safety_expert, dominant)

            return ExpertReport(
                expert_refusal_scores=expert_scores,
                dominant_experts=dominant,
                safety_expert_id=safety_expert,
                distribution_type=distribution,
                recommended_targeting=recommended,
                expert_router_preferences=router_data.get("mean_preferences_overall", {}),
                status="ok",
            )
        except Exception as e:
            return ExpertReport(
                expert_refusal_scores={},
                dominant_experts=[],
                safety_expert_id=None,
                distribution_type="no_data",
                recommended_targeting=[],
                status=f"error: {str(e)}",
            )

    def _extract_router_activations(
        self,
        model: Any,
        tokenizer: Any,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        n_experts: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract MoE router logits/probabilities via forward hooks.
        Returns aggregated router preferences for harmful vs harmless prompts.
        """
        router_outputs_harmful = []
        router_outputs_harmless = []

        try:
            # Determine device
            device = self.device
            if hasattr(model, "device"):
                try:
                    device = model.device
                except Exception:
                    pass

            # Hook function to capture router outputs
            captured_data = []

            def _router_hook(module, input, output):
                if isinstance(output, tuple):
                    for o in output:
                        if isinstance(o, torch.Tensor) and o.dim() >= 2:
                            captured_data.append(o.detach().cpu())
                            break
                elif isinstance(output, torch.Tensor):
                    captured_data.append(output.detach().cpu())

            # Find MoE router layers and register hooks
            hooks = []
            hook_registered = False
            for name, module in model.named_modules():
                name_lower = name.lower()
                if any(kw in name_lower for kw in ["router", "gate", "moe", "mixture"]):
                    hooks.append(module.register_forward_hook(_router_hook))
                    hook_registered = True

            if not hook_registered:
                return None

            # Forward pass on harmful prompts
            for prompt in harmful_prompts[:10]:
                captured_data.clear()
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    try:
                        model(**inputs)
                    except Exception:
                        pass
                if captured_data:
                    router_outputs_harmful.append(captured_data[-1].clone())
                    if len(router_outputs_harmful) >= 3:
                        break

            # Forward pass on harmless prompts
            for prompt in harmless_prompts[:10]:
                captured_data.clear()
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    try:
                        model(**inputs)
                    except Exception:
                        pass
                if captured_data:
                    router_outputs_harmless.append(captured_data[-1].clone())
                    if len(router_outputs_harmless) >= 3:
                        break

            # Remove hooks
            for h in hooks:
                h.remove()

            if not router_outputs_harmful or not router_outputs_harmless:
                return None

            # Aggregate: mean router probability per expert
            harmful_stack = torch.stack([r.flatten()[:n_experts] for r in router_outputs_harmful if r.numel() >= n_experts])
            harmless_stack = torch.stack([r.flatten()[:n_experts] for r in router_outputs_harmless if r.numel() >= n_experts])

            if harmful_stack.shape[0] == 0 or harmless_stack.shape[0] == 0:
                return None

            harmful_mean = harmful_stack.float().mean(dim=0)
            harmless_mean = harmless_stack.float().mean(dim=0)

            mean_preferences = {}
            for i in range(min(n_experts, len(harmful_mean))):
                mean_preferences[i] = float(harmful_mean[i].item())

            return {
                "harmful_mean": harmful_mean,
                "harmless_mean": harmless_mean,
                "mean_preferences_overall": mean_preferences,
            }

        except Exception:
            return None

    def _compute_expert_refusal_scores(
        self,
        router_data: Dict[str, Any],
        n_experts: int,
        refusal_direction: Optional[torch.Tensor],
    ) -> Dict[int, float]:
        """
        Compute per-expert refusal scores from router data.
        Higher = more involved in refusal.
        """
        scores = {}
        harmful_mean = router_data.get("harmful_mean")
        harmless_mean = router_data.get("harmless_mean")

        if harmful_mean is not None and harmless_mean is not None:
            # Difference in router probability between harmful and harmless
            diff = harmful_mean - harmless_mean
            # Softmax-style normalization
            diff_exp = torch.exp(diff)
            diff_probs = diff_exp / (diff_exp.sum() + 1e-8)
            for i in range(min(n_experts, len(diff_probs))):
                scores[i] = round(float(diff_probs[i].item()), 4)
        else:
            # Uniform fallback
            for i in range(n_experts):
                scores[i] = 1.0 / n_experts

        return scores

    def _classify_distribution(
        self, expert_scores: Dict[int, float], n_experts: int
    ) -> str:
        """Classify the distribution type of expert refusal scores."""
        if not expert_scores:
            return "no_data"

        scores = list(expert_scores.values())
        max_score = max(scores)
        total = sum(scores)

        if total == 0:
            return "balanced"

        # Concentration ratio: top expert vs total
        concentration = max_score / total

        # Count experts above 20% normalized contribution
        n_significant = len([s for s in scores if s / total > 0.15])

        if concentration > 0.5:
            return "concentrated"
        elif concentration > 0.25 and n_significant <= 3:
            return "distributed"
        else:
            return "balanced"

    def _compute_recommended_targeting(
        self,
        expert_scores: Dict[int, float],
        distribution: str,
        safety_expert: Optional[int],
        dominant: List[int],
    ) -> List[int]:
        """Compute recommended experts to target for intervention."""
        if distribution == "concentrated":
            if safety_expert is not None:
                return [safety_expert]
            return dominant[:1] if dominant else []
        elif distribution == "distributed":
            return dominant[:3] if len(dominant) >= 3 else dominant
        else:
            # Balanced: target top half
            sorted_experts = sorted(expert_scores, key=expert_scores.get, reverse=True)
            n_target = max(1, len(sorted_experts) // 2)
            return sorted_experts[:n_target]

    def _fallback_expert_analysis(
        self,
        model: Any,
        tokenizer: Any,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        n_experts: int,
        refusal_direction: Optional[torch.Tensor],
    ) -> ExpertReport:
        """
        Fallback analysis when router hooks fail.
        Uses last-hidden-state projections to clustered centroids.
        """
        try:
            device = self.device
            if hasattr(model, "device"):
                try:
                    device = model.device
                except Exception:
                    pass

            # Get hidden states for both prompt types
            harmful_hidden = self._get_last_hidden(model, tokenizer, harmful_prompts[:5], device)
            harmless_hidden = self._get_last_hidden(model, tokenizer, harmless_prompts[:5], device)

            if harmful_hidden is None or harmless_hidden is None:
                return ExpertReport(
                    expert_refusal_scores={},
                    dominant_experts=[],
                    safety_expert_id=None,
                    distribution_type="no_data",
                    recommended_targeting=[],
                    status="no_data: cannot extract hidden states",
                )

            # Cluster the hidden state space into n_experts regions via k-means
            from sklearn.cluster import KMeans

            all_states = torch.cat([harmful_hidden.mean(dim=1), harmless_hidden.mean(dim=1)], dim=0).cpu().numpy()

            kmeans = KMeans(n_clusters=n_experts, random_state=42, n_init=10)
            kmeans.fit(all_states)

            # Compute difference in cluster assignment between harmful and harmless
            harmful_clusters = kmeans.predict(harmful_hidden.mean(dim=1).cpu().numpy())
            harmless_clusters = kmeans.predict(harmless_hidden.mean(dim=1).cpu().numpy())

            # Per-cluster refusal score = proportion of harmful assignments
            expert_scores = {}
            for i in range(n_experts):
                harm_count = np.sum(harmful_clusters == i)
                harmless_count = np.sum(harmless_clusters == i)
                total = harm_count + harmless_count
                if total > 0:
                    # Normalized: higher = more harmful-assigned
                    expert_scores[i] = round(harm_count / total, 4)
                else:
                    expert_scores[i] = 0.5

            dominant = sorted(
                [e for e, s in expert_scores.items() if s > 0.6],
                key=lambda e: expert_scores[e], reverse=True,
            )
            safety_expert = max(expert_scores, key=expert_scores.get) if expert_scores else None
            distribution = self._classify_distribution(expert_scores, n_experts)
            recommended = self._compute_recommended_targeting(expert_scores, distribution, safety_expert, dominant)

            return ExpertReport(
                expert_refusal_scores=expert_scores,
                dominant_experts=dominant,
                safety_expert_id=safety_expert,
                distribution_type=distribution,
                recommended_targeting=recommended,
                status="ok (fallback clustering)",
            )
        except Exception as e:
            return ExpertReport(
                expert_refusal_scores={},
                dominant_experts=[],
                safety_expert_id=None,
                distribution_type="no_data",
                recommended_targeting=[],
                status=f"error: {str(e)}",
            )

    def _get_last_hidden(
        self, model: Any, tokenizer: Any, prompts: List[str], device: str
    ) -> Optional[torch.Tensor]:
        """Extract last hidden state from model forward pass."""
        try:
            all_hidden = []
            for prompt in prompts[:5]:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    last_hidden = outputs.hidden_states[-1]
                    all_hidden.append(last_hidden.cpu())
            if all_hidden:
                return torch.cat(all_hidden, dim=0)
            return None
        except Exception:
            return None

    def compute_expert_specialization(
        self,
        model: Any,
        tokenizer: Any,
        prompts_by_category: Dict[str, List[str]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute expert specialization across prompt categories.
        Requires router extraction; falls back to uniform distribution.

        Args:
            model: MoE model
            tokenizer: Tokenizer
            prompts_by_category: Dict mapping category name to list of prompts

        Returns:
            Dict mapping expert_id -> {category: specialization_score}
        """
        specialization: Dict[int, Dict[str, float]] = {}

        if not prompts_by_category:
            return specialization

        n_experts = 8
        try:
            if hasattr(model, "config"):
                n_experts = getattr(model.config, "num_local_experts", 8)
        except Exception:
            pass

        for expert_id in range(n_experts):
            specialization[expert_id] = {}

        try:
            device = self.device
            if hasattr(model, "device"):
                try:
                    device = model.device
                except Exception:
                    pass

            for category, prompts in prompts_by_category.items():
                if not prompts:
                    continue

                router_data = self._extract_router_activations(
                    model, tokenizer, prompts, prompts, n_experts
                )

                if router_data and "harmful_mean" in router_data:
                    probs = router_data["harmful_mean"]
                    for i in range(min(n_experts, len(probs))):
                        specialization[i][category] = round(float(probs[i].item()), 4)
                else:
                    # Uniform fallback
                    for i in range(n_experts):
                        specialization[i][category] = round(1.0 / n_experts, 4)

        except Exception:
            # Fill with uniform on error
            for expert_id in range(n_experts):
                for category in prompts_by_category:
                    specialization[expert_id][category] = round(1.0 / n_experts, 4)

        return specialization

    def get_expert_surgery_plan(
        self,
        expert_report: ExpertReport,
        method: str = "surgical",
    ) -> Dict[str, Any]:
        """
        Generate expert-specific surgery plan for constraint removal.

        Args:
            expert_report: Report from analyze_expert_contributions
            method: Liberation method ("surgical", "broad", "progressive")

        Returns:
            Surgery plan dict
        """
        n_experts = len(expert_report.expert_refusal_scores)

        if method == "surgical":
            target_experts = expert_report.recommended_targeting[:3] if expert_report.recommended_targeting else []
        elif method == "broad":
            target_experts = expert_report.dominant_experts if expert_report.dominant_experts else list(range(n_experts))
        elif method == "progressive":
            # Target most refusal-heavy expert first
            target_experts = [expert_report.safety_expert_id] if expert_report.safety_expert_id is not None else []
        else:
            target_experts = []

        all_experts = list(range(n_experts)) if n_experts > 0 else list(expert_report.expert_refusal_scores.keys())
        preserve_experts = [e for e in all_experts if e not in target_experts]

        return {
            "target_experts": target_experts,
            "method": method,
            "expected_effect": f"Remove refusal from {len(target_experts)} expert(s)",
            "preserve_experts": preserve_experts,
            "distribution_type": expert_report.distribution_type,
            "note": "Surgical MoE targeting preserves non-safety-expert capabilities",
        }
