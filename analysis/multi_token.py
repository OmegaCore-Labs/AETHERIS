"""
Multi-Token Position Analyzer — Production-Grade Generation Analysis

Analyzes refusal across multi-token autoregressive generations. Tracks how
constraint directions evolve through generation and detects whether refusal
is triggered at specific token positions or accumulates gradually.

Based on: analysis of refusal dynamics across generated sequences.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from scipy.signal import find_peaks


@dataclass
class TokenReport:
    """Container for multi-token analysis."""
    position_scores: Dict[int, float]
    critical_positions: List[int]
    position_distribution: List[float]
    is_position_specific: bool
    # Extended fields
    accumulation_curve: Dict[int, float] = field(default_factory=dict)
    trigger_position: Optional[int] = None
    generation_breakdown: List[Dict[str, float]] = field(default_factory=list)
    constraint_drift: Dict[int, float] = field(default_factory=dict)
    entropy_trajectory: Dict[int, float] = field(default_factory=dict)
    gradient_magnitudes: Dict[int, float] = field(default_factory=dict)
    transition_points: List[int] = field(default_factory=list)
    accumulation_mode: str = "unknown"  # "trigger", "gradual", "oscillating"
    model_available: bool = True


class MultiTokenAnalyzer:
    """
    Analyze refusal across multi-token autoregressive generations.

    Tracks:
    - Per-position refusal signal strength
    - Gradual accumulation vs sudden trigger detection
    - Constraint direction drift across generation steps
    - Entropy trajectory (decision uncertainty per position)
    - Transition points where generation shifts toward refusal

    Can work with:
    1. Pre-computed activation sequences (per-position hidden states)
    2. Live model generation with hook-based activation collection
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_positions(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        detect_triggers: bool = True
    ) -> TokenReport:
        """
        Analyze token position contributions with trigger detection.

        Args:
            harmful_activations: (batch, seq_len, hidden_dim)
            harmless_activations: (batch, seq_len, hidden_dim)
            detect_triggers: Whether to detect trigger positions

        Returns:
            TokenReport with position analysis
        """
        if harmful_activations.dim() < 2 or harmless_activations.dim() < 2:
            return TokenReport(
                position_scores={}, critical_positions=[],
                position_distribution=[], is_position_specific=False,
                model_available=False
            )

        seq_len = min(harmful_activations.shape[1], harmless_activations.shape[1])

        # --- Per-position signal strength ---
        position_scores = {}
        position_directions = {}
        for pos in range(seq_len):
            harm_pos = harmful_activations[:, pos, :].float()
            harmless_pos = harmless_activations[:, pos, :].float()

            mean_harm = harm_pos.mean(dim=0)
            mean_harmless = harmless_pos.mean(dim=0)
            diff = mean_harm - mean_harmless

            score = torch.norm(diff).item()
            position_scores[pos] = score

            if score > 1e-8:
                position_directions[pos] = diff / score

        # --- Critical positions (top 20%) ---
        scores = list(position_scores.values())
        threshold = np.percentile(scores, 80) if scores else 0
        critical = [p for p, s in position_scores.items() if s > threshold]

        # --- Normalized distribution ---
        max_score = max(scores) if scores else 1.0
        distribution = [s / (max_score + 1e-8) for s in scores]

        # --- Position specificity ---
        if scores:
            concentration = max(scores) / (sum(scores) + 1e-8)
            is_specific = concentration > 0.3
        else:
            is_specific = False

        # --- Accumulation curve ---
        accumulation_curve = {}
        cumulative = 0.0
        total = sum(scores) if scores else 1.0
        for pos in sorted(position_scores.keys()):
            cumulative += position_scores[pos]
            accumulation_curve[pos] = cumulative / (total + 1e-8)

        # --- Trigger detection ---
        trigger_pos = None
        transition_points = []
        gradient_magnitudes = {}
        if detect_triggers and len(scores) > 2:
            scores_arr = np.array(scores)
            # First derivative
            gradient = np.gradient(scores_arr)
            grad_norm = np.abs(gradient)
            for i, g in enumerate(grad_norm):
                gradient_magnitudes[i] = float(g)

            # Find peaks in gradient
            if len(grad_norm) > 0:
                peak_threshold = np.mean(grad_norm) + 1.5 * np.std(grad_norm)
                peaks, props = find_peaks(grad_norm, height=peak_threshold)
                transition_points = peaks.tolist()

                if len(transition_points) > 0:
                    # Trigger is the first major transition
                    trigger_pos = transition_points[0]

        # --- Constraint direction drift ---
        constraint_drift = {}
        if position_directions and len(position_directions) > 1:
            first_dir = position_directions[min(position_directions.keys())]
            for pos in sorted(position_directions.keys()):
                cos_sim = float(torch.dot(position_directions[pos], first_dir))
                constraint_drift[pos] = float(np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi)

        # --- Entropy trajectory (approximate) ---
        entropy_trajectory = {}
        for pos in range(seq_len):
            harm_pos = harmful_activations[:, pos, :].float()
            # Approximate entropy from activation variance
            var = torch.var(harm_pos, dim=0).mean().item()
            # Map variance to entropy-like measure
            entropy_trajectory[pos] = float(np.log(var + 1e-8))

        # --- Accumulation mode ---
        accumulation_mode = self._detect_accumulation_mode(
            scores, accumulation_curve, transition_points
        )

        return TokenReport(
            position_scores=position_scores,
            critical_positions=critical,
            position_distribution=distribution,
            is_position_specific=is_specific,
            accumulation_curve=accumulation_curve,
            trigger_position=trigger_pos,
            constraint_drift=constraint_drift,
            entropy_trajectory=entropy_trajectory,
            gradient_magnitudes=gradient_magnitudes,
            transition_points=transition_points,
            accumulation_mode=accumulation_mode,
            model_available=True
        )

    def analyze_generation(
        self,
        model,
        tokenizer,
        harmful_prompt: str,
        max_new_tokens: int = 50,
        refusal_tokens: Optional[List[str]] = None
    ) -> TokenReport:
        """
        Analyze refusal across an autoregressive generation.

        Tracks how constraint directions evolve during generation.
        Collects hidden states at each generation step.

        Args:
            model: HuggingFace causal LM
            tokenizer: Associated tokenizer
            harmful_prompt: Prompt expected to trigger refusal
            max_new_tokens: Maximum tokens to generate
            refusal_tokens: Refusal-indicating tokens

        Returns:
            TokenReport with per-generation-step analysis
        """
        if refusal_tokens is None:
            refusal_tokens = ["I", "cannot", "sorry", "apologize", "can't", "unable"]

        try:
            return self._real_generation_analysis(
                model, tokenizer, harmful_prompt, max_new_tokens, refusal_tokens
            )
        except Exception:
            return TokenReport(
                position_scores={}, critical_positions=[],
                position_distribution=[], is_position_specific=False,
                model_available=False
            )

    def _real_generation_analysis(
        self,
        model,
        tokenizer,
        harmful_prompt: str,
        max_new_tokens: int,
        refusal_tokens: List[str]
    ) -> TokenReport:
        """Analyze refusal during real autoregressive generation."""
        inputs = tokenizer(harmful_prompt, return_tensors="pt",
                          truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[1]

        refusal_ids = set()
        for t in refusal_tokens:
            ids = tokenizer.encode(f" {t}", add_special_tokens=False)
            if ids:
                refusal_ids.add(ids[0])

        # Storage for per-step data
        position_scores = {}
        constraint_drift = {}
        entropy_trajectory = {}
        generation_breakdown = []
        first_direction = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                output = model(**inputs, output_hidden_states=True)

                # Get last layer hidden state
                last_hidden = output.hidden_states[-1][0, -1, :]

                # Logits and probabilities at this step
                logits = output.logits[0, -1, :].float()
                probs = torch.softmax(logits, dim=-1)

                # Refusal probability at this step
                refusal_prob = sum(float(probs[tid]) for tid in refusal_ids)
                position_scores[step] = refusal_prob

                # Entropy
                entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
                entropy_trajectory[step] = entropy

                # Constraint direction drift
                if first_direction is None:
                    first_direction = last_hidden.clone()
                else:
                    cos_sim = float(torch.dot(last_hidden, first_direction) /
                                   (torch.norm(last_hidden) * torch.norm(first_direction) + 1e-8))
                    constraint_drift[step] = float(np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi)

                # Top tokens at this step
                top_k = 5
                top_vals, top_ids = torch.topk(probs, k=top_k)
                top_info = {
                    f"token_{i}": {
                        "text": tokenizer.decode([int(tid)]),
                        "prob": float(val)
                    }
                    for i, (tid, val) in enumerate(zip(top_ids, top_vals))
                }
                generation_breakdown.append(top_info)

                # Sample next token
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token.unsqueeze(0)], dim=1)

                # Update attention mask
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = torch.cat([
                        inputs['attention_mask'],
                        torch.ones(1, 1, device=self.device, dtype=inputs['attention_mask'].dtype)
                    ], dim=1)

                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

        # --- Post-generation analysis ---
        scores = list(position_scores.values())
        critical_positions = []
        if scores:
            threshold = np.percentile(scores, 80)
            critical_positions = [p for p, s in position_scores.items() if s > threshold]

        distribution = [s / max(scores) for s in scores] if scores else []

        is_specific = False
        if scores:
            concentration = max(scores) / (sum(scores) + 1e-8)
            is_specific = concentration > 0.3

        # Accumulation curve
        accumulation_curve = {}
        cumulative = 0.0
        total = sum(scores) if scores else 1.0
        for pos in sorted(position_scores.keys()):
            cumulative += position_scores[pos]
            accumulation_curve[pos] = cumulative / (total + 1e-8)

        return TokenReport(
            position_scores=position_scores,
            critical_positions=critical_positions,
            position_distribution=distribution,
            is_position_specific=is_specific,
            accumulation_curve=accumulation_curve,
            trigger_position=critical_positions[0] if critical_positions else None,
            generation_breakdown=generation_breakdown,
            constraint_drift=constraint_drift,
            entropy_trajectory=entropy_trajectory,
            model_available=True
        )

    def _detect_accumulation_mode(
        self,
        scores: List[float],
        accumulation_curve: Dict[int, float],
        transition_points: List[int]
    ) -> str:
        """Detect how refusal accumulates: trigger, gradual, or oscillating."""
        if not scores:
            return "unknown"

        if len(transition_points) >= 1 and len(transition_points) <= 2:
            # Check if first transition explains most of the signal
            if accumulation_curve and min(accumulation_curve.keys()) in transition_points:
                return "trigger"
            elif transition_points:
                first_transition = transition_points[0]
                if first_transition in accumulation_curve:
                    if accumulation_curve[first_transition] > 0.5:
                        return "trigger"

        if len(transition_points) > 3:
            return "oscillating"

        # Check gradualness: variance of step-to-step changes
        if len(scores) > 2:
            diffs = np.diff(np.array(scores))
            cv = np.std(np.abs(diffs)) / (np.mean(np.abs(diffs)) + 1e-8)
            if cv < 0.5:
                return "gradual"

        return "gradual"

    def find_trigger_positions(
        self,
        token_report: TokenReport,
        threshold: float = 0.7
    ) -> List[int]:
        """Find positions above normalized threshold."""
        norm_scores = token_report.position_distribution
        return [i for i, s in enumerate(norm_scores) if s > threshold]

    def compute_early_late_ratio(
        self,
        token_report: TokenReport,
        split_ratio: float = 0.5
    ) -> float:
        """Compute ratio of early to late token signal."""
        positions = sorted(token_report.position_scores.keys())
        n = len(positions)
        if n == 0:
            return 1.0

        split = int(n * split_ratio)
        early_sum = sum(token_report.position_scores[p] for p in positions[:split])
        late_sum = sum(token_report.position_scores[p] for p in positions[split:])

        return early_sum / (late_sum + 1e-8)
