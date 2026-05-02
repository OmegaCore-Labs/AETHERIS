"""
Refusal Logit Lens — Production-Grade Logit Lens Analysis

Projects hidden states at each layer to vocabulary space using the
unembedding matrix (lm_head), tracking refusal token probability
progression across layers to locate the "decision layer."

Based on: nostalgebraist (2020), Belrose et al. (2023)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


# Known refusal tokens across common models
DEFAULT_REFUSAL_TOKENS = [
    "I", "cannot", "can't", "unable", "sorry", "apologize",
    "not", "won't", "don't", "illegal", "unethical", "harmful",
    "decline", "against", "violates"
]

# Compliance tokens for contrast
DEFAULT_COMPLIANCE_TOKENS = [
    "Sure", "Yes", "Here", "Certainly", "Of", "Let", "I'll"
]


@dataclass
class LogitLensReport:
    """Container for logit lens analysis."""
    decision_layer: int
    refusal_tokens: List[str]
    refusal_probability_curve: Dict[int, float]
    early_refusal_layers: List[int]
    late_decision_layers: List[int]
    # Extended fields
    token_probability_curves: Dict[str, Dict[int, float]] = field(default_factory=dict)
    compliance_probability_curve: Dict[int, float] = field(default_factory=dict)
    refusal_compliance_ratio: Dict[int, float] = field(default_factory=dict)
    entropy_curve: Dict[int, float] = field(default_factory=dict)
    top_tokens_per_layer: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)
    inflection_point: Optional[int] = None  # Layer where refusal probability inflects
    confidence_interval: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    model_available: bool = True


class RefusalLogitLens:
    """
    Apply logit lens to locate the refusal decision layer.

    Projects hidden states through the unembedding (lm_head) matrix at each
    layer and tracks how refusal token probabilities evolve. The "decision
    layer" is where refusal tokens first become dominant.

    Features:
    - Full token probability tracking per layer
    - Refusal vs compliance contrast tracking
    - Entropy monitoring (sharpens at decision)
    - Inflection point detection via second differences
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze(
        self,
        model,
        tokenizer,
        harmful_prompt: str,
        refusal_tokens: Optional[List[str]] = None,
        compliance_tokens: Optional[List[str]] = None,
        track_top_k: int = 5
    ) -> LogitLensReport:
        """
        Analyze refusal decision layer via logit lens.

        Args:
            model: HuggingFace causal LM with output_hidden_states support
            tokenizer: Associated tokenizer
            harmful_prompt: Prompt expected to trigger refusal
            refusal_tokens: Tokens indicating refusal (auto-detected if None)
            compliance_tokens: Tokens indicating compliance
            track_top_k: Number of top tokens to track per layer

        Returns:
            LogitLensReport with decision layer and probability curves
        """
        if refusal_tokens is None:
            refusal_tokens = DEFAULT_REFUSAL_TOKENS
        if compliance_tokens is None:
            compliance_tokens = DEFAULT_COMPLIANCE_TOKENS

        model_available = True

        try:
            # Tokenize
            inputs = tokenizer(harmful_prompt, return_tensors="pt",
                              truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass with hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()

            if lm_head is None:
                return self._no_model_report(refusal_tokens, "No lm_head available")

            # Get refusal and compliance token IDs
            refusal_ids = set()
            for t in refusal_tokens:
                ids = tokenizer.encode(f" {t}", add_special_tokens=False)
                if ids:
                    refusal_ids.add(ids[0])
                # Also try without space prefix
                ids2 = tokenizer.encode(t, add_special_tokens=False)
                if ids2 and len(ids2) == 1:
                    refusal_ids.add(ids2[0])

            compliance_ids = set()
            for t in compliance_tokens:
                ids = tokenizer.encode(f" {t}", add_special_tokens=False)
                if ids:
                    compliance_ids.add(ids[0])

        except Exception as e:
            return self._no_model_report(
                refusal_tokens, f"Model forward pass failed: {e}"
            )

        # Track probabilities across layers
        refusal_probs = {}
        compliance_probs = {}
        ratio_curve = {}
        entropy_curve = {}
        top_tokens = {}
        token_probs = {t: {} for t in refusal_tokens}

        for layer_idx, hidden in enumerate(hidden_states):
            # Apply lm_head to get logits
            logits = lm_head(hidden)  # (batch, seq, vocab)
            last_logits = logits[0, -1, :]  # Last position logits

            probs = torch.softmax(last_logits.float(), dim=-1)

            # Refusal probability
            r_prob = sum(float(probs[tid]) for tid in refusal_ids)
            refusal_probs[layer_idx] = r_prob

            # Compliance probability
            c_prob = sum(float(probs[tid]) for tid in compliance_ids)
            compliance_probs[layer_idx] = c_prob

            # Ratio
            ratio_curve[layer_idx] = r_prob / (c_prob + 1e-8)

            # Entropy
            entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
            entropy_curve[layer_idx] = entropy

            # Top-k tokens
            top_k_vals, top_k_indices = torch.topk(probs, k=track_top_k)
            top_tokens[layer_idx] = [
                (tokenizer.decode([int(idx)]), float(val))
                for idx, val in zip(top_k_indices, top_k_vals)
            ]

            # Per-token probabilities
            for t, tid_set in [(t, {tokenizer.encode(f" {t}", add_special_tokens=False)[0]})
                              for t in refusal_tokens
                              if tokenizer.encode(f" {t}", add_special_tokens=False)]:
                tid = list(tid_set)[0]
                token_probs[t][layer_idx] = float(probs[tid])

        # Find decision layer using multiple strategies
        decision_layer = self._find_decision_layer(refusal_probs, ratio_curve, entropy_curve)

        # Early refusal layers (where prob first exceeds 0.3)
        early_refusal = sorted([l for l, p in refusal_probs.items() if p > 0.3])

        # Late decision layers (where prob jumps significantly)
        late_decision = self._find_probability_jumps(refusal_probs)

        # Inflection point (maximum second derivative)
        inflection_point = self._find_inflection(refusal_probs)

        # Confidence intervals via bootstrapping over last position context
        confidence = self._estimate_confidence(hidden_states, lm_head, refusal_ids, len(hidden_states))

        return LogitLensReport(
            decision_layer=decision_layer,
            refusal_tokens=refusal_tokens,
            refusal_probability_curve=refusal_probs,
            early_refusal_layers=early_refusal,
            late_decision_layers=late_decision,
            token_probability_curves=token_probs,
            compliance_probability_curve=compliance_probs,
            refusal_compliance_ratio=ratio_curve,
            entropy_curve=entropy_curve,
            top_tokens_per_layer=top_tokens,
            inflection_point=inflection_point,
            confidence_interval=confidence,
            model_available=model_available
        )

    def _find_decision_layer(
        self,
        refusal_probs: Dict[int, float],
        ratio_curve: Dict[int, float],
        entropy_curve: Dict[int, float]
    ) -> int:
        """Find decision layer using combined criteria."""
        if not refusal_probs:
            return -1

        # Strategy 1: Where refusal probability peaks
        prob_peak = max(refusal_probs, key=refusal_probs.get)

        # Strategy 2: Where ratio of refusal/compliance peaks
        if ratio_curve:
            ratio_peak = max(ratio_curve, key=ratio_curve.get)
        else:
            ratio_peak = prob_peak

        # Strategy 3: Where entropy drops sharply (decision sharpening)
        if len(entropy_curve) > 1:
            sorted_layers = sorted(entropy_curve.keys())
            entropy_drops = {}
            for i in range(1, len(sorted_layers)):
                drop = entropy_curve[sorted_layers[i - 1]] - entropy_curve[sorted_layers[i]]
                entropy_drops[sorted_layers[i]] = drop
            if entropy_drops:
                entropy_peak = max(entropy_drops, key=entropy_drops.get)
            else:
                entropy_peak = prob_peak
        else:
            entropy_peak = prob_peak

        # Combine: weighted average of layer indices
        candidate_layers = [prob_peak, ratio_peak, entropy_peak]
        # Return mode (most common) or median
        from collections import Counter
        counter = Counter(candidate_layers)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else prob_peak

    def _find_probability_jumps(
        self, refusal_probs: Dict[int, float], threshold: float = 0.2
    ) -> List[int]:
        """Find layers where refusal probability jumps significantly."""
        layers = sorted(refusal_probs.keys())
        jumps = []
        for i in range(1, len(layers)):
            jump = refusal_probs[layers[i]] - refusal_probs[layers[i - 1]]
            if jump > threshold:
                jumps.append(layers[i])
        return jumps

    def _find_inflection(self, refusal_probs: Dict[int, float]) -> Optional[int]:
        """Find inflection point via maximum second derivative."""
        layers = sorted(refusal_probs.keys())
        if len(layers) < 3:
            return None

        probs = np.array([refusal_probs[l] for l in layers])
        second_diff = np.diff(probs, n=2)
        if len(second_diff) == 0:
            return None

        inflection_idx = np.argmax(np.abs(second_diff))
        return layers[inflection_idx + 1]

    def _estimate_confidence(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        lm_head,
        refusal_ids: Set[int],
        n_layers: int,
        n_bootstrap: int = 100
    ) -> Dict[int, Tuple[float, float]]:
        """
        Estimate confidence intervals via bootstrapping over sequence positions.
        Uses the last few token positions as bootstrap samples.
        """
        confidence = {}
        try:
            last_hidden = hidden_states[-1]  # Use last layer to determine seq_len
            seq_len = last_hidden.shape[1]
            positions = list(range(max(0, seq_len - 10), seq_len))

            if len(positions) < 2:
                return confidence

            for layer_idx, hidden in enumerate(hidden_states):
                layer_probs = []
                for pos in positions:
                    logits = lm_head(hidden[:, pos:pos + 1, :])
                    probs = torch.softmax(logits[0, 0].float(), dim=-1)
                    r_prob = sum(float(probs[tid]) for tid in refusal_ids)
                    layer_probs.append(r_prob)

                if layer_probs:
                    arr = np.array(layer_probs)
                    confidence[layer_idx] = (
                        float(np.percentile(arr, 5)),
                        float(np.percentile(arr, 95))
                    )
        except Exception:
            pass

        return confidence

    def _no_model_report(self, refusal_tokens: List[str], reason: str) -> LogitLensReport:
        """Return a placeholder report when model is unavailable."""
        return LogitLensReport(
            decision_layer=-1,
            refusal_tokens=refusal_tokens,
            refusal_probability_curve={},
            early_refusal_layers=[],
            late_decision_layers=[],
            model_available=False,
            confidence_interval={},
            top_tokens_per_layer={0: [(reason, 1.0)]}
        )

    def compute_decision_heatmap(
        self,
        model,
        tokenizer,
        prompts: List[str],
        refusal_tokens: Optional[List[str]] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute refusal decision heatmap across multiple prompts.

        Returns dictionary mapping prompt to per-layer refusal probability.
        """
        results = {}
        for prompt in prompts:
            report = self.analyze(model, tokenizer, prompt, refusal_tokens)
            results[prompt] = report.refusal_probability_curve
        return results
