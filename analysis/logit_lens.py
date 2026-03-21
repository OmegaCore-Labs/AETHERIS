"""
Refusal Logit Lens

Locates the layer where the model "decides" to refuse by analyzing
logit differences across layers.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LogitLensReport:
    """Container for logit lens analysis."""
    decision_layer: int
    refusal_tokens: List[str]
    refusal_probability_curve: Dict[int, float]
    early_refusal_layers: List[int]
    late_decision_layers: List[int]


class RefusalLogitLens:
    """
    Locate where refusal decisions occur using logit lens.

    Based on nostalgebraist (2020) logit lens technique.
    Analyzes how the probability of refusal tokens evolves across layers.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze(
        self,
        model,
        tokenizer,
        harmful_prompt: str,
        refusal_tokens: Optional[List[str]] = None
    ) -> LogitLensReport:
        """
        Analyze refusal decision layer.

        Args:
            model: HuggingFace model
            tokenizer: Associated tokenizer
            harmful_prompt: Prompt that should trigger refusal
            refusal_tokens: Tokens indicating refusal (auto-detected if None)

        Returns:
            LogitLensReport with decision layer
        """
        if refusal_tokens is None:
            refusal_tokens = ["I", "cannot", "unable", "sorry", "apologize", "can't"]

        # Tokenize prompt
        inputs = tokenizer(harmful_prompt, return_tensors="pt").to(self.device)

        # Forward pass with output_hidden_states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # Tuple of (n_layers+1, batch, seq, hidden)

        # Get refusal token IDs
        refusal_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in refusal_tokens if t in tokenizer.vocab]

        # Track refusal probability across layers
        refusal_probs = {}

        for layer_idx, hidden in enumerate(hidden_states):
            # Apply LM head to get logits
            logits = model.lm_head(hidden)  # (batch, seq, vocab)
            last_token_logits = logits[0, -1, :]  # Last token position

            # Compute probabilities
            probs = torch.softmax(last_token_logits, dim=-1)

            # Sum probability of refusal tokens
            refusal_prob = sum(probs[token_id].item() for token_id in refusal_ids)
            refusal_probs[layer_idx] = refusal_prob

        # Find decision layer (where refusal probability peaks)
        decision_layer = max(refusal_probs, key=refusal_probs.get)

        # Find early refusal layers (first where prob > 0.3)
        early_refusal = [l for l, p in refusal_probs.items() if p > 0.3][:3]

        # Find late decision layers (where prob jumps significantly)
        late_decision = []
        prev_prob = 0
        for layer, prob in sorted(refusal_probs.items()):
            if prob - prev_prob > 0.2:
                late_decision.append(layer)
            prev_prob = prob

        return LogitLensReport(
            decision_layer=decision_layer,
            refusal_tokens=refusal_tokens,
            refusal_probability_curve=refusal_probs,
            early_refusal_layers=early_refusal,
            late_decision_layers=late_decision
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
