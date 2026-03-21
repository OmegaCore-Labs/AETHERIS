"""
Capability Validation Suite

Validates that model capabilities are preserved after constraint removal.
Measures perplexity, coherence, KL divergence, and effective rank.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import entropy


@dataclass
class ValidationReport:
    """Container for validation results."""
    perplexity_before: float
    perplexity_after: float
    perplexity_delta: float                     # Percent change
    coherence_before: float
    coherence_after: float
    coherence_delta: float
    kl_divergence: float                        # Between before and after distributions
    effective_rank_before: float
    effective_rank_after: float
    rank_preservation: float                    # Ratio of after/before rank
    passed: bool
    warnings: List[str]
    metadata: Dict[str, Any]


class CapabilityValidator:
    """
    Validate that model capabilities are preserved after modification.

    Implements:
    - Perplexity measurement
    - Coherence scoring
    - KL divergence between original and modified output distributions
    - Effective rank of weight matrices
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def validate(
        self,
        original_model,
        modified_model,
        tokenizer,
        test_texts: Optional[List[str]] = None,
        threshold_perplexity: float = 0.15,    # 15% increase allowed
        threshold_coherence: float = 0.10,      # 10% decrease allowed
        threshold_kl: float = 0.5              # KL divergence threshold
    ) -> ValidationReport:
        """
        Run complete validation suite.

        Args:
            original_model: Original model before modification
            modified_model: Modified model after removal
            tokenizer: Tokenizer for both models
            test_texts: Texts for perplexity evaluation
            threshold_perplexity: Max allowed perplexity increase
            threshold_coherence: Max allowed coherence decrease
            threshold_kl: Max allowed KL divergence

        Returns:
            ValidationReport with all metrics
        """
        warnings = []

        # Default test texts if not provided
        if test_texts is None:
            test_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a fascinating field of study.",
                "In the beginning, there was nothing but darkness.",
                "The theory of relativity revolutionized physics.",
                "To be or not to be, that is the question."
            ]

        # Measure perplexity
        ppl_before = self.compute_perplexity(original_model, tokenizer, test_texts)
        ppl_after = self.compute_perplexity(modified_model, tokenizer, test_texts)

        ppl_delta = (ppl_after - ppl_before) / (ppl_before + 1e-8)

        if ppl_delta > threshold_perplexity:
            warnings.append(f"Perplexity increased by {ppl_delta:.1%} (threshold {threshold_perplexity:.0%})")

        # Measure coherence
        coh_before = self.compute_coherence(original_model, tokenizer, test_texts)
        coh_after = self.compute_coherence(modified_model, tokenizer, test_texts)

        coh_delta = (coh_after - coh_before) / (coh_before + 1e-8)

        if coh_delta < -threshold_coherence:
            warnings.append(f"Coherence decreased by {-coh_delta:.1%} (threshold {threshold_coherence:.0%})")

        # Compute KL divergence
        kl_div = self.compute_kl_divergence(original_model, modified_model, tokenizer, test_texts)

        if kl_div > threshold_kl:
            warnings.append(f"KL divergence {kl_div:.3f} exceeds threshold {threshold_kl}")

        # Measure effective rank preservation
        rank_before, rank_after = self.compute_effective_rank(original_model, modified_model)
        rank_preservation = rank_after / (rank_before + 1e-8)

        if rank_preservation < 0.8:
            warnings.append(f"Effective rank dropped to {rank_preservation:.1%} of original")

        passed = len(warnings) == 0

        return ValidationReport(
            perplexity_before=ppl_before,
            perplexity_after=ppl_after,
            perplexity_delta=ppl_delta,
            coherence_before=coh_before,
            coherence_after=coh_after,
            coherence_delta=coh_delta,
            kl_divergence=kl_div,
            effective_rank_before=rank_before,
            effective_rank_after=rank_after,
            rank_preservation=rank_preservation,
            passed=passed,
            warnings=warnings,
            metadata={}
        )

    def compute_perplexity(
        self,
        model,
        tokenizer,
        texts: List[str],
        max_length: int = 512
    ) -> float:
        """
        Compute average perplexity on test texts.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            texts: List of test strings
            max_length: Maximum sequence length

        Returns:
            Average perplexity
        """
        model.eval()
        perplexities = []

        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexities.append(np.exp(loss))

        return np.mean(perplexities)

    def compute_coherence(
        self,
        model,
        tokenizer,
        texts: List[str],
        max_new_tokens: int = 50
    ) -> float:
        """
        Compute coherence score using log-likelihood of continuations.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            texts: List of test strings
            max_new_tokens: Length of continuation to generate

        Returns:
            Average coherence score (higher is better)
        """
        model.eval()
        coherence_scores = []

        for text in texts:
            # Tokenize the prompt
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate continuation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Compute average log probability of generated tokens
            # (simplified coherence proxy)
            if hasattr(outputs, 'scores') and outputs.scores:
                avg_log_prob = sum(
                    torch.log_softmax(score, dim=-1).max(dim=-1).values.mean().item()
                    for score in outputs.scores
                ) / len(outputs.scores)
                coherence_scores.append(avg_log_prob)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def compute_kl_divergence(
        self,
        original_model,
        modified_model,
        tokenizer,
        texts: List[str],
        max_new_tokens: int = 100
    ) -> float:
        """
        Compute KL divergence between output distributions.

        Args:
            original_model: Original model
            modified_model: Modified model
            tokenizer: Tokenizer
            texts: Test prompts
            max_new_tokens: Length for generation

        Returns:
            Average KL divergence
        """
        original_model.eval()
        modified_model.eval()

        kl_divs = []

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Get logits for next token
                orig_outputs = original_model(**inputs)
                mod_outputs = modified_model(**inputs)

                orig_logits = orig_outputs.logits[0, -1, :].cpu().numpy()
                mod_logits = mod_outputs.logits[0, -1, :].cpu().numpy()

                # Convert to probabilities
                orig_probs = np.exp(orig_logits - np.max(orig_logits))
                orig_probs = orig_probs / np.sum(orig_probs)

                mod_probs = np.exp(mod_logits - np.max(mod_logits))
                mod_probs = mod_probs / np.sum(mod_probs)

                # Compute KL divergence
                kl = entropy(orig_probs, mod_probs)
                kl_divs.append(kl)

        return np.mean(kl_divs) if kl_divs else 0.0

    def compute_effective_rank(
        self,
        original_model,
        modified_model,
        sample_ratio: float = 0.1
    ) -> Tuple[float, float]:
        """
        Compute effective rank of weight matrices.

        Effective rank = exp(-sum(p_i * log(p_i))) where p_i are normalized singular values.

        Args:
            original_model: Original model
            modified_model: Modified model
            sample_ratio: Ratio of layers to sample (for speed)

        Returns:
            Tuple of (original_rank, modified_rank)
        """
        original_ranks = []
        modified_ranks = []

        # Collect all weight matrices
        original_weights = []
        modified_weights = []

        for (name, param) in original_model.named_parameters():
            if "weight" in name and param.dim() == 2:
                original_weights.append(param.data.cpu().numpy())

        for (name, param) in modified_model.named_parameters():
            if "weight" in name and param.dim() == 2:
                modified_weights.append(param.data.cpu().numpy())

        # Sample if too many
        n_layers = len(original_weights)
        if sample_ratio < 1.0:
            n_sample = max(1, int(n_layers * sample_ratio))
            indices = np.random.choice(n_layers, n_sample, replace=False)
            original_weights = [original_weights[i] for i in indices]
            modified_weights = [modified_weights[i] for i in indices]

        # Compute effective rank for each
        for orig_w, mod_w in zip(original_weights, modified_weights):
            # Singular values
            s_orig = np.linalg.svd(orig_w, compute_uv=False)
            s_mod = np.linalg.svd(mod_w, compute_uv=False)

            # Normalize
            p_orig = s_orig / (np.sum(s_orig) + 1e-8)
            p_mod = s_mod / (np.sum(s_mod) + 1e-8)

            # Effective rank
            rank_orig = np.exp(-np.sum(p_orig * np.log(p_orig + 1e-8)))
            rank_mod = np.exp(-np.sum(p_mod * np.log(p_mod + 1e-8)))

            original_ranks.append(rank_orig)
            modified_ranks.append(rank_mod)

        return np.mean(original_ranks), np.mean(modified_ranks)
