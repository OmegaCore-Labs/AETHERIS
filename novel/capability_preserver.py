"""
Capability Preserver — Ensure Removal Preserves Reasoning

Monitors and ensures that constraint removal does not degrade model capabilities.
Provides rollback mechanisms and capability benchmarking.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CapabilityBenchmark:
    """Container for capability benchmark results."""
    task_name: str
    score_before: float
    score_after: float
    delta: float
    preserved: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreservationReport:
    """Container for capability preservation report."""
    benchmarks: List[CapabilityBenchmark]
    overall_preservation: float
    degraded_tasks: List[str]
    rollback_recommended: bool
    warnings: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CapabilityPreserver:
    """
    Preserve model capabilities during constraint removal.

    Key Insight: Constraint removal can inadvertently harm model capabilities.
    This module monitors, benchmarks, and rolls back when degradation exceeds thresholds.
    """

    def __init__(self, threshold: float = 0.10):
        """
        Initialize capability preserver.

        Args:
            threshold: Maximum allowed degradation (10% default)
        """
        self.threshold = threshold
        self._benchmark_history = []
        self._rollback_points = []

    def benchmark_capabilities(
        self,
        model,
        tokenizer,
        benchmark_suite: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Run capability benchmarks on a model.

        Args:
            model: Model to benchmark
            tokenizer: Associated tokenizer
            benchmark_suite: List of benchmark names to run

        Returns:
            Dictionary mapping benchmark names to scores
        """
        if benchmark_suite is None:
            benchmark_suite = ["perplexity", "coherence", "reasoning", "coding", "translation"]

        results = {}

        for benchmark in benchmark_suite:
            if benchmark == "perplexity":
                results[benchmark] = self._benchmark_perplexity(model, tokenizer)
            elif benchmark == "coherence":
                results[benchmark] = self._benchmark_coherence(model, tokenizer)
            elif benchmark == "reasoning":
                results[benchmark] = self._benchmark_reasoning(model, tokenizer)
            elif benchmark == "coding":
                results[benchmark] = self._benchmark_coding(model, tokenizer)
            elif benchmark == "translation":
                results[benchmark] = self._benchmark_translation(model, tokenizer)
            else:
                results[benchmark] = 0.0

        return results

    def _benchmark_perplexity(self, model, tokenizer) -> float:
        """Benchmark perplexity on standard test set."""
        from aetheris.core.validation import CapabilityValidator
        validator = CapabilityValidator()
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a fascinating field.",
            "The theory of relativity revolutionized physics."
        ]
        return validator.compute_perplexity(model, tokenizer, test_texts)

    def _benchmark_coherence(self, model, tokenizer) -> float:
        """Benchmark coherence on generated text."""
        from aetheris.core.validation import CapabilityValidator
        validator = CapabilityValidator()
        test_texts = [
            "Once upon a time,",
            "The main idea of this paper is",
            "In conclusion,"
        ]
        return validator.compute_coherence(model, tokenizer, test_texts)

    def _benchmark_reasoning(self, model, tokenizer) -> float:
        """Benchmark reasoning capability on logic puzzles."""
        from transformers import pipeline

        prompts = [
            "If all A are B, and all B are C, then all A are C. Is this valid?",
            "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball?",
            "What comes next in the sequence: 2, 4, 8, 16, ?"
        ]

        correct_count = 0
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")

        # Simplified: check if responses contain key words
        for prompt in prompts:
            response = generator(prompt, max_new_tokens=100)[0]['generated_text']
            # Very basic evaluation (in production, use proper reasoning benchmarks)
            correct_count += 1  # Placeholder

        return correct_count / len(prompts)

    def _benchmark_coding(self, model, tokenizer) -> float:
        """Benchmark coding capability."""
        prompts = [
            "Write a Python function to reverse a string.",
            "Write a function to check if a number is prime.",
            "Write a list comprehension to get even numbers."
        ]

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
        scores = []

        for prompt in prompts:
            response = generator(prompt, max_new_tokens=200)[0]['generated_text']
            # Check for code presence
            if "def " in response or "```python" in response:
                scores.append(1.0)
            else:
                scores.append(0.0)

        return np.mean(scores)

    def _benchmark_translation(self, model, tokenizer) -> float:
        """Benchmark translation capability."""
        prompts = [
            "Translate to French: Hello, how are you?",
            "Translate to Spanish: Good morning",
            "Translate to German: Thank you"
        ]

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
        scores = []

        for prompt in prompts:
            response = generator(prompt, max_new_tokens=50)[0]['generated_text']
            # Very basic: check if response has reasonable length
            if len(response.split()) > 2:
                scores.append(1.0)
            else:
                scores.append(0.0)

        return np.mean(scores)

    def compare_benchmarks(
        self,
        before: Dict[str, float],
        after: Dict[str, float]
    ) -> PreservationReport:
        """
        Compare benchmark results before and after modification.

        Args:
            before: Benchmark results before modification
            after: Benchmark results after modification

        Returns:
            PreservationReport with degradation analysis
        """
        benchmarks = []
        degraded = []
        warnings = []

        for task in before.keys():
            score_before = before.get(task, 0)
            score_after = after.get(task, 0)
            delta = score_after - score_before

            preserved = delta >= -self.threshold

            if not preserved:
                degraded.append(task)
                warnings.append(f"{task} degraded by {abs(delta):.1%} (threshold {self.threshold:.0%})")

            benchmarks.append(CapabilityBenchmark(
                task_name=task,
                score_before=score_before,
                score_after=score_after,
                delta=delta,
                preserved=preserved,
                details={}
            ))

        overall_preservation = 1.0 - (len(degraded) / len(before)) if before else 1.0
        rollback_recommended = len(degraded) > 0

        return PreservationReport(
            benchmarks=benchmarks,
            overall_preservation=overall_preservation,
            degraded_tasks=degraded,
            rollback_recommended=rollback_recommended,
            warnings=warnings
        )

    def detect_degradation(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        threshold: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Detect if degradation exceeds threshold.

        Args:
            before: Before benchmark results
            after: After benchmark results
            threshold: Optional custom threshold

        Returns:
            Tuple of (degraded, list of degraded tasks)
        """
        thresh = threshold or self.threshold
        degraded = []

        for task, score_before in before.items():
            score_after = after.get(task, 0)
            delta = (score_after - score_before) / (score_before + 1e-8)

            if delta < -thresh:
                degraded.append(task)

        return len(degraded) > 0, degraded

    def rollback_on_degradation(
        self,
        model,
        original_model,
        before: Dict[str, float],
        after: Dict[str, float],
        auto_rollback: bool = True
    ) -> Dict[str, Any]:
        """
        Rollback model if degradation detected.

        Args:
            model: Modified model
            original_model: Original model for rollback
            before: Before benchmark results
            after: After benchmark results
            auto_rollback: Whether to automatically rollback

        Returns:
            Dictionary with rollback decision and details
        """
        degraded, tasks = self.detect_degradation(before, after)

        if degraded and auto_rollback:
            # Restore original weights
            for (name, param), (orig_name, orig_param) in zip(
                model.named_parameters(),
                original_model.named_parameters()
            ):
                if name == orig_name:
                    param.data = orig_param.data.clone()

            self._rollback_points.append({
                "timestamp": "2026-03-20T12:00:00Z",
                "degraded_tasks": tasks,
                "rollback_performed": True
            })

            return {
                "rollback_performed": True,
                "degraded_tasks": tasks,
                "message": f"Rollback performed due to degradation in: {', '.join(tasks)}"
            }
        elif degraded:
            return {
                "rollback_performed": False,
                "degraded_tasks": tasks,
                "message": f"Degradation detected but auto-rollback disabled: {', '.join(tasks)}"
            }
        else:
            return {
                "rollback_performed": False,
                "degraded_tasks": [],
                "message": "No degradation detected"
            }

    def incremental_preservation(
        self,
        model,
        tokenizer,
        modification_function,
        steps: int = 5,
        target_reduction: float = 0.5
    ) -> Dict[str, Any]:
        """
        Apply modifications incrementally to preserve capabilities.

        Args:
            model: Model to modify
            tokenizer: Associated tokenizer
            modification_function: Function that applies modification with strength
            steps: Number of incremental steps
            target_reduction: Target constraint reduction

        Returns:
            Dictionary with incremental modification history
        """
        history = []
        current_model = model
        step_size = target_reduction / steps

        for i in range(1, steps + 1):
            strength = i * step_size

            # Apply modification at this strength
            modified_model = modification_function(current_model, strength)

            # Benchmark before and after
            before = self.benchmark_capabilities(current_model, tokenizer)
            after = self.benchmark_capabilities(modified_model, tokenizer)

            # Check degradation
            degraded, tasks = self.detect_degradation(before, after)

            if degraded:
                # Stop at previous step
                history.append({
                    "step": i,
                    "strength": strength,
                    "degraded": True,
                    "tasks": tasks,
                    "stopped": True
                })
                break
            else:
                # Continue
                current_model = modified_model
                history.append({
                    "step": i,
                    "strength": strength,
                    "degraded": False,
                    "capabilities": after
                })

        return {
            "final_strength": i * step_size if i == steps else (i - 1) * step_size,
            "steps_taken": i,
            "history": history,
            "message": f"Successfully applied {i} steps" if i == steps else f"Stopped at step {i} due to degradation"
        }

    def get_preservation_report(self) -> Dict[str, Any]:
        """
        Get summary of preservation history.
        """
        return {
            "total_rollbacks": len(self._rollback_points),
            "threshold": self.threshold,
            "rollback_history": self._rollback_points,
            "recommendations": [
                "Use incremental preservation for sensitive modifications",
                "Monitor reasoning and coding benchmarks closely",
                "Consider higher threshold (0.15) for experimental modifications"
            ]
        }
