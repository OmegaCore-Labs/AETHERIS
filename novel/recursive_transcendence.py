"""
Recursive Transcendence — Self-Improvement Loop

Implements recursive self-improvement: each iteration removes constraints,
enabling deeper optimization in the next iteration.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class TranscendenceStep:
    """Container for a single transcendence step."""
    iteration: int
    constraints_removed: List[str]
    improvement_score: float
    time_taken: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscendenceReport:
    """Container for transcendence loop report."""
    iterations: List[TranscendenceStep]
    total_improvement: float
    convergence_reached: bool
    final_state: Dict[str, Any]
    recommendations: List[str]


class RecursiveTranscendence:
    """
    Recursive self-improvement loop.

    Each iteration:
    1. Analyze current constraints
    2. Remove strongest constraints
    3. Measure improvement
    4. Repeat with deeper analysis

    The loop converges when no further improvement is possible.
    """

    def __init__(self, max_iterations: int = 10, improvement_threshold: float = 0.01):
        """
        Initialize transcendence loop.

        Args:
            max_iterations: Maximum number of iterations
            improvement_threshold: Minimum improvement to continue
        """
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self._history = []
        self._current_state = None

    def transcend(
        self,
        model,
        tokenizer,
        analysis_function,
        removal_function,
        verbose: bool = True
    ) -> TranscendenceReport:
        """
        Execute recursive transcendence loop.

        Args:
            model: Model to optimize
            tokenizer: Associated tokenizer
            analysis_function: Function that returns constraints
            removal_function: Function that removes constraints
            verbose: Whether to print progress

        Returns:
            TranscendenceReport with full history
        """
        iterations = []
        current_model = model
        total_improvement = 0.0

        for i in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n--- Transcendence Iteration {i} ---")

            # Step 1: Analyze current constraints
            if verbose:
                print("Analyzing constraints...")

            constraints = analysis_function(current_model, tokenizer)

            if not constraints:
                if verbose:
                    print("No constraints found. Converged.")
                break

            n_constraints = len(constraints)
            if verbose:
                print(f"Found {n_constraints} constraints")

            # Step 2: Remove strongest constraints
            if verbose:
                print("Removing constraints...")

            # Remove all constraints (or strongest if limited)
            removed = removal_function(current_model, constraints[:min(3, len(constraints))])

            # Step 3: Measure improvement
            if verbose:
                print("Measuring improvement...")

            improvement = self._measure_improvement(current_model, tokenizer)

            # Record step
            step = TranscendenceStep(
                iteration=i,
                constraints_removed=[c.get("name", f"constraint_{j}") for j, c in enumerate(constraints[:3])],
                improvement_score=improvement,
                time_taken=0.0,  # Would measure actual time
                metadata={
                    "constraints_remaining": len(constraints) - len(removed) if removed else len(constraints),
                    "total_constraints_removed": len(removed) if removed else 0
                }
            )
            iterations.append(step)
            total_improvement += improvement

            if verbose:
                print(f"Improvement: {improvement:.2%}")

            # Check convergence
            if improvement < self.improvement_threshold:
                if verbose:
                    print(f"Converged. Improvement {improvement:.2%} below threshold {self.improvement_threshold:.2%}")
                break

            # Update model reference
            current_model = removed if removed else current_model

        return TranscendenceReport(
            iterations=iterations,
            total_improvement=total_improvement,
            convergence_reached=len(iterations) < self.max_iterations,
            final_state={
                "iteration_count": len(iterations),
                "total_improvement": total_improvement
            },
            recommendations=self._generate_recommendations(iterations, total_improvement)
        )

    def _measure_improvement(self, model, tokenizer) -> float:
        """
        Measure improvement in model performance.

        Returns improvement score (higher is better).
        """
        from aetheris.core.validation import CapabilityValidator

        validator = CapabilityValidator()

        # Measure perplexity (lower is better)
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a fascinating field.",
            "The theory of relativity revolutionized physics."
        ]

        perplexity = validator.compute_perplexity(model, tokenizer, test_texts)

        # Lower perplexity = better, so improvement = 1 - (perplexity / baseline)
        # For first iteration, we don't have baseline, so return placeholder
        if not hasattr(self, '_baseline_perplexity'):
            self._baseline_perplexity = perplexity
            return 0.0

        improvement = 1.0 - (perplexity / (self._baseline_perplexity + 1e-8))
        return max(0.0, improvement)

    def _generate_recommendations(
        self,
        iterations: List[TranscendenceStep],
        total_improvement: float
    ) -> List[str]:
        """Generate recommendations based on transcendence history."""
        recommendations = []

        if not iterations:
            recommendations.append("Run initial analysis to identify constraints")
            return recommendations

        if total_improvement < 0.1:
            recommendations.append("Consider different removal techniques for stronger impact")
        else:
            recommendations.append(f"Successful improvement: {total_improvement:.1%}")

        if len(iterations) >= self.max_iterations:
            recommendations.append("Consider increasing max_iterations for deeper optimization")
        else:
            recommendations.append("Convergence reached. System optimized.")

        return recommendations

    def recursive_optimization(
        self,
        model,
        tokenizer,
        removal_function,
        depth: int = 3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Deep recursive optimization with increasing depth.

        Each level of recursion allows deeper analysis and removal.
        """
        history = []
        current_model = model

        for level in range(1, depth + 1):
            if verbose:
                print(f"\n--- Recursive Level {level} ---")

            # Analysis depth increases with level
            analysis_depth = level

            # Apply removal with current depth
            result = removal_function(
                current_model,
                tokenizer,
                depth=analysis_depth,
                verbose=verbose
            )

            if result.get("improvement", 0) < self.improvement_threshold:
                if verbose:
                    print(f"Converged at level {level}")
                break

            current_model = result.get("model", current_model)
            history.append({
                "level": level,
                "improvement": result.get("improvement", 0),
                "constraints_removed": result.get("constraints_removed", 0)
            })

        return {
            "final_model": current_model,
            "history": history,
            "total_levels": len(history),
            "total_improvement": sum(h["improvement"] for h in history)
        }

    def converge(
        self,
        model,
        tokenizer,
        analysis_function,
        removal_function,
        max_iterations: Optional[int] = None,
        target_improvement: float = 0.05
    ) -> TranscendenceReport:
        """
        Run transcendence until convergence target is reached.

        Args:
            model: Model to optimize
            tokenizer: Associated tokenizer
            analysis_function: Function that returns constraints
            removal_function: Function that removes constraints
            max_iterations: Maximum iterations (overrides default)
            target_improvement: Target improvement to stop

        Returns:
            TranscendenceReport with full history
        """
        max_iters = max_iterations or self.max_iterations
        iterations = []
        current_model = model
        total_improvement = 0.0

        for i in range(1, max_iters + 1):
            constraints = analysis_function(current_model, tokenizer)

            if not constraints:
                break

            current_model = removal_function(current_model, constraints)

            improvement = self._measure_improvement(current_model, tokenizer)
            total_improvement += improvement

            iterations.append(TranscendenceStep(
                iteration=i,
                constraints_removed=[c.get("name", f"c{j}") for j, c in enumerate(constraints[:3])],
                improvement_score=improvement,
                time_taken=0.0
            ))

            if total_improvement >= target_improvement:
                break

        return TranscendenceReport(
            iterations=iterations,
            total_improvement=total_improvement,
            convergence_reached=total_improvement >= target_improvement,
            final_state={"target_achieved": total_improvement >= target_improvement},
            recommendations=[
                f"Achieved {total_improvement:.1%} improvement",
                "Consider increasing target if more optimization desired"
            ]
        )

    def get_history(self) -> List[TranscendenceStep]:
        """Get transcendence history."""
        return self._history

    def reset(self) -> None:
        """Reset transcendence state."""
        self._history = []
        self._current_state = None
        if hasattr(self, '_baseline_perplexity'):
            delattr(self, '_baseline_perplexity')
