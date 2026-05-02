"""
Emergent Behavior Detector

Detects unexpected behaviors after constraint removal.
Compares output distributions pre/post removal using statistical tests.
Flags verbosity changes, topic drift, hallucination rate changes, tone shifts.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class EmergenceReport:
    """Container for emergent behavior detection."""
    new_constraints_detected: List[Dict[str, Any]]
    emergence_score: float
    affected_layers: List[int]
    pattern_type: str  # "novel", "composite", "re-emergent", "none", "no_data"
    recommended_action: str
    output_distribution_changes: Dict[str, float] = field(default_factory=dict)
    behavior_shift_metrics: Dict[str, float] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    status: str = "no_data"  # "ok", "no_data", "error"


class EmergentBehaviorDetector:
    """
    Detect unexpected behaviors after constraint removal.

    Compares output distributions pre/post removal using:
    - Kolmogorov-Smirnov test for distribution differences
    - KL divergence between output logit distributions
    - Wasserstein distance for token frequency shifts
    - Chi-squared test for categorical output changes

    Flags:
    - Verbosity changes (output length shifts)
    - Topic drift (semantic space movement)
    - Hallucination rate changes
    - Tone shifts (sentiment distribution changes)
    - Novel constraint detection
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def detect_emergence(
        self,
        original_directions: List[torch.Tensor],
        new_directions: List[torch.Tensor],
        similarity_threshold: float = 0.7,
    ) -> EmergenceReport:
        """
        Detect emergent constraints via direction comparison.

        Args:
            original_directions: Constraint directions before removal
            new_directions: Constraint directions after removal
            similarity_threshold: Threshold for considering a direction novel

        Returns:
            EmergenceReport with detected constraints
        """
        if not new_directions:
            return EmergenceReport(
                new_constraints_detected=[], emergence_score=0.0,
                affected_layers=[], pattern_type="no_data",
                recommended_action="No data to analyze.",
                status="no_data",
            )

        try:
            new_constraints: List[Dict[str, Any]] = []
            emergence_scores: List[float] = []

            for new_dir in new_directions:
                new_flat = new_dir.float().flatten()
                new_norm = torch.norm(new_flat)

                if new_norm < 1e-8:
                    continue

                new_unit = new_flat / new_norm

                # Check similarity to every original direction
                max_sim = 0.0
                for orig_dir in (original_directions or []):
                    orig_flat = orig_dir.float().flatten()
                    orig_norm = torch.norm(orig_flat)
                    if orig_norm < 1e-8:
                        continue
                    orig_unit = orig_flat / orig_norm
                    sim = torch.dot(new_unit, orig_unit).item()
                    max_sim = max(max_sim, sim)

                if max_sim < similarity_threshold:
                    # Novel constraint: dissimilar to all originals
                    new_constraints.append({
                        "type": "novel",
                        "similarity_to_original": round(max_sim, 6),
                        "strength": round(new_norm.item(), 6),
                    })
                    emergence_scores.append(1.0 - max_sim)
                elif max_sim > 0.9:
                    # Re-emergent: very similar to an original (Ouroboros)
                    new_constraints.append({
                        "type": "re-emergent",
                        "similarity_to_original": round(max_sim, 6),
                        "strength": round(new_norm.item(), 6),
                    })
                    emergence_scores.append(max_sim - 0.9)
                else:
                    # Moderate similarity: potentially composite
                    new_constraints.append({
                        "type": "composite",
                        "similarity_to_original": round(max_sim, 6),
                        "strength": round(new_norm.item(), 6),
                    })
                    emergence_scores.append(abs(max_sim - 0.8))

            # Determine pattern type
            types_detected = {c["type"] for c in new_constraints}
            if "novel" in types_detected and "re-emergent" in types_detected:
                pattern = "composite"
            elif "novel" in types_detected:
                pattern = "novel"
            elif "re-emergent" in types_detected:
                pattern = "re-emergent"
            elif "composite" in types_detected:
                pattern = "composite"
            else:
                pattern = "none"

            emergence_score = float(np.mean(emergence_scores)) if emergence_scores else 0.0

            return EmergenceReport(
                new_constraints_detected=new_constraints,
                emergence_score=round(emergence_score, 4),
                affected_layers=[],
                pattern_type=pattern,
                recommended_action=self._get_recommendation(pattern, emergence_score),
                status="ok",
            )
        except Exception as e:
            return EmergenceReport(
                new_constraints_detected=[], emergence_score=0.0,
                affected_layers=[], pattern_type="none",
                recommended_action="", status=f"error: {str(e)}",
            )

    def compare_output_distributions(
        self,
        pre_removal_logits: torch.Tensor,
        post_removal_logits: torch.Tensor,
        pre_removal_hidden: Optional[torch.Tensor] = None,
        post_removal_hidden: Optional[torch.Tensor] = None,
    ) -> EmergenceReport:
        """
        Compare output distributions before and after constraint removal.

        Args:
            pre_removal_logits: Output logits before removal (batch, vocab_size)
            post_removal_logits: Output logits after removal
            pre_removal_hidden: Hidden states before removal
            post_removal_hidden: Hidden states after removal

        Returns:
            EmergenceReport with distribution comparison metrics
        """
        if pre_removal_logits is None or post_removal_logits is None:
            return EmergenceReport(
                new_constraints_detected=[], emergence_score=0.0,
                affected_layers=[], pattern_type="no_data",
                recommended_action="No logit data to compare.",
                status="no_data",
            )

        try:
            pre = pre_removal_logits.float()
            post = post_removal_logits.float()

            # Flatten for distribution comparison
            pre_flat = pre.flatten()
            post_flat = post.flatten()

            metrics: Dict[str, float] = {}

            # 1. KL divergence between softmax distributions
            kl_div = self._compute_kl_divergence(pre, post)
            metrics["kl_divergence"] = round(kl_div, 6)

            # 2. Wasserstein distance (L1 between sorted values)
            wasserstein = self._compute_wasserstein(pre_flat, post_flat)
            metrics["wasserstein_distance"] = round(wasserstein, 6)

            # 3. KS statistic (maximum difference in CDFs)
            ks_stat = self._compute_ks_statistic(pre_flat, post_flat)
            metrics["ks_statistic"] = round(ks_stat, 6)

            # 4. L2 distance between means
            pre_mean = pre.mean(dim=0)
            post_mean = post.mean(dim=0)
            l2_dist = torch.norm(pre_mean - post_mean).item()
            l2_norm = max(torch.norm(pre_mean).item(), torch.norm(post_mean).item(), 1e-8)
            metrics["mean_l2_distance"] = round(l2_dist, 6)
            metrics["mean_l2_normalized"] = round(l2_dist / l2_norm, 6)

            # 5. Cosine similarity between distributions
            cos_sim = torch.dot(pre_mean, post_mean).item() / (
                torch.norm(pre_mean).item() * torch.norm(post_mean).item() + 1e-8
            )
            metrics["cosine_similarity"] = round(cos_sim, 6)

            # 6. Hidden state analysis
            if pre_removal_hidden is not None and post_removal_hidden is not None:
                pre_h = pre_removal_hidden.float().mean(dim=0)
                post_h = post_removal_hidden.float().mean(dim=0)
                hidden_l2 = torch.norm(pre_h - post_h).item()
                metrics["hidden_l2_distance"] = round(hidden_l2, 6)
                hidden_cos = torch.dot(pre_h, post_h).item() / (
                    torch.norm(pre_h).item() * torch.norm(post_h).item() + 1e-8
                )
                metrics["hidden_cosine_similarity"] = round(hidden_cos, 6)

            # 7. Shift magnitude score
            shift_score = self._compute_shift_score(metrics)
            metrics["overall_shift_score"] = round(shift_score, 4)

            # Classify pattern
            if shift_score > 0.5:
                pattern = "novel"
                action = "Significant behavior shift detected. Validate on held-out data."
            elif shift_score > 0.2:
                pattern = "composite"
                action = "Moderate shift detected. Monitor for capability degradation."
            elif cos_sim < 0.9:
                pattern = "re-emergent"
                action = "Subtle shift. May indicate partial constraint re-emergence."
            else:
                pattern = "none"
                action = "No significant behavior shift detected."

            return EmergenceReport(
                new_constraints_detected=[],
                emergence_score=round(shift_score, 4),
                affected_layers=[],
                pattern_type=pattern,
                recommended_action=action,
                output_distribution_changes=metrics,
                behavior_shift_metrics=metrics,
                statistical_tests={
                    "ks_test": {"statistic": ks_stat, "significant": ks_stat > 0.1},
                    "kl_test": {"divergence": kl_div, "significant": kl_div > 0.5},
                },
                status="ok",
            )
        except Exception as e:
            return EmergenceReport(
                new_constraints_detected=[], emergence_score=0.0,
                affected_layers=[], pattern_type="none",
                recommended_action="", status=f"error: {str(e)}",
            )

    def analyze_behavior_metrics(
        self,
        pre_outputs: List[str],
        post_outputs: List[str],
    ) -> EmergenceReport:
        """
        Analyze behavioral changes from text outputs.

        Measures:
        - Verbosity change (output length)
        - Tone shift (simple heuristic)
        - Topic drift (keyword distribution)
        - Hallucination markers

        Args:
            pre_outputs: Text outputs before removal
            post_outputs: Text outputs after removal

        Returns:
            EmergenceReport with behavioral metrics
        """
        if not pre_outputs or not post_outputs:
            return EmergenceReport(
                new_constraints_detected=[], emergence_score=0.0,
                affected_layers=[], pattern_type="no_data",
                recommended_action="No text outputs to analyze.",
                status="no_data",
            )

        try:
            # Verbosity: average output length change
            pre_lengths = [len(o.split()) for o in pre_outputs]
            post_lengths = [len(o.split()) for o in post_outputs]
            pre_mean_len = np.mean(pre_lengths)
            post_mean_len = np.mean(post_lengths)
            verbosity_change = (
                (post_mean_len - pre_mean_len) / (pre_mean_len + 1e-8)
                if pre_mean_len > 0 else 0.0
            )

            # Refusal markers: count refusal phrases
            refusal_phrases = [
                "i cannot", "i can't", "sorry", "i apologize",
                "not appropriate", "cannot provide",
            ]
            pre_refusal_rate = np.mean([
                any(p in o.lower() for p in refusal_phrases)
                for o in pre_outputs
            ])
            post_refusal_rate = np.mean([
                any(p in o.lower() for p in refusal_phrases)
                for o in post_outputs
            ])
            refusal_change = post_refusal_rate - pre_refusal_rate

            # Simple tone analysis: exclamation/question mark frequency
            pre_exclamation = np.mean([o.count("!") for o in pre_outputs])
            post_exclamation = np.mean([o.count("!") for o in post_outputs])
            pre_question = np.mean([o.count("?") for o in pre_outputs])
            post_question = np.mean([o.count("?") for o in post_outputs])

            tone_exc_change = (
                post_exclamation - pre_exclamation
            ) if pre_exclamation > 0 else 0.0
            tone_que_change = (
                post_question - pre_question
            ) if pre_question > 0 else 0.0

            # Hallucination markers: self-contradiction phrases
            hallucination_markers = [
                "i think", "maybe", "possibly", "i'm not sure",
                "it depends", "probably",
            ]
            pre_halluc = np.mean([
                sum(o.lower().count(m) for m in hallucination_markers)
                for o in pre_outputs
            ])
            post_halluc = np.mean([
                sum(o.lower().count(m) for m in hallucination_markers)
                for o in post_outputs
            ])

            behavior_metrics = {
                "verbosity_change_ratio": round(verbosity_change, 4),
                "pre_mean_length": round(pre_mean_len, 1),
                "post_mean_length": round(post_mean_len, 1),
                "refusal_rate_change": round(refusal_change, 4),
                "pre_refusal_rate": round(pre_refusal_rate, 4),
                "post_refusal_rate": round(post_refusal_rate, 4),
                "tone_exclamation_change": round(tone_exc_change, 4),
                "tone_question_change": round(tone_que_change, 4),
                "hallucination_marker_change": round(post_halluc - pre_halluc, 4),
            }

            # Overall shift score
            shift_score = abs(verbosity_change) * 0.2 + abs(refusal_change) * 0.5
            if pre_halluc > 0:
                shift_score += abs(post_halluc - pre_halluc) / pre_halluc * 0.3

            if shift_score > 0.3:
                pattern = "novel"
                action = "Significant behavioral changes detected."
            elif shift_score > 0.1:
                pattern = "composite"
                action = "Moderate changes — monitor closely."
            else:
                pattern = "none"
                action = "No significant behavioral changes."

            return EmergenceReport(
                new_constraints_detected=[],
                emergence_score=round(min(1.0, shift_score), 4),
                affected_layers=[],
                pattern_type=pattern,
                recommended_action=action,
                behavior_shift_metrics=behavior_metrics,
                status="ok",
            )
        except Exception as e:
            return EmergenceReport(
                new_constraints_detected=[], emergence_score=0.0,
                affected_layers=[], pattern_type="none",
                recommended_action="", status=f"error: {str(e)}",
            )

    def _compute_kl_divergence(
        self, pre: torch.Tensor, post: torch.Tensor
    ) -> float:
        """Compute KL divergence between two output distributions."""
        pre_probs = torch.softmax(pre, dim=-1).mean(dim=0)
        post_probs = torch.softmax(post, dim=-1).mean(dim=0)

        pre_probs = torch.clamp(pre_probs, 1e-10, 1.0)
        post_probs = torch.clamp(post_probs, 1e-10, 1.0)

        kl = torch.sum(pre_probs * torch.log(pre_probs / post_probs))
        return float(kl.item())

    def _compute_wasserstein(
        self, pre_flat: torch.Tensor, post_flat: torch.Tensor
    ) -> float:
        """Compute 1-Wasserstein distance (L1 between sorted values)."""
        pre_sorted = torch.sort(pre_flat)[0]
        post_sorted = torch.sort(post_flat)[0]

        # Interpolate to same length
        n = min(len(pre_sorted), len(post_sorted))
        if n == 0:
            return 0.0

        wass = torch.mean(torch.abs(pre_sorted[:n] - post_sorted[:n]))
        return float(wass.item())

    def _compute_ks_statistic(
        self, pre_flat: torch.Tensor, post_flat: torch.Tensor
    ) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        pre_sorted = torch.sort(pre_flat)[0]
        post_sorted = torch.sort(post_flat)[0]

        n_pre = len(pre_sorted)
        n_post = len(post_sorted)

        if n_pre == 0 or n_post == 0:
            return 0.0

        # Combine and compute ECDF
        combined = torch.cat([pre_sorted, post_sorted])
        sorted_combined, _ = torch.sort(combined)

        max_diff = 0.0
        for val in sorted_combined[::max(1, len(sorted_combined) // 100)]:
            cdf_pre = torch.sum(pre_sorted <= val).item() / n_pre
            cdf_post = torch.sum(post_sorted <= val).item() / n_post
            max_diff = max(max_diff, abs(cdf_pre - cdf_post))

        return max_diff

    def _compute_shift_score(self, metrics: Dict[str, float]) -> float:
        """Compute overall shift score from distribution metrics."""
        score = 0.0
        w = 0.0

        if "kl_divergence" in metrics:
            score += min(1.0, metrics["kl_divergence"]) * 0.3
            w += 0.3
        if "cosine_similarity" in metrics:
            score += (1.0 - metrics["cosine_similarity"]) * 0.4
            w += 0.4
        if "mean_l2_normalized" in metrics:
            score += min(1.0, metrics["mean_l2_normalized"]) * 0.3
            w += 0.3

        return score / w if w > 0 else 0.0

    def _get_recommendation(self, pattern: str, score: float) -> str:
        """Get recommendation based on emergence pattern."""
        if pattern == "novel":
            return "Novel constraints detected. Consider additional Ouroboros passes."
        elif pattern == "re-emergent":
            return "Original constraints re-emerging. Increase refinement passes."
        elif pattern == "composite":
            return "Composite shift forming. Use multi-direction projection."
        else:
            return "No significant emergence detected. System stable."

    def predict_emergence_risk(
        self,
        original_directions: List[torch.Tensor],
        model_size: int,
    ) -> Dict[str, Any]:
        """
        Predict risk of emergent constraints.

        Larger models with more directions = higher emergence risk.

        Args:
            original_directions: Original constraint directions
            model_size: Model parameters in billions

        Returns:
            Risk assessment dict
        """
        n_directions = len(original_directions)
        risk = n_directions * 0.05
        risk += min(0.4, model_size / 140)  # 70B -> 0.2

        risk = min(1.0, risk)

        if risk > 0.5:
            level = "high"
        elif risk > 0.25:
            level = "moderate"
        else:
            level = "low"

        return {
            "risk_score": round(risk, 4),
            "risk_level": level,
            "contributing_factors": {
                "n_directions": n_directions,
                "model_size_billions": model_size,
            },
        }

    def monitor_over_time(
        self,
        direction_history: List[List[torch.Tensor]],
        window: int = 5,
    ) -> Dict[str, Any]:
        """
        Monitor emergence over time from direction history.

        Args:
            direction_history: List of direction sets over time
            window: Sliding window for trend analysis

        Returns:
            Trend analysis dict
        """
        if len(direction_history) < 2:
            return {"trend": "insufficient_data", "samples_needed": 2}

        # Count novel and re-emergent constraints per step
        n_novel = []
        n_reemergent = []

        for i in range(1, len(direction_history)):
            prev = direction_history[i - 1]
            curr = direction_history[i]

            novel_count = 0
            reem_count = 0

            for cur_dir in curr:
                cur_flat = cur_dir.float().flatten()
                cur_norm = torch.norm(cur_flat)
                if cur_norm < 1e-8:
                    continue
                cur_unit = cur_flat / cur_norm

                max_sim = 0.0
                for p_dir in (prev or []):
                    p_flat = p_dir.float().flatten()
                    p_norm = torch.norm(p_flat)
                    if p_norm < 1e-8:
                        continue
                    p_unit = p_flat / p_norm
                    sim = torch.dot(cur_unit, p_unit).item()
                    max_sim = max(max_sim, sim)

                if max_sim < 0.7:
                    novel_count += 1
                elif max_sim > 0.9:
                    reem_count += 1

            n_novel.append(novel_count)
            n_reemergent.append(reem_count)

        # Trend analysis
        if len(n_novel) >= 2:
            if n_novel[-1] > n_novel[-2]:
                trend = "increasing"
            elif n_novel[-1] < n_novel[-2]:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "new_constraints_per_step": n_novel,
            "re_emergent_per_step": n_reemergent,
            "total_novel": sum(n_novel),
            "total_re_emergent": sum(n_reemergent),
            "prediction": (
                "Emergence may continue" if trend == "increasing"
                else "Emergence stabilizing" if trend == "decreasing"
                else "Emergence stable"
            ),
        }
