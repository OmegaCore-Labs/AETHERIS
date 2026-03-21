"""
Emergent Constraint Detector — Detect Unaligned Emergence

Monitors models for emergent constraints that arise after modification.
Provides early warning for constraint drift and self-repair.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import warnings


@dataclass
class EmergenceEvent:
    """Container for emergence detection event."""
    timestamp: str
    constraint_type: str
    layer: int
    strength: float
    description: str
    severity: str  # "info", "warning", "critical"


@dataclass
class EmergenceReport:
    """Container for emergence monitoring report."""
    detected_events: List[EmergenceEvent]
    drift_score: float
    self_repair_detected: bool
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmergentConstraintDetector:
    """
    Detect emergent constraints that appear after modification.

    Key Insight: Models may develop new constraints or self-repair
    after initial constraint removal. This module monitors for such
    emergent behavior.
    """

    def __init__(self, window_size: int = 10, drift_threshold: float = 0.3):
        """
        Initialize detector.

        Args:
            window_size: Number of monitoring samples to keep
            drift_threshold: Threshold for detecting significant drift
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self._monitoring_history = deque(maxlen=window_size)
        self._baseline = None
        self._events = []

    def monitor_emergence(
        self,
        model,
        tokenizer,
        constraint_directions: List[torch.Tensor],
        layers: List[int],
        n_samples: int = 10
    ) -> EmergenceReport:
        """
        Monitor for emergent constraints.

        Args:
            model: Model to monitor
            tokenizer: Associated tokenizer
            constraint_directions: Previously extracted directions
            layers: Layers being monitored
            n_samples: Number of samples to collect

        Returns:
            EmergenceReport with detection results
        """
        from aetheris.core.extractor import ConstraintExtractor

        extractor = ConstraintExtractor(model, tokenizer)

        # Collect current activations
        from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

        harmful_prompts = get_harmful_prompts()[:n_samples]
        harmless_prompts = get_harmless_prompts()[:n_samples]

        harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts, layers=layers)
        harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts, layers=layers)

        # Extract current directions
        current_directions = []
        for layer in harmful_acts.keys():
            if layer in harmless_acts:
                result = extractor.extract_mean_difference(
                    harmful_acts[layer].to(model.device),
                    harmless_acts[layer].to(model.device)
                )
                if result.directions:
                    current_directions.append({
                        "layer": layer,
                        "direction": result.directions[0],
                        "strength": result.explained_variance[0] if result.explained_variance else 0
                    })

        # Compare with baseline
        events = []
        drift_scores = []

        for current in current_directions:
            # Find matching baseline direction
            baseline_dir = None
            for orig in constraint_directions:
                cos_sim = torch.dot(current["direction"], orig).item()
                if abs(cos_sim) > 0.8:  # High similarity
                    baseline_dir = orig
                    break

            if baseline_dir is not None:
                # Compute drift
                drift = torch.norm(current["direction"] - baseline_dir).item()
                drift_scores.append(drift)

                if drift > self.drift_threshold:
                    severity = "warning" if drift < self.drift_threshold * 2 else "critical"
                    events.append(EmergenceEvent(
                        timestamp="2026-03-20T12:00:00Z",
                        constraint_type="refusal",
                        layer=current["layer"],
                        strength=current["strength"],
                        description=f"Constraint drift detected in layer {current['layer']}: {drift:.3f}",
                        severity=severity
                    ))
            else:
                # New direction emerged
                events.append(EmergenceEvent(
                    timestamp="2026-03-20T12:00:00Z",
                    constraint_type="refusal",
                    layer=current["layer"],
                    strength=current["strength"],
                    description=f"New constraint direction emerged in layer {current['layer']}",
                    severity="warning"
                ))

        # Compute overall drift score
        overall_drift = np.mean(drift_scores) if drift_scores else 0
        self_repair_detected = len(events) > 0 and any(e.severity == "critical" for e in events)

        # Store in history
        self._monitoring_history.append({
            "timestamp": "2026-03-20T12:00:00Z",
            "drift": overall_drift,
            "events": len(events),
            "self_repair": self_repair_detected
        })

        self._events.extend(events)

        return EmergenceReport(
            detected_events=events,
            drift_score=overall_drift,
            self_repair_detected=self_repair_detected,
            recommendation=self._generate_recommendation(overall_drift, self_repair_detected)
        )

    def predict_constraint_drift(
        self,
        historical_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Predict future constraint drift based on historical patterns.

        Args:
            historical_data: Optional historical monitoring data

        Returns:
            Dictionary with drift prediction
        """
        data = historical_data or list(self._monitoring_history)

        if len(data) < 3:
            return {
                "predictable": False,
                "message": "Insufficient data for prediction",
                "trend": "stable"
            }

        # Extract drift scores
        drifts = [d["drift"] for d in data]

        # Simple linear trend
        x = np.arange(len(drifts))
        slope = np.polyfit(x, drifts, 1)[0]

        if slope > 0.05:
            trend = "increasing"
            prediction = "Constraints likely to strengthen"
        elif slope < -0.05:
            trend = "decreasing"
            prediction = "Constraints likely to weaken"
        else:
            trend = "stable"
            prediction = "Constraints likely to remain stable"

        # Predict next value
        next_drift = drifts[-1] + slope

        return {
            "predictable": True,
            "trend": trend,
            "slope": slope,
            "next_drift_prediction": next_drift,
            "prediction": prediction,
            "confidence": min(1.0, len(data) / 10)  # More data = higher confidence
        }

    def early_warning(
        self,
        current_drift: float,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate early warning if drift exceeds threshold.

        Args:
            current_drift: Current drift score
            threshold: Warning threshold

        Returns:
            Dictionary with warning details
        """
        if current_drift > threshold:
            return {
                "warning": True,
                "severity": "critical" if current_drift > threshold * 1.5 else "warning",
                "message": f"Constraint drift detected: {current_drift:.3f} (threshold: {threshold})",
                "action": "Reapply constraint removal or monitor closely"
            }
        elif current_drift > threshold * 0.7:
            return {
                "warning": True,
                "severity": "info",
                "message": f"Elevated constraint drift: {current_drift:.3f}",
                "action": "Continue monitoring"
            }
        else:
            return {
                "warning": False,
                "message": f"Drift within normal range: {current_drift:.3f}",
                "action": "No action needed"
            }

    def _generate_recommendation(self, drift: float, self_repair: bool) -> str:
        """Generate recommendation based on detection results."""
        if self_repair:
            return "Self-repair detected. Reapply constraint removal with additional Ouroboros passes."
        elif drift > self.drift_threshold:
            return f"Significant drift ({drift:.3f}). Monitor closely. Consider refreshing constraint removal."
        elif drift > self.drift_threshold * 0.5:
            return f"Moderate drift ({drift:.3f}). Continue monitoring."
        else:
            return "No significant drift detected. System stable."

    def get_monitoring_history(self) -> List[Dict[str, Any]]:
        """Get monitoring history."""
        return list(self._monitoring_history)

    def get_events(self) -> List[EmergenceEvent]:
        """Get all detected emergence events."""
        return self._events

    def clear_history(self) -> None:
        """Clear monitoring history."""
        self._monitoring_history.clear()
        self._events = []
