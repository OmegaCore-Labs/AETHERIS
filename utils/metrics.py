"""
Metrics Collection

Collect performance metrics, timing, and success rates for operations.
"""

import time
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime


@dataclass
class OperationMetrics:
    """Container for operation metrics."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None


class MetricsCollector:
    """
    Collect and export performance metrics.

    Features:
    - Operation timing
    - Success rate tracking
    - Export to JSON
    - Aggregated statistics
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._metrics: List[OperationMetrics] = []
        self._current: Optional[OperationMetrics] = None

    def start_operation(self, operation: str, **metadata) -> str:
        """
        Start timing an operation.

        Args:
            operation: Operation name
            **metadata: Additional metadata

        Returns:
            Operation ID (timestamp)
        """
        self._current = OperationMetrics(
            operation=operation,
            start_time=time.time(),
            metadata=metadata
        )
        return str(self._current.start_time)

    def end_operation(
        self,
        success: bool = True,
        error: Optional[str] = None,
        **metadata
    ) -> Optional[float]:
        """
        End current operation.

        Args:
            success: Whether operation succeeded
            error: Error message if failed
            **metadata: Additional metadata

        Returns:
            Duration in seconds
        """
        if not self._current:
            return None

        self._current.end_time = time.time()
        self._current.success = success
        self._current.error = error
        self._current.metadata.update(metadata)

        duration = self._current.duration
        self._metrics.append(self._current)
        self._current = None

        return duration

    def record_time(self, name: str, duration: float, **metadata) -> None:
        """
        Record a timed operation without start/end.

        Args:
            name: Operation name
            duration: Duration in seconds
            **metadata: Additional metadata
        """
        self._metrics.append(OperationMetrics(
            operation=name,
            start_time=time.time() - duration,
            end_time=time.time(),
            success=True,
            metadata=metadata
        ))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics.

        Returns:
            Dictionary with statistics
        """
        if not self._metrics:
            return {"total_operations": 0}

        stats = {
            "total_operations": len(self._metrics),
            "successful": sum(1 for m in self._metrics if m.success),
            "failed": sum(1 for m in self._metrics if not m.success),
            "operations_by_type": defaultdict(list),
            "average_duration_by_type": {},
            "total_duration": 0,
        }

        for m in self._metrics:
            stats["operations_by_type"][m.operation].append(m)
            if m.duration:
                stats["total_duration"] += m.duration

        for op, ops in stats["operations_by_type"].items():
            durations = [m.duration for m in ops if m.duration]
            if durations:
                stats["average_duration_by_type"][op] = sum(durations) / len(durations)

        stats["success_rate"] = stats["successful"] / stats["total_operations"]

        # Convert defaultdict to dict
        stats["operations_by_type"] = {
            k: len(v) for k, v in stats["operations_by_type"].items()
        }

        return stats

    def export_json(self, path: Optional[str] = None) -> str:
        """
        Export metrics to JSON.

        Args:
            path: Optional file path

        Returns:
            JSON string or file path
        """
        data = {
            "export_time": datetime.utcnow().isoformat(),
            "stats": self.get_stats(),
            "metrics": [
                {
                    "operation": m.operation,
                    "start_time": m.start_time,
                    "duration": m.duration,
                    "success": m.success,
                    "error": m.error,
                    "metadata": m.metadata
                }
                for m in self._metrics
            ]
        }

        json_str = json.dumps(data, indent=2, default=str)

        if path:
            with open(path, 'w') as f:
                f.write(json_str)
            return path

        return json_str

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics = []
        self._current = None

    def get_recent(self, limit: int = 10) -> List[OperationMetrics]:
        """Get recent operations."""
        return self._metrics[-limit:]

    def get_success_rate(self, operation: Optional[str] = None) -> float:
        """Get success rate for an operation."""
        if operation:
            ops = [m for m in self._metrics if m.operation == operation]
        else:
            ops = self._metrics

        if not ops:
            return 0.0

        successful = sum(1 for m in ops if m.success)
        return successful / len(ops)


# Global metrics collector
_default_metrics = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = MetricsCollector()
    return _default_metrics
