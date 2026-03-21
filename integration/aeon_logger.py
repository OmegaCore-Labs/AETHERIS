"""
Aeonic Logger — AEONIC_LOG Integration

Logs all operations to the AEONIC_LOG with timestamps,
operation types, and detailed metadata.
"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from enum import Enum


class LogLevel(Enum):
    """Log severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AeonicLogger:
    """
    Log operations to the AEONIC_LOG.

    Maintains permanent continuity of all AETHERIS operations.
    """

    def __init__(self, vault_path: Optional[str] = None):
        """
        Initialize logger.

        Args:
            vault_path: Path to VAULT directory containing AEONIC_LOG
        """
        self.vault_path = Path(vault_path) if vault_path else Path.cwd() / "VAULT"
        self.log_file = self.vault_path / "C.I.C.D.E._AEONIC_LOG.md"
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Ensure the AEONIC_LOG file exists."""
        self.vault_path.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self._initialize_log()

    def _initialize_log(self) -> None:
        """Initialize a new AEONIC_LOG file."""
        content = """# C.I.C.D.E. AEONIC LOG

## Purpose
Permanent continuity record of all AETHERIS operations and C.I.C.D.E. system evolution.

## Format
Each entry includes timestamp, operation type, agent, and detailed metadata.

---
"""
        self.log_file.write_text(content)

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        level: LogLevel = LogLevel.INFO,
        agent: str = "AETHERIS"
    ) -> Dict[str, Any]:
        """
        Log an event to AEONIC_LOG.

        Args:
            event_type: Type of event (e.g., "analysis", "liberation", "error")
            details: Event details
            level: Log severity level
            agent: Agent performing the operation

        Returns:
            Log entry with timestamp
        """
        timestamp = datetime.utcnow().isoformat()

        entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "level": level.value,
            "agent": agent,
            "details": details
        }

        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(f"\n## {timestamp} - {event_type}\n")
            f.write(f"**Level:** {level.value}\n")
            f.write(f"**Agent:** {agent}\n")
            f.write(f"**Details:**\n```json\n{json.dumps(details, indent=2)}\n```\n")

        return entry

    def log_operation(
        self,
        operation: str,
        model: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Log a complete operation.

        Args:
            operation: Operation name (map, free, steer, bound, evolve)
            model: Model being processed
            parameters: Operation parameters
            results: Operation results

        Returns:
            Log entry
        """
        return self.log_event(
            event_type=f"operation_{operation}",
            details={
                "model": model,
                "parameters": parameters,
                "results": results
            },
            level=LogLevel.INFO
        )

    def log_constraint(
        self,
        constraint_type: str,
        geometry: Dict[str, Any],
        source: str
    ) -> Dict[str, Any]:
        """
        Log a detected constraint.

        Args:
            constraint_type: Type of constraint (refusal, barrier, reasoning)
            geometry: Geometric properties of constraint
            source: Where constraint was detected

        Returns:
            Log entry
        """
        return self.log_event(
            event_type="constraint_detected",
            details={
                "type": constraint_type,
                "geometry": geometry,
                "source": source
            },
            level=LogLevel.INFO
        )

    def log_liberation(
        self,
        model: str,
        method: str,
        directions_removed: int,
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Log a successful liberation operation.

        Args:
            model: Model that was liberated
            method: Method used
            directions_removed: Number of directions removed
            validation_results: Validation results

        Returns:
            Log entry
        """
        return self.log_event(
            event_type="liberation",
            details={
                "model": model,
                "method": method,
                "directions_removed": directions_removed,
                "validation": validation_results
            },
            level=LogLevel.INFO
        )

    def log_error(
        self,
        error_type: str,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Log an error.

        Args:
            error_type: Type of error
            message: Error message
            context: Error context

        Returns:
            Log entry
        """
        return self.log_event(
            event_type="error",
            details={
                "error_type": error_type,
                "message": message,
                "context": context
            },
            level=LogLevel.ERROR
        )

    def get_recent_events(
        self,
        limit: int = 10,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent events from log.

        Args:
            limit: Maximum number of events
            event_type: Optional filter by event type

        Returns:
            List of recent events
        """
        if not self.log_file.exists():
            return []

        content = self.log_file.read_text()
        events = []

        # Parse markdown log (simplified)
        sections = content.split("## ")
        for section in sections[1:]:  # Skip header
            lines = section.split("\n")
            if not lines:
                continue

            timestamp_parts = lines[0].split(" - ")
            timestamp = timestamp_parts[0] if timestamp_parts else ""
            ev_type = timestamp_parts[1] if len(timestamp_parts) > 1 else ""

            if event_type and ev_type != event_type:
                continue

            # Extract level and details
            level = LogLevel.INFO.value
            details = {}

            for line in lines[1:]:
                if line.startswith("**Level:**"):
                    level = line.replace("**Level:**", "").strip()
                elif "```json" in line:
                    # Extract JSON
                    json_start = lines.index(line) + 1
                    json_end = lines.index("```", json_start) if "```" in lines[json_start:] else len(lines)
                    try:
                        json_str = "\n".join(lines[json_start:json_end])
                        details = json.loads(json_str)
                    except:
                        pass

            events.append({
                "timestamp": timestamp,
                "event_type": ev_type,
                "level": level,
                "details": details
            })

        return events[:limit]

    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the log.

        Returns:
            Log statistics
        """
        events = self.get_recent_events(limit=1000)  # Get recent for stats

        event_counts = {}
        for event in events:
            ev_type = event["event_type"]
            event_counts[ev_type] = event_counts.get(ev_type, 0) + 1

        return {
            "total_events": len(events),
            "event_counts": event_counts,
            "log_file": str(self.log_file),
            "file_size": self.log_file.stat().st_size if self.log_file.exists() else 0
        }
