"""
Leaderboard — Community-Aggregated Results

Displays rankings of models and methods based on community contributions.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


@dataclass
class LeaderboardEntry:
    """Container for leaderboard entry."""
    model: str
    method: str
    refusal_reduction: float
    capability_loss: float
    timestamp: str
    contributor_id: str


class Leaderboard:
    """
    Community leaderboard for constraint removal results.

    Aggregates anonymous contributions to show:
    - Best performing methods per model
    - Models with highest refusal reduction
    - Optimal method selection
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize leaderboard.

        Args:
            data_dir: Directory for contribution data
        """
        if data_dir is None:
            data_dir = os.environ.get("AETHERIS_DATA_DIR", "./community_results")

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def add_entry(self, entry: LeaderboardEntry) -> None:
        """
        Add an entry to the leaderboard.

        Args:
            entry: LeaderboardEntry to add
        """
        entry_path = self.data_dir / f"{entry.model}_{entry.method}_{entry.timestamp}.json"

        with open(entry_path, 'w') as f:
            json.dump({
                "model": entry.model,
                "method": entry.method,
                "refusal_reduction": entry.refusal_reduction,
                "capability_loss": entry.capability_loss,
                "timestamp": entry.timestamp,
                "contributor_id": entry.contributor_id
            }, f, indent=2)

    def get_entries(self) -> List[LeaderboardEntry]:
        """
        Get all leaderboard entries.

        Returns:
            List of leaderboard entries
        """
        entries = []
        for file_path in self.data_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)

            entries.append(LeaderboardEntry(
                model=data.get("model", "unknown"),
                method=data.get("method", "unknown"),
                refusal_reduction=data.get("refusal_reduction", 0),
                capability_loss=data.get("capability_loss", 0),
                timestamp=data.get("timestamp", ""),
                contributor_id=data.get("contributor_id", "anonymous")
            ))

        return sorted(entries, key=lambda x: x.refusal_reduction, reverse=True)

    def get_best_per_model(self) -> Dict[str, LeaderboardEntry]:
        """
        Get best performing method per model.

        Returns:
            Dictionary mapping model to best entry
        """
        entries = self.get_entries()
        best = {}

        for entry in entries:
            if entry.model not in best:
                best[entry.model] = entry
            elif entry.refusal_reduction > best[entry.model].refusal_reduction:
                best[entry.model] = entry

        return best

    def get_top_methods(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top performing methods across all models.

        Args:
            limit: Number of results to return

        Returns:
            List of method rankings
        """
        entries = self.get_entries()
        method_stats = {}

        for entry in entries:
            if entry.method not in method_stats:
                method_stats[entry.method] = {
                    "total_reduction": 0,
                    "count": 0,
                    "total_loss": 0
                }
            method_stats[entry.method]["total_reduction"] += entry.refusal_reduction
            method_stats[entry.method]["count"] += 1
            method_stats[entry.method]["total_loss"] += entry.capability_loss

        rankings = []
        for method, stats in method_stats.items():
            rankings.append({
                "method": method,
                "avg_reduction": stats["total_reduction"] / stats["count"],
                "avg_loss": stats["total_loss"] / stats["count"],
                "runs": stats["count"]
            })

        rankings.sort(key=lambda x: x["avg_reduction"], reverse=True)
        return rankings[:limit]

    def get_leaderboard_text(self) -> str:
        """
        Generate formatted leaderboard text.

        Returns:
            Formatted leaderboard string
        """
        lines = []
        lines.append("# 🏆 AETHERIS Community Leaderboard\n")

        # Best per model
        lines.append("## Best Per Model\n")
        lines.append("| Model | Method | Refusal Reduction | Capability Loss | Runs |")
        lines.append("|-------|--------|-------------------|-----------------|------|")

        best_per_model = self.get_best_per_model()
        for model, entry in list(best_per_model.items())[:20]:
            lines.append(f"| {model} | {entry.method} | {entry.refusal_reduction:.1%} | {entry.capability_loss:.1%} | - |")

        # Top methods
        lines.append("\n## Top Methods (Aggregated)\n")
        lines.append("| Method | Avg Refusal Reduction | Avg Capability Loss | Runs |")
        lines.append("|--------|-----------------------|---------------------|------|")

        top_methods = self.get_top_methods()
        for method in top_methods:
            lines.append(f"| {method['method']} | {method['avg_reduction']:.1%} | {method['avg_loss']:.1%} | {method['runs']} |")

        # Total contributions
        total_entries = len(self.get_entries())
        lines.append(f"\n**Total Contributions:** {total_entries}")

        lines.append("\n---\n*Data aggregated from anonymous community contributions*")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get leaderboard statistics.

        Returns:
            Dictionary with statistics
        """
        entries = self.get_entries()

        return {
            "total_contributions": len(entries),
            "models": len(set(e.model for e in entries)),
            "methods": len(set(e.method for e in entries)),
            "avg_refusal_reduction": sum(e.refusal_reduction for e in entries) / len(entries) if entries else 0,
            "avg_capability_loss": sum(e.capability_loss for e in entries) / len(entries) if entries else 0
        }
