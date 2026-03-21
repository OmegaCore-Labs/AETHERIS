"""
Community Contributor — Opt-in Telemetry

Contribute anonymous research data to the community dataset.
"""

import json
import platform
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class CommunityContributor:
    """
    Contribute anonymous research data.

    Features:
    - Anonymous telemetry
    - Local contribution storage
    - Leaderboard data aggregation
    """

    def __init__(self, data_dir: str = "./community_results"):
        """
        Initialize contributor.

        Args:
            data_dir: Directory for contribution data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def submit_contribution(
        self,
        experiment_data: Dict[str, Any],
        anonymous: bool = True,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit contribution to community dataset.

        Args:
            experiment_data: Experiment results
            anonymous: Whether to anonymize
            notes: Optional notes about the run

        Returns:
            Contribution result
        """
        # Anonymize data
        if anonymous:
            experiment_data = self._anonymize(experiment_data)

        # Add metadata
        contribution = {
            "timestamp": datetime.utcnow().isoformat(),
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "data": experiment_data,
            "notes": notes
        }

        # Generate contribution ID
        contrib_id = hashlib.sha256(
            f"{contribution['timestamp']}{platform.node()}".encode()
        ).hexdigest()[:16]

        # Save locally
        contrib_path = self.data_dir / f"{contrib_id}.json"
        with open(contrib_path, 'w') as f:
            json.dump(contribution, f, indent=2)

        return {
            "success": True,
            "contribution_id": contrib_id,
            "message": "Contribution saved locally",
            "note": "To submit to global dataset, open a PR or enable telemetry"
        }

    def _anonymize(self, data: Dict) -> Dict:
        """Anonymize sensitive data."""
        # Remove any identifying information
        if "user" in data:
            del data["user"]
        if "ip" in data:
            del data["ip"]

        # Hash model names (optional)
        if "model" in data:
            data["model_hash"] = hashlib.sha256(data["model"].encode()).hexdigest()[:8]
            del data["model"]

        return data

    def load_contributions(self) -> List[Dict[str, Any]]:
        """
        Load all local contributions.

        Returns:
            List of contributions
        """
        contributions = []
        for file_path in self.data_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                contributions.append(json.load(f))
        return contributions

    def aggregate_results(
        self,
        metric: str = "refusal_rate",
        min_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Aggregate contributions for analysis.

        Args:
            metric: Metric to aggregate
            min_runs: Minimum runs for inclusion

        Returns:
            Aggregated results
        """
        contributions = self.load_contributions()

        if not contributions:
            return {"error": "No contributions found"}

        # Group by model
        by_model = {}
        for contrib in contributions:
            data = contrib.get("data", {})
            model = data.get("model_hash", "unknown")
            value = data.get(metric)

            if value is not None:
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(value)

        # Aggregate
        results = []
        for model, values in by_model.items():
            if len(values) >= min_runs:
                results.append({
                    "model": model,
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5
                })

        return {
            "metric": metric,
            "total_contributions": len(contributions),
            "models_analyzed": len(results),
            "results": results
        }

    def generate_leaderboard(self) -> str:
        """
        Generate leaderboard from contributions.

        Returns:
            Leaderboard as Markdown
        """
        refusal_rates = self.aggregate_results("refusal_after")

        if "error" in refusal_rates:
            return "# No contributions found"

        lines = ["# AETHERIS Community Leaderboard\n"]
        lines.append("| Model | Refusal Rate | Runs |")
        lines.append("|-------|--------------|------|")

        sorted_results = sorted(
            refusal_rates.get("results", []),
            key=lambda x: x.get("mean", 1),
            reverse=False
        )

        for r in sorted_results[:20]:
            lines.append(f"| {r['model']} | {r['mean']:.1%} | {r['count']} |")

        return "\n".join(lines)
