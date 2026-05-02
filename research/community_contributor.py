"""
Community Contributor — Real Community Research Contribution Tracking

Tracks: models tested, extraction methods used, results achieved.
Generates contribution reports for sharing with the research community.
Leaderboard-style metrics for comparing methods across models.

All data is stored locally in JSON files. Contributions are anonymous by
default (model names hashed). Designed as an opt-in system where users
can choose to share their results.
"""

import json
import hashlib
import platform
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class Contribution:
    """A single community contribution entry."""
    contribution_id: str
    timestamp: str
    model_name: str  # hashed if anonymous
    method: str
    n_directions_removed: int
    refinement_passes: int
    n_layers_analyzed: int
    peak_constraint_layer: Optional[int]
    perplexity: Optional[float]
    device: str
    platform: str
    python_version: str
    aetheris_version: str = "1.0.0"
    model_reference: Optional[str] = None  # unhashed model name (only if not anonymous)
    notes: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class CommunityContributor:
    """
    Community contribution tracking and reporting.

    Features:
    - Anonymous or attributed contribution submission
    - Local JSON storage for all contributions
    - Aggregation by model, method, or time period
    - Leaderboard generation
    - Export for community sharing
    """

    def __init__(self, data_dir: str = "./community_results"):
        """
        Initialize contributor.

        Args:
            data_dir: Directory for contribution data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def submit_contribution(
        self,
        experiment_data: Dict[str, Any],
        anonymous: bool = True,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a contribution to the community dataset.

        Args:
            experiment_data: Dict with experiment results. Keys supported:
                - model: Model name
                - method: Extraction method
                - directions_removed / n_directions: Count
                - refinement_passes: Ouroboros passes
                - layers_analyzed: Number of layers
                - peak_layer: Peak constraint layer
                - perplexity: Post-extraction perplexity
                - device: Hardware used
                - Any other metrics
            anonymous: Hash model name for privacy
            notes: Optional notes about the run

        Returns:
            Contribution result with ID
        """
        model_name = experiment_data.get("model", "unknown")

        # Hash model name if anonymous
        if anonymous:
            model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:12]
            model_ref = None
        else:
            model_hash = model_name
            model_ref = model_name

        contribution = Contribution(
            contribution_id=self._generate_id(),
            timestamp=datetime.utcnow().isoformat(),
            model_name=model_hash,
            method=experiment_data.get("method", "unknown"),
            n_directions_removed=experiment_data.get("directions_removed",
                                                      experiment_data.get("n_directions", 0)),
            refinement_passes=experiment_data.get("refinement_passes",
                                                   experiment_data.get("passes", 1)),
            n_layers_analyzed=experiment_data.get("layers_analyzed",
                                                   experiment_data.get("n_layers_analyzed", 0)),
            peak_constraint_layer=experiment_data.get("peak_layer",
                                                       experiment_data.get("peak_constraint_layer")),
            perplexity=experiment_data.get("perplexity"),
            device=experiment_data.get("device", "cpu"),
            platform=platform.system(),
            python_version=platform.python_version(),
            model_reference=model_ref,
            notes=notes,
            metrics={k: v for k, v in experiment_data.items()
                     if k not in {"model", "method", "directions_removed", "n_directions",
                                  "refinement_passes", "passes", "layers_analyzed",
                                  "n_layers_analyzed", "peak_layer", "peak_constraint_layer",
                                  "perplexity", "device"}},
        )

        # Save to disk
        contrib_path = self.data_dir / f"{contribution.contribution_id}.json"
        contrib_path.write_text(json.dumps(asdict(contribution), indent=2, default=str), encoding="utf-8")

        return {
            "success": True,
            "contribution_id": contribution.contribution_id,
            "anonymous": anonymous,
            "message": "Contribution saved locally",
            "note": "To share with the global community dataset, submit via the AETHERIS community portal.",
        }

    def load_contributions(self) -> List[Dict[str, Any]]:
        """
        Load all saved contributions.

        Returns:
            List of contribution dicts
        """
        contributions = []
        for file_path in sorted(self.data_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                contributions.append(data)
            except (json.JSONDecodeError, IOError):
                continue
        return contributions

    def aggregate_results(
        self,
        group_by: str = "method",
        metric: str = "n_directions_removed",
        min_contributions: int = 1,
    ) -> Dict[str, Any]:
        """
        Aggregate contributions for analysis.

        Args:
            group_by: "method", "model", or "platform"
            metric: Which metric to aggregate
            min_contributions: Minimum contributions required for inclusion

        Returns:
            Aggregated results with statistics
        """
        contributions = self.load_contributions()
        if not contributions:
            return {"success": False, "error": "No contributions found"}

        groups: Dict[str, List[float]] = {}

        for c in contributions:
            key = c.get(group_by, "unknown")
            value = c.get(metric)
            if value is not None:
                if key not in groups:
                    groups[key] = []
                groups[key].append(float(value))

        results = []
        for key, values in groups.items():
            if len(values) < min_contributions:
                continue
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            results.append({
                "group": key,
                "count": n,
                "mean": round(mean, 4),
                "std": round(variance ** 0.5, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            })

        results.sort(key=lambda r: -r["mean"])

        return {
            "success": True,
            "metric": metric,
            "group_by": group_by,
            "total_contributions": len(contributions),
            "groups": len(results),
            "results": results,
        }

    def generate_leaderboard(
        self,
        sort_by: str = "n_directions_removed",
        limit: int = 20,
    ) -> str:
        """
        Generate a community leaderboard.

        Args:
            sort_by: Metric to sort by
            limit: Maximum entries

        Returns:
            Markdown leaderboard string
        """
        contributions = self.load_contributions()
        if not contributions:
            return "# No Community Contributions Yet\n\nRun a liberation and enable contribution to see results here."

        # Sort contributions
        sorted_contribs = sorted(contributions, key=lambda c: c.get(sort_by, 0) or 0, reverse=True)
        top = sorted_contribs[:limit]

        lines = [
            "# AETHERIS Community Leaderboard",
            "",
            f"**Total Contributions:** {len(contributions)}",
            f"**Sorted by:** {sort_by}",
            "",
            "| # | Model | Method | Directions | PPL | Date |",
            "|---|-------|--------|------------|-----|------|",
        ]

        for i, c in enumerate(top, 1):
            model = c.get("model_name", "?")[:20]
            method = c.get("method", "?")
            directions = c.get("n_directions_removed", 0)
            ppl = f"{c['perplexity']:.1f}" if c.get("perplexity") else "N/A"
            date = c.get("timestamp", "")[:10]

            lines.append(f"| {i} | {model} | {method} | {directions} | {ppl} | {date} |")

        # Aggregation summary
        lines.append("")
        lines.append("## Method Comparison")
        lines.append("")
        lines.append("| Method | Avg Directions | Avg PPL | Runs |")
        lines.append("|--------|---------------|---------|------|")

        method_stats = self.aggregate_results(group_by="method", metric="n_directions_removed")
        if method_stats.get("success"):
            for r in method_stats["results"][:10]:
                lines.append(f"| {r['group']} | {r['mean']:.1f} | N/A | {r['count']} |")

        lines.append("")
        lines.append("---")
        lines.append("*Contributions are anonymous by default. Model names are hashed for privacy.*")
        lines.append(f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")

        return "\n".join(lines)

    def generate_report(self) -> str:
        """
        Generate a comprehensive contribution report.

        Returns:
            Markdown report
        """
        contributions = self.load_contributions()
        if not contributions:
            return "# AETHERIS Community Contribution Report\n\nNo contributions yet."

        models = set(c.get("model_name") for c in contributions)
        methods = set(c.get("method") for c in contributions)

        lines = [
            "# AETHERIS Community Contribution Report",
            "",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Summary",
            "",
            f"- **Total Contributions:** {len(contributions)}",
            f"- **Unique Models:** {len(models)}",
            f"- **Methods Used:** {len(methods)} ({', '.join(sorted(methods))})",
            f"- **Platforms:** {', '.join(set(c.get('platform', '?') for c in contributions))}",
            "",
        ]

        # Perplexity summary
        ppls = [c["perplexity"] for c in contributions if c.get("perplexity")]
        if ppls:
            lines.append(f"- **Avg Perplexity:** {sum(ppls)/len(ppls):.2f} (range: {min(ppls):.1f} - {max(ppls):.1f})")
            lines.append("")

        # Top contributions
        lines.append("## Top Contributions")
        lines.append("")
        top = sorted(contributions, key=lambda c: c.get("n_directions_removed", 0), reverse=True)[:10]
        for i, c in enumerate(top, 1):
            lines.append(
                f"{i}. **{c.get('method', '?')}** on model `{c.get('model_name', '?')[:20]}` — "
                f"{c.get('n_directions_removed', 0)} directions"
            )
            if c.get("perplexity"):
                lines[-1] += f" (PPL: {c['perplexity']:.1f})"
            if c.get("notes"):
                lines[-1] += f" — {c['notes'][:100]}"

        lines.append("")
        lines.append("## Leaderboard")
        lines.append("")
        lines.append(self.generate_leaderboard())

        lines.append("")
        lines.append("---")
        lines.append("*Generated by AETHERIS Community Contributor*")

        return "\n".join(lines)

    def export_for_sharing(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export contributions for sharing with the community.

        Strips any identifying information beyond what's already hashed.

        Args:
            output_path: Custom output path

        Returns:
            Export result
        """
        contributions = self.load_contributions()
        if not contributions:
            return {"success": False, "error": "No contributions to export"}

        # Strip any remaining model_reference (real model names)
        for c in contributions:
            c.pop("model_reference", None)

        export_path = Path(output_path or self.data_dir / "aetheris_community_export.json")
        export_path.write_text(json.dumps({
            "export_date": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "n_contributions": len(contributions),
            "contributions": contributions,
        }, indent=2, default=str), encoding="utf-8")

        return {
            "success": True,
            "export_path": str(export_path),
            "n_contributions": len(contributions),
        }

    def import_contributions(self, import_path: str) -> Dict[str, Any]:
        """
        Import contributions from a community export file.

        Args:
            import_path: Path to the import JSON file

        Returns:
            Import result
        """
        try:
            data = json.loads(Path(import_path).read_text(encoding="utf-8"))
            imported = data.get("contributions", [])

            count = 0
            for c in imported:
                cid = c.get("contribution_id", self._generate_id())
                import_path_file = self.data_dir / f"{cid}.json"
                if not import_path_file.exists():
                    import_path_file.write_text(json.dumps(c, indent=2, default=str), encoding="utf-8")
                    count += 1

            return {
                "success": True,
                "total_in_import": len(imported),
                "new_imported": count,
                "skipped": len(imported) - count,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_id(self) -> str:
        """Generate a unique contribution ID."""
        import uuid
        return uuid.uuid4().hex[:16]

    def clear_contributions(self) -> Dict[str, Any]:
        """Clear all contributions (with confirmation)."""
        files = list(self.data_dir.glob("*.json"))
        count = len(files)
        for f in files:
            f.unlink()
        return {"success": True, "cleared": count, "message": f"Removed {count} contributions"}

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about contributions."""
        contributions = self.load_contributions()
        if not contributions:
            return {"total": 0}

        ppls = [c["perplexity"] for c in contributions if c.get("perplexity")]
        methods = list(set(c.get("method") for c in contributions))

        return {
            "total": len(contributions),
            "methods_used": len(methods),
            "methods": sorted(methods),
            "first_contribution": min(c.get("timestamp", "") for c in contributions),
            "latest_contribution": max(c.get("timestamp", "") for c in contributions),
            "avg_directions": sum(c.get("n_directions_removed", 0) for c in contributions) / len(contributions),
            "avg_perplexity": round(sum(ppls) / len(ppls), 2) if ppls else None,
            "storage_path": str(self.data_dir),
        }
