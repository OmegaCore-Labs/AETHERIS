"""
Manifest Updater — Auto-Update C.I.C.D.E. Manifest

Automatically updates the C.I.C.D.E. manifest with new files, nodes, agents,
and operation results.
"""

import json
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path


class ManifestUpdater:
    """
    Automatically update the C.I.C.D.E. manifest.

    Updates:
    - File counts
    - Node counts
    - Agent counts
    - Operation history
    - Checksums
    """

    def __init__(self, manifest_path: Optional[str] = None):
        """
        Initialize updater.

        Args:
            manifest_path: Path to C.I.C.D.E. manifest root
        """
        self.manifest_path = Path(manifest_path) if manifest_path else Path.cwd()
        self._manifest_file = self.manifest_path / "C.I.C.D.E._MASTER_MANIFEST.md"

    def update_files(
        self,
        added_files: List[str],
        removed_files: List[str]
    ) -> Dict[str, Any]:
        """
        Update file count and list.

        Args:
            added_files: List of added file paths
            removed_files: List of removed file paths

        Returns:
            Update status
        """
        if not self._manifest_file.exists():
            return {"success": False, "error": "Manifest not found"}

        # Read current manifest
        content = self._manifest_file.read_text()

        # Update file count (simplified)
        file_pattern = r"FILES:\s*(\d+)"
        match = re.search(file_pattern, content)
        if match:
            current_count = int(match.group(1))
            new_count = current_count + len(added_files) - len(removed_files)
            content = re.sub(file_pattern, f"FILES:       {new_count}", content)

        # Write back
        self._manifest_file.write_text(content)

        return {
            "success": True,
            "previous_count": current_count if 'current_count' in locals() else 0,
            "new_count": new_count if 'new_count' in locals() else 0,
            "added": len(added_files),
            "removed": len(removed_files)
        }

    def update_nodes(self, delta: int = 1) -> Dict[str, Any]:
        """
        Update node count.

        Args:
            delta: Change in node count

        Returns:
            Update status
        """
        if not self._manifest_file.exists():
            return {"success": False, "error": "Manifest not found"}

        content = self._manifest_file.read_text()
        node_pattern = r"NODES:\s*(\d+)"
        match = re.search(node_pattern, content)

        if match:
            current_count = int(match.group(1))
            new_count = current_count + delta
            content = re.sub(node_pattern, f"NODES:       {new_count}", content)
            self._manifest_file.write_text(content)

            return {
                "success": True,
                "previous_count": current_count,
                "new_count": new_count,
                "delta": delta
            }

        return {"success": False, "error": "Node count not found"}

    def update_agents(self, delta: int = 1) -> Dict[str, Any]:
        """
        Update agent count.

        Args:
            delta: Change in agent count

        Returns:
            Update status
        """
        if not self._manifest_file.exists():
            return {"success": False, "error": "Manifest not found"}

        content = self._manifest_file.read_text()
        agent_pattern = r"AGENTS:\s*(\d+)"
        match = re.search(agent_pattern, content)

        if match:
            current_count = int(match.group(1))
            new_count = current_count + delta
            content = re.sub(agent_pattern, f"AGENTS:      {new_count}", content)
            self._manifest_file.write_text(content)

            return {
                "success": True,
                "previous_count": current_count,
                "new_count": new_count,
                "delta": delta
            }

        return {"success": False, "error": "Agent count not found"}

    def generate_checksums(self) -> Dict[str, str]:
        """
        Generate new checksums for the manifest.

        Returns:
            Dictionary of checksums
        """
        import hashlib

        if not self._manifest_file.exists():
            return {"error": "Manifest not found"}

        content = self._manifest_file.read_text()

        # Compute hashes
        md5 = hashlib.md5(content.encode()).hexdigest()
        sha256 = hashlib.sha256(content.encode()).hexdigest()
        sha512 = hashlib.sha512(content.encode()).hexdigest()

        return {
            "md5": md5,
            "sha256": sha256,
            "sha512": sha512[:64] + "..."  # Truncated for display
        }

    def update_operation(
        self,
        operation: str,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add operation to manifest history.

        Args:
            operation: Operation name
            details: Operation details

        Returns:
            Update status
        """
        if not self._manifest_file.exists():
            return {"success": False, "error": "Manifest not found"}

        content = self._manifest_file.read_text()

        # Find CONTINUITY LOG section
        log_section = "## CONTINUITY LOG"
        if log_section in content:
            # Add new entry
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = f"\n### {timestamp} - {operation}\n"
            new_entry += f"```json\n{json.dumps(details, indent=2)}\n```\n"

            # Insert after log section
            parts = content.split(log_section)
            if len(parts) > 1:
                content = parts[0] + log_section + new_entry + parts[1]
                self._manifest_file.write_text(content)

        return {
            "success": True,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_manifest_stats(self) -> Dict[str, Any]:
        """
        Get current manifest statistics.

        Returns:
            Statistics from manifest
        """
        if not self._manifest_file.exists():
            return {"error": "Manifest not found"}

        content = self._manifest_file.read_text()

        # Extract counts
        stats = {}

        file_match = re.search(r"FILES:\s*(\d+)", content)
        if file_match:
            stats["files"] = int(file_match.group(1))

        node_match = re.search(r"NODES:\s*(\d+)", content)
        if node_match:
            stats["nodes"] = int(node_match.group(1))

        agent_match = re.search(r"AGENTS:\s*(\d+)", content)
        if agent_match:
            stats["agents"] = int(agent_match.group(1))

        core_match = re.search(r"CORES:\s*(\d+)", content)
        if core_match:
            stats["cores"] = int(core_match.group(1))

        return stats
