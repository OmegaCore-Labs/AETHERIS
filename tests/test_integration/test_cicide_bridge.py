"""
Tests for CICIDEBridge
"""

import pytest
import tempfile
from pathlib import Path
from aetheris.integration.cicide_bridge import CICIDEBridge


class TestCICIDEBridge:
    """Test suite for CICIDEBridge."""

    def test_initialization(self):
        """Test bridge initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = CICIDEBridge(tmpdir)
            assert bridge.manifest_path == Path(tmpdir)

    def test_update_aeonic_log(self):
        """Test AEONIC_LOG update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = CICIDEBridge(tmpdir)

            result = bridge.update_aeonic_log(
                operation="test",
                details={"key": "value"}
            )

            assert result is True

            # Verify file was created
            log_file = Path(tmpdir) / "VAULT" / "C.I.C.D.E._AEONIC_LOG.md"
            assert log_file.exists()

    def test_register_operation(self):
        """Test operation registration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = CICIDEBridge(tmpdir)

            result = bridge.register_operation(
                operation="map",
                model_name="gpt2",
                parameters={"method": "basic"},
                results={"success": True}
            )

            assert result["success"] is True
            assert "operation_id" in result

    def test_get_manifest_status(self):
        """Test manifest status retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = CICIDEBridge(tmpdir)
            status = bridge.get_manifest_status()

            assert "manifest_path" in status
            assert "manifest_exists" in status
            assert "last_sync" in status
