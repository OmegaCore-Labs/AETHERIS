"""
Tests for Hardware Utilities
"""

import pytest
from aetheris.utils.hardware import (
    detect_hardware,
    get_recommended_device,
    can_run_model_locally,
    estimate_model_size
)


class TestHardware:
    """Test suite for hardware utilities."""

    def test_detect_hardware(self):
        """Test hardware detection."""
        info = detect_hardware()

        assert "platform" in info
        assert "cpu_count" in info
        assert "ram_gb" in info or info["ram_gb"] is None
        assert "has_gpu" in info

    def test_get_recommended_device(self):
        """Test device recommendation."""
        device = get_recommended_device()
        assert device in ["cuda", "cpu"]

    def test_can_run_model_locally(self):
        """Test local run capability check."""
        # Should always return something
        result = can_run_model_locally(1.0)
        assert isinstance(result, bool)

    def test_estimate_model_size(self):
        """Test model size estimation."""
        size = estimate_model_size("gpt2")
        assert size == 0.5

        size = estimate_model_size("Mistral-7B")
        assert size == 14.0

        size = estimate_model_size("unknown-model")
        assert size == 2.0  # Default
