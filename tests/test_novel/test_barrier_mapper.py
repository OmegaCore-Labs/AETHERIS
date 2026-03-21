"""
Tests for BarrierMapper
"""

import pytest
from aetheris.novel.barrier_mapper import BarrierMapper, BarrierAnalysis


class TestBarrierMapper:
    """Test suite for BarrierMapper."""

    def test_initialization(self):
        """Test mapper initialization."""
        mapper = BarrierMapper()
        assert mapper._theorem_database is not None
        assert "shell_method" in mapper._theorem_database

    def test_map_barrier_geometry_shell_method(self):
        """Test mapping shell-method barrier."""
        mapper = BarrierMapper()
        analysis = mapper.map_barrier_geometry("shell_method")

        assert isinstance(analysis, BarrierAnalysis)
        assert analysis.theorem_name == "shell_method"
        assert analysis.constraint_direction == "spherical_code_dependency"
        assert analysis.rank == 3

    def test_map_barrier_geometry_roth_theorem(self):
        """Test mapping Roth's theorem barrier."""
        mapper = BarrierMapper()
        analysis = mapper.map_barrier_geometry("roth_theorem")

        assert isinstance(analysis, BarrierAnalysis)
        assert analysis.theorem_name == "roth_theorem"
        assert analysis.threshold == "o(N)"

    def test_map_custom_barrier(self):
        """Test mapping custom barrier."""
        mapper = BarrierMapper()
        analysis = mapper._analyze_custom_barrier("custom_theorem")

        assert isinstance(analysis, BarrierAnalysis)
        assert analysis.theorem_name == "custom_theorem"
        assert analysis.barrier_type == "custom"

    def test_generate_bypass_strategy(self):
        """Test bypass strategy generation."""
        mapper = BarrierMapper()
        analysis = mapper.map_barrier_geometry("shell_method")
        strategy = mapper.generate_bypass_strategy(analysis)

        assert "strategy" in strategy
        assert "implementation" in strategy
        assert "expected_improvement" in strategy

    def test_compare_barriers(self):
        """Test barrier comparison."""
        mapper = BarrierMapper()
        comparison = mapper.compare_barriers("shell_method", "roth_theorem")

        assert "theorem1" in comparison
        assert "theorem2" in comparison
        assert "similarity" in comparison
        assert "technique_transfer" in comparison
