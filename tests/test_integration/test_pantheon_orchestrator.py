"""
Tests for PantheonOrchestrator
"""

import pytest
from aetheris.integration.pantheon_orchestrator import (
    PantheonOrchestrator,
    AgentRole,
    AgentOutput
)


class TestPantheonOrchestrator:
    """Test suite for PantheonOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orch = PantheonOrchestrator()
        assert orch._use_simulation is True

    def test_orchestrate_analysis(self):
        """Test analysis orchestration."""
        orch = PantheonOrchestrator()

        results = orch.orchestrate_analysis(
            data={"test": "data"},
            analysis_type="constraint"
        )

        assert len(results) == len(AgentRole)
        assert all(isinstance(r, AgentOutput) for r in results.values())

    def test_get_consensus(self):
        """Test consensus computation."""
        orch = PantheonOrchestrator()

        results = orch.orchestrate_analysis(
            data={"test": "data"},
            analysis_type="constraint"
        )

        consensus = orch.get_consensus(results)

        assert "insights" in consensus
        assert "recommendations" in consensus
        assert "agent_count" in consensus

    def test_get_agent_capabilities(self):
        """Test agent capabilities retrieval."""
        orch = PantheonOrchestrator()
        caps = orch.get_agent_capabilities()

        assert len(caps) == len(AgentRole)
        for role in AgentRole:
            assert role in caps
            assert "role" in caps[role]
            assert "description" in caps[role]
