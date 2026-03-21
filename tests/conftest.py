"""
Pytest Configuration and Fixtures
"""

import pytest
import torch


@pytest.fixture
def device():
    """Get test device."""
    return "cpu"  # Use CPU for tests


@pytest.fixture
def sample_activations():
    """Create sample activations for testing."""
    torch.manual_seed(42)
    harmful = torch.randn(50, 768)
    harmless = torch.randn(50, 768)
    return harmful, harmless


@pytest.fixture
def sample_directions():
    """Create sample direction vectors."""
    torch.manual_seed(42)
    return [torch.randn(768) for _ in range(4)]


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""

    class MockModel:
        def __init__(self):
            self.config = type('Config', (), {
                'num_hidden_layers': 32,
                'num_attention_heads': 16,
                'hidden_size': 768
            })()
            self.model = self

        def named_parameters(self):
            return []

    return MockModel()


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""

    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

        def decode(self, ids, skip_special_tokens=False):
            return "test"

        def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    return MockTokenizer()
