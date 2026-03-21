"""
AETHERIS Research Module

Auto-publishing and research tools:
- Paper generation
- LaTeX templates
- Community contribution
- Citation management
"""

from aetheris.research.paper_generator import PaperGenerator
from aetheris.research.community_contributor import CommunityContributor
from aetheris.research.citation_manager import CitationManager

__all__ = [
    "PaperGenerator",
    "CommunityContributor",
    "CitationManager",
]
