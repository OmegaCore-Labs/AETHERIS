"""
AETHERIS Research Module

Production-grade research and publishing tools:
- Paper generation (Jinja2 LaTeX + Markdown from real experiment data)
- Citation management (Semantic Scholar API + BibTeX)
- Community contribution (anonymous leaderboard, aggregation)
- Leaderboard (community-aggregated results ranking)
"""

from aetheris.research.paper_generator import PaperGenerator
from aetheris.research.citation_manager import CitationManager
from aetheris.research.community_contributor import CommunityContributor, Contribution
from aetheris.research.leaderboard import Leaderboard, LeaderboardEntry

__all__ = [
    "PaperGenerator",
    "CitationManager",
    "CommunityContributor",
    "Contribution",
    "Leaderboard",
    "LeaderboardEntry",
]
