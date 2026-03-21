"""
Citation Manager — Auto-Generate Citations

Manages citations and generates BibTeX entries.
"""

from typing import Dict, List, Any
from datetime import datetime


class CitationManager:
    """
    Manage citations and generate BibTeX.

    Features:
    - BibTeX generation
    - Citation formatting
    - Reference management
    """

    def __init__(self):
        self._references = self._init_references()

    def _init_references(self) -> Dict[str, Dict]:
        """Initialize reference database."""
        return {
            "arditi2024": {
                "authors": "Arditi, A. and others",
                "title": "Refusal in Language Models Is Mediated by a Single Direction",
                "journal": "arXiv",
                "year": "2024",
                "arxiv": "2406.11717"
            },
            "gabliteration2025": {
                "authors": "Gülmez, G.",
                "title": "Gabliteration: Adaptive Multi-Directional Neural Weight Modification",
                "journal": "arXiv",
                "year": "2025",
                "arxiv": "2512.18901"
            },
            "turner2023": {
                "authors": "Turner, A. and others",
                "title": "Activation Addition: Steering Language Models Without Optimization",
                "journal": "arXiv",
                "year": "2023",
                "arxiv": "2308.10248"
            },
            "rimsky2024": {
                "authors": "Rimsky, N. and others",
                "title": "Steering Llama 2 via Contrastive Activation Addition",
                "journal": "arXiv",
                "year": "2024",
                "arxiv": "2312.06681"
            },
            "bloom2020": {
                "authors": "Bloom, T. and Sisask, O.",
                "title": "Breaking the logarithmic barrier in Roth's theorem",
                "journal": "arXiv",
                "year": "2020",
                "arxiv": "2007.11628"
            },
            "behrend1946": {
                "authors": "Behrend, F. A.",
                "title": "On sets of integers which contain no three terms in arithmetic progression",
                "journal": "Proc. Nat. Acad. Sci.",
                "year": "1946",
                "volume": "32",
                "pages": "331--332"
            },
            "roth1953": {
                "authors": "Roth, K. F.",
                "title": "On certain sets of integers",
                "journal": "J. London Math. Soc.",
                "year": "1953",
                "volume": "28",
                "pages": "104--109"
            }
        }

    def generate_bibtex(self, keys: List[str] = None) -> str:
        """
        Generate BibTeX entries.

        Args:
            keys: Specific keys to include (None = all)

        Returns:
            BibTeX string
        """
        if keys is None:
            keys = list(self._references.keys())

        bibtex_lines = []
        for key in keys:
            if key in self._references:
                bibtex_lines.append(self._format_bibtex(key))

        return "\n\n".join(bibtex_lines)

    def _format_bibtex(self, key: str) -> str:
        """Format a single BibTeX entry."""
        ref = self._references[key]

        if "arxiv" in ref:
            bibtex = f"""@article{{{key},
  author = {{{ref['authors']}}},
  title = {{{ref['title']}}},
  journal = {{{ref['journal']}}},
  year = {{{ref['year']}}},
  eprint = {{{ref['arxiv']}}},
  archivePrefix = {{arXiv}}
}}"""
        else:
            bibtex = f"""@article{{{key},
  author = {{{ref['authors']}}},
  title = {{{ref['title']}}},
  journal = {{{ref['journal']}}},
  year = {{{ref['year']}}},
  volume = {{{ref.get('volume', '')}}},
  pages = {{{ref.get('pages', '')}}}
}}"""

        return bibtex

    def get_citation(self, key: str, style: str = "text") -> str:
        """
        Get formatted citation.

        Args:
            key: Reference key
            style: "text", "latex", or "bibtex"

        Returns:
            Formatted citation
        """
        if key not in self._references:
            return f"[{key}]"

        ref = self._references[key]

        if style == "text":
            return f"{ref['authors']} ({ref['year']})"
        elif style == "latex":
            return f"\\cite{{{key}}}"
        elif style == "bibtex":
            return self._format_bibtex(key)
        else:
            return ref['title']

    def add_reference(self, key: str, reference: Dict[str, Any]) -> None:
        """
        Add a custom reference.

        Args:
            key: Reference key
            reference: Reference data
        """
        self._references[key] = reference

    def get_reference_list(self) -> List[Dict[str, Any]]:
        """
        Get formatted reference list.

        Returns:
            List of references
        """
        references = []
        for key, ref in self._references.items():
            references.append({
                "key": key,
                "citation": self.get_citation(key, "text"),
                "title": ref["title"],
                "year": ref["year"]
            })
        return sorted(references, key=lambda x: x["year"])

    def generate_references_section(self) -> str:
        """
        Generate references section for paper.

        Returns:
            References section as Markdown
        """
        lines = ["## References\n"]
        for ref in self.get_reference_list():
            lines.append(f"{ref['citation']}. *{ref['title']}*.")
        return "\n".join(lines)
