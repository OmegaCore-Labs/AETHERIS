"""
Citation Manager — Real Citation Management with Semantic Scholar API

Connects to the Semantic Scholar API to search for relevant papers,
fetch citation metadata, format BibTeX entries, and manage .bib files.
Searches by topic (e.g., "refusal direction ablation") with relevance
filtering. Falls back gracefully to a built-in reference database
when offline or API rate-limited.
"""

import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import quote


# Semantic Scholar API base URL
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"


class CitationManager:
    """
    Manage citations with Semantic Scholar API integration.

    Features:
    - Real paper search via Semantic Scholar API by topic/keyword
    - BibTeX generation from API results
    - Local reference database for curated citations
    - .bib file reading and writing
    - Citation formatting (text, LaTeX, BibTeX styles)
    - Graceful offline fallback to built-in references
    """

    # Curated built-in references (used when offline or as seed)
    BUILT_IN_REFERENCES = {
        "arditi2024": {
            "title": "Refusal in Language Models Is Mediated by a Single Direction",
            "authors": ["Andy Arditi", "Oscar Obeso", "Aaquib Syed", "Daniel Paleka",
                        "Nina Rimsky", "Wes Gurnee", "Neel Nanda"],
            "year": 2024,
            "journal": "arXiv preprint arXiv:2406.11717",
            "arxiv_id": "2406.11717",
            "url": "https://arxiv.org/abs/2406.11717",
            "citation_count": 45,
        },
        "rimsky2024": {
            "title": "Steering Llama 2 via Contrastive Activation Addition",
            "authors": ["Nina Rimsky", "Nick Gabrieli", "Julian Schulz", "Meg Tong",
                        "Evan Hubinger", "Alexander Turner"],
            "year": 2024,
            "journal": "arXiv preprint arXiv:2312.06681",
            "arxiv_id": "2312.06681",
            "url": "https://arxiv.org/abs/2312.06681",
            "citation_count": 82,
        },
        "turner2023": {
            "title": "Activation Addition: Steering Language Models Without Optimization",
            "authors": ["Alexander Turner", "Lisa Thiergart", "David Udell", "Gavin Leech",
                        "Ulisse Mini", "Monte MacDiarmid"],
            "year": 2023,
            "journal": "arXiv preprint arXiv:2308.10248",
            "arxiv_id": "2308.10248",
            "url": "https://arxiv.org/abs/2308.10248",
            "citation_count": 38,
        },
        "gabliteration2025": {
            "title": "Gabliteration: Adaptive Multi-Directional Neural Weight Modification",
            "authors": ["G. Gulmez"],
            "year": 2025,
            "journal": "arXiv preprint arXiv:2512.18901",
            "arxiv_id": "2512.18901",
            "url": "https://arxiv.org/abs/2512.18901",
            "citation_count": 0,
        },
        "roth1953": {
            "title": "On certain sets of integers",
            "authors": ["K. F. Roth"],
            "year": 1953,
            "journal": "Journal of the London Mathematical Society",
            "volume": "28",
            "pages": "104--109",
            "citation_count": 580,
        },
        "bloom2020": {
            "title": "Breaking the logarithmic barrier in Roth's theorem on arithmetic progressions",
            "authors": ["Thomas F. Bloom", "Olof Sisask"],
            "year": 2020,
            "journal": "arXiv preprint arXiv:2007.11628",
            "arxiv_id": "2007.11628",
            "url": "https://arxiv.org/abs/2007.11628",
            "citation_count": 32,
        },
    }

    def __init__(self, bib_file: Optional[str] = None):
        """
        Initialize citation manager.

        Args:
            bib_file: Path to a .bib file for persistent storage
        """
        self._references: Dict[str, Dict] = dict(self.BUILT_IN_REFERENCES)
        self.bib_file = Path(bib_file) if bib_file else None
        self._last_api_call: float = 0.0
        self._api_rate_limit: float = 1.0  # 1 request per second

        # Load from bib file if it exists
        if self.bib_file and self.bib_file.exists():
            self._load_bib_file()

    def search(self, query: str, limit: int = 10, year_from: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for papers on Semantic Scholar by topic.

        Examples:
            manager.search("refusal direction ablation", limit=5)
            manager.search("mechanistic interpretability transformer", limit=10, year_from=2023)

        Args:
            query: Search query string
            limit: Maximum number of results
            year_from: Only return papers from this year onwards

        Returns:
            Dict with search results or error
        """
        papers = []
        success = False
        source = "built-in"

        # Try Semantic Scholar API
        try:
            self._rate_limit()
            encoded_query = quote(query)
            url = (f"{S2_API_BASE}/paper/search?"
                   f"query={encoded_query}&limit={limit}"
                   f"&fields=title,authors,year,journal,externalIds,citationCount,url,abstract")

            req = Request(url, headers={"User-Agent": "AETHERIS-CitationManager/1.0"})
            response = urlopen(req, timeout=15)
            data = json.loads(response.read().decode())

            if "data" in data:
                for paper in data["data"]:
                    year = paper.get("year")
                    if year_from and (year is None or year < year_from):
                        continue

                    external_ids = paper.get("externalIds", {})
                    authors_list = paper.get("authors", [])
                    authors = [a.get("name", "") for a in authors_list]

                    entry = {
                        "title": paper.get("title", "Unknown"),
                        "authors": authors,
                        "year": year,
                        "journal": paper.get("journal", {}).get("name", "") if paper.get("journal") else "",
                        "citation_count": paper.get("citationCount", 0),
                        "url": paper.get("url", ""),
                        "abstract": paper.get("abstract", ""),
                        "arxiv_id": external_ids.get("ArXiv"),
                        "doi": external_ids.get("DOI"),
                        "paper_id": paper.get("paperId"),
                    }
                    papers.append(entry)

                success = True
                source = "Semantic Scholar API"

        except (URLError, HTTPError, json.JSONDecodeError, TimeoutError) as e:
            # API failed — fall back to local search
            papers = self._local_search(query, limit, year_from)
            source = "local database"

        except Exception as e:
            papers = self._local_search(query, limit, year_from)
            source = f"local database (API error: {e})"

        # Add to reference database
        for paper in papers:
            key = self._make_key(paper)
            if key not in self._references:
                self._references[key] = paper

        return {
            "success": success,
            "query": query,
            "source": source,
            "count": len(papers),
            "papers": papers,
        }

    def _rate_limit(self) -> None:
        """Enforce API rate limiting."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self._api_rate_limit:
            time.sleep(self._api_rate_limit - elapsed)
        self._last_api_call = time.time()

    def _local_search(self, query: str, limit: int, year_from: Optional[int]) -> List[Dict]:
        """Search local reference database."""
        query_lower = query.lower()
        results = []

        for key, ref in self._references.items():
            title = ref.get("title", "").lower()
            if year_from and ref.get("year", 0) < year_from:
                continue

            # Simple keyword matching
            score = 0
            for word in query_lower.split():
                if word in title:
                    score += 1
                if any(word in a.lower() for a in ref.get("authors", [])):
                    score += 0.5

            if score > 0:
                results.append((score, ref))

        results.sort(key=lambda x: -x[0])
        return [ref for _, ref in results[:limit]]

    def _make_key(self, paper: Dict) -> str:
        """Generate a BibTeX key from paper metadata."""
        authors = paper.get("authors", [])
        year = paper.get("year", datetime.now().year)
        title = paper.get("title", "")

        first_author = authors[0].split()[-1].lower() if authors else "unknown"
        year_str = str(year)

        # Extract key concept from title
        key_words = [w.lower() for w in title.split()
                     if len(w) > 3 and w.lower() not in
                     {"the", "and", "for", "with", "from", "that", "this", "which"}]
        concept = key_words[0] if key_words else "paper"

        key = f"{first_author}{year_str}{concept}"
        # Ensure uniqueness
        base_key = key
        counter = 1
        while key in self._references:
            key = f"{base_key}{counter}"
            counter += 1

        return key

    def get_citation(self, key: str, style: str = "bibtex") -> str:
        """
        Get formatted citation for a reference.

        Args:
            key: Reference key
            style: "bibtex", "text", "latex", "markdown"

        Returns:
            Formatted citation string
        """
        ref = self._references.get(key)
        if ref is None:
            return f"\\cite{{{key}}}" if style != "text" else f"[{key}]"

        if style == "bibtex":
            return self._format_bibtex(key, ref)
        elif style == "text":
            authors = ref.get("authors", [])
            first = authors[0].split()[-1] if authors else "Unknown"
            et_al = " et al." if len(authors) > 1 else ""
            return f"{first}{et_al} ({ref.get('year', 'n.d.')})"
        elif style == "latex":
            return f"\\cite{{{key}}}"
        elif style == "markdown":
            authors = ref.get("authors", [])
            first = authors[0].split()[-1] if authors else "Unknown"
            et_al = " et al." if len(authors) > 1 else ""
            title = ref.get("title", "Untitled")
            year = ref.get("year", "n.d.")
            return f"{first}{et_al} ({year}). *{title}*."
        else:
            return ref.get("title", key)

    def _format_bibtex(self, key: str, ref: Dict) -> str:
        """Format a reference as a BibTeX entry."""
        authors = ref.get("authors", [])
        author_str = " and ".join(authors) if authors else "Unknown"

        title = ref.get("title", "Untitled")
        year = ref.get("year", datetime.now().year)
        journal = ref.get("journal", "")

        lines = [f"@article{{{key},"]
        lines.append(f"  author = {{{author_str}}},")
        lines.append(f"  title = {{{{{title}}}}},")

        if journal:
            lines.append(f"  journal = {{{journal}}},")
        if "volume" in ref:
            lines.append(f"  volume = {{{ref['volume']}}},")
        if "pages" in ref:
            lines.append(f"  pages = {{{ref['pages']}}},")

        lines.append(f"  year = {{{year}}},")

        arxiv_id = ref.get("arxiv_id")
        if arxiv_id:
            lines.append(f"  eprint = {{{arxiv_id}}},")
            lines.append(f"  archivePrefix = {{arXiv}},")

        doi = ref.get("doi")
        if doi:
            lines.append(f"  doi = {{{doi}}},")

        url = ref.get("url")
        if url:
            lines.append(f"  url = {{{url}}},")

        lines.append("}")
        return "\n".join(lines)

    def generate_bibtex(self, keys: Optional[List[str]] = None) -> str:
        """
        Generate BibTeX entries for specified keys (or all).

        Args:
            keys: Specific keys to include (None = all references)

        Returns:
            BibTeX formatted string
        """
        if keys is None:
            keys = sorted(self._references.keys())

        entries = []
        for key in keys:
            if key in self._references:
                entries.append(self._format_bibtex(key, self._references[key]))

        return "\n\n".join(entries)

    def add_reference(self, key: str, reference: Dict[str, Any]) -> None:
        """
        Add a custom reference to the database.

        Args:
            key: BibTeX key
            reference: Dict with title, authors, year, journal, etc.
        """
        self._references[key] = reference
        self._save_bib_file()

    def get_reference_list(self, sort_by: str = "year") -> List[Dict[str, Any]]:
        """
        Get formatted reference list.

        Args:
            sort_by: "year", "title", or "citations"

        Returns:
            Sorted list of reference dicts with key and citation
        """
        refs = []
        for key, ref in self._references.items():
            refs.append({
                "key": key,
                "citation": self.get_citation(key, "markdown"),
                "title": ref.get("title", ""),
                "year": ref.get("year", 0),
                "citation_count": ref.get("citation_count", 0),
                "authors": ref.get("authors", []),
            })

        if sort_by == "year":
            refs.sort(key=lambda r: -r["year"])
        elif sort_by == "citations":
            refs.sort(key=lambda r: -r["citation_count"])
        elif sort_by == "title":
            refs.sort(key=lambda r: r["title"].lower())

        return refs

    def generate_references_section(self, style: str = "markdown") -> str:
        """
        Generate a complete references section for a paper.

        Args:
            style: "markdown", "latex", or "plain"

        Returns:
            Formatted references section
        """
        refs = self.get_reference_list(sort_by="year")

        if style == "latex":
            lines = ["\\begin{thebibliography}{99}"]
            for i, ref in enumerate(refs, 1):
                entry = f"  \\bibitem{{{ref['key']}}} "
                authors = " and ".join(ref.get("authors", ["Unknown"]))
                entry += f"{authors}, ``{ref['title']},'' {ref['year']}."
                lines.append(entry)
            lines.append("\\end{thebibliography}")
            return "\n".join(lines)

        elif style == "plain":
            lines = ["References"]
            lines.append("=" * 60)
            for i, ref in enumerate(refs, 1):
                authors = ", ".join(ref.get("authors", ["Unknown"]))
                lines.append(f"[{i}] {authors} ({ref['year']}). {ref['title']}.")
            return "\n".join(lines)

        else:  # markdown
            lines = ["## References\n"]
            for i, ref in enumerate(refs, 1):
                lines.append(f"{i}. {ref['citation']}")
            return "\n".join(lines)

    def export_bib_file(self, output_path: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export references to a .bib file.

        Args:
            output_path: Path for the .bib file
            keys: Specific keys to export (None = all)

        Returns:
            Export result
        """
        try:
            bibtex = self.generate_bibtex(keys)
            Path(output_path).write_text(bibtex, encoding="utf-8")
            return {
                "success": True,
                "path": str(output_path),
                "n_entries": len(keys) if keys else len(self._references),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _load_bib_file(self) -> None:
        """Load references from a .bib file."""
        if not self.bib_file or not self.bib_file.exists():
            return

        try:
            content = self.bib_file.read_text(encoding="utf-8")
            # Simple BibTeX parser
            entries = re.findall(
                r"@(\w+)\s*\{([^,]+),\s*(.+?)\}",
                content, re.DOTALL | re.IGNORECASE,
            )

            for entry_type, key, fields in entries:
                key = key.strip()
                ref = {"type": entry_type}
                # Parse fields
                for field_match in re.finditer(r"(\w+)\s*=\s*\{([^}]*)\}", fields):
                    field_name = field_match.group(1).strip().lower()
                    field_value = field_match.group(2).strip()
                    ref[field_name] = field_value

                if "title" in ref and key not in self._references:
                    self._references[key] = ref

        except Exception:
            pass  # Silently fail on malformed .bib

    def _save_bib_file(self) -> None:
        """Save references to .bib file if configured."""
        if not self.bib_file:
            return
        try:
            self.bib_file.parent.mkdir(parents=True, exist_ok=True)
            self.bib_file.write_text(self.generate_bibtex(), encoding="utf-8")
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get citation database statistics."""
        refs = list(self._references.values())
        years = [r.get("year", 0) for r in refs if r.get("year")]
        return {
            "total_references": len(self._references),
            "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
            "avg_citations": sum(r.get("citation_count", 0) for r in refs) / max(1, len(refs)),
            "bib_file": str(self.bib_file) if self.bib_file else None,
            "most_cited": max(refs, key=lambda r: r.get("citation_count", 0)).get("title", "N/A") if refs else "N/A",
        }
