"""Sparse vector generator for hybrid search (BM25/TF-IDF).

Generates sparse vectors for Qdrant's hybrid search. These complement
the dense Voyage-4 embeddings by capturing exact keyword matches that
semantic embeddings miss.

Critical for manufacturing manuals:
- Part numbers (AB-4521-CX) — exact match, not semantic
- Torque values (50 Nm) — specific numbers
- Equipment model names (Caterpillar C7) — proper nouns
- Technical abbreviations (MP, SOP, OT) — domain-specific

Uses a simple TF-IDF approach with a vocabulary built from the corpus.
Qdrant's IDF modifier handles the IDF weighting server-side, so we
only need to compute term frequencies.

Reference: AUDIT.md Section 2, Qdrant hybrid search docs.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Tokenization pattern: split on non-alphanumeric, keep hyphens for part numbers.
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][\w\-]*[a-zA-Z0-9]|[a-zA-Z0-9]")

# Common stop words in Spanish and English to exclude.
_STOP_WORDS = frozenset(
    {
        # Spanish
        "el",
        "la",
        "los",
        "las",
        "de",
        "del",
        "en",
        "y",
        "a",
        "que",
        "es",
        "por",
        "con",
        "para",
        "un",
        "una",
        "se",
        "no",
        "al",
        "lo",
        "su",
        "como",
        "mas",
        "o",
        "pero",
        "sus",
        "le",
        "ha",
        "me",
        "si",
        "sin",
        "sobre",
        "este",
        "ya",
        "entre",
        "cuando",
        "todo",
        "esta",
        "ser",
        "son",
        "dos",
        "tambien",
        "fue",
        "hay",
        "desde",
        "estan",
        "nos",
        "durante",
        "todos",
        "uno",
        "les",
        "ni",
        "contra",
        "otros",
        "ese",
        "eso",
        "ante",
        "ellos",
        "e",
        "esto",
        "antes",
        "algunos",
        "que",
        "unos",
        "yo",
        "otro",
        "otras",
        "otra",
        "tanto",
        "esa",
        "estos",
        "mucho",
        "quienes",
        "nada",
        "muchos",
        "cual",
        "poco",
        "ella",
        "estar",
        "estas",
        "algunas",
        "algo",
        "nosotros",
        # English
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "is",
        "are",
        "was",
        "were",
        "been",
        "has",
        "had",
        "did",
        "does",
    }
)


@dataclass
class SparseVector:
    """A sparse vector with indices and values for Qdrant."""

    indices: list[int]
    values: list[float]


@dataclass
class SparseVectorizer:
    """Generates sparse vectors from text using term frequency.

    Uses a hash-based vocabulary (no pre-built dictionary needed).
    Qdrant's IDF modifier handles corpus-level weighting server-side.
    """

    # Maximum vocabulary size (hash space).
    vocab_size: int = 30000

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase terms, excluding stop words."""
        tokens = _TOKEN_PATTERN.findall(text.lower())
        return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    def _hash_token(self, token: str) -> int:
        """Hash a token to a vocabulary index."""
        # Use Python's built-in hash, modulo vocab_size.
        return abs(hash(token)) % self.vocab_size

    def vectorize(self, text: str) -> SparseVector:
        """Generate a sparse vector from text.

        Returns term frequencies as sparse vector values. Qdrant's
        IDF modifier will weight these by inverse document frequency
        at query time.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # Count term frequencies.
        tf = Counter(self._hash_token(t) for t in tokens)

        # Apply log(1 + tf) normalization to prevent long documents
        # from dominating.
        indices = sorted(tf.keys())
        values = [math.log1p(tf[idx]) for idx in indices]

        return SparseVector(indices=indices, values=values)

    def vectorize_batch(self, texts: list[str]) -> list[SparseVector]:
        """Generate sparse vectors for a batch of texts."""
        return [self.vectorize(text) for text in texts]
