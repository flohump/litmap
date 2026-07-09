from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import numpy as np

from litmap.embedder import (
    load_all_embeddings,
    load_chunk_vectors,
    EMBEDDINGS_DB,
    MANUSCRIPT_KEY,
)


def find_similar(
    query_embedding: np.ndarray,
    db_path: Path = EMBEDDINGS_DB,
    scope_keys: Optional[list[str]] = None,
    top_k: int = 10,
    exclude_key: Optional[str] = None,
) -> list[dict]:
    """Return the top_k papers most similar to query_embedding.

    A paper's score is the cosine of its BEST full-text chunk, not of a single
    vector averaged over the whole document. Averaging buried a paper's one
    relevant section under everything else it discussed, so long, specific papers
    lost to short, vaguely-on-topic ones. Papers with no full text are scored on
    their title+abstract vector.

    Each result: {"key": str, "similarity": float}
    """
    ta_matrix, ta_keys = load_all_embeddings(db_path, scope_keys)
    chunk_matrix, chunk_owners = load_chunk_vectors(db_path, scope_keys)

    chunked = set(chunk_owners)
    fallback_rows = [i for i, k in enumerate(ta_keys) if k not in chunked]

    parts: list[np.ndarray] = []
    owners: list[str] = []
    if len(chunk_owners):
        parts.append(chunk_matrix)
        owners.extend(chunk_owners)
    if fallback_rows:
        parts.append(ta_matrix[fallback_rows])
        owners.extend(ta_keys[i] for i in fallback_rows)
    if not owners:
        return []

    matrix = np.vstack(parts)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    matrix_norm = matrix / norms

    q_norm = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-10)
    similarities = matrix_norm @ q_norm  # one score per row (chunk or fallback)

    best: dict[str, float] = {}
    for key, sim in zip(owners, similarities):
        if key == exclude_key or key == MANUSCRIPT_KEY:
            continue
        sim = float(sim)
        if sim > best.get(key, -np.inf):
            best[key] = sim

    results = [{"key": k, "similarity": s} for k, s in best.items()]
    results.sort(key=lambda x: (-x["similarity"], x["key"]))
    return results[:top_k]


_DOI_PREFIX_RE = re.compile(r'^https?://(?:dx\.)?doi\.org/', re.IGNORECASE)


def _normalise_doi(raw: str) -> str:
    return _DOI_PREFIX_RE.sub("", raw.strip()).lower()


def deduplicate_results(
    enriched: list[dict],
    top_k: int,
) -> list[dict]:
    """Collapse duplicate papers from enriched search results.

    Two passes of deduplication:
    1. Normalised DOI — strips https://doi.org/ and http://dx.doi.org/ prefixes
       so the same paper stored as both a URL-form and bare DOI is caught.
    2. Normalised title — catches preprint/published-version pairs that share a
       title but carry different DOIs (e.g. bioRxiv preprint vs journal article).

    Input must already be sorted by descending similarity — the first occurrence
    of each canonical key is kept (highest score wins).
    Returns at most top_k entries.
    """
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[dict] = []
    for r in enriched:
        doi = _normalise_doi(r.get("doi") or "")
        title = (r.get("title") or "").strip().lower()
        if (doi and doi in seen_dois) or (title and title in seen_titles):
            continue
        if doi:
            seen_dois.add(doi)
        if title:
            seen_titles.add(title)
        unique.append(r)
        if len(unique) == top_k:
            break
    return unique


# Matches: single ALL-CAPS acronyms (>=2 chars), or capitalised words/phrases
_PROPER_NOUN_RE = re.compile(
    r'\b[A-Z]{2,}\b'                          # ALL-CAPS acronyms: GBIF, IPCC
    r'|'
    r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}\b' # Title Case phrases: MaxEnt, Climate Change
)


def extract_proper_nouns(text: str) -> list[str]:
    """Extract capitalised phrases and acronyms from a sentence."""
    matches = _PROPER_NOUN_RE.findall(text)
    # Deduplicate, filter common English sentence-starters
    _SKIP = {"The", "A", "An", "In", "We", "Our", "This", "These", "For", "To"}
    seen: set[str] = set()
    result = []
    for m in matches:
        if m not in _SKIP and m not in seen:
            seen.add(m)
            result.append(m)
    return result
