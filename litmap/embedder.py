from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from tqdm import tqdm

from litmap.zotero import get_all_items, Item, ZOTERO_DB

EMBEDDINGS_DB = Path.home() / "LitLake" / "embeddings.db"
MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"
DIMS = 768
_BATCH_SIZE = 32
# GTE-ModernBERT context window; leave a small margin
_MAX_TOKENS = 8000

# Full-text chunking. Retrieval scores a paper by its BEST chunk, so chunks want
# to be about the size of one argument -- a few paragraphs -- not one paper.
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_CHUNK_OVERLAP = 64
# Chunks are encoded in small batches. Attention cost is quadratic *within* a
# sequence, so 8 x 512 tokens is far lighter than the single 8000-token encode
# this replaced. Set to 1 to restore strict one-at-a-time behaviour if an MPS
# out-of-memory error ever appears.
_CHUNK_BATCH_SIZE = 8

# Synthetic key for the transient manuscript vector inserted by `litmap map -m`.
MANUSCRIPT_KEY = "__manuscript__"

_model = None
_tokenizer = None


class ModelMismatchError(RuntimeError):
    """embeddings.db holds vectors from a different embedding model."""


def _detect_device() -> str:
    """Pick a torch device. LITMAP_DEVICE overrides; otherwise mps > cuda > cpu."""
    override = os.environ.get("LITMAP_DEVICE")
    if override:
        return override
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device=_detect_device())
    return _model


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def init_db(db_path: Path = EMBEDDINGS_DB, *, expect_model: bool = True) -> None:
    """Create the schema and verify the database's embedding model.

    Raises ModelMismatchError when the stored model differs from MODEL_NAME:
    vectors from two models share no coordinate system, so similarity across
    them is meaningless. Pass expect_model=False only when about to re-embed
    everything (sync --force), which then adopts the new model.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            zotero_key  TEXT PRIMARY KEY,
            vector      BLOB NOT NULL,
            embedded_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fulltext_chunks (
            zotero_key  TEXT    NOT NULL,
            chunk_idx   INTEGER NOT NULL,
            vector      BLOB    NOT NULL,
            n_tokens    INTEGER NOT NULL,
            embedded_at TEXT    NOT NULL,
            PRIMARY KEY (zotero_key, chunk_idx)
        );
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    # `fulltext_embeddings` (one mean-pooled vector per paper) is retired: never
    # created, never written, never read. Existing databases keep the table --
    # dropping it is the user's call -- but it no longer affects any result.
    stored = conn.execute("SELECT value FROM meta WHERE key = 'model'").fetchone()
    if stored is None:
        conn.execute("INSERT INTO meta VALUES ('model', ?)", (MODEL_NAME,))
        conn.execute("INSERT INTO meta VALUES ('dims', ?)", (str(DIMS),))
    elif stored[0] != MODEL_NAME and expect_model:
        conn.close()
        raise ModelMismatchError(
            f"{db_path} was built with embedding model '{stored[0]}', "
            f"but litmap now uses '{MODEL_NAME}'. The stored vectors are not "
            f"comparable to new ones. Run 'litmap sync --force' to re-embed the "
            f"library with the current model."
        )
    conn.commit()
    conn.close()


def _adopt_model(db_path: Path) -> None:
    """Record the current model and drop vectors built with the previous one."""
    conn = sqlite3.connect(db_path)
    for key, value in (("model", MODEL_NAME), ("dims", str(DIMS))):
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
    conn.commit()
    conn.close()


def embed_text(text: str, db_path: Path = EMBEDDINGS_DB) -> np.ndarray:
    model = _get_model()
    vecs = model.encode([text], normalize_embeddings=True)
    return np.array(vecs[0], dtype=np.float32)


def get_embedding(zotero_key: str, db_path: Path = EMBEDDINGS_DB) -> Optional[np.ndarray]:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE zotero_key = ?", (zotero_key,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).copy()


def load_all_embeddings(
    db_path: Path = EMBEDDINGS_DB,
    scope_keys: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    conn = sqlite3.connect(db_path)
    if scope_keys:
        placeholders = ",".join("?" * len(scope_keys))
        rows = conn.execute(
            f"SELECT zotero_key, vector FROM embeddings WHERE zotero_key IN ({placeholders})",
            scope_keys,
        ).fetchall()
    else:
        rows = conn.execute("SELECT zotero_key, vector FROM embeddings").fetchall()
    conn.close()
    if not rows:
        return np.empty((0, DIMS), dtype=np.float32), []
    keys = [r[0] for r in rows]
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return matrix, keys


def _stored_model(db_path: Path) -> Optional[str]:
    """The model name recorded in an existing DB, or None if unknown."""
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'model'").fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    return row[0] if row else None


def _purge_fulltext(db_path: Path) -> None:
    """Drop full-text vectors left over from a previous embedding model."""
    conn = sqlite3.connect(db_path)
    for table in ("fulltext_chunks", "fulltext_embeddings"):
        try:
            conn.execute(f"DELETE FROM {table}")
        except sqlite3.OperationalError:
            pass  # legacy table absent on databases created after it was retired
    conn.commit()
    conn.close()


# A prune wider than this fraction of the index is refused unless explicitly
# allowed. `sync` runs unattended from a SessionStart hook, and pruning a paper
# also drops its full-text chunks -- hours of work. A partial or transient read of
# zotero.sqlite (mid-write, mid-sync, wrong path) looks exactly like "the user
# deleted almost everything", and must not be acted on silently.
_MAX_PRUNE_FRACTION = 0.25
# Below this many rows the fraction is meaningless; a fresh index prunes freely.
_PRUNE_GUARD_MIN_ROWS = 50


@dataclass
class SyncReport:
    n_embedded: int = 0
    n_pruned: int = 0
    # Non-zero when the prune guard tripped: this many stale vectors were left alone.
    n_prune_refused: int = 0


def _prune_stale(
    db_path: Path,
    valid_keys: set[str],
    allow_large_prune: bool = False,
) -> tuple[int, int]:
    """Delete vectors whose paper is no longer a citable item in Zotero.

    `sync` only ever inserted, so an index accumulated vectors for papers the
    user deleted, for items that were never papers, and for anything a past bug
    let through. They stayed searchable forever. Embeddings are derived data --
    deleting one costs a re-embed, keeping one corrupts every ranking.

    Returns (n_pruned, n_refused).
    """
    conn = sqlite3.connect(db_path)
    all_keys = [k for (k,) in conn.execute("SELECT zotero_key FROM embeddings")]
    stale = [k for k in all_keys if k not in valid_keys and k != MANUSCRIPT_KEY]

    if (
        stale
        and not allow_large_prune
        and len(all_keys) >= _PRUNE_GUARD_MIN_ROWS
        and len(stale) / len(all_keys) > _MAX_PRUNE_FRACTION
    ):
        conn.close()
        return 0, len(stale)

    if stale:
        params = [(k,) for k in stale]
        conn.executemany("DELETE FROM embeddings WHERE zotero_key = ?", params)
        try:
            conn.executemany("DELETE FROM fulltext_chunks WHERE zotero_key = ?", params)
        except sqlite3.OperationalError:
            pass
        conn.commit()
    conn.close()
    return len(stale), 0


def sync(
    db_path: Path = EMBEDDINGS_DB,
    zotero_db: Path = ZOTERO_DB,
    force: bool = False,
    allow_large_prune: bool = False,
) -> SyncReport:
    previous_model = _stored_model(db_path)
    # A mismatch is fatal unless we are about to re-embed everything anyway.
    init_db(db_path, expect_model=not force)
    model_changed = bool(previous_model) and previous_model != MODEL_NAME

    all_items = get_all_items(zotero_db)
    if force:
        items_to_embed = all_items
    else:
        existing_keys = _existing_keys(db_path)
        items_to_embed = [i for i in all_items if i.key not in existing_keys]

    if items_to_embed:
        _embed_and_store(items_to_embed, db_path)

    # Adopt the new model only once every vector has been rewritten under it.
    if force and model_changed:
        _adopt_model(db_path)
        _purge_fulltext(db_path)

    # An empty library means the read failed or the DB is wrong. Never take that
    # as licence to delete the whole index.
    n_pruned, n_refused = (
        _prune_stale(db_path, {i.key for i in all_items}, allow_large_prune)
        if all_items else (0, 0)
    )
    return SyncReport(
        n_embedded=len(items_to_embed), n_pruned=n_pruned, n_prune_refused=n_refused
    )


def _existing_keys(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT zotero_key FROM embeddings").fetchall()
    conn.close()
    return {r[0] for r in rows}


def _embed_and_store(items: list[Item], db_path: Path) -> None:
    model = _get_model()
    texts = [f"{i.title} {i.abstract} {i.keywords}".strip() for i in items]
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with tqdm(total=len(items), desc="Syncing new papers", unit="paper") as bar:
        for batch_start in range(0, len(items), _BATCH_SIZE):
            batch_items = items[batch_start:batch_start + _BATCH_SIZE]
            batch_texts = texts[batch_start:batch_start + _BATCH_SIZE]
            vecs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
            for item, vec in zip(batch_items, vecs):
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (zotero_key, vector, embedded_at) VALUES (?, ?, ?)",
                    (item.key, np.array(vec, dtype=np.float32).tobytes(), now),
                )
                bar.update(1)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Full-text PDF embedding
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: Path) -> tuple[str, Optional[str]]:
    """Extract plain text from a PDF using PyMuPDF.

    Returns (text, error). Failures used to be swallowed and returned as "",
    which made an unreadable PDF indistinguishable from one with no text -- so
    papers silently never got embedded and nothing said so.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages), None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"[:200]


def _chunk_spans(n_tokens: int, chunk_tokens: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Half-open [start, end) token spans of an overlapping sliding window.

    A trailing window that would contain nothing but overlap is dropped, so a
    document of exactly chunk_tokens yields one chunk rather than two.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if not 0 <= chunk_overlap < chunk_tokens:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be >= 0 and < chunk_tokens ({chunk_tokens})"
        )
    if n_tokens <= 0:
        return []
    stride = chunk_tokens - chunk_overlap
    starts = [
        s for s in range(0, n_tokens, stride)
        if s == 0 or s + chunk_overlap < n_tokens
    ]
    return [(s, min(s + chunk_tokens, n_tokens)) for s in starts]


def _embed_fulltext_chunks(
    item: Item,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[Optional[list[tuple[np.ndarray, int]]], Optional[str]]:
    """Embed one paper's PDF as a list of (unit vector, n_tokens) chunks.

    Returns (None, None) when the item has no local PDF, and (None, reason) when
    text could not be extracted.
    """
    if item.pdf_path is None:
        return None, None
    raw_text, err = _extract_pdf_text(item.pdf_path)
    if err:
        return None, err
    if not raw_text.strip():
        return None, "no extractable text (scanned PDF?)"

    tokenizer = _get_tokenizer()
    all_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
    if not all_tokens:
        return None, "no extractable text (scanned PDF?)"

    spans = _chunk_spans(len(all_tokens), chunk_tokens, chunk_overlap)
    texts = [
        tokenizer.decode(all_tokens[start:end], skip_special_tokens=True)
        for start, end in spans
    ]

    model = _get_model()
    vectors: list[np.ndarray] = []
    for batch_start in range(0, len(texts), _CHUNK_BATCH_SIZE):
        batch = texts[batch_start:batch_start + _CHUNK_BATCH_SIZE]
        encoded = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vectors.extend(np.asarray(v, dtype=np.float32) for v in encoded)

    return [(v, end - start) for v, (start, end) in zip(vectors, spans)], None


def chunked_keys(db_path: Path, keys: list[str]) -> set[str]:
    """Which of `keys` have full-text chunks.

    A paper scored on its best chunk sits systematically higher than one scored on
    its title and abstract, so a caller comparing two scores needs to know which is
    which.
    """
    if not keys:
        return set()
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" * len(keys))
        rows = conn.execute(
            f"SELECT DISTINCT zotero_key FROM fulltext_chunks WHERE zotero_key IN ({placeholders})",
            keys,
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _existing_chunk_keys(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT zotero_key FROM fulltext_chunks").fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    return {r[0] for r in rows}


def load_chunk_vectors(
    db_path: Path = EMBEDDINGS_DB,
    scope_keys: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """All full-text chunk vectors, one matrix row per CHUNK.

    Returns (matrix, owners) where owners[i] is the zotero_key that row i belongs to.
    """
    conn = sqlite3.connect(db_path)
    try:
        if scope_keys is not None:
            if not scope_keys:
                return np.empty((0, DIMS), dtype=np.float32), []
            placeholders = ",".join("?" * len(scope_keys))
            rows = conn.execute(
                f"SELECT zotero_key, vector FROM fulltext_chunks "
                f"WHERE zotero_key IN ({placeholders}) ORDER BY zotero_key, chunk_idx",
                scope_keys,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT zotero_key, vector FROM fulltext_chunks ORDER BY zotero_key, chunk_idx"
            ).fetchall()
    except sqlite3.OperationalError:
        return np.empty((0, DIMS), dtype=np.float32), []
    finally:
        conn.close()
    if not rows:
        return np.empty((0, DIMS), dtype=np.float32), []
    owners = [r[0] for r in rows]
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return matrix, owners


def _mean_chunk_vectors(matrix: np.ndarray, owners: list[str]) -> dict[str, np.ndarray]:
    """One L2-normalised mean vector per paper, from its chunk vectors."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for owner, vec in zip(owners, matrix):
        if owner in sums:
            sums[owner] += vec
            counts[owner] += 1
        else:
            sums[owner] = vec.astype(np.float64).copy()
            counts[owner] = 1
    means = {}
    for owner, total in sums.items():
        mean = total / counts[owner]
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        means[owner] = mean.astype(np.float32)
    return means


def load_all_fulltext_embeddings(
    db_path: Path = EMBEDDINGS_DB,
    scope_keys: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """One vector per paper: the mean of its full-text chunks, else title+abstract.

    Used by `map` and `cluster`, where each paper must be a single point. Search
    does NOT use this -- it scores papers by their best chunk (see find_similar).
    """
    ta_matrix, ta_keys = load_all_embeddings(db_path, scope_keys)
    if not ta_keys:
        return ta_matrix, ta_keys
    chunk_matrix, owners = load_chunk_vectors(db_path, scope_keys)
    if not owners:
        return ta_matrix, ta_keys
    means = _mean_chunk_vectors(chunk_matrix, owners)
    matrix = np.stack([means.get(k, v) for k, v in zip(ta_keys, ta_matrix)])
    return matrix, ta_keys


@dataclass
class FulltextSyncReport:
    n_embedded: int = 0
    n_skipped_no_pdf: int = 0
    # (zotero_key, reason) -- opaque keys only, never titles
    failures: list[tuple[str, str]] = field(default_factory=list)


def sync_fulltext(
    db_path: Path = EMBEDDINGS_DB,
    zotero_db: Path = ZOTERO_DB,
    force: bool = False,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    collection: Optional[str] = None,
) -> FulltextSyncReport:
    """Embed each local PDF as overlapping chunks in the fulltext_chunks table.

    One transaction per paper, so an interrupted run resumes where it stopped and
    a re-run replaces a paper's chunks wholesale (no stale trailing chunk_idx).
    Papers already chunked are skipped unless force=True.
    """
    init_db(db_path)
    # Validate the window before doing hours of work.
    _chunk_spans(1, chunk_tokens, chunk_overlap)

    if collection:
        from litmap.zotero import get_collection
        all_items = get_collection(collection, zotero_db)
    else:
        all_items = get_all_items(zotero_db)

    items_with_pdf = [i for i in all_items if i.pdf_path is not None]
    report = FulltextSyncReport(n_skipped_no_pdf=len(all_items) - len(items_with_pdf))

    if not force:
        existing = _existing_chunk_keys(db_path)
        items_with_pdf = [i for i in items_with_pdf if i.key not in existing]
    if not items_with_pdf:
        return report

    conn = sqlite3.connect(db_path)
    with tqdm(total=len(items_with_pdf), desc="Embedding full-text PDFs", unit="paper") as bar:
        for item in items_with_pdf:
            bar.set_postfix({"file": item.pdf_path.name[:40] if item.pdf_path else ""})
            try:
                chunks, err = _embed_fulltext_chunks(item, chunk_tokens, chunk_overlap)
            except Exception as e:  # a single bad PDF must not end the run
                chunks, err = None, f"{type(e).__name__}: {e}"[:200]

            if chunks:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("DELETE FROM fulltext_chunks WHERE zotero_key = ?", (item.key,))
                conn.executemany(
                    "INSERT INTO fulltext_chunks "
                    "(zotero_key, chunk_idx, vector, n_tokens, embedded_at) VALUES (?, ?, ?, ?, ?)",
                    [
                        (item.key, idx, vec.tobytes(), n_tokens, now)
                        for idx, (vec, n_tokens) in enumerate(chunks)
                    ],
                )
                conn.commit()
                report.n_embedded += 1
            elif err:
                report.failures.append((item.key, err))
            bar.update(1)

    conn.close()
    return report
