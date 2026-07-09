"""Known-bug test for the full-text pooling defect.

Full-text search stored ONE vector per paper: the mean of its chunk vectors.
Averaging is a low-pass filter. A 40-page paper whose one relevant section
answers the query is dominated by the ~19 sections that do not, so its stored
vector points nowhere near the query. A shallow paper that is vaguely on-topic
throughout outranks it -- exactly inverting what a literature search is for.
The longer and more specific the paper, the worse it scores.

Fixture, in 768-d space (e1, e2 orthonormal):

  NEEDLE  20 chunks: one == e1 (answers the query), nineteen == e2.
          mean = normalize(e1 + 19*e2), so cos(mean, e1) ~= 0.053
  HAY     no chunks; title+abstract vector = 0.6*e1 + 0.8*e2, cos = 0.6

Query = e1.

  mean-pool (old): HAY (0.60) beats NEEDLE (0.053)  -- the buried section loses
  max-pool  (new): NEEDLE (1.00) beats HAY (0.60)   -- the buried section wins

Against the pre-fix scorer, test_buried_relevant_chunk_wins FAILS. That red run
is the point: it distinguishes "the new scorer works" from "the fixture never
exercised the scorer at all".
"""
import sqlite3

import numpy as np
import pytest

from litmap.search import find_similar
from tests.conftest import store_vector

DIMS = 768


def _axis(i: int) -> np.ndarray:
    v = np.zeros(DIMS, dtype=np.float32)
    v[i] = 1.0
    return v


E1 = _axis(0)
E2 = _axis(1)
N_IRRELEVANT = 19


def _store_chunks(db_path, key: str, vectors: list[np.ndarray]) -> None:
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO fulltext_chunks (zotero_key, chunk_idx, vector, n_tokens, embedded_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        [(key, i, v.tobytes(), 512) for i, v in enumerate(vectors)],
    )
    conn.commit()
    conn.close()


def _store_legacy_mean(db_path, key: str, vectors: list[np.ndarray]) -> None:
    """What the old sync-fulltext persisted: the L2-normalised mean chunk vector."""
    mean = np.mean(np.stack(vectors), axis=0).astype(np.float32)
    mean /= np.linalg.norm(mean)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO fulltext_embeddings "
        "(zotero_key, vector, embedded_at, n_tokens, n_chunks) VALUES (?, ?, 'x', ?, ?)",
        (key, mean.tobytes(), 512 * len(vectors), len(vectors)),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def buried_needle_db(embeddings_db):
    chunks = [E1] + [E2] * N_IRRELEVANT

    # NEEDLE: irrelevant title+abstract, one relevant chunk buried in the body.
    store_vector(embeddings_db, "NEEDLE", E2)
    _store_chunks(embeddings_db, "NEEDLE", chunks)
    _store_legacy_mean(embeddings_db, "NEEDLE", chunks)

    # HAY: no full text, moderately on-topic title+abstract.
    hay = (0.6 * E1 + 0.8 * E2).astype(np.float32)
    store_vector(embeddings_db, "HAY", hay)
    return embeddings_db


def test_fixture_actually_buries_the_needle(buried_needle_db):
    """Guard the fixture: the mean vector must genuinely lose to HAY.

    If this ever passes trivially, the regression test below proves nothing.
    """
    conn = sqlite3.connect(buried_needle_db)
    blob = conn.execute(
        "SELECT vector FROM fulltext_embeddings WHERE zotero_key = 'NEEDLE'"
    ).fetchone()[0]
    conn.close()
    mean_vec = np.frombuffer(blob, dtype=np.float32)

    cos_mean = float(mean_vec @ E1)
    cos_hay = 0.6
    assert cos_mean == pytest.approx(1 / np.sqrt(1 + N_IRRELEVANT**2), abs=1e-4)
    assert cos_mean < cos_hay


def test_buried_relevant_chunk_wins(buried_needle_db):
    """THE regression test. Fails against mean-pooling, passes against max-pooling."""
    results = find_similar(E1, buried_needle_db, top_k=2)
    keys = [r["key"] for r in results]

    assert keys[0] == "NEEDLE", (
        "a paper containing a chunk identical to the query must rank first; "
        "mean-pooling drowns it"
    )
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert keys[1] == "HAY"
    assert results[1]["similarity"] == pytest.approx(0.6, abs=1e-4)


def test_legacy_fulltext_table_is_ignored(buried_needle_db):
    """The retired mean-pooled table must no longer influence ranking.

    Poison it with a vector that would win if it were still read.
    """
    conn = sqlite3.connect(buried_needle_db)
    conn.execute(
        "INSERT OR REPLACE INTO fulltext_embeddings "
        "(zotero_key, vector, embedded_at) VALUES ('HAY', ?, 'x')",
        (E1.tobytes(),),
    )
    conn.commit()
    conn.close()

    results = find_similar(E1, buried_needle_db, top_k=2)
    assert results[0]["key"] == "NEEDLE"
    assert results[1]["similarity"] == pytest.approx(0.6, abs=1e-4), (
        "HAY scored from the poisoned legacy table instead of title+abstract"
    )
