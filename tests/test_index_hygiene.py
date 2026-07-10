"""The embeddings index must contain citable papers, and only citable papers.

Three defects, all found by auditing a real 2,898-row index against the library
it was built from. 44% of those rows were not papers:

  644 attachments  -- itemTypeIDs were hardcoded as (14, 26). In Zotero 9 those
                      are `email` and `newspaperArticle`; attachment is 3 and
                      note is 27. So attachments were embedded, and their
                      "titles" are PDF filenames. (Fixed by resolving by name;
                      covered in test_author_fidelity.py.)
  606 orphans      -- items deleted from Zotero. `sync` only ever INSERTed; it
                      had no way to remove a vector, so deleted papers were
                      searchable forever.
   16 trashed      -- items sitting in Zotero's Trash. They remain in `items`
                      until the trash is emptied, and litmap never excluded them.

Every one of these competes with real papers for a place in the top-K.
"""
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from litmap.embedder import DIMS, MANUSCRIPT_KEY, init_db, sync
from litmap.zotero import get_all_items, get_collection, get_item
from tests.conftest import store_vector


# ---------------------------------------------------------------------------
# Trash exclusion
# ---------------------------------------------------------------------------

def test_get_all_items_excludes_trashed(zotero_db):
    keys = {i.key for i in get_all_items(zotero_db)}
    assert "TRASH0005" not in keys, "a paper in Zotero's Trash must not be indexed"


def test_get_item_refuses_a_trashed_key(zotero_db):
    assert get_item("TRASH0005", zotero_db) is None


def test_get_item_refuses_a_trashed_doi(zotero_db):
    assert get_item("10.2222/trash", zotero_db) is None


def test_get_collection_excludes_trashed(zotero_db):
    """A trashed item still has its collectionItems row."""
    conn = sqlite3.connect(zotero_db)
    conn.execute("INSERT INTO collectionItems VALUES (1, 50)")
    conn.commit()
    conn.close()
    keys = {i.key for i in get_collection("My Papers", zotero_db)}
    assert "TRASH0005" not in keys


def test_works_when_deleteditems_table_is_absent(tmp_path):
    """Older Zotero schemas have no deletedItems table; do not crash."""
    import shutil
    from litmap.zotero import _connect, _trash_clause

    src = tmp_path / "no_trash.sqlite"
    conn = sqlite3.connect(src)
    conn.executescript("""
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        INSERT INTO itemTypes VALUES (2, 'journalArticle');
    """)
    conn.commit()
    conn.close()
    with _connect(src) as c:
        assert _trash_clause(c) == ""


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def _fake_model():
    m = patch("litmap.embedder._get_model").start()
    m.return_value.encode.side_effect = lambda texts, **kw: np.stack(
        [np.ones(DIMS, dtype=np.float32) for _ in texts]
    )
    return m


@pytest.fixture
def synced_db(tmp_path, zotero_db):
    db_path = tmp_path / "embeddings.db"
    init_db(db_path)
    with patch("litmap.embedder._get_model") as m:
        m.return_value.encode.side_effect = lambda texts, **kw: np.stack(
            [np.ones(DIMS, dtype=np.float32) for _ in texts]
        )
        sync(db_path, zotero_db)
    return db_path


def _keys(db_path):
    conn = sqlite3.connect(db_path)
    keys = {r[0] for r in conn.execute("SELECT zotero_key FROM embeddings")}
    conn.close()
    return keys


def test_sync_reports_embedded_and_pruned(synced_db, zotero_db):
    report = sync(synced_db, zotero_db)
    assert report.n_embedded == 0
    assert report.n_pruned == 0


def test_sync_prunes_vectors_for_items_no_longer_in_zotero(synced_db, zotero_db, make_vector_fn):
    """The 606-orphan case: a paper was deleted from Zotero after being embedded."""
    store_vector(synced_db, "GONE0001", make_vector_fn(1))
    store_vector(synced_db, "GONE0002", make_vector_fn(2))
    assert {"GONE0001", "GONE0002"} <= _keys(synced_db)

    report = sync(synced_db, zotero_db)

    assert report.n_pruned == 2
    assert not {"GONE0001", "GONE0002"} & _keys(synced_db)
    assert "AAAA0001" in _keys(synced_db), "real papers must survive the prune"


def test_sync_prunes_attachment_vectors_left_by_the_old_type_id_bug(
    synced_db, zotero_db, make_vector_fn
):
    store_vector(synced_db, "AAAA0003", make_vector_fn(3))   # the attachment
    report = sync(synced_db, zotero_db)
    assert report.n_pruned == 1
    assert "AAAA0003" not in _keys(synced_db)


def test_sync_prunes_a_trashed_paper_that_was_embedded_before_deletion(
    synced_db, zotero_db, make_vector_fn
):
    store_vector(synced_db, "TRASH0005", make_vector_fn(5))
    sync(synced_db, zotero_db)
    assert "TRASH0005" not in _keys(synced_db)


def test_prune_also_removes_that_paper_s_fulltext_chunks(synced_db, zotero_db, make_vector_fn):
    store_vector(synced_db, "GONE0001", make_vector_fn(1))
    conn = sqlite3.connect(synced_db)
    conn.execute(
        "INSERT INTO fulltext_chunks (zotero_key, chunk_idx, vector, n_tokens, embedded_at) "
        "VALUES ('GONE0001', 0, ?, 512, 'x')",
        (make_vector_fn(1).tobytes(),),
    )
    conn.commit()
    conn.close()

    sync(synced_db, zotero_db)

    conn = sqlite3.connect(synced_db)
    n = conn.execute("SELECT COUNT(*) FROM fulltext_chunks WHERE zotero_key='GONE0001'").fetchone()[0]
    conn.close()
    assert n == 0, "orphaned chunks must go with their paper"


def test_prune_never_touches_the_manuscript_row(synced_db, zotero_db, make_vector_fn):
    """`map --manuscript` stores a synthetic key that is not a Zotero item."""
    store_vector(synced_db, MANUSCRIPT_KEY, make_vector_fn(9))
    sync(synced_db, zotero_db)
    assert MANUSCRIPT_KEY in _keys(synced_db)


def test_prune_is_skipped_when_zotero_returns_nothing(synced_db, zotero_db, monkeypatch):
    """A failed or empty library read must never empty the index."""
    before = _keys(synced_db)
    monkeypatch.setattr("litmap.embedder.get_all_items", lambda db: [])
    report = sync(synced_db, zotero_db)
    assert report.n_pruned == 0
    assert _keys(synced_db) == before
