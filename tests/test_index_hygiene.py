"""The embeddings index must contain citable papers, and only citable papers.

Four defects, all found by auditing a real 2,898-row index against the library it
was built from. 48% of those rows were not papers the user had chosen to keep:

  644 attachments  -- itemTypeIDs were hardcoded as (14, 26). In Zotero 9 those
                      are `email` and `newspaperArticle`; attachment is 3 and
                      note is 27. So attachments were embedded, and their
                      "titles" are PDF filenames. (Fixed by resolving by name;
                      covered in test_author_fidelity.py.)
  606 orphans      -- items deleted from Zotero. `sync` only ever INSERTed; it
                      had no way to remove a vector, so deleted papers were
                      searchable forever.
  116 feed items   -- subscribed journal tables-of-contents. Zotero deletes them
                      again after a few days, so indexing them means chasing a
                      rolling window -- and is where most of the orphans came from.
   16 trashed      -- items sitting in Zotero's Trash. They remain in `items`
                      until the trash is emptied, and litmap never excluded them.

Every one of these competes with real papers for a place in the top-K.

The prune that fixes them is itself dangerous once `sync` runs unattended, so it
carries a safety valve: a prune wider than a quarter of the index is refused.
"""
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from litmap.embedder import DIMS, MANUSCRIPT_KEY, init_db, sync
from litmap.embedder import _MAX_PRUNE_FRACTION, _PRUNE_GUARD_MIN_ROWS  # noqa: F401
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


# ---------------------------------------------------------------------------
# Feed exclusion
# ---------------------------------------------------------------------------

def test_feed_items_are_not_indexed(zotero_db):
    """A subscribed journal feed is a rolling window of papers the user has NOT saved.

    Zotero deletes feed items again after `cleanupReadAfter` days, so indexing them
    means the index permanently chases churn -- and generates orphan vectors for
    papers that were never in the library.
    """
    keys = {i.key for i in get_all_items(zotero_db)}
    assert "FEEDITEM1" not in keys


def test_group_library_papers_are_still_indexed(zotero_db):
    """Guard against over-exclusion: only `feed` libraries go, not every non-user one."""
    keys = {i.key for i in get_all_items(zotero_db)}
    assert "GROUPPAPR" in keys


def test_get_item_refuses_a_feed_key(zotero_db):
    assert get_item("FEEDITEM1", zotero_db) is None


def test_sync_prunes_feed_vectors_left_by_the_old_code(synced_db, zotero_db, make_vector_fn):
    store_vector(synced_db, "FEEDITEM1", make_vector_fn(11))
    report = sync(synced_db, zotero_db)
    assert report.n_pruned == 1
    assert "FEEDITEM1" not in _keys(synced_db)


def test_feed_exclusion_when_libraries_table_is_absent(tmp_path):
    """Old Zotero schemas predate feeds; fall back to the `feeds` table, then to nothing."""
    from litmap.zotero import _connect, _feed_clause

    db_path = tmp_path / "no_libraries.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript("CREATE TABLE placeholder (x INTEGER);")
    conn.commit(); conn.close()
    with _connect(db_path) as c:
        assert _feed_clause(c) == ""

    db2 = tmp_path / "feeds_only.sqlite"
    conn = sqlite3.connect(db2)
    conn.executescript("CREATE TABLE feeds (libraryID INTEGER PRIMARY KEY, name TEXT);")
    conn.commit(); conn.close()
    with _connect(db2) as c:
        assert "feeds" in _feed_clause(c)


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


# ---------------------------------------------------------------------------
# Prune safety valve
# ---------------------------------------------------------------------------

def _index_with(tmp_path, make_vector_fn, n_valid, n_stale):
    """A bare index: n_valid keys that exist in Zotero, n_stale that do not."""
    db_path = tmp_path / "guard.db"
    init_db(db_path)
    valid = {f"REAL{i:04d}" for i in range(n_valid)}
    for i, k in enumerate(sorted(valid)):
        store_vector(db_path, k, make_vector_fn(i))
    for i in range(n_stale):
        store_vector(db_path, f"GONE{i:04d}", make_vector_fn(1000 + i))
    return db_path, valid


def test_prune_refuses_to_delete_more_than_a_quarter_of_the_index(tmp_path, make_vector_fn):
    """A partial read of zotero.sqlite looks exactly like a mass deletion.

    sync runs unattended from a SessionStart hook, and pruning a paper also drops
    its full-text chunks. Acting on a transient read would throw away hours of
    embedding work, silently and without anyone watching.
    """
    from litmap.embedder import _prune_stale
    db_path, valid = _index_with(tmp_path, make_vector_fn, n_valid=60, n_stale=40)
    before = _keys(db_path)

    n_pruned, n_refused = _prune_stale(db_path, valid)

    assert (n_pruned, n_refused) == (0, 40)          # 40 of 100 rows = 40% > 25%
    assert _keys(db_path) == before, "nothing may be deleted when the guard trips"


def test_allow_large_prune_overrides_the_guard(tmp_path, make_vector_fn):
    from litmap.embedder import _prune_stale
    db_path, valid = _index_with(tmp_path, make_vector_fn, n_valid=60, n_stale=40)
    n_pruned, n_refused = _prune_stale(db_path, valid, allow_large_prune=True)
    assert (n_pruned, n_refused) == (40, 0)
    assert _keys(db_path) == valid


def test_a_prune_under_the_limit_proceeds_without_the_flag(tmp_path, make_vector_fn):
    from litmap.embedder import _prune_stale
    db_path, valid = _index_with(tmp_path, make_vector_fn, n_valid=90, n_stale=10)
    n_pruned, n_refused = _prune_stale(db_path, valid)
    assert (n_pruned, n_refused) == (10, 0)          # 10% < 25%
    assert _keys(db_path) == valid


def test_guard_does_not_apply_to_a_small_index(tmp_path, make_vector_fn):
    """A fresh index of a handful of rows must still be able to prune all of them."""
    from litmap.embedder import _prune_stale
    db_path, valid = _index_with(tmp_path, make_vector_fn, n_valid=3, n_stale=2)
    n_pruned, n_refused = _prune_stale(db_path, valid)
    assert (n_pruned, n_refused) == (2, 0)           # 5 rows, below the guard minimum


def test_guard_keeps_the_manuscript_row_out_of_the_fraction(tmp_path, make_vector_fn):
    from litmap.embedder import _prune_stale
    db_path, valid = _index_with(tmp_path, make_vector_fn, n_valid=60, n_stale=1)
    store_vector(db_path, MANUSCRIPT_KEY, make_vector_fn(7))
    n_pruned, n_refused = _prune_stale(db_path, valid)
    assert (n_pruned, n_refused) == (1, 0)
    assert MANUSCRIPT_KEY in _keys(db_path)


def test_sync_surfaces_a_refused_prune(synced_db, zotero_db, make_vector_fn):
    for i in range(60):
        store_vector(synced_db, f"GONE{i:04d}", make_vector_fn(2000 + i))
    report = sync(synced_db, zotero_db)
    assert report.n_pruned == 0
    assert report.n_prune_refused == 60
    assert "AAAA0001" in _keys(synced_db)


def test_cli_reports_a_refused_prune_and_names_the_override(
    synced_db, zotero_db, make_vector_fn
):
    from typer.testing import CliRunner
    from litmap.cli import app

    for i in range(60):
        store_vector(synced_db, f"GONE{i:04d}", make_vector_fn(2000 + i))
    result = CliRunner().invoke(
        app, ["sync", "--db-path", str(synced_db), "--zotero-db", str(zotero_db)]
    )
    assert result.exit_code == 0
    assert "Refused to prune" in result.output
    assert "--allow-large-prune" in result.output
    assert "AAAA0001" in _keys(synced_db)
