"""`map --manuscript` must not leave its synthetic vector in the library.

The manuscript is embedded into the `embeddings` table under the key
"__manuscript__" so it can be laid out among the papers. It was never deleted,
so a stale non-paper vector accumulated in the database and turned up in later
searches and clusters.
"""
import sqlite3

import numpy as np
import pytest
from typer.testing import CliRunner

from litmap import cli as cli_mod
from litmap import embedder
from litmap.cli import app
from litmap.embedder import MANUSCRIPT_KEY
from litmap.search import find_similar
from tests.conftest import store_vector


def _manuscript_rows(db_path) -> int:
    conn = sqlite3.connect(db_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE zotero_key = ?", (MANUSCRIPT_KEY,)
    ).fetchone()[0]
    conn.close()
    return n


@pytest.fixture
def map_env(monkeypatch, tmp_path, embeddings_db, zotero_db, make_vector_fn):
    for i, key in enumerate(["AAAA0001", "AAAA0002", "AAAA0004"]):
        store_vector(embeddings_db, key, make_vector_fn(i))

    class _FakeModel:
        def encode(self, texts, **kw):
            return np.stack([make_vector_fn(7) for _ in texts])

    monkeypatch.setattr(embedder, "_get_model", lambda: _FakeModel())

    # UMAP needs more points than this fixture has, and it is not what these
    # tests are about. Stub the layout so the command reaches its render step.
    monkeypatch.setattr(
        "litmap.layout.compute_layout",
        lambda matrix, keys, **kw: {k: (float(i), 0.0) for i, k in enumerate(keys)},
    )
    monkeypatch.setattr("litmap.layout.build_graph", lambda matrix, keys, k=3: [])

    monkeypatch.setattr("litmap.manuscript.extract_manuscript_text", lambda p: "manuscript text")
    monkeypatch.setattr(
        "litmap.manuscript.parse_bibliography",
        lambda p: [{"key": "AAAA0001"}, {"key": "AAAA0002"}],
    )
    monkeypatch.setattr(
        "litmap.manuscript.match_items_to_zotero",
        lambda entries, db: (
            [i for i in __import__("litmap.zotero", fromlist=["x"]).get_all_items(db)
             if i.key in ("AAAA0001", "AAAA0002")],
            [],
        ),
    )
    ms = tmp_path / "paper.docx"
    ms.write_text("manuscript")
    return embeddings_db, zotero_db, ms, tmp_path


def _invoke_map(map_env, fmt="html"):
    embeddings_db, zotero_db, ms, tmp_path = map_env
    return CliRunner().invoke(
        app,
        ["map", "--manuscript", str(ms), "--format", fmt,
         "--output", str(tmp_path / "out"), "--no-label-clusters",
         "--db-path", str(embeddings_db), "--zotero-db", str(zotero_db)],
    )


def test_manuscript_row_removed_after_successful_map(map_env):
    embeddings_db = map_env[0]
    result = _invoke_map(map_env)
    assert result.exit_code == 0, result.output
    assert _manuscript_rows(embeddings_db) == 0


def test_manuscript_row_removed_when_rendering_fails(map_env, monkeypatch):
    embeddings_db = map_env[0]

    def _boom(*a, **k):
        raise RuntimeError("render exploded")

    monkeypatch.setattr("litmap.renderer.render_html", _boom)
    result = _invoke_map(map_env)

    assert result.exit_code != 0
    assert _manuscript_rows(embeddings_db) == 0, "cleanup must run in a finally block"


def test_stale_manuscript_row_purged_at_start(map_env, make_vector_fn):
    """A row orphaned by an older crashed run is cleared on the next map."""
    embeddings_db = map_env[0]
    store_vector(embeddings_db, MANUSCRIPT_KEY, make_vector_fn(99))
    assert _manuscript_rows(embeddings_db) == 1

    _invoke_map(map_env)
    assert _manuscript_rows(embeddings_db) == 0


def test_search_never_returns_a_stale_manuscript_row(embeddings_db, make_vector_fn):
    """Defence in depth: even if a row survives, it is not a search result."""
    query = make_vector_fn(3)
    store_vector(embeddings_db, MANUSCRIPT_KEY, query)  # perfect match
    store_vector(embeddings_db, "AAAA0001", make_vector_fn(4))

    results = find_similar(query, embeddings_db, top_k=5)
    assert [r["key"] for r in results] == ["AAAA0001"]
