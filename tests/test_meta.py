"""embeddings.db must refuse to mix vectors from different embedding models.

The meta row was written with INSERT OR IGNORE, so it recorded whichever model
ran first and never updated. Changing MODEL_NAME then silently produced a table
holding two incompatible vector spaces, and every similarity score computed
across them was meaningless -- with no error and no way to tell after the fact.
"""
import sqlite3

import numpy as np
import pytest

from litmap import embedder
from litmap.embedder import ModelMismatchError, init_db


def _meta(db_path) -> dict:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    conn.close()
    return dict(rows)


def _set_model(db_path, name: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('model', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (name,),
    )
    conn.commit()
    conn.close()


def test_init_db_adopts_model_when_absent(tmp_path):
    db_path = tmp_path / "fresh.db"
    init_db(db_path)
    meta = _meta(db_path)
    assert meta["model"] == embedder.MODEL_NAME
    assert meta["dims"] == str(embedder.DIMS)


def test_init_db_accepts_matching_model(embeddings_db):
    init_db(embeddings_db)  # fixture already stores the current model name


def test_init_db_raises_on_model_mismatch(embeddings_db):
    _set_model(embeddings_db, "some-other/model-v2")
    with pytest.raises(ModelMismatchError) as exc:
        init_db(embeddings_db)
    msg = str(exc.value)
    assert "some-other/model-v2" in msg
    assert embedder.MODEL_NAME in msg
    assert "--force" in msg


def test_init_db_expect_model_false_skips_check(embeddings_db):
    """The escape hatch sync(force=True) uses to adopt a new model."""
    _set_model(embeddings_db, "some-other/model-v2")
    init_db(embeddings_db, expect_model=False)  # must not raise


def test_sync_force_adopts_new_model_and_purges_fulltext(
    monkeypatch, embeddings_db, zotero_db, make_vector_fn
):
    """After --force rewrites every vector, the meta row must follow."""
    _set_model(embeddings_db, "some-other/model-v2")
    conn = sqlite3.connect(embeddings_db)
    conn.execute(
        "INSERT INTO fulltext_embeddings (zotero_key, vector, embedded_at) "
        "VALUES ('AAAA0001', ?, 'x')",
        (make_vector_fn(1).tobytes(),),
    )
    conn.commit()
    conn.close()

    class _FakeModel:
        def encode(self, texts, **kw):
            return np.stack([make_vector_fn(i) for i in range(len(texts))])

    monkeypatch.setattr(embedder, "_get_model", lambda: _FakeModel())
    embedder.sync(embeddings_db, zotero_db, force=True)

    assert _meta(embeddings_db)["model"] == embedder.MODEL_NAME
    conn = sqlite3.connect(embeddings_db)
    n_stale = conn.execute("SELECT COUNT(*) FROM fulltext_embeddings").fetchone()[0]
    conn.close()
    assert n_stale == 0, "full-text vectors from the old model must not survive"


def test_sync_without_force_refuses_on_mismatch(monkeypatch, embeddings_db, zotero_db):
    _set_model(embeddings_db, "some-other/model-v2")
    with pytest.raises(ModelMismatchError):
        embedder.sync(embeddings_db, zotero_db, force=False)


def test_sync_same_model_force_keeps_fulltext(
    monkeypatch, embeddings_db, zotero_db, make_vector_fn
):
    """Re-embedding titles under the SAME model must not destroy full-text work."""
    conn = sqlite3.connect(embeddings_db)
    conn.execute(
        "INSERT INTO fulltext_embeddings (zotero_key, vector, embedded_at) "
        "VALUES ('AAAA0001', ?, 'x')",
        (make_vector_fn(1).tobytes(),),
    )
    conn.commit()
    conn.close()

    class _FakeModel:
        def encode(self, texts, **kw):
            return np.stack([make_vector_fn(i) for i in range(len(texts))])

    monkeypatch.setattr(embedder, "_get_model", lambda: _FakeModel())
    embedder.sync(embeddings_db, zotero_db, force=True)

    conn = sqlite3.connect(embeddings_db)
    n = conn.execute("SELECT COUNT(*) FROM fulltext_embeddings").fetchone()[0]
    conn.close()
    assert n == 1


def test_search_cmd_reports_mismatch_without_traceback(embeddings_db, zotero_db):
    from typer.testing import CliRunner
    from litmap.cli import app

    _set_model(embeddings_db, "some-other/model-v2")
    result = CliRunner().invoke(
        app,
        ["search", "--query", "anything",
         "--db-path", str(embeddings_db), "--zotero-db", str(zotero_db)],
    )
    assert result.exit_code == 2
    assert "some-other/model-v2" in result.output
    assert "Traceback" not in result.output
