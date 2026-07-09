"""Full-text chunking, storage, resumability and failure reporting."""
import sqlite3
import zlib

import numpy as np
import pytest

from litmap import embedder
from litmap.embedder import (
    DIMS,
    _chunk_spans,
    init_db,
    load_all_fulltext_embeddings,
    sync_fulltext,
)


# --------------------------------------------------------------------------
# Chunk window arithmetic (pure)
# --------------------------------------------------------------------------

def test_chunk_spans_overlapping_windows():
    assert _chunk_spans(1000, 512, 64) == [(0, 512), (448, 960), (896, 1000)]


def test_chunk_spans_short_document_is_one_chunk():
    assert _chunk_spans(300, 512, 64) == [(0, 300)]


def test_chunk_spans_drops_pure_overlap_tail():
    """A document of exactly chunk_tokens must not spawn an all-overlap second chunk."""
    assert _chunk_spans(512, 512, 64) == [(0, 512)]


def test_chunk_spans_zero_overlap_is_contiguous():
    assert _chunk_spans(1000, 512, 0) == [(0, 512), (512, 1000)]


def test_chunk_spans_empty_document():
    assert _chunk_spans(0, 512, 64) == []


@pytest.mark.parametrize("chunk_tokens,overlap", [(512, 512), (512, 600), (512, -1), (0, 0)])
def test_chunk_spans_rejects_invalid_window(chunk_tokens, overlap):
    with pytest.raises(ValueError):
        _chunk_spans(1000, chunk_tokens, overlap)


# --------------------------------------------------------------------------
# Hermetic doubles
# --------------------------------------------------------------------------

def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(DIMS).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeTokenizer:
    """One token per whitespace-separated word; decode is reversible enough."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


class FakeModel:
    def __init__(self, fail_on=None):
        self.batches = []
        self._fail_on = fail_on or set()

    def encode(self, texts, **kw):
        texts = list(texts)
        for t in texts:
            if t in self._fail_on:
                raise RuntimeError("simulated encode failure")
        self.batches.append(texts)
        return np.stack([_unit(zlib.crc32(t.encode())) for t in texts])


@pytest.fixture
def zotero_db_with_pdfs(tmp_path):
    """Two papers, each with a local PDF attachment that exists on disk."""
    db_path = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        INSERT INTO itemTypes VALUES (2, 'journalArticle');
        INSERT INTO itemTypes VALUES (14, 'attachment');
        INSERT INTO itemTypes VALUES (26, 'note');

        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        INSERT INTO fields VALUES (1, 'title');
        INSERT INTO fields VALUES (2, 'abstractNote');
        INSERT INTO fields VALUES (6, 'date');
        INSERT INTO fields VALUES (8, 'DOI');

        CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
        INSERT INTO creatorTypes VALUES (1, 'author');
        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, creatorTypeID INTEGER, orderIndex INTEGER);

        CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INTEGER, libraryID INTEGER DEFAULT 1, key TEXT);
        INSERT INTO items VALUES (1, 2, 1, 'PAPER001');
        INSERT INTO items VALUES (2, 2, 1, 'PAPER002');
        INSERT INTO items VALUES (3, 2, 1, 'NOPDF003');
        INSERT INTO items VALUES (10, 14, 1, 'ATTA0001');
        INSERT INTO items VALUES (11, 14, 1, 'ATTA0002');

        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO itemDataValues VALUES (1, 'Confidential Paper Title One');
        INSERT INTO itemDataValues VALUES (2, 'Confidential Paper Title Two');
        INSERT INTO itemDataValues VALUES (3, 'Confidential Paper Title Three');

        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        INSERT INTO itemData VALUES (1, 1, 1);
        INSERT INTO itemData VALUES (2, 1, 2);
        INSERT INTO itemData VALUES (3, 1, 3);

        CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, collectionName TEXT,
                                  parentCollectionID INTEGER, libraryID INTEGER DEFAULT 1);
        CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);

        CREATE TABLE itemAttachments (
            itemID INTEGER PRIMARY KEY, parentItemID INTEGER,
            linkMode INTEGER, contentType TEXT, path TEXT
        );
        INSERT INTO itemAttachments VALUES (10, 1, 1, 'application/pdf', 'storage:one.pdf');
        INSERT INTO itemAttachments VALUES (11, 2, 1, 'application/pdf', 'storage:two.pdf');
    """)
    conn.commit()
    conn.close()

    for att_key, name in (("ATTA0001", "one.pdf"), ("ATTA0002", "two.pdf")):
        d = tmp_path / "storage" / att_key
        d.mkdir(parents=True)
        (d / name).write_bytes(b"%PDF-1.4 stub")
    return db_path


@pytest.fixture
def fake_encoders(monkeypatch):
    """Install fake tokenizer + model; return the model so tests can inspect calls."""
    model = FakeModel()
    monkeypatch.setattr(embedder, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(embedder, "_get_model", lambda: model)
    # 1200 "words" -> 1200 tokens -> 3 chunks at 512/64
    monkeypatch.setattr(
        embedder, "_extract_pdf_text",
        lambda p: (" ".join(f"word{i}" for i in range(1200)), None),
    )
    return model


def _chunk_rows(db_path, key=None):
    conn = sqlite3.connect(db_path)
    if key:
        rows = conn.execute(
            "SELECT chunk_idx, n_tokens FROM fulltext_chunks WHERE zotero_key = ? ORDER BY chunk_idx",
            (key,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT zotero_key, chunk_idx FROM fulltext_chunks ORDER BY zotero_key, chunk_idx"
        ).fetchall()
    conn.close()
    return rows


# --------------------------------------------------------------------------
# sync_fulltext
# --------------------------------------------------------------------------

def test_sync_fulltext_writes_chunks_and_never_the_legacy_table(
    embeddings_db, zotero_db_with_pdfs, fake_encoders
):
    report = sync_fulltext(embeddings_db, zotero_db_with_pdfs)

    assert report.n_embedded == 2
    assert report.n_skipped_no_pdf == 1  # NOPDF003
    assert report.failures == []
    # 1200 tokens, window 512 stride 448 -> spans (0,512) (448,960) (896,1200)
    assert _chunk_rows(embeddings_db, "PAPER001") == [(0, 512), (1, 512), (2, 304)]

    conn = sqlite3.connect(embeddings_db)
    n_legacy = conn.execute("SELECT COUNT(*) FROM fulltext_embeddings").fetchone()[0]
    conn.close()
    assert n_legacy == 0


def test_sync_fulltext_skips_already_chunked_papers(
    embeddings_db, zotero_db_with_pdfs, fake_encoders
):
    sync_fulltext(embeddings_db, zotero_db_with_pdfs)
    fake_encoders.batches.clear()

    report = sync_fulltext(embeddings_db, zotero_db_with_pdfs)
    assert report.n_embedded == 0
    assert fake_encoders.batches == [], "already-chunked papers must not be re-encoded"


def test_sync_fulltext_force_leaves_no_stale_chunk_rows(
    embeddings_db, zotero_db_with_pdfs, fake_encoders, monkeypatch
):
    """A re-run producing fewer chunks must not leave the old trailing ones behind."""
    sync_fulltext(embeddings_db, zotero_db_with_pdfs)
    assert len(_chunk_rows(embeddings_db, "PAPER001")) == 3

    # Same PDF now yields much less text -> a single chunk.
    monkeypatch.setattr(
        embedder, "_extract_pdf_text",
        lambda p: (" ".join(f"word{i}" for i in range(100)), None),
    )
    sync_fulltext(embeddings_db, zotero_db_with_pdfs, force=True)
    assert _chunk_rows(embeddings_db, "PAPER001") == [(0, 100)]


def test_sync_fulltext_commits_per_item_and_resumes(
    embeddings_db, zotero_db_with_pdfs, monkeypatch
):
    """One paper failing must not roll back the paper that already succeeded."""
    tok = FakeTokenizer()
    monkeypatch.setattr(embedder, "_get_tokenizer", lambda: tok)

    texts = {
        "one.pdf": " ".join(f"a{i}" for i in range(600)),
        "two.pdf": " ".join(f"b{i}" for i in range(600)),
    }
    monkeypatch.setattr(embedder, "_extract_pdf_text", lambda p: (texts[p.name], None))

    # Poison every chunk text produced from two.pdf's token count... simpler: fail on
    # the second paper by counting encode calls.
    class FailSecondPaper(FakeModel):
        def __init__(self):
            super().__init__()
            self.papers_seen = 0

        def encode(self, texts, **kw):
            # each paper is one batch here (2 chunks < _CHUNK_BATCH_SIZE)
            self.papers_seen += 1
            if self.papers_seen == 2:
                raise RuntimeError("simulated encode failure")
            return super().encode(texts, **kw)

    model = FailSecondPaper()
    monkeypatch.setattr(embedder, "_get_model", lambda: model)

    report = sync_fulltext(embeddings_db, zotero_db_with_pdfs)
    assert report.n_embedded == 1
    assert [k for k, _ in report.failures] == ["PAPER002"]
    assert len(_chunk_rows(embeddings_db, "PAPER001")) == 2, "committed work must survive"
    assert _chunk_rows(embeddings_db, "PAPER002") == []

    # Resume: only the failed paper is retried.
    good = FakeModel()
    monkeypatch.setattr(embedder, "_get_model", lambda: good)
    report2 = sync_fulltext(embeddings_db, zotero_db_with_pdfs)
    assert report2.n_embedded == 1
    assert len(_chunk_rows(embeddings_db, "PAPER002")) == 2


def test_sync_fulltext_reports_failures_by_key_never_title(
    embeddings_db, zotero_db_with_pdfs, monkeypatch
):
    monkeypatch.setattr(embedder, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(embedder, "_get_model", lambda: FakeModel())

    def _broken(path):
        if path.name == "one.pdf":
            return "", "FileDataError: cannot open broken document"
        return " ".join(f"w{i}" for i in range(100)), None

    monkeypatch.setattr(embedder, "_extract_pdf_text", _broken)

    report = sync_fulltext(embeddings_db, zotero_db_with_pdfs)
    assert report.n_embedded == 1
    assert len(report.failures) == 1
    key, reason = report.failures[0]
    assert key == "PAPER001"
    assert "FileDataError" in reason
    assert "Confidential" not in reason, "failure reasons must not leak titles"


def test_sync_fulltext_flags_scanned_pdf_with_no_text(
    embeddings_db, zotero_db_with_pdfs, monkeypatch
):
    monkeypatch.setattr(embedder, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(embedder, "_get_model", lambda: FakeModel())
    monkeypatch.setattr(embedder, "_extract_pdf_text", lambda p: ("   \n  ", None))

    report = sync_fulltext(embeddings_db, zotero_db_with_pdfs)
    assert report.n_embedded == 0
    assert len(report.failures) == 2
    assert all("scanned" in reason for _, reason in report.failures)


def test_sync_fulltext_rejects_bad_window_before_any_work(
    embeddings_db, zotero_db_with_pdfs, fake_encoders
):
    with pytest.raises(ValueError):
        sync_fulltext(embeddings_db, zotero_db_with_pdfs, chunk_tokens=100, chunk_overlap=100)
    assert _chunk_rows(embeddings_db) == []


# --------------------------------------------------------------------------
# Schema + map/cluster loader
# --------------------------------------------------------------------------

def test_fresh_db_has_chunks_table_and_no_legacy_table(tmp_path):
    db_path = tmp_path / "fresh.db"
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "fulltext_chunks" in names
    assert "fulltext_embeddings" not in names


def test_load_all_fulltext_embeddings_means_chunks_and_falls_back(embeddings_db):
    """map/cluster still get exactly one vector per paper."""
    from tests.conftest import store_vector

    e1 = np.zeros(DIMS, dtype=np.float32); e1[0] = 1.0
    e2 = np.zeros(DIMS, dtype=np.float32); e2[1] = 1.0

    store_vector(embeddings_db, "CHUNKED", e2)
    store_vector(embeddings_db, "PLAIN", e1)
    conn = sqlite3.connect(embeddings_db)
    conn.executemany(
        "INSERT INTO fulltext_chunks (zotero_key, chunk_idx, vector, n_tokens, embedded_at) "
        "VALUES ('CHUNKED', ?, ?, 512, 'x')",
        [(0, e1.tobytes()), (1, e2.tobytes())],
    )
    conn.commit()
    conn.close()

    matrix, keys = load_all_fulltext_embeddings(embeddings_db)
    by_key = dict(zip(keys, matrix))
    assert len(keys) == 2

    # CHUNKED: normalised mean of e1 and e2
    expected = (e1 + e2) / np.linalg.norm(e1 + e2)
    np.testing.assert_allclose(by_key["CHUNKED"], expected, atol=1e-6)
    # PLAIN: untouched title+abstract vector
    np.testing.assert_allclose(by_key["PLAIN"], e1, atol=1e-6)
