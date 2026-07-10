import os
import sqlite3
import pytest
from pathlib import Path

# The suite never loads the embedding model (tests patch _get_model/_get_tokenizer).
# These guarantee a stray import can never reach the HuggingFace hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture
def zotero_db(tmp_path):
    """Minimal Zotero-schema SQLite DB with two items in one collection."""
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

        CREATE TABLE items (
            itemID INTEGER PRIMARY KEY,
            itemTypeID INTEGER,
            libraryID INTEGER DEFAULT 1,
            key TEXT
        );
        INSERT INTO items VALUES (1, 2, 1, 'AAAA0001');
        INSERT INTO items VALUES (2, 2, 1, 'AAAA0002');
        INSERT INTO items VALUES (3, 14, 1, 'AAAA0003');
        INSERT INTO items VALUES (4, 2, 1, 'AAAA0004');
        -- A paper the user moved to the Trash. Zotero keeps it in `items` until
        -- the trash is emptied, so every query must exclude it explicitly.
        -- (itemID 50 / valueIDs 50x: tests add their own rows at 5 / 40x.)
        INSERT INTO items VALUES (50, 2, 1, 'TRASH0005');

        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO itemDataValues VALUES (101, 'Ecology of Networks');
        INSERT INTO itemDataValues VALUES (102, 'Abstract about ecology');
        INSERT INTO itemDataValues VALUES (103, '2021');
        INSERT INTO itemDataValues VALUES (104, '10.1234/eco');
        INSERT INTO itemDataValues VALUES (201, 'Climate and Change');
        INSERT INTO itemDataValues VALUES (202, 'Abstract about climate');
        INSERT INTO itemDataValues VALUES (203, '2022');
        INSERT INTO itemDataValues VALUES (204, '10.5678/cli');
        INSERT INTO itemDataValues VALUES (301, 'Trait Based Analysis');
        INSERT INTO itemDataValues VALUES (302, 'Abstract about traits');
        INSERT INTO itemDataValues VALUES (303, '2023');
        INSERT INTO itemDataValues VALUES (304, '10.1111/trait');
        INSERT INTO itemDataValues VALUES (501, 'Discarded Duplicate Paper');
        INSERT INTO itemDataValues VALUES (502, 'Abstract of a trashed item');
        INSERT INTO itemDataValues VALUES (503, '2024');
        INSERT INTO itemDataValues VALUES (504, '10.2222/trash');

        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        INSERT INTO itemData VALUES (1, 1, 101);
        INSERT INTO itemData VALUES (1, 2, 102);
        INSERT INTO itemData VALUES (1, 6, 103);
        INSERT INTO itemData VALUES (1, 8, 104);
        INSERT INTO itemData VALUES (2, 1, 201);
        INSERT INTO itemData VALUES (2, 2, 202);
        INSERT INTO itemData VALUES (2, 6, 203);
        INSERT INTO itemData VALUES (2, 8, 204);
        INSERT INTO itemData VALUES (4, 1, 301);
        INSERT INTO itemData VALUES (4, 2, 302);
        INSERT INTO itemData VALUES (4, 6, 303);
        INSERT INTO itemData VALUES (4, 8, 304);
        INSERT INTO itemData VALUES (50, 1, 501);
        INSERT INTO itemData VALUES (50, 2, 502);
        INSERT INTO itemData VALUES (50, 6, 503);
        INSERT INTO itemData VALUES (50, 8, 504);

        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY, dateDeleted TEXT);
        INSERT INTO deletedItems VALUES (50, '2026-07-01 00:00:00');

        -- A subscribed journal feed. Its items arrive automatically, have no
        -- attachments, and Zotero deletes them again after a few days. They are
        -- not the user's library and must never be indexed.
        CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT);
        INSERT INTO libraries VALUES (1, 'user');
        INSERT INTO libraries VALUES (7, 'group');
        INSERT INTO libraries VALUES (9, 'feed');
        CREATE TABLE feeds (
            libraryID INTEGER PRIMARY KEY, name TEXT NOT NULL, url TEXT NOT NULL,
            cleanupReadAfter INT, cleanupUnreadAfter INT
        );
        INSERT INTO feeds VALUES (9, 'Nature', 'https://example.org/nature.rss', 3, 30);

        INSERT INTO items VALUES (60, 2, 9, 'FEEDITEM1');
        INSERT INTO itemDataValues VALUES (601, 'A Paper I Have Not Saved');
        INSERT INTO itemDataValues VALUES (602, 'Abstract from a journal feed');
        INSERT INTO itemDataValues VALUES (603, '2026');
        INSERT INTO itemDataValues VALUES (604, '10.3333/feed');
        INSERT INTO itemData VALUES (60, 1, 601);
        INSERT INTO itemData VALUES (60, 2, 602);
        INSERT INTO itemData VALUES (60, 6, 603);
        INSERT INTO itemData VALUES (60, 8, 604);

        -- A genuine paper in a group library: NOT a feed, must stay indexed.
        INSERT INTO items VALUES (70, 2, 7, 'GROUPPAPR');
        INSERT INTO itemDataValues VALUES (701, 'Shared Group Library Paper');
        INSERT INTO itemDataValues VALUES (702, 'Abstract from a group library');
        INSERT INTO itemDataValues VALUES (703, '2025');
        INSERT INTO itemDataValues VALUES (704, '10.4444/group');
        INSERT INTO itemData VALUES (70, 1, 701);
        INSERT INTO itemData VALUES (70, 2, 702);
        INSERT INTO itemData VALUES (70, 6, 703);
        INSERT INTO itemData VALUES (70, 8, 704);

        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        INSERT INTO creators VALUES (1, 'Jane', 'Smith');
        INSERT INTO creators VALUES (2, 'Bob', 'Jones');
        INSERT INTO creators VALUES (3, 'Kim', 'Trashed');

        CREATE TABLE itemCreators (
            itemID INTEGER, creatorID INTEGER, creatorTypeID INTEGER, orderIndex INTEGER
        );
        INSERT INTO itemCreators VALUES (1, 1, 1, 0);
        INSERT INTO itemCreators VALUES (2, 2, 1, 0);
        INSERT INTO itemCreators VALUES (50, 3, 1, 0);

        CREATE TABLE collections (
            collectionID INTEGER PRIMARY KEY,
            collectionName TEXT,
            parentCollectionID INTEGER,
            libraryID INTEGER DEFAULT 1
        );
        INSERT INTO collections VALUES (1, 'My Papers', NULL, 1);
        INSERT INTO collections VALUES (2, 'Sub A', 1, 1);

        CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);
        INSERT INTO collectionItems VALUES (1, 1);
        INSERT INTO collectionItems VALUES (1, 2);
        INSERT INTO collectionItems VALUES (2, 1);
        INSERT INTO collectionItems VALUES (2, 4);

        CREATE TABLE itemAttachments (
            itemID       INTEGER PRIMARY KEY,
            parentItemID INTEGER,
            linkMode     INTEGER,
            contentType  TEXT,
            path         TEXT
        );
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def zotero_db_authors(tmp_path):
    """Zotero DB exercising two author-fidelity traps.

    MULTI001: creators inserted with DECREASING orderIndex, so any query that
      relies on insertion/rowid order yields the last author first.
      Correct authorship order: Muller-Karger (0), Apple (1), Zhao (2).
    INST001: a single-field (institutional) creator with no firstName. Zotero
      stores organisations this way. `lastName || ', ' || firstName` evaluates
      to NULL for such a row, and GROUP_CONCAT skips NULLs, so the author
      vanishes entirely.
    """
    db_path = tmp_path / "zotero_authors.sqlite"
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
        INSERT INTO creatorTypes VALUES (3, 'editor');

        CREATE TABLE items (
            itemID INTEGER PRIMARY KEY, itemTypeID INTEGER,
            libraryID INTEGER DEFAULT 1, key TEXT
        );
        INSERT INTO items VALUES (1, 2, 1, 'MULTI001');
        INSERT INTO items VALUES (2, 2, 1, 'INST001');

        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO itemDataValues VALUES (1, 'Ordered Authors Paper');
        INSERT INTO itemDataValues VALUES (2, 'Abstract one');
        INSERT INTO itemDataValues VALUES (3, '2018');
        INSERT INTO itemDataValues VALUES (4, '10.9999/multi');
        INSERT INTO itemDataValues VALUES (5, 'Institutional Report');
        INSERT INTO itemDataValues VALUES (6, 'Abstract two');
        INSERT INTO itemDataValues VALUES (7, '2020');
        INSERT INTO itemDataValues VALUES (8, '10.9999/inst');

        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        INSERT INTO itemData VALUES (1, 1, 1);
        INSERT INTO itemData VALUES (1, 2, 2);
        INSERT INTO itemData VALUES (1, 6, 3);
        INSERT INTO itemData VALUES (1, 8, 4);
        INSERT INTO itemData VALUES (2, 1, 5);
        INSERT INTO itemData VALUES (2, 2, 6);
        INSERT INTO itemData VALUES (2, 6, 7);
        INSERT INTO itemData VALUES (2, 8, 8);

        -- Highest creatorID is the FIRST author; insertion order is reversed.
        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        INSERT INTO creators VALUES (10, 'Xavier', 'Zhao');
        INSERT INTO creators VALUES (11, 'Alice', 'Apple');
        INSERT INTO creators VALUES (12, 'Frank E.', 'Muller-Karger');
        INSERT INTO creators VALUES (20, NULL, 'Global Science Council');
        INSERT INTO creators VALUES (21, 'Rita', 'Editor');

        CREATE TABLE itemCreators (
            itemID INTEGER, creatorID INTEGER, creatorTypeID INTEGER, orderIndex INTEGER
        );
        INSERT INTO itemCreators VALUES (1, 10, 1, 2);
        INSERT INTO itemCreators VALUES (1, 11, 1, 1);
        INSERT INTO itemCreators VALUES (1, 12, 1, 0);
        INSERT INTO itemCreators VALUES (2, 20, 1, 0);
        INSERT INTO itemCreators VALUES (2, 21, 3, 0);

        CREATE TABLE collections (
            collectionID INTEGER PRIMARY KEY, collectionName TEXT,
            parentCollectionID INTEGER, libraryID INTEGER DEFAULT 1
        );
        CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);
        CREATE TABLE itemAttachments (
            itemID       INTEGER PRIMARY KEY,
            parentItemID INTEGER,
            linkMode     INTEGER,
            contentType  TEXT,
            path         TEXT
        );
    """)
    conn.commit()
    conn.close()
    return db_path


import numpy as np


@pytest.fixture
def embeddings_db(tmp_path):
    """Empty embeddings DB with correct schema."""
    db_path = tmp_path / "embeddings.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE embeddings (
            zotero_key TEXT PRIMARY KEY,
            vector     BLOB NOT NULL,
            embedded_at TEXT NOT NULL
        );
        -- Legacy mean-pooled table. Retired (never written, never read) but kept
        -- in the fixture so the "we no longer read it" tests can poison it.
        CREATE TABLE fulltext_embeddings (
            zotero_key  TEXT PRIMARY KEY,
            vector      BLOB NOT NULL,
            embedded_at TEXT NOT NULL,
            n_tokens    INTEGER,
            n_chunks    INTEGER
        );
        CREATE TABLE fulltext_chunks (
            zotero_key  TEXT    NOT NULL,
            chunk_idx   INTEGER NOT NULL,
            vector      BLOB    NOT NULL,
            n_tokens    INTEGER NOT NULL,
            embedded_at TEXT    NOT NULL,
            PRIMARY KEY (zotero_key, chunk_idx)
        );
        CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO meta VALUES ('model', 'Alibaba-NLP/gte-modernbert-base');
        INSERT INTO meta VALUES ('dims', '768');
    """)
    conn.commit()
    conn.close()
    return db_path


def make_vector(seed: int, dims: int = 768) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(dims).astype(np.float32)


def store_vector(db_path, key: str, vector: np.ndarray):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (zotero_key, vector, embedded_at) VALUES (?, ?, datetime('now'))",
        (key, vector.tobytes()),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def make_vector_fn():
    return make_vector
