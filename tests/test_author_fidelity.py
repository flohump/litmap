"""Author-fidelity regression tests.

Two independent defects, both in the `authors` column of _ITEM_SELECT:

1. Ordering. GROUP_CONCAT has no defined output order. With creators inserted
   in reverse orderIndex the aggregate returned the LAST author first, so
   `item.authors[0]` -- the field a caller reads as "first author" -- named a
   senior author instead.

2. Single-field creators. `lastName || ', ' || firstName` is NULL when
   firstName is NULL (SQL NULL propagation) and GROUP_CONCAT silently skips
   NULLs, so institutional authors ("Global Science Council") disappeared and
   the item came back with an empty author list.

Both tests fail against the pre-fix _ITEM_SELECT.
"""
from litmap.zotero import get_all_items, get_item


def _by_key(items):
    return {i.key: i for i in items}


def test_authors_follow_orderindex_not_insertion_order(zotero_db_authors):
    item = _by_key(get_all_items(zotero_db_authors))["MULTI001"]
    assert item.authors == [
        "Muller-Karger, Frank E.",
        "Apple, Alice",
        "Zhao, Xavier",
    ]


def test_first_author_is_orderindex_zero(zotero_db_authors):
    item = get_item("MULTI001", zotero_db_authors)
    assert item.authors[0].startswith("Muller-Karger")


def test_institutional_author_is_not_dropped(zotero_db_authors):
    item = _by_key(get_all_items(zotero_db_authors))["INST001"]
    assert item.authors == ["Global Science Council"]


def test_editors_are_not_listed_as_authors(zotero_db_authors):
    """INST001 has an editor (Rita Editor); only the author creatorType counts."""
    item = _by_key(get_all_items(zotero_db_authors))["INST001"]
    assert not any("Editor" in a for a in item.authors)


def test_get_item_and_get_all_items_agree_on_authors(zotero_db_authors):
    from_all = _by_key(get_all_items(zotero_db_authors))["MULTI001"].authors
    from_one = get_item("MULTI001", zotero_db_authors).authors
    assert from_all == from_one


def test_excluded_types_resolved_by_name_not_hardcoded_id(tmp_path):
    """Zotero renumbers itemTypeIDs across major versions (it did so in 9).

    Here 'attachment' and 'note' carry IDs 40/41, not the historical 14/26.
    Resolving by name keeps them excluded; a hardcoded (14, 26) would leak an
    attachment into the paper list.
    """
    import sqlite3
    from litmap.zotero import _connect, _excluded_type_ids

    db_path = tmp_path / "renumbered.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        INSERT INTO itemTypes VALUES (7, 'journalArticle');
        INSERT INTO itemTypes VALUES (40, 'attachment');
        INSERT INTO itemTypes VALUES (41, 'note');
    """)
    conn.commit()
    conn.close()

    with _connect(db_path) as c:
        assert sorted(_excluded_type_ids(c)) == [40, 41]


def test_excluded_types_fall_back_when_table_missing(tmp_path):
    import sqlite3
    from litmap.zotero import _connect, _excluded_type_ids, _EXCLUDED_TYPES_FALLBACK

    db_path = tmp_path / "no_itemtypes.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE placeholder (x INTEGER)")
    conn.commit()
    conn.close()

    with _connect(db_path) as c:
        assert _excluded_type_ids(c) == _EXCLUDED_TYPES_FALLBACK
