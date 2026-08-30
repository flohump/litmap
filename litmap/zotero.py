from dataclasses import dataclass, field
import os
from pathlib import Path
import sqlite3
from typing import Optional

ZOTERO_DB = Path.home() / "Zotero" / "zotero.sqlite"
# Index only the Personal library by default. A large shared group library
# otherwise swamps a curated personal one and pollutes every search result.
# Widen deliberately with LITMAP_LIBRARY_IDS="1,579642", or "all" for every library.
PERSONAL_LIBRARY_ID = 1
# Item types excluded from all paper-level queries: neither is a citable paper.
# Resolved by name at runtime -- Zotero assigns numeric IDs per database, from the
# global schema's order when the itemTypes table is built or migrated, so they are
# not stable across Zotero versions or profile histories.
EXCLUDED_TYPE_NAMES = ("attachment", "note")
# Last resort if the itemTypes table cannot be read: upstream's original literals.
# No known numbering makes both correct (see FORK-CHANGES.md); kept only so that
# failure mode degrades to upstream's behaviour rather than a crash.
_EXCLUDED_TYPES_FALLBACK = (14, 26)


@dataclass
class Item:
    key: str
    title: str
    abstract: str
    authors: list[str]
    year: str
    doi: str
    keywords: str = ""
    pdf_path: Optional[Path] = None


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _field_ids(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT fieldName, fieldID FROM fields "
        "WHERE fieldName IN ('title','abstractNote','date','DOI','keywords')"
    ).fetchall()
    return {r["fieldName"]: r["fieldID"] for r in rows}


def _author_type_id(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT creatorTypeID FROM creatorTypes WHERE creatorType = 'author'"
    ).fetchone()
    return row["creatorTypeID"] if row else 1


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (name,)
    ).fetchone() is not None


def _trash_clause(conn: sqlite3.Connection) -> str:
    """SQL excluding items sitting in Zotero's Trash.

    Deleting a paper in Zotero only adds a row to `deletedItems`; the item stays
    in `items` until the trash is emptied. Without this, papers the user threw
    away keep coming back in search results.
    """
    if not _has_table(conn, "deletedItems"):
        return ""
    return " AND i.itemID NOT IN (SELECT itemID FROM deletedItems)"


def _feed_clause(conn: sqlite3.Connection) -> str:
    """SQL excluding items belonging to a Zotero feed library.

    A feed is a subscribed journal table-of-contents, not a library the user
    curates. Its items arrive automatically, never have attachments, and Zotero
    deletes them again after a few days (`feeds.cleanupReadAfter` /
    `cleanupUnreadAfter`). Indexing them means the index is permanently chasing a
    rolling window of papers the user has not chosen to keep.
    """
    if _has_table(conn, "libraries"):
        columns = {row[1] for row in conn.execute("PRAGMA table_info(libraries)")}
        if "type" in columns:
            return " AND i.libraryID NOT IN (SELECT libraryID FROM libraries WHERE type = 'feed')"
    if _has_table(conn, "feeds"):
        return " AND i.libraryID NOT IN (SELECT libraryID FROM feeds)"
    return ""


def configured_library_ids() -> Optional[tuple[int, ...]]:
    """Libraries to index, or None for every library.

    Defaults to the Personal library alone; LITMAP_LIBRARY_IDS overrides.
    """
    raw = os.environ.get("LITMAP_LIBRARY_IDS")
    if raw is None:
        return (PERSONAL_LIBRARY_ID,)
    if raw.strip().lower() == "all":
        return None
    try:
        ids = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError:
        raise ValueError(
            f"LITMAP_LIBRARY_IDS must be comma-separated integers or 'all', got {raw!r}"
        ) from None
    if not ids:
        raise ValueError("LITMAP_LIBRARY_IDS is empty; unset it or use 'all'")
    return ids


def _library_clause(conn: sqlite3.Connection) -> str:
    """SQL restricting items to the configured libraries."""
    ids = configured_library_ids()
    if ids is None:
        return ""
    return " AND i.libraryID IN (" + ",".join(str(i) for i in ids) + ")"


def _exclusions(conn: sqlite3.Connection) -> str:
    """Everything that is in `items` but is not one of the user's papers."""
    return _trash_clause(conn) + _feed_clause(conn) + _library_clause(conn)


def _excluded_type_ids(conn: sqlite3.Connection) -> tuple[int, ...]:
    placeholders = ",".join("?" * len(EXCLUDED_TYPE_NAMES))
    try:
        rows = conn.execute(
            f"SELECT itemTypeID FROM itemTypes WHERE typeName IN ({placeholders})",
            EXCLUDED_TYPE_NAMES,
        ).fetchall()
    except sqlite3.OperationalError:
        return _EXCLUDED_TYPES_FALLBACK
    ids = tuple(r["itemTypeID"] for r in rows)
    return ids or _EXCLUDED_TYPES_FALLBACK


def _format_author(last: Optional[str], first: Optional[str]) -> str:
    """Render one creator as "Last, First".

    Institutional creators carry only a lastName. Concatenating them in SQL
    (`lastName || ', ' || firstName`) yields NULL, and GROUP_CONCAT drops NULLs,
    which silently erased organisation authors -- hence the formatting lives here.
    """
    last = (last or "").strip()
    first = (first or "").strip()
    if last and first:
        return f"{last}, {first}"
    return last or first


def _authors_by_item(
    conn: sqlite3.Connection,
    author_type: int,
    item_ids: Optional[list[int]] = None,
) -> dict[int, list[str]]:
    """Map itemID -> authors in authorship order.

    Ordering is applied in Python rather than with `GROUP_CONCAT(... ORDER BY ...)`,
    which is a syntax error before SQLite 3.44 (the interpreter shipping this
    package links 3.42). Passing item_ids=None reads every creator row, which
    avoids an IN-clause wider than SQLITE_MAX_VARIABLE_NUMBER on a large library.
    """
    sql = """
        SELECT ic.itemID AS item_id, c.lastName AS last_name, c.firstName AS first_name
        FROM itemCreators ic
        JOIN creators c ON c.creatorID = ic.creatorID
        WHERE ic.creatorTypeID = ?
    """
    params: list = [author_type]
    if item_ids is not None:
        if not item_ids:
            return {}
        sql += f" AND ic.itemID IN ({','.join('?' * len(item_ids))})"
        params.extend(item_ids)
    sql += " ORDER BY ic.itemID, ic.orderIndex"

    out: dict[int, list[str]] = {}
    for r in conn.execute(sql, params).fetchall():
        name = _format_author(r["last_name"], r["first_name"])
        if name:
            out.setdefault(r["item_id"], []).append(name)
    return out


def _rows_to_items(
    rows,
    zotero_base: Optional[Path] = None,
    authors_by_id: Optional[dict[int, list[str]]] = None,
) -> list[Item]:
    authors_by_id = authors_by_id or {}
    items = []
    for r in rows:
        authors = list(authors_by_id.get(dict(r).get("item_id"), []))
        # Resolve PDF path from storage:filename pattern
        pdf_path: Optional[Path] = None
        raw_path = dict(r).get("pdf_path") or ""
        if raw_path and zotero_base is not None:
            if raw_path.startswith("storage:"):
                filename = raw_path[len("storage:"):]
                # Zotero stores attachments in storage/<8-char-key>/<filename>
                # We use the attachment key stored alongside
                att_key = dict(r).get("att_key") or ""
                candidate = zotero_base / "storage" / att_key / filename
                if candidate.exists():
                    pdf_path = candidate
        items.append(Item(
            key=r["key"],
            title=r["title"] or "",
            abstract=r["abstract"] or "",
            authors=authors,
            year=(r["year"] or "")[:4],
            doi=r["doi"] or "",
            keywords=r["keywords"] or "",
            pdf_path=pdf_path,
        ))
    return items


def _item_select(excluded_ids: tuple[int, ...], extra_where: str = "") -> str:
    """Paper-level SELECT. Creators are NOT joined here.

    Joining itemCreators multiplied every item row by its author count and forced
    a GROUP BY whose GROUP_CONCAT had no defined output order. Authors are fetched
    separately by _authors_by_item and attached in Python.
    """
    excluded_sql = "(" + ",".join(str(int(t)) for t in excluded_ids) + ")"
    return """
    SELECT
        i.itemID  AS item_id,
        i.key,
        tv.value  AS title,
        av.value  AS abstract,
        dv.value  AS year,
        doiv.value AS doi,
        kv.value  AS keywords,
        att.path  AS pdf_path,
        atti.key  AS att_key
    FROM items i
    LEFT JOIN itemData    td   ON td.itemID   = i.itemID AND td.fieldID   = :title_id
    LEFT JOIN itemDataValues tv ON tv.valueID = td.valueID
    LEFT JOIN itemData    ad   ON ad.itemID   = i.itemID AND ad.fieldID   = :abs_id
    LEFT JOIN itemDataValues av ON av.valueID = ad.valueID
    LEFT JOIN itemData    dd   ON dd.itemID   = i.itemID AND dd.fieldID   = :date_id
    LEFT JOIN itemDataValues dv ON dv.valueID = dd.valueID
    LEFT JOIN itemData    doid ON doid.itemID = i.itemID AND doid.fieldID = :doi_id
    LEFT JOIN itemDataValues doiv ON doiv.valueID = doid.valueID
    LEFT JOIN itemData    kd   ON kd.itemID   = i.itemID AND kd.fieldID   = :keywords_id
    LEFT JOIN itemDataValues kv ON kv.valueID = kd.valueID
    LEFT JOIN (
        SELECT parentItemID, MIN(itemID) AS itemID, path
        FROM itemAttachments
        WHERE contentType = 'application/pdf'
        GROUP BY parentItemID
    ) att ON att.parentItemID = i.itemID
    LEFT JOIN items atti ON atti.itemID = att.itemID
    WHERE i.itemTypeID NOT IN """ + excluded_sql + """
      AND tv.value IS NOT NULL""" + extra_where + """
"""


def _field_params(fids: dict[str, int]) -> dict:
    return {
        "title_id": fids["title"], "abs_id": fids["abstractNote"],
        "date_id": fids["date"], "doi_id": fids["DOI"],
        "keywords_id": fids.get("keywords"),
    }


def get_all_items(db_path: Path = ZOTERO_DB) -> list[Item]:
    zotero_base = db_path.parent
    with _connect(db_path) as conn:
        fids = _field_ids(conn)
        atid = _author_type_id(conn)
        rows = conn.execute(
            _item_select(_excluded_type_ids(conn), _exclusions(conn)),
            _field_params(fids),
        ).fetchall()
        # item_ids=None: read every creator row rather than build a 16k-wide IN clause
        authors = _authors_by_item(conn, atid)
    return _rows_to_items(rows, zotero_base, authors)


def get_collection(name: str, db_path: Path = ZOTERO_DB) -> list[Item]:
    zotero_base = db_path.parent
    with _connect(db_path) as conn:
        fids = _field_ids(conn)
        atid = _author_type_id(conn)
        rows = conn.execute(
            _item_select(_excluded_type_ids(conn), _exclusions(conn)) + """
              AND i.itemID IN (
                  SELECT ci.itemID FROM collectionItems ci
                  JOIN collections col ON col.collectionID = ci.collectionID
                  WHERE col.collectionName = :name
              )
            """,
            {**_field_params(fids), "name": name},
        ).fetchall()
        authors = _authors_by_item(conn, atid, [r["item_id"] for r in rows])
    return _rows_to_items(rows, zotero_base, authors)


def get_item(key_or_doi: str, db_path: Path = ZOTERO_DB) -> Optional[Item]:
    zotero_base = db_path.parent
    with _connect(db_path) as conn:
        fids = _field_ids(conn)
        atid = _author_type_id(conn)
        rows = conn.execute(
            _item_select(_excluded_type_ids(conn), _exclusions(conn)) + """
              AND (i.key = :val OR doiv.value = :val)
            LIMIT 1
            """,
            {**_field_params(fids), "val": key_or_doi},
        ).fetchall()
        authors = _authors_by_item(conn, atid, [r["item_id"] for r in rows])
    items = _rows_to_items(rows, zotero_base, authors)
    return items[0] if items else None


def get_subcollection_map(db_path: Path = ZOTERO_DB) -> dict[str, list[str]]:
    """Return {zotero_key: [collection_names]} for every non-attachment item.

    Lists only the collections the item directly belongs to — parents of
    a child collection are not included by inheritance.
    """
    with _connect(db_path) as conn:
        excluded_sql = "(" + ",".join(str(int(t)) for t in _excluded_type_ids(conn)) + ")"
        rows = conn.execute(
            """
            SELECT i.key AS key, col.collectionName AS name
            FROM items i
            JOIN collectionItems ci ON ci.itemID = i.itemID
            JOIN collections col ON col.collectionID = ci.collectionID
            WHERE i.itemTypeID NOT IN """ + excluded_sql + _exclusions(conn) + """
            ORDER BY i.key, col.collectionName
            """
        ).fetchall()
    mapping: dict[str, list[str]] = {}
    for r in rows:
        mapping.setdefault(r["key"], []).append(r["name"])
    return mapping
