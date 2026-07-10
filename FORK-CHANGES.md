# Changes in this fork

Forked from [`dougwyu/litmap`](https://github.com/dougwyu/litmap) at commit `efc3c2a6`.
Apache-2.0, same as upstream. Every change below ships with tests; the suite runs
fully offline (`uv sync --extra dev && uv run pytest`).

Upstream's own suite was red at `efc3c2a6` — 20 of 64 tests failed because two
fixtures omitted tables the code queries (`itemAttachments`, `fulltext_embeddings`).
That is repaired first, so later regressions are distinguishable from inherited ones.

---

## 1. Correctness bugs in the index

These silently corrupted the search index. They are the reason to take this fork
seriously even if you do not want the retrieval changes.

### Attachments were embedded as papers

`zotero.py` excluded item types with the literals `itemTypeID NOT IN (14, 26)`.
Those IDs were correct for Zotero 7-8. **Zotero 9 renumbered every item type.** On a
current library, `attachment` is 3 and `note` is 27; `14` and `26` are `email` and
`newspaperArticle`. So the filter excluded emails and newspaper articles while
letting every attachment through — and an attachment's "title" is its PDF filename.

Measured on one real library: **644 attachments** embedded as papers, and every
email and newspaper article wrongly missing.

Fixed by resolving type IDs by name from `itemTypes`, with the old literals kept
only as a fallback when that table cannot be read.

### Deleted papers stayed searchable forever

`sync` only ever `INSERT`ed. Nothing in litmap could remove a vector, so a paper
deleted from Zotero remained in the index indefinitely. Same library: **606 orphan
vectors**.

`sync` now prunes any vector whose key is no longer a citable item, dropping that
paper's full-text chunks with it. An empty library read never triggers a prune, so
a failed or misdirected read cannot empty the index. `sync()` returns a
`SyncReport(n_embedded, n_pruned)`.

### Trashed papers were indexed

Deleting an item in Zotero only adds a row to `deletedItems`; the item stays in
`items` until the trash is emptied. litmap never consulted that table. Now excluded
from `get_all_items`, `get_collection`, `get_item` and `get_subcollection_map`,
guarded for schemas predating the table.

Together these three accounted for **1,266 of 2,898 rows (44%) of a real index**.
Every search ranked real papers against them.

### Authorship order, and vanishing institutional authors

`GROUP_CONCAT` has no defined output order, so `item.authors[0]` — read by callers
as the first author — could name a senior author instead.

Separately, `lastName || ', ' || firstName` is `NULL` when `firstName` is `NULL`,
and `GROUP_CONCAT` skips `NULL`s. Creators stored in Zotero's single-field mode —
organisations, consortia, agencies — therefore **disappeared entirely**, leaving the
item with an empty author list.

Upstream fixes the ordering half in `b4f48cc` using `GROUP_CONCAT(... ORDER BY ...)`.
**That syntax requires SQLite ≥ 3.44.** `pyproject.toml` declares support for Python
3.11, and the python.org 3.11 build links SQLite 3.42, where it is a hard syntax
error — every `get_all_items` call raises. Upstream's fix breaks a configuration
upstream claims to support.

This fork instead fetches authors in a separate `ORDER BY ic.orderIndex` query and
assembles them in Python. It works on every SQLite, fixes both halves of the bug,
and removes the creator join that was multiplying item rows.

---

## 2. Retrieval: score a paper by its best chunk

`sync-fulltext` used to store **one vector per paper**: the mean of its ~3000-token
chunks. Averaging is a low-pass filter. A forty-page paper whose one relevant
section answers the query is dominated by the thirty-nine pages that do not, so its
stored vector points nowhere near the query, and a shallow paper that is vaguely
on-topic throughout outranks it. The more specific the paper, the worse it scored.

Chunks are now stored individually (`fulltext_chunks`, 512 tokens with 64 overlap,
both configurable) and a paper is scored by the **cosine of its best chunk**. Papers
with no full text fall back to their title+abstract vector, so a part-finished
backfill still yields a coherent index. `map` and `cluster`, which need one point per
paper, take the mean of its chunks at load time.

`tests/test_pooling_regression.py` is the known-bug test: a paper with one relevant
chunk among nineteen irrelevant ones scores 0.053 under mean-pooling and 1.0 under
max-pooling, losing to a 0.6 decoy in the first case. It fails against the old
scorer and passes against the new one.

The legacy `fulltext_embeddings` table is retired: never created, never written,
never read. Existing rows are inert. Drop it when convenient:
`DROP TABLE fulltext_embeddings; VACUUM;`

`scripts/retrieval_panel.py` measures the change against planted answers, with a
verbatim-passage control that must return rank 1 once chunks exist — without it,
"no improvement" cannot be distinguished from "the index was never consulted".

---

## 3. Robustness

- **Device selection.** `device="mps"` was hardcoded, so litmap could not run
  anywhere but Apple Silicon. Now `LITMAP_DEVICE` > mps > cuda > cpu, with torch
  imported lazily and any probe failure degrading to cpu.
- **Model mismatch.** The `meta` model row was written with `INSERT OR IGNORE`, so it
  recorded whichever model ran first and never updated. Changing `MODEL_NAME` then
  left the database holding two incompatible vector spaces, silently, with every
  cross-model similarity meaningless. Opening such a database is now refused (exit 2,
  no traceback), naming both models; `sync --force` re-embeds, adopts the new model
  only after all vectors are rewritten, and clears full-text chunks built under the
  old one. Forcing under the *same* model leaves full-text work intact.
- **Unreadable PDFs.** `_extract_pdf_text` swallowed every exception and returned
  `""`, making a corrupt PDF indistinguishable from one with no text — papers
  silently never got embedded. `sync-fulltext` now reports them by Zotero key.
- **Resumability.** `sync-fulltext` commits per paper, and replaces a paper's chunks
  wholesale, so an interrupted run resumes and a shortened document leaves no stale
  trailing chunks.
- **`map --manuscript` leaked.** It inserted the manuscript's vector under the key
  `__manuscript__` and never removed it, so a synthetic non-paper accumulated in the
  library and surfaced in later searches and clusters. Now deleted in a `finally`
  block; stale rows from earlier crashed runs are purged at start; `find_similar`
  refuses the key regardless.

---

## 4. New: `litmap search --judge`

Embedding similarity measures whether two texts are *about the same things*. It
cannot tell whether a paper bears on a claim — a paper arguing the opposite embeds
close to it. `--judge` passes the shortlist to a local [ollama](https://ollama.com)
model which reads title and abstract and scores relevance 0-10; results re-sort by
that score.

Off by default. Standard library only — no new dependency, no lockfile change.
Nothing leaves the machine. If ollama is not running the search returns its unjudged
results and says why, rather than failing. JSON output is strictly additive
(`judge_score`, `judge_reason`, and a top-level `judge` block), and a contract test
pins the pre-existing keys.

---

## Upgrading an existing `~/LitLake/embeddings.db`

Nothing is destroyed; back it up first anyway.

```bash
cp ~/LitLake/embeddings.db ~/LitLake/embeddings.db.bak
uv run litmap sync            # embeds newly-eligible items, prunes the junk
uv run litmap sync-fulltext   # builds chunks; hours; resumable
```

`sync` will report how many stale vectors it pruned. A large number is expected on
any index built by the old code.
