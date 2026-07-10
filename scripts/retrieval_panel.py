#!/usr/bin/env python3
"""Ground-truth retrieval panel for litmap, with planted answers.

Why this exists
---------------
A semantic top-K can look entirely plausible and still miss the right paper.
The only way to know whether a retrieval change helped is to plant answers you
already know and measure where they rank, before and after.

Two panels are run over the same sampled papers:

  A. CONTROL -- the query is a verbatim passage from deep inside the paper's PDF.
     After a full-text backfill this passage lives inside one of that paper's own
     chunks, so the paper MUST come back at rank 1. If it does not, the index is
     broken, and any Panel-B result is meaningless. This is the positive control:
     without it, "no improvement" is ambiguous between "no benefit" and
     "nothing was measured".

  B. ESTIMATE -- the query is a research question written by a local LLM from that
     same passage, forbidden from copying long runs of its wording. This is the
     honest measurement: it approximates a human asking a conceptual question whose
     answer sits in a paper's methods or results rather than its abstract.

Panel B is an optimistic proxy in one direction (the question is generated from the
passage, so it retains its conceptual specificity) and pessimistic in another (it
never sees the paper's title). Report it as an estimate, not a user-facing metric.

Privacy
-------
No paper text, title, or author is ever printed. Passages and generated questions
are written to a queries file so the before/after runs use IDENTICAL queries; that
file is not meant to be read by the operator. Output carries opaque Zotero keys,
ranks and similarity scores only.

Usage
-----
    # BEFORE the backfill, while fulltext_chunks is empty:
    uv run --project ~/src/Cowork/litmap python scripts/retrieval_panel.py \
        --label baseline --n 24

    # AFTER `litmap sync-fulltext`, reusing the same queries:
    uv run --project ~/src/Cowork/litmap python scripts/retrieval_panel.py \
        --label chunked

    # Compare:
    uv run --project ~/src/Cowork/litmap python scripts/retrieval_panel.py \
        --compare baseline chunked
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from litmap.embedder import EMBEDDINGS_DB, _extract_pdf_text, embed_text  # noqa: E402
from litmap.search import find_similar  # noqa: E402
from litmap.zotero import ZOTERO_DB, get_all_items  # noqa: E402

PANEL_DIR = Path.home() / "LitLake"
QUERIES_FILE = PANEL_DIR / "panel_queries.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "gemma4:26b"

PASSAGE_CHARS = 1200
MIN_DOC_CHARS = 8000          # need a document long enough to have a "deep" passage
BODY_START, BODY_END = 0.15, 0.75   # skip front matter and the reference list
RANK_CEILING = 5000           # effectively "rank everything"

_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}

_SYSTEM = (
    "You turn an excerpt from the body of a scientific paper into the research "
    "question it helps answer. Write ONE question of 15-30 words, phrased the way a "
    "researcher would type it into a literature search. Do not copy any run of more "
    "than three consecutive words from the excerpt. Do not mention the paper, the "
    "excerpt, or the authors."
)


def generate_question(passage: str) -> str:
    payload = {
        "model": JUDGE_MODEL,
        "stream": False,
        "format": _QUESTION_SCHEMA,
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Excerpt:\n{passage}"},
        ],
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            reply = json.loads(response.read().decode())
    except urllib.error.URLError as e:
        raise SystemExit(f"ollama unreachable at {OLLAMA_URL}: {e.reason}. Is `ollama serve` running?")
    return json.loads(reply["message"]["content"])["question"].strip()


def _words(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]


def longest_shared_ngram(a: str, b: str, cap: int = 12) -> int:
    """Longest run of consecutive words the question copied from the passage."""
    aw, bw = _words(a), _words(b)
    bgrams = {n: {tuple(bw[i:i + n]) for i in range(len(bw) - n + 1)} for n in range(1, cap + 1)}
    best = 0
    for n in range(1, min(cap, len(aw)) + 1):
        if any(tuple(aw[i:i + n]) in bgrams[n] for i in range(len(aw) - n + 1)):
            best = n
    return best


def build_queries(n: int, seed: int) -> dict:
    """Sample papers with a long PDF, cut one deep passage each, write a question."""
    items = [i for i in get_all_items(ZOTERO_DB) if i.pdf_path is not None]
    items.sort(key=lambda i: i.key)                     # deterministic order
    random.Random(seed).shuffle(items)

    queries: dict[str, dict] = {}
    scanned = 0
    for item in items:
        if len(queries) >= n:
            break
        scanned += 1
        text, err = _extract_pdf_text(item.pdf_path)
        if err or len(text) < MIN_DOC_CHARS:
            continue
        lo, hi = int(len(text) * BODY_START), int(len(text) * BODY_END) - PASSAGE_CHARS
        if hi <= lo:
            continue
        start = random.Random(seed + len(queries)).randint(lo, hi)
        passage = " ".join(text[start:start + PASSAGE_CHARS].split())
        if len(passage) < PASSAGE_CHARS * 0.6:
            continue
        question = generate_question(passage)
        queries[item.key] = {
            "passage": passage,
            "question": question,
            "leak_ngram": longest_shared_ngram(question, passage),
        }
        print(f"  built {len(queries):>3}/{n}  key={item.key}  leak={queries[item.key]['leak_ngram']}w",
              flush=True)

    print(f"  scanned {scanned} PDFs to build {len(queries)} usable queries")
    return queries


def rank_of(key: str, query_text: str) -> tuple[int | None, float | None, float]:
    vec = embed_text(query_text)
    results = find_similar(vec, EMBEDDINGS_DB, top_k=RANK_CEILING)
    top1 = results[0]["similarity"] if results else float("nan")
    for position, r in enumerate(results, start=1):
        if r["key"] == key:
            return position, r["similarity"], top1
    return None, None, top1


def metrics(ranks: list[int | None]) -> dict:
    found = [r for r in ranks if r is not None]
    reciprocal = [1.0 / r for r in found] + [0.0] * (len(ranks) - len(found))
    return {
        "n": len(ranks),
        "hit@1": sum(r == 1 for r in found) / len(ranks),
        "hit@5": sum(r <= 5 for r in found) / len(ranks),
        "hit@10": sum(r <= 10 for r in found) / len(ranks),
        "hit@25": sum(r <= 25 for r in found) / len(ranks),
        "MRR": sum(reciprocal) / len(ranks),
        "median_rank": statistics.median(found) if found else None,
    }


def run_panel(label: str) -> dict:
    queries = json.loads(QUERIES_FILE.read_text())
    rows = []
    for i, (key, q) in enumerate(sorted(queries.items()), start=1):
        rank_a, sim_a, top1_a = rank_of(key, q["passage"])
        rank_b, sim_b, top1_b = rank_of(key, q["question"])
        rows.append({
            "key": key, "leak_ngram": q["leak_ngram"],
            "A_rank": rank_a, "A_sim_gt": sim_a, "A_sim_top1": top1_a,
            "B_rank": rank_b, "B_sim_gt": sim_b, "B_sim_top1": top1_b,
        })
        print(f"  {i:>3}/{len(queries)}  {key}  A_rank={rank_a}  B_rank={rank_b}", flush=True)

    out = {
        "label": label,
        "n_chunked_papers": _chunked_paper_count(),
        "A_control_verbatim_passage": metrics([r["A_rank"] for r in rows]),
        "B_estimate_generated_question": metrics([r["B_rank"] for r in rows]),
        "leak_ngram_median": statistics.median([r["leak_ngram"] for r in rows]),
        "rows": rows,
    }
    path = PANEL_DIR / f"panel_{label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return out


def _chunked_paper_count() -> int:
    import sqlite3
    conn = sqlite3.connect(EMBEDDINGS_DB)
    try:
        return conn.execute("SELECT COUNT(DISTINCT zotero_key) FROM fulltext_chunks").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def _fmt(m: dict) -> str:
    med = m["median_rank"]
    return (f"hit@1={m['hit@1']:.2f}  hit@5={m['hit@5']:.2f}  hit@10={m['hit@10']:.2f}  "
            f"hit@25={m['hit@25']:.2f}  MRR={m['MRR']:.3f}  median_rank={med}")


def show(out: dict) -> None:
    print(f"\n--- panel '{out['label']}'  (papers with chunks: {out['n_chunked_papers']}) ---")
    print(f"A control  (verbatim passage): {_fmt(out['A_control_verbatim_passage'])}")
    print(f"B estimate (LLM question)    : {_fmt(out['B_estimate_generated_question'])}")
    print(f"median longest copied phrase : {out['leak_ngram_median']} words")


def compare(before: str, after: str) -> None:
    b = json.loads((PANEL_DIR / f"panel_{before}.json").read_text())
    a = json.loads((PANEL_DIR / f"panel_{after}.json").read_text())
    show(b); show(a)

    print(f"\n--- {before} -> {after} ---")
    for panel in ("A_control_verbatim_passage", "B_estimate_generated_question"):
        print(f"\n{panel}")
        for k in ("hit@1", "hit@5", "hit@10", "hit@25", "MRR"):
            print(f"  {k:<7} {b[panel][k]:.3f} -> {a[panel][k]:.3f}   ({a[panel][k] - b[panel][k]:+.3f})")
        print(f"  median  {b[panel]['median_rank']} -> {a[panel]['median_rank']}")

    if a["A_control_verbatim_passage"]["hit@1"] < 0.9 and a["n_chunked_papers"] > 0:
        print("\n!! CONTROL FAILED: a verbatim passage should retrieve its own paper at rank 1.")
        print("   The full-text index is not being used. Do not trust panel B.")

    moved = [(r["key"], rb["B_rank"], r["B_rank"])
             for r, rb in zip(sorted(a["rows"], key=lambda x: x["key"]),
                              sorted(b["rows"], key=lambda x: x["key"]))
             if r["B_rank"] and rb["B_rank"] and r["B_rank"] > rb["B_rank"]]
    if moved:
        print(f"\n{len(moved)} papers got WORSE on panel B (investigate before tuning):")
        for key, before_rank, after_rank in moved:
            print(f"  {key}  {before_rank} -> {after_rank}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", help="name this run (e.g. baseline, chunked)")
    p.add_argument("--n", type=int, default=24, help="papers to sample when building queries")
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--rebuild-queries", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.label:
        p.error("--label is required unless --compare is used")

    if args.rebuild_queries or not QUERIES_FILE.exists():
        print(f"building {args.n} planted queries (this reads PDFs and calls {JUDGE_MODEL})")
        QUERIES_FILE.write_text(json.dumps(build_queries(args.n, args.seed), indent=2))
        print(f"wrote {QUERIES_FILE} (contains paper text -- not for display)")
    else:
        print(f"reusing existing queries from {QUERIES_FILE}")

    show(run_panel(args.label))


if __name__ == "__main__":
    main()
