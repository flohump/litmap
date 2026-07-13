# Validating a retrieval change

A change to how papers are scored either helps or it doesn't, and you cannot tell by
reading the diff. Neither can you tell by running a few searches and liking the
results — a semantic top-K is *designed* to look plausible. The only way to know is
to plant answers you already know and measure where they rank.

`scripts/retrieval_panel.py` does that.

## The two panels

The script samples papers that have a local PDF, cuts one passage from deep inside
each (past the abstract, before the references), and runs two panels over the same
papers:

**A. Control — the query is the verbatim passage.** Once chunks exist, that passage
lives inside one of the paper's own chunks, so the paper *must* come back at rank 1.
If it doesn't, the full-text index is not being consulted and nothing else you
measure means anything.

This is the part people skip, and it is the part that matters. Without a control,
"no improvement" and "I never actually measured anything" produce identical output.

**B. Estimate — the query is a research question written from that passage**, in
someone else's words. The script rejects any question that copies more than five
consecutive words. This approximates a human asking a conceptual question whose
answer sits in a paper's methods or results rather than its abstract.

## Running it

```bash
# 1. Sample papers and cut one deep passage from each.
python scripts/retrieval_panel.py --extract --n 14

# 2. Write ~/LitLake/panel_queries.json:  {"<key>": {"passage": "...", "question": "..."}}
#    Write each question yourself, without reusing the passage's phrasing.
#    (--use-ollama will draft them with a local model, for unattended runs.)

# 3. Baseline, before the change:
python scripts/retrieval_panel.py --label baseline --title-abstract-only

# 4. After `litmap sync-fulltext`:
python scripts/retrieval_panel.py --label chunked
python scripts/retrieval_panel.py --compare baseline chunked
```

`--title-abstract-only` reproduces the pre-chunking scorer exactly, so the "before"
panel can be re-run at any time without deleting the chunks. Both runs must use
identical queries, or you have measured the queries rather than the index.

`--compare` prints the delta and **names any paper that got worse**. That is a
finding to investigate — a chunk boundary splitting the supporting passage, or a
chunk of boilerplate matching the query — not noise to tune away.

## Reading the result honestly

Two biases are built in, and neither can be removed:

- **Only papers with a local PDF can be sampled**, and they are the only papers
  chunking can help. The panel measures the best case, not the library-wide effect.
- **The question is written with the passage in view**, so it retains conceptual
  specificity a cold user query would not have. It is optimistic in that direction
  and pessimistic in another: it never sees the paper's title.

Report it as an estimate. State `n`. State that it is one library and one embedding
model.

## Similarity thresholds

The panel also yields the data to calibrate a "nothing relevant" cut-off. Add a
handful of deliberately out-of-domain questions — topics your library certainly does
not cover — and compare their top-1 similarity against the in-domain ones.

### What the calibration measured

Fourteen in-domain questions — each with a known answer in the library — and eight
deliberately out-of-domain ones (superconducting qubits, Byzantine architecture,
myeloma, baroque flute fingering) separate cleanly, but only on the **top-1
similarity of the result list**:

- out-of-domain top-1 never exceeded **0.711**;
- in-domain top-1 was always **≥ 0.767**, and that floor held in *both* regimes —
  chunked full text and title+abstract only.

The separating margin is therefore just **0.056**. Gate the top-1 at about **0.74**
and it answers one question reliably, and one only: *does this library contain
anything on topic?* Because the margin is so narrow, treat **0.71–0.78** as a grey
zone — read the abstract rather than trust the number.

The gate is a property of the whole result list, never of an individual candidate: it
does not say whether any one paper is relevant (warning 1 below), and candidate scores
are not comparable across papers, because full-text and title+abstract-only papers sit
on different scales. Same caveats as the panel above: n = 14 in-domain + 8
out-of-domain queries, one library, one embedding model.

Two warnings, both learned the hard way on a real library:

1. **Gate on the top-1 of the result list, not on a candidate's own score.** A paper
   scored on its best full-text chunk sits systematically above one scored on its
   title and abstract. Measured: the *same* correct answers scored 0.811–0.881 with
   full text and as low as 0.641 without — beneath the 0.711 that an unrelated paper
   reaches for a nonsense query. So a relevant paper with no PDF can rank below an
   irrelevant one that has a PDF. Every search result carries `full_text: true|false`;
   compare only within a group.

2. **Do not invent a gap-based rule.** The gap between top-1 and top-2, the gap
   between top-1 and top-10, the lift over the tail, and the z-score of top-1 against
   its tail were all tested. **All four overlap** between real and nonsense queries.
   The absolute top-1 was the only statistic that separated.
