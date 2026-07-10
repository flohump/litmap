# litmap

> ### This is a patched fork
>
> Forked from [`dougwyu/litmap`](https://github.com/dougwyu/litmap) at `efc3c2a6`.
> **[`FORK-CHANGES.md`](FORK-CHANGES.md) lists what differs and why** — including four
> bugs that silently corrupt the search index on any Zotero 9 library, and a
> retrieval change measured against planted answers ([`docs/VALIDATION.md`](docs/VALIDATION.md)).
>
> If you are running upstream litmap today, the thirty-second read-only check in
> `FORK-CHANGES.md` will tell you how much of your index is not papers. On the
> library that prompted this fork, it was 48%.
>
> Branches: `fulltext-chunks` is the work. `master` is the pristine upstream commit
> this was forked from, kept for provenance. `upstream-master` parks upstream's
> current tip, unaudited — **do not merge it**: its author-order fix requires
> SQLite ≥ 3.44 and is a hard syntax error on the Python 3.11 that `pyproject.toml`
> declares support for.

A local Python CLI for semantic mapping and search over your [Zotero](https://www.zotero.org/) library. Generate interactive 2D maps of papers positioned by meaning, find papers similar to a query or focal paper, and understand how your manuscript sits within its citation landscape.

Everything runs locally — no background daemon, no network calls after the first model download, no opaque installers.

---

## Features

- **`litmap map`** — UMAP scatter plot of papers with k-nearest-neighbour edges, coloured by semantic position. Cluster labels generated automatically via HDBSCAN + TF-IDF. Outputs interactive Plotly HTML and publication-quality PNG/PDF (300 DPI).
- **`litmap search`** — Cosine similarity search over your Zotero library for a query sentence, passage, or focal paper. A paper is scored by its single best full-text chunk, so a relevant section is found even in a long paper mostly about something else; papers without full text fall back to title+abstract. Optional `--judge` reranks the shortlist with a local ollama model. Outputs a ranked table or JSON.
- **`litmap cluster`** — Hierarchical semantic clustering (≤2 levels) of a collection, bibliography, or the whole library. Outputs an interactive dendrogram (HTML), static dendrograms (PNG/PDF), a labelled outline (Markdown + JSON), and a `.linkage.npy` cache for downstream analyses.
- **`litmap info`** — Show embedding status for a single paper.
- **`litmap sync`** — Manually trigger title+abstract embedding of all Zotero items.
- **`litmap sync-fulltext`** — Embed full PDF text for items with a local PDF, as overlapping chunks. Used automatically in place of title+abstract wherever it exists. Safe to interrupt and resume; PDFs that cannot be read are reported rather than skipped in silence.
- **Auto-sync** — Every command automatically embeds any Zotero items not yet in the cache before running. A `tqdm` progress bar appears during sync; silent if already up to date.
- **Four map modes** — collection only, manuscript bibliography only, intersection, or full union.
- **Manuscript node** — When `--manuscript` is provided, your paper appears as a red star in the map, positioned semantically among its cited works. Papers in the library that have the manuscript among their k nearest neighbours in embedding space will also draw edges to it, so the manuscript typically accumulates more edges than regular nodes.

---

## Requirements

- Python 3.11–3.13
- [uv](https://github.com/astral-sh/uv) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Zotero installed with a library at `~/Zotero/zotero.sqlite`
- Optional, for `litmap search --judge`: [ollama](https://ollama.com) running locally with a chat model pulled

---

## Installation

```bash
git clone <this-repo> ~/your/path/litmap
cd ~/your/path/litmap
uv venv
uv pip install -e .
```

On first use, `sentence-transformers` downloads the `Alibaba-NLP/gte-modernbert-base` embedding model (~570 MB). Subsequent runs are fully offline.

Inference picks a device automatically: Metal (MPS) on Apple Silicon, else CUDA, else CPU. Override with `LITMAP_DEVICE=cpu litmap ...`.

---

## Quick Start

`uv run litmap` must be run from inside the project directory (or a subdirectory). The easiest way to make it available everywhere is a shell alias:

```bash
echo 'alias litmap="uv run --project ~/your/path/litmap litmap"' >> ~/.zshrc
source ~/.zshrc
```

After that, you can run `litmap` from anywhere:

```bash
# Search your library
litmap search --query "biodiversity measurement remote sensing" --top-k 5

# Map a Zotero collection
litmap map --collection "My Papers" --output ~/Desktop/litmap

# Map a manuscript's bibliography
litmap map --manuscript paper.pdf --output ~/Desktop/litmap

# Map a manuscript positioned within its cited collection
litmap map --collection "My Papers" --manuscript paper.pdf --output ~/Desktop/litmap

# Cluster a collection into a labelled semantic hierarchy
litmap cluster --collection "My Papers" --output ~/Desktop/clusters
```

Without the alias, prefix every command with `uv run` and run it from `~/your/path/litmap/`.

Open `litmap.html` in a browser for the interactive version. `litmap.png` and `litmap.pdf` are ready for publication.

See [docs/tutorial.md](docs/tutorial.md) for a full walkthrough.

---

## CLI Reference

### `litmap map`

Requires at least one of `--collection` or `--manuscript`.

```
Options:
  -c, --collection TEXT          Zotero collection name
  -m, --manuscript PATH          Manuscript file (PDF, DOCX, .bib, .tex)
      --union                    Use collection ∪ manuscript bibliography
  -o, --output PATH              Output base path [default: litmap_output]
      --n-neighbors INT          UMAP n_neighbors [default: 3]
      --edge-k INT               k-NN edges per node [default: 3]
  -f, --format TEXT              html | png | pdf | all [default: all]
      --label-clusters           Annotate HDBSCAN clusters with TF-IDF keywords [default: on]
      --no-label-clusters        Disable cluster annotations
      --min-cluster-size INT     HDBSCAN min_cluster_size [default: 5]
      --center-manuscript        Bias manuscript node toward centre of UMAP layout
```

**Paper set.** The paper set depends on which flags are provided:

- `--collection` only — maps papers in the collection.
- `--manuscript` only — parses the manuscript bibliography and maps those papers.
- `--collection --manuscript` (no `--union`) — maps the collection only, with the manuscript added as a red star node.
- `--collection --manuscript --union` — maps the union of the collection and the manuscript bibliography, with the manuscript as a red star node.

**Layout.** UMAP reduces the high-dimensional embedding vectors to 2D. `--n-neighbors` controls how many neighbours each point considers: lower values (e.g. 3) produce tighter, more separated clusters by emphasising local structure; higher values (e.g. 30–50) produce smoother, more globally coherent layouts.

**Edges.** Each paper is connected to its `--edge-k` nearest neighbours in embedding space. In the static PNG/PDF output, edge opacity is weighted by cosine similarity — darker edges indicate stronger semantic overlap. The manuscript node (red star) accumulates both its own outgoing edges and incoming edges from papers that count it among their nearest neighbours, so it typically has more connections than regular nodes.

**Cluster labels.** Generated automatically by running HDBSCAN on the 2D layout coordinates, then labelling each cluster with its top-3 TF-IDF keyword phrases. Title and abstract are used as the TF-IDF corpus (full PDF text produces noisy labels from reference lists and boilerplate). Noise points (papers HDBSCAN couldn't assign to any cluster) are left unlabelled. Labels are rendered in dark teal with a white background box. Use `--no-label-clusters` to disable, or `--min-cluster-size` to control granularity — larger values produce fewer, broader clusters.

### `litmap search`

Requires either `--query` or `--paper`.

```
Options:
  -q, --query TEXT          Query sentence or passage
  -p, --paper TEXT          Title or DOI of a focal paper in your library
  -c, --collection TEXT     Scope search to a collection
  -k, --top-k INT           Number of results [default: 10]
  -f, --format TEXT         table | json [default: table]
      --judge               Rerank results with a local ollama judge model
      --judge-model TEXT    Ollama model used by --judge [default: gemma4:26b]
      --judge-url TEXT      Ollama base URL [default: http://localhost:11434]
```

Provide either `--query` (free text) or `--paper` (title fragment or DOI of a paper already in your library). When using `--paper`, the focal paper itself is excluded from results. Results are deduplicated by DOI and by title before returning.

**Scoring.** A paper's score is the cosine of its *best-matching full-text chunk*, not of a vector averaged over the whole document. Averaging is a low-pass filter: a forty-page paper whose one relevant section answers your query is drowned out by the thirty-nine pages that do not, so it loses to a shallow paper that is vaguely on-topic throughout. Scoring by the best chunk finds the section. Papers with no full text are scored on their title+abstract vector, so a partially-embedded library degrades gracefully rather than ranking inconsistently.

**`--judge`.** Embedding similarity measures whether two texts are *about the same things*. It cannot tell whether a paper actually bears on your claim — a paper arguing the opposite of your query embeds close to it. With `--judge`, the shortlist is passed to a local [ollama](https://ollama.com) model that reads each title and abstract and scores relevance from 0 to 10; results re-sort by that score, and `judge_score` / `judge_reason` are added to the JSON output. Nothing leaves the machine. Requires `ollama serve` and the model pulled (`ollama pull gemma4:26b`); if either is missing, the search returns its unjudged results and explains why on stderr rather than failing.

```bash
# Two-stage retrieval: embed wide, then judge the shortlist
litmap search --query "does nitrogen addition reduce plant diversity" --top-k 15 --judge
```

### `litmap cluster`

```
Options:
  -c, --collection TEXT          Zotero collection name
  -m, --manuscript PATH          Manuscript file (PDF, DOCX, .bib, .tex)
      --union                    Use collection ∪ bibliography as paper set
      --top-clusters INT         Number of level-1 clusters [default: auto]
      --subcluster-threshold INT Min cluster size that triggers level-2 [default: 20]
  -o, --output PATH              Output base path [default: litmap_cluster]
  -f, --format TEXT              html | pdf | png | md | json | all [default: all]
```

Writes `<output>.html` (interactive dendrogram), `.pdf` + `.png` (static 300 DPI), `.md` + `.json` (labelled outline), and `.linkage.npy` (scipy linkage cache). If neither `--collection` nor `--manuscript` is given, the entire library is clustered. If both are given, `--union` is required.

Level-1 cluster count defaults to `max(2, round(sqrt(N/2)))`; any level-1 cluster with at least `--subcluster-threshold` papers is split into sub-clusters. Cluster labels are TF-IDF keyword triplets derived from member title and abstract text.

### `litmap info <paper>`

Show embedding status for a single paper. Accepts a title fragment, DOI, or Zotero key. Reports the PDF path, whether a title+abstract embedding exists, and whether a full-text embedding exists (with token count and chunk count).

```bash
litmap info "Chung 2026"
litmap info 10.1038/s41586-024-12345-6
litmap info ABC12DEF
```

### `litmap sync`

```
Options:
      --force    Re-embed all papers, even those already in the cache
```

Embed all Zotero items (title + abstract) not yet in the cache. Runs automatically before every command, so manual invocation is rarely needed.

`--force` regenerates every embedding, and is the supported way to switch models: change `MODEL_NAME` in `embedder.py`, then run `litmap sync --force`. Once all vectors have been rewritten, the new model is recorded in `meta` and any full-text chunks built under the old model are cleared (re-run `sync-fulltext` to rebuild them). Forcing under the *same* model leaves full-text work untouched.

### `litmap sync-fulltext`

```
Options:
  -c, --collection TEXT    Scope to a Zotero collection
      --chunk-tokens INT   Tokens per chunk [default: 512]
      --chunk-overlap INT  Token overlap between consecutive chunks [default: 64]
      --force              Re-embed even already-processed PDFs
```

Embeds the full text of every Zotero item that has a local PDF attachment. Text is extracted with PyMuPDF and split into overlapping windows of `--chunk-tokens` tokens (stepping forward by `chunk-tokens - chunk-overlap` each time, so a passage straddling a boundary still lands whole inside some chunk). Every chunk is encoded and stored as its own row in `fulltext_chunks`. `search` scores a paper by its best chunk; `map` and `cluster`, which need one point per paper, average its chunks at load time.

The job is safe to interrupt and resume: each paper is committed in its own transaction, and already-embedded papers are skipped unless `--force` is passed. A re-run replaces a paper's chunks wholesale, so shortening a document never leaves stale chunks behind. Use `--collection` to process one collection at a time (personal or group), which is how to try a chunk size on a small set before committing to a full-library run.

PDFs that cannot be read — corrupt files, scanned images with no text layer — are listed at the end of the run with their Zotero keys, instead of being silently dropped:

```
3 PDFs could not be read and have no full text:
  ABCD1234  no extractable text (scanned PDF?)
  EFGH5678  FileDataError: cannot open broken document
Inspect one with: litmap info <key>
```

**Choosing `--chunk-tokens`**

A chunk is the unit that gets scored, so it should be about the size of *one argument* — a few paragraphs — not one paper. Small chunks are what make a buried result findable; the point of storing chunks separately is lost if each one spans half a paper. Overlap exists so a claim split across a boundary is not cut in half.

- **512 tokens (~380 words)** — roughly a long paragraph or a short subsection. The default.
- **256 tokens** — sharper localisation of a specific claim, at more rows and more encoding calls.
- **1024+ tokens** — coarser; a chunk starts to average over several distinct points, which is the problem chunking exists to solve. Also disproportionately slower: attention cost grows with the square of sequence length, so one 1024-token chunk costs about as much as four 512-token ones.

Chunks are encoded in small batches. If a machine with little GPU memory ever reports an out-of-memory error, set `_CHUNK_BATCH_SIZE = 1` in `embedder.py` to restore strict one-chunk-at-a-time encoding.

Full-library runs take hours. Throughput depends on machine, chunk size and PDF length; measure it on your own library rather than trusting a number here.

```bash
# Try the default on one collection first
litmap sync-fulltext --collection "My Papers"

# Sharper chunks for a collection you are working through closely
litmap sync-fulltext --collection "My Papers" --chunk-tokens 256 --force

# The whole library, resumable — safe to interrupt
litmap sync-fulltext
```

---

## Storage

Embeddings are stored in `~/LitLake/embeddings.db` (SQLite), created automatically on first run. The database holds:

- **`embeddings`** — title + abstract vectors. One row per Zotero item, ~3 KB each (768 float32 values). Populated by `litmap sync` and auto-sync.
- **`fulltext_chunks`** — full-text vectors, **one row per chunk**, keyed `(zotero_key, chunk_idx)`, ~3 KB each plus `n_tokens`. A 6000-token paper at the default window yields about 13 chunks, so roughly 40 KB per paper. Populated by `litmap sync-fulltext`.
- **`meta`** — the embedding model and dimensionality this database was built with. Opening it with a different `MODEL_NAME` is refused: vectors from two models share no coordinate system, so similarity across them is meaningless. `litmap sync --force` re-embeds and adopts the new model.

`search` uses `fulltext_chunks` where a paper has it and `embeddings` otherwise, per paper — the two can be populated independently, and a half-finished `sync-fulltext` still leaves a coherent index.

> **Legacy.** Databases created before chunked storage also contain a `fulltext_embeddings` table holding one mean-pooled vector per paper. It is no longer written or read, and no longer affects any result. Re-run `litmap sync-fulltext` to build chunks; once you are satisfied, reclaim the space with `sqlite3 ~/LitLake/embeddings.db "DROP TABLE fulltext_embeddings; VACUUM;"`.

To inspect coverage:

```bash
sqlite3 ~/LitLake/embeddings.db "
SELECT
  (SELECT COUNT(*) FROM embeddings) AS title_abstract_embedded,
  (SELECT COUNT(DISTINCT zotero_key) FROM fulltext_chunks) AS fulltext_papers,
  (SELECT COUNT(*) FROM fulltext_chunks) AS total_chunks,
  (SELECT value FROM meta WHERE key = 'model') AS model;"
```

---

## Scripts

### `scripts/manuscript_to_bib.py`

Export a BibTeX file of all papers cited in a manuscript, matched against your Zotero library.

```bash
uv run scripts/manuscript_to_bib.py manuscript.docx refs.bib
uv run scripts/manuscript_to_bib.py manuscript.docx refs.bib --zotero-db ~/Zotero/zotero.sqlite
```

Works with two document types:

- **Word documents with live Zotero field codes** — citations inserted via the Zotero Word or Google Docs plugin, not yet unlinked. Keys are extracted directly from the field code JSON and matched exactly.
- **Google Docs exports (Download as .docx)** — field codes are stripped on export, so the script falls back to extracting `https://doi.org/` links from the Zotero-generated bibliography and matching them against Zotero by DOI. Requires the document to contain a Zotero bibliography that includes DOI numbers (e.g. Methods in Ecology & Evolution style).

The script tries field codes first and falls back to DOI extraction automatically. Output is a `.bib` file importable via Zotero **File → Import**. Errors if the output file already exists.

---

## Architecture

```
litmap/
├── zotero.py          Read-only access to ~/Zotero/zotero.sqlite; resolves PDF paths
├── embedder.py        Embedding model, embeddings.db cache, fulltext pipeline
├── layout.py          UMAP 2D layout + HDBSCAN cluster labels + k-NN graph
├── search.py          Best-chunk (max-pool) similarity search + deduplication
├── judge.py           Optional local-ollama reranker for `search --judge`
├── manuscript.py      Bibliography parser (PDF/DOCX/BibTeX/LaTeX)
├── renderer.py        Plotly HTML + matplotlib PNG/PDF (for `map`)
├── cluster.py         Hierarchical clustering, TF-IDF labelling, outline builder
├── cluster_render.py  Dendrograms (Plotly HTML + matplotlib) + outline renderers
├── cli.py             Typer CLI — map, search, cluster, sync, sync-fulltext, info
└── scripts/
    └── manuscript_to_bib.py  Export cited papers to BibTeX (Google Docs + Word)
```

---

## Credits

- **[lit-lake](https://github.com/ElliotRoe/lit-lake)** by Elliot Roe — the original inspiration for this project. `litmap` replicates lit-lake's core embedding and search functionality as an auditable, dependency-light Python package, without the `.mcpb` installer or background daemon.
- **[Claude](https://claude.ai)** (Anthropic) — this codebase was designed and implemented using Claude (Cowork mode), including full-text PDF embedding, chunked encoding, MPS memory management, and HDBSCAN cluster labelling.
- **[sentence-transformers](https://www.sbert.net/)** + **[Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)** — local embedding inference; 768-dim GTE-ModernBERT (149M params, 8192-token context) with Metal GPU acceleration on Apple Silicon.
- **[UMAP](https://github.com/lmcinnes/umap)** — dimensionality reduction for the semantic layout.
- **[HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan)** (via scikit-learn) — unsupervised clustering for map annotations.

---

## License

Apache 2.0
