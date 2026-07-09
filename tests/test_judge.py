"""Judge reranking, and the JSON contract the Zotero skills depend on.

No socket is ever opened: urlopen is monkeypatched. `--judge` must degrade to
unjudged results rather than failing the search, and must only ADD keys to the
JSON output -- the zotero and manuscript-audit skills parse the existing ones.
"""
import io
import json
import urllib.error

import numpy as np
import pytest
from typer.testing import CliRunner

from litmap import embedder, judge as judge_mod
from litmap.cli import app
from litmap.judge import JudgeError, judge_results

# Keys the zotero + manuscript-audit skills read out of each result.
SKILL_RESULT_KEYS = {"zotero_key", "title", "authors", "year", "abstract", "similarity", "doi"}


def _reply(scores) -> dict:
    return {"message": {"content": json.dumps({"scores": scores})}}


def _fake_urlopen(reply: dict):
    class _Response:
        def read(self):
            return json.dumps(reply).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return lambda request, timeout=None: _Response()


def _results(n=3):
    return [
        {"zotero_key": f"K{i}", "title": f"Paper {i}", "abstract": f"Abstract {i}",
         "authors": [], "year": "2020", "doi": f"10.1/{i}", "similarity": 0.9 - 0.1 * i}
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# judge_results
# --------------------------------------------------------------------------

def test_judge_reorders_by_score(monkeypatch):
    monkeypatch.setattr(
        judge_mod.urllib.request, "urlopen",
        _fake_urlopen(_reply([
            {"index": 0, "score": 2, "reason": "topical only"},
            {"index": 1, "score": 9, "reason": "directly supports"},
            {"index": 2, "score": 5, "reason": "partial"},
        ])),
    )
    out = judge_results("q", _results())
    assert [r["zotero_key"] for r in out] == ["K1", "K2", "K0"]
    assert out[0]["judge_score"] == 9
    assert out[0]["judge_reason"] == "directly supports"


def test_judge_preserves_original_fields(monkeypatch):
    monkeypatch.setattr(
        judge_mod.urllib.request, "urlopen",
        _fake_urlopen(_reply([{"index": i, "score": 5, "reason": "x"} for i in range(3)])),
    )
    before = _results()
    out = judge_results("q", before)
    for r in out:
        original = next(b for b in before if b["zotero_key"] == r["zotero_key"])
        for key, value in original.items():
            assert r[key] == value, f"judge mutated {key}"


def test_judge_ties_break_on_similarity(monkeypatch):
    monkeypatch.setattr(
        judge_mod.urllib.request, "urlopen",
        _fake_urlopen(_reply([{"index": i, "score": 7, "reason": ""} for i in range(3)])),
    )
    out = judge_results("q", _results())
    assert [r["zotero_key"] for r in out] == ["K0", "K1", "K2"]


def test_judge_unscored_results_sort_last_and_survive(monkeypatch):
    monkeypatch.setattr(
        judge_mod.urllib.request, "urlopen",
        _fake_urlopen(_reply([{"index": 1, "score": 4, "reason": "ok"}])),
    )
    out = judge_results("q", _results())
    assert len(out) == 3, "results the judge skipped must not disappear"
    assert out[0]["zotero_key"] == "K1"
    assert {r["judge_score"] for r in out[1:]} == {0}


def test_judge_out_of_range_index_ignored(monkeypatch):
    monkeypatch.setattr(
        judge_mod.urllib.request, "urlopen",
        _fake_urlopen(_reply([
            {"index": 99, "score": 10, "reason": "hallucinated"},
            {"index": 0, "score": 6, "reason": "real"},
        ])),
    )
    out = judge_results("q", _results())
    assert len(out) == 3
    assert out[0]["zotero_key"] == "K0"


def test_judge_clamps_scores(monkeypatch):
    monkeypatch.setattr(
        judge_mod.urllib.request, "urlopen",
        _fake_urlopen(_reply([{"index": 0, "score": 42, "reason": ""},
                              {"index": 1, "score": -5, "reason": ""}])),
    )
    out = judge_results("q", _results(2))
    assert out[0]["judge_score"] == 10
    assert out[1]["judge_score"] == 0


def test_judge_connection_refused_raises_actionable_error(monkeypatch):
    def _refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(judge_mod.urllib.request, "urlopen", _refuse)
    with pytest.raises(JudgeError, match="ollama serve"):
        judge_results("q", _results())


def test_judge_missing_model_says_how_to_pull(monkeypatch):
    def _404(request, timeout=None):
        raise urllib.error.HTTPError("u", 404, "not found", {}, io.BytesIO(b"no such model"))

    monkeypatch.setattr(judge_mod.urllib.request, "urlopen", _404)
    with pytest.raises(JudgeError, match="ollama pull"):
        judge_results("q", _results())


def test_judge_unparseable_reply_raises(monkeypatch):
    monkeypatch.setattr(
        judge_mod.urllib.request, "urlopen",
        _fake_urlopen({"message": {"content": "not json at all"}}),
    )
    with pytest.raises(JudgeError, match="parse"):
        judge_results("q", _results())


def test_judge_empty_results_short_circuits(monkeypatch):
    def _explode(*a, **k):
        raise AssertionError("must not call ollama for an empty result set")

    monkeypatch.setattr(judge_mod.urllib.request, "urlopen", _explode)
    assert judge_results("q", []) == []


def test_abstract_budget_shrinks_with_result_count():
    """Prompt must stay inside the context window as top-k grows."""
    assert judge_mod._abstract_budget(1) == judge_mod._MAX_ABSTRACT_CHARS
    assert judge_mod._abstract_budget(500) == judge_mod._MIN_ABSTRACT_CHARS  # floor binds
    assert judge_mod._abstract_budget(20) < judge_mod._abstract_budget(5)


def test_prompt_stays_within_char_budget_at_typical_top_k():
    long_abstract = "x" * 5000
    results = [
        {"title": "T" * 80, "abstract": long_abstract, "zotero_key": f"K{i}"}
        for i in range(25)
    ]
    prompt = judge_mod._build_prompt("a query", results)
    assert len(prompt) < judge_mod._PROMPT_CHAR_BUDGET * 1.3


# --------------------------------------------------------------------------
# CLI wiring + JSON contract
# --------------------------------------------------------------------------

@pytest.fixture
def cli_db(monkeypatch, embeddings_db, zotero_db, make_vector_fn):
    """A search-ready DB whose model calls are faked."""
    from tests.conftest import store_vector

    for i, key in enumerate(["AAAA0001", "AAAA0002", "AAAA0004"]):
        store_vector(embeddings_db, key, make_vector_fn(i))

    class _FakeModel:
        def encode(self, texts, **kw):
            return np.stack([make_vector_fn(0) for _ in texts])

    monkeypatch.setattr(embedder, "_get_model", lambda: _FakeModel())
    return embeddings_db, zotero_db


def _run(cli_db, *extra):
    embeddings_db, zotero_db = cli_db
    return CliRunner().invoke(
        app,
        ["search", "--query", "networks", "--format", "json",
         "--db-path", str(embeddings_db), "--zotero-db", str(zotero_db), *extra],
    )


def test_json_contract_without_judge_is_unchanged(cli_db):
    result = _run(cli_db)
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    assert set(payload) == {"query", "results"}, "top-level JSON shape is a skill contract"
    for r in payload["results"]:
        assert set(r) == SKILL_RESULT_KEYS


def test_json_with_judge_is_additive_only(cli_db, monkeypatch):
    baseline = {r["zotero_key"]: r for r in json.loads(_run(cli_db).stdout)["results"]}

    monkeypatch.setattr(
        judge_mod.urllib.request, "urlopen",
        _fake_urlopen(_reply([{"index": i, "score": 5, "reason": "r"} for i in range(3)])),
    )
    payload = json.loads(_run(cli_db, "--judge").stdout)

    assert set(payload) == {"query", "results", "judge"}
    assert payload["judge"] == {"applied": True, "model": "gemma4:26b"}
    for r in payload["results"]:
        assert set(r) == SKILL_RESULT_KEYS | {"judge_score", "judge_reason"}
        for key, value in baseline[r["zotero_key"]].items():
            assert r[key] == value, f"--judge changed the legacy field {key}"


def test_search_degrades_gracefully_when_ollama_is_down(cli_db, monkeypatch):
    def _refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(judge_mod.urllib.request, "urlopen", _refuse)
    result = _run(cli_db, "--judge")

    assert result.exit_code == 0, "a missing judge must not fail the search"
    payload = json.loads(result.stdout)
    assert payload["judge"]["applied"] is False
    assert payload["results"], "unjudged results are still returned"
    assert "judge_score" not in payload["results"][0]
