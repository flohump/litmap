"""Rerank search results with a local ollama model.

Embedding similarity answers "is this text about the same things as the query".
It cannot tell whether a paper actually *bears on the claim* -- a paper arguing
the opposite of the query embeds close to it. A judge reads title and abstract
and scores relevance directly, which is why retrieval here is a cascade: embed
wide, then judge the shortlist.

Opt-in (`litmap search --judge`) and degrades to unjudged results if ollama is
not running. Uses only the standard library, so litmap gains no dependency.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = "gemma4:26b"

# Context budget for the whole prompt, in characters (~4 chars/token, 8192 ctx).
_PROMPT_CHAR_BUDGET = 24000
_MIN_ABSTRACT_CHARS = 200
_MAX_ABSTRACT_CHARS = 1500
_MAX_REASON_CHARS = 200
_TIMEOUT_S = 300.0

_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "score", "reason"],
            },
        }
    },
    "required": ["scores"],
}

_SYSTEM = (
    "You rank scientific papers by how well they support or bear on a specific "
    "query. Judge relevance to the query's claim, not topical overlap: a paper "
    "that merely shares vocabulary is not relevant, and a paper that contradicts "
    "the claim IS relevant to it. Score 0-10. Be strict; most papers are not a "
    "strong match. Return one entry per paper, using the given index."
)


class JudgeError(RuntimeError):
    """The judge could not run. Callers should degrade to unjudged results."""


def _abstract_budget(n_results: int) -> int:
    """Chars of abstract per result, so the whole prompt fits the context window."""
    if n_results <= 0:
        return _MAX_ABSTRACT_CHARS
    per_result = _PROMPT_CHAR_BUDGET // n_results
    return max(_MIN_ABSTRACT_CHARS, min(_MAX_ABSTRACT_CHARS, per_result))


def _build_prompt(query: str, results: list[dict]) -> str:
    budget = _abstract_budget(len(results))
    lines = [f"Query: {query}", "", "Papers:"]
    for i, r in enumerate(results):
        abstract = (r.get("abstract") or "").strip()[:budget]
        lines.append(f"[{i}] {r.get('title') or '(untitled)'}")
        if abstract:
            lines.append(f"    {abstract}")
    lines.append("")
    lines.append(f"Score every paper from 0 to 10. Use each index exactly once (0-{len(results) - 1}).")
    return "\n".join(lines)


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        if e.code == 404:
            raise JudgeError(
                f"judge model not found on the ollama server. Run: ollama pull <model>. ({body})"
            ) from e
        raise JudgeError(f"ollama returned HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise JudgeError(
            f"ollama is not reachable at {url.rsplit('/api/', 1)[0]} "
            f"- is `ollama serve` running? ({e.reason})"
        ) from e
    except TimeoutError as e:
        raise JudgeError(f"ollama timed out after {timeout:.0f}s") from e


def _parse_scores(reply: dict, n_results: int) -> dict[int, tuple[int, str]]:
    content = (reply.get("message") or {}).get("content", "")
    try:
        parsed = json.loads(content)
        entries = parsed["scores"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise JudgeError(f"could not parse the judge's reply: {e}") from e

    scores: dict[int, tuple[int, str]] = {}
    for entry in entries:
        try:
            idx = int(entry["index"])
            score = int(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < n_results:
            reason = str(entry.get("reason", ""))[:_MAX_REASON_CHARS]
            scores[idx] = (max(0, min(10, score)), reason)
    if not scores:
        raise JudgeError("the judge returned no usable scores")
    return scores


def judge_results(
    query: str,
    results: list[dict],
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = _TIMEOUT_S,
) -> list[dict]:
    """Return results with judge_score/judge_reason added, best first.

    Existing keys are never modified, so downstream JSON consumers keep working.
    Papers the judge skipped sort last rather than disappearing.
    """
    if not results:
        return []

    reply = _post(
        f"{base_url.rstrip('/')}/api/chat",
        {
            "model": model,
            "stream": False,
            "format": _SCHEMA,
            "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _build_prompt(query, results)},
            ],
        },
        timeout,
    )
    scores = _parse_scores(reply, len(results))

    judged = []
    for i, r in enumerate(results):
        score, reason = scores.get(i, (0, "not scored by the judge"))
        judged.append({**r, "judge_score": score, "judge_reason": reason})

    judged.sort(key=lambda r: (-r["judge_score"], -r.get("similarity", 0.0)))
    return judged
