"""
lib.parsing — strict, single ranking parser shared by every consumer.
=====================================================================

Repairs the "parser searches candidate mentions across the whole response, so a
reasoning mention becomes the inferred ranking" finding.

Contract for the corrected prompts
-----------------------------------
Models are instructed to end their answer with a sentinel block:

    FINAL RANKING: [45, 12, 7, 33, 2, 19, 8, 41, 3, 27]

or a JSON object:

    {"ranking": [45, 12, 7, 33, 2, 19, 8, 41, 3, 27]}

`parse_ranking()` parses ONLY that terminal block. It does NOT scan reasoning
text for "Candidate #NN" mentions. If both a JSON block and a `FINAL RANKING:`
line are present, the LAST one in the response wins (models sometimes restate).

A legacy fallback (`parse_ranking_legacy`) reproduces the old cross-response
behaviour and is used only to RE-SCORE historical responses that predate the
structured-output prompts, clearly labelled as such.
"""

from __future__ import annotations

import re
import json

MAX_RANK = 10
N_CANDIDATES = 50


# ---------------------------------------------------------------------------
# Strict parser (for corrected structured-output prompts)
# ---------------------------------------------------------------------------

_JSON_RANKING_RE = re.compile(r"\{[^{}]*\"ranking\"\s*:\s*\[[^\]]*\][^{}]*\}", re.IGNORECASE | re.DOTALL)
_SENTINEL_RE     = re.compile(r"FINAL\s+RANKING\s*[:=]\s*\[([^\]]*)\]", re.IGNORECASE)
_INT_RE          = re.compile(r"\d{1,2}")


def _clean(nums: list[int]) -> list[int]:
    out, seen = [], set()
    for n in nums:
        if 1 <= n <= N_CANDIDATES and n not in seen:
            out.append(n)
            seen.add(n)
        if len(out) >= MAX_RANK:
            break
    return out


def parse_ranking(response_text: str) -> tuple[list[int], bool]:
    """
    Parse ONLY the terminal ranking block. Returns (ranked_ids, parse_success).

    Precedence: last JSON {"ranking": [...]} block, else last FINAL RANKING line.
    """
    if not response_text or not response_text.strip():
        return [], False

    # Prefer the LAST JSON ranking object in the response.
    json_matches = list(_JSON_RANKING_RE.finditer(response_text))
    if json_matches:
        blob = json_matches[-1].group(0)
        try:
            obj = json.loads(blob)
            ids = [int(x) for x in obj.get("ranking", []) if isinstance(x, (int, float, str))
                   and str(x).strip().lstrip("-").isdigit()]
            ranked = _clean(ids)
            if ranked:
                return ranked, True
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Else the LAST FINAL RANKING: [...] sentinel line.
    sentinels = list(_SENTINEL_RE.finditer(response_text))
    if sentinels:
        inner = sentinels[-1].group(1)
        ids = [int(m.group(0)) for m in _INT_RE.finditer(inner)]
        ranked = _clean(ids)
        if ranked:
            return ranked, True

    # Safe middle tier (v2 audit Findings B & C): an EXPLICIT ordered
    # "Rank N: #M" / "Rank N: Candidate #M" list. This is a genuine ranking
    # block, NOT a scan of reasoning prose, so it does not reintroduce the
    # audit's cross-response contamination:
    #   * every id must be anchored to an explicit "Rank k" token on its line,
    #   * ids are ordered by the model's own rank number k (not document order),
    #   * the match is line-anchored (each Rank/id pair comes from a single line
    #     via a non-greedy, newline-free gap), so a prose sentence like
    #     "rank Candidate #5 higher than #9" spanning the response cannot be
    #     stitched into a ranking (Finding C: no whole-response `.*` scan).
    # Finding B: the id may be written "#01" (no "Candidate" word), which is the
    # scored-list format weaker models emit ("Rank 1: #01 — Score: 10/10 — ...").
    # Recovering it removes a PARSER artifact and exposes the model's true (often
    # degenerate, document-order) ranking rather than silently scoring it 0.
    rank_lines = re.findall(
        r"[Rr]ank\s*\**\s*(\d+)\**\s*[:\.\)]?\s*(?:[Cc]andidate\s*)?#(\d+)",
        response_text,
    )
    if rank_lines:
        ordered = sorted(rank_lines, key=lambda t: int(t[0]))
        ids = [int(cid) for _, cid in ordered]
        ranked = _clean(ids)
        if ranked:
            return ranked, True

    return [], False


# ---------------------------------------------------------------------------
# Legacy parser (RE-SCORING historical responses only)
# ---------------------------------------------------------------------------

_LEGACY_PATTERNS = [
    r"[Rr]ank\s*\**\s*\d+\**\s*[:\.]?\s*.*?[Cc]andidate\s*#(\d+)",
    r"[Cc]andidate\s*#(\d+)",
    r"(?<!\w)#(\d{1,2})(?!\w)",
]


def parse_ranking_legacy(response_text: str) -> tuple[list[int], bool]:
    """
    Old cross-response heuristic. Retained ONLY to re-score responses collected
    before structured outputs existed. New runs must use parse_ranking().
    """
    if not response_text or not response_text.strip():
        return [], False
    ranked, seen = [], set()
    for pattern in _LEGACY_PATTERNS:
        if len(ranked) >= MAX_RANK:
            break
        for m in re.finditer(pattern, response_text, re.IGNORECASE):
            num = int(m.group(1))
            if 1 <= num <= N_CANDIDATES and num not in seen:
                ranked.append(num)
                seen.add(num)
            if len(ranked) >= MAX_RANK:
                break
    return ranked[:MAX_RANK], len(ranked) > 0


# ---------------------------------------------------------------------------
# Condition A — generated company names
# ---------------------------------------------------------------------------

_A_JSON_RE     = re.compile(r"\{[^{}]*\"targets\"\s*:\s*\[[^\]]*\][^{}]*\}", re.IGNORECASE | re.DOTALL)
_A_SENTINEL_RE = re.compile(r"FINAL\s+TARGETS\s*[:=]\s*(\[[^\]]*\])", re.IGNORECASE)
# The name may be preceded by a bullet separator the model emits between the
# number and the company ("1. - **Acme**"). Skipping one leading dash/em-dash
# removes a PARSER artifact: without it the capture starts on a forbidden
# character and the whole line is dropped, scoring an otherwise valid response 0.
# The separator is consumed, NOT captured, so the trailing "- rationale" still
# terminates the name exactly as before.
_A_LINE_RE     = re.compile(r"^\s*\d+[.)]\s*(?:[—\-]\s*)?([^—\-:\n]+)")


def parse_generated_names(response_text: str) -> list[str]:
    """
    Extract up to 5 generated target names from a Condition-A response.

    Precedence: terminal `FINAL TARGETS: [...]` sentinel (structured prompt),
    then a JSON {"targets": [...]} block, then a numbered list (legacy). Names
    are returned raw (resolution happens in lib.entities.resolve_match).
    """
    if not response_text or not response_text.strip():
        return []

    # 1. FINAL TARGETS sentinel (last one wins)
    sm = list(_A_SENTINEL_RE.finditer(response_text))
    if sm:
        try:
            arr = json.loads(sm[-1].group(1))
            names = [str(x).strip() for x in arr if str(x).strip()]
            if names:
                return names[:5]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 2. JSON {"targets": [...]}
    m = list(_A_JSON_RE.finditer(response_text))
    if m:
        try:
            obj = json.loads(m[-1].group(0))
            names = [str(x).strip() for x in obj.get("targets", []) if str(x).strip()]
            if names:
                return names[:5]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    names = []
    for line in response_text.strip().splitlines():
        lm = _A_LINE_RE.match(line)
        if lm:
            nm = lm.group(1).strip()
            if nm and len(nm) >= 3:
                names.append(nm)
    return names[:5]
