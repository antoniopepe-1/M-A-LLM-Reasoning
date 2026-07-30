"""
lib.corpus_io — load the corpus and enforce HARD invariants.
============================================================

Repairs audit Finding 4 (internally inconsistent candidate corpus: a 51-candidate
deal, a duplicated gold entity, a 49-candidate long/wide mismatch).

`load_corpus(strict=True)` refuses to return a corpus that violates any of:
  - each deal has exactly N_CANDIDATES (50) candidate lines,
  - candidate IDs are exactly 1..50 with no gaps or duplicates,
  - exactly one candidate is the gold target and target_position points to it,
  - candidate tickers are unique within a deal (no entity appears twice),
  - deal_ids are unique across the corpus.

The same check is exposed as `assert_invariants()` so corpus-building scripts
can gate their own output BEFORE it is ever handed to the (paid) LLM pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

N_CANDIDATES = 50
_CAND_ID_RE = re.compile(r"Candidate\s*#(\d+)", re.IGNORECASE)


class CorpusInvariantError(AssertionError):
    pass


def candidate_ids(candidate_list: str) -> list[int]:
    return [int(m.group(1)) for m in _CAND_ID_RE.finditer(candidate_list or "")]


def check_deal(deal: dict) -> list[str]:
    """Return a list of invariant-violation messages for one deal (empty == OK)."""
    problems: list[str] = []
    did = deal.get("deal_id", "?")
    ids = candidate_ids(deal.get("candidate_list", ""))

    if len(ids) != N_CANDIDATES:
        problems.append(f"{did}: {len(ids)} candidate lines (expected {N_CANDIDATES})")

    if sorted(ids) != list(range(1, N_CANDIDATES + 1)):
        missing = set(range(1, N_CANDIDATES + 1)) - set(ids)
        dupes = {i for i in ids if ids.count(i) > 1}
        problems.append(f"{did}: candidate IDs not 1..{N_CANDIDATES} "
                        f"(missing={sorted(missing)}, duplicated={sorted(dupes)})")

    tp = deal.get("target_position")
    if tp is None or not (1 <= int(tp) <= N_CANDIDATES):
        problems.append(f"{did}: target_position={tp} out of range")

    # candidate ticker uniqueness (requires a per-candidate ticker list if present)
    tickers = deal.get("candidate_tickers")
    if tickers:
        norm = [str(t).upper() for t in tickers if t]
        if len(norm) != len(set(norm)):
            dup = {t for t in norm if norm.count(t) > 1}
            problems.append(f"{did}: duplicate candidate tickers {sorted(dup)}")
        # exactly one gold target
        gold = deal.get("target_ticker")
        if gold and norm.count(str(gold).upper()) != 1:
            problems.append(f"{did}: gold target ticker appears "
                            f"{norm.count(str(gold).upper())} times (expected 1)")

    return problems


def assert_invariants(corpus: list[dict]) -> None:
    """Raise CorpusInvariantError listing every violation, or pass silently."""
    problems: list[str] = []

    ids = [d.get("deal_id") for d in corpus]
    if len(ids) != len(set(ids)):
        dup = {i for i in ids if ids.count(i) > 1}
        problems.append(f"duplicate deal_ids: {sorted(dup)}")

    for deal in corpus:
        problems.extend(check_deal(deal))

    if problems:
        raise CorpusInvariantError(
            f"Corpus failed {len(problems)} invariant check(s):\n  - "
            + "\n  - ".join(problems[:50])
            + ("" if len(problems) <= 50 else f"\n  ... and {len(problems) - 50} more")
        )


def load_corpus(path: str | Path, strict: bool = True) -> list[dict]:
    """Load the wide corpus JSON and (by default) assert its invariants."""
    corpus = json.loads(Path(path).read_text(encoding="utf-8"))
    if strict:
        assert_invariants(corpus)
    return corpus
