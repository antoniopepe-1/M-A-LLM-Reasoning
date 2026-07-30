"""
18_audit_leakage.py  (rewritten)
================================

Leakage GATE for the exact prompts that will be sent to the models. Replaces the
old audit, which ran on the wrong (297-deal) corpus, tested only acquirer
masking, and never gated the pipeline (audit: "leakage audit does not validate
the actual experiment").

What it does now
----------------
1. Loads the FINAL corpus and, for every deal, SERIALIZES the exact Condition B
   user prompt via the same code path the pipeline uses (lib.profile). The audit
   therefore inspects the literal bytes the model sees, not an approximation.

2. Runs seven leakage channels over each serialized prompt / corpus:
     - acquirer identity        (name / ticker survived masking in the profile)
     - target identity          (gold target name/ticker/alias in its own line)
     - candidate identity       (a candidate ticker in its own description)
     - deal / outcome language  ("was acquired by", "reverse merger", SPAC, ...)
     - future timestamps        (a year strictly after the deal year in a line)
     - duplicate entities       (a ticker appearing twice within a deal)
     - candidate-count invariant(exactly 50 unique IDs 1..50, one gold target)

3. Writes a per-deal report and a summary, and EXITS NON-ZERO if any hard gate
   is breached, so it can be wired before the paid run:
       python scripts/18_audit_leakage.py
       python scripts/16c_v3_experiment_pipeline.py --condition all --model all --strategy all
"""

from __future__ import annotations

import re
import sys
import json
import logging
import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import profile as _profile          # noqa: E402
from lib import entities as _entities         # noqa: E402
from lib import corpus_io as _corpus_io       # noqa: E402

CORPUS_PATH = BASE_DIR / "data" / "candidate_corpus" / "corpus_wide_final.json"
AUDIT_DIR   = BASE_DIR / "data" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

AUDIT_MAX_OUTCOME_FRAC   = 0.01   # post-deal language virtually eliminated
AUDIT_MAX_CANDIDATE_FRAC = 0.05   # residual candidate identity
N_CANDIDATES = 50


def target_line(deal: dict) -> str:
    tp = deal.get("target_position")
    for ln in (deal.get("candidate_list") or "").splitlines():
        m = re.match(r"Candidate #(\d+)", ln)
        if m and int(m.group(1)) == tp:
            return ln
    return ""


def audit_deal(deal: dict) -> dict:
    cand_lines = [ln for ln in (deal.get("candidate_list") or "").splitlines() if ln.strip()]

    # 1. acquirer identity survived masking in the profile (exact serialized text)
    profile = _profile.build_acquirer_profile(deal)
    acq_name = (deal.get("acquirer_name") or "").split("(")[0].strip()
    acq_tokens = [t for t in _entities.normalize_name(acq_name).split()
                  if t not in _entities.STOP_TOKENS and len(t) >= 5]
    prof_low = f" {profile.lower()} "
    acq_leak = any(f" {t} " in prof_low for t in acq_tokens)

    # 2. target identity in its own candidate line
    tline = target_line(deal).split("| Description:", 1)[-1]
    _, target_leak = _entities.redact_identity(
        tline, deal.get("target_name", ""),
        _entities.name_aliases(deal.get("target_name", "")), deal.get("target_ticker"))

    # 3. candidate identity (ticker) + 4. outcome language in descriptions
    cand_id_leaks = outcome_leaks = 0
    tickers = deal.get("candidate_tickers") or []
    for i, ln in enumerate(cand_lines):
        desc = ln.split("| Description:", 1)[-1]
        if _entities._POSTDEAL_RE.search(desc):
            outcome_leaks += 1
        tk = tickers[i] if i < len(tickers) else None
        if tk and re.search(r"\b" + re.escape(str(tk)) + r"\b", desc):
            cand_id_leaks += 1

    # 5. future timestamps
    fut = 0
    dyear = int(deal.get("deal_year") or 0)
    for ln in cand_lines:
        if any(int(y) > dyear for y in re.findall(r"\b(20\d{2})\b", ln)):
            fut += 1

    # 6. duplicate entities within the deal
    dup = 0
    if tickers:
        norm = [str(t).upper() for t in tickers if t]
        dup = len(norm) - len(set(norm))

    # 7. candidate-count invariant
    ids = _corpus_io.candidate_ids(deal.get("candidate_list", ""))
    count_ok = (sorted(ids) == list(range(1, N_CANDIDATES + 1)))

    return {
        "deal_id": deal.get("deal_id"),
        "n_candidate_lines": len(cand_lines),
        "acquirer_leak": bool(acq_leak),
        "target_leak": bool(target_leak),
        "candidate_id_leaks": cand_id_leaks,
        "outcome_leaks": outcome_leaks,
        "future_timestamp_leaks": fut,
        "duplicate_entities": dup,
        "count_invariant_ok": count_ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Serialized-prompt leakage gate.")
    ap.add_argument("--corpus", default=str(CORPUS_PATH))
    ap.add_argument("--no-gate", action="store_true",
                    help="Report only; do not exit non-zero on breach.")
    args = ap.parse_args()

    corpus = _corpus_io.load_corpus(args.corpus, strict=True)
    log.info("auditing %d deals (serialized Condition-B prompts)", len(corpus))

    df = pd.DataFrame([audit_deal(d) for d in corpus])
    df.to_csv(AUDIT_DIR / "leakage_report.csv", index=False)

    total = int(df["n_candidate_lines"].sum())
    summary = {
        "n_deals": len(df),
        "n_candidate_lines": total,
        "target_identity_leak_deals": int(df["target_leak"].sum()),
        "acquirer_identity_leak_deals": int(df["acquirer_leak"].sum()),
        "candidate_identity_leak_lines": int(df["candidate_id_leaks"].sum()),
        "candidate_identity_leak_frac": round(df["candidate_id_leaks"].sum() / total, 4),
        "outcome_language_lines": int(df["outcome_leaks"].sum()),
        "outcome_language_frac": round(df["outcome_leaks"].sum() / total, 4),
        "future_timestamp_lines": int(df["future_timestamp_leaks"].sum()),
        "duplicate_entity_count": int(df["duplicate_entities"].sum()),
        "count_invariant_violations": int((~df["count_invariant_ok"]).sum()),
    }
    (AUDIT_DIR / "leakage_summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== LEAKAGE AUDIT (serialized final prompts) ===")
    for k, v in summary.items():
        print(f"  {k:<34} {v}")

    breaches = []
    if summary["target_identity_leak_deals"]:   breaches.append("target identity leak")
    if summary["acquirer_identity_leak_deals"]: breaches.append("acquirer identity leak")
    if summary["duplicate_entity_count"]:       breaches.append("duplicate entities")
    if summary["count_invariant_violations"]:   breaches.append("candidate-count violation")
    if summary["outcome_language_frac"] > AUDIT_MAX_OUTCOME_FRAC:
        breaches.append(f"outcome language {summary['outcome_language_frac']} > {AUDIT_MAX_OUTCOME_FRAC}")
    if summary["candidate_identity_leak_frac"] > AUDIT_MAX_CANDIDATE_FRAC:
        breaches.append(f"candidate identity {summary['candidate_identity_leak_frac']} > {AUDIT_MAX_CANDIDATE_FRAC}")

    if breaches:
        print("\nGATE: FAILED\n  - " + "\n  - ".join(breaches))
        if not args.no_gate:
            log.error("leakage gate FAILED — aborting before any paid run")
            return 1
    else:
        print("\nGATE: PASSED — corpus is clean for the main run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
