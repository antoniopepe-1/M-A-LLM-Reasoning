"""
14b_rebuild_pointintime_corpus.py
=================================

Canonical, validity-repaired corpus builder. Replaces the 14 -> 15 -> 23
chain for the *final* corpus. Directly addresses audit Findings 2, 3 and 4, and
removes the outcome-leakage surface for Finding 1 (the acquirer profile itself is
built leakage-free in lib.profile at prompt time).

What it fixes
-------------
Finding 2 (Condition B not anonymous)
    Every candidate description is passed through lib.entities.redact_identity(),
    which masks legal names, aliases, d/b/a operands, tickers and URLs. A residual
    -identity flag is recorded per candidate; deals whose GOLD target still leaks
    are dropped or sent to review (pre-specified rule below).

Finding 3 (descriptions not point-in-time)
    For each candidate we take the LAST HuggingFace 10-K Item 1 filed STRICTLY
    BEFORE the deal's announced_date. Candidates with no pre-announcement filing
    fall back to a redacted CIQ description ONLY IF that description survives the
    post-deal sentence filter; otherwise they carry "No description available."
    Post-deal language ("was acquired by", "reverse merger", "SPAC", dated
    "as of <date>, ...") is stripped by the redactor as defence in depth.

Finding 4 (internally inconsistent corpus)
    Long and wide representations are built from ONE canonical per-deal entity
    table. Candidate tickers are deduplicated WITH replacement so every deal ends
    with exactly 50 unique entities, IDs 1..50 without gaps, and exactly one gold
    target. lib.corpus_io.assert_invariants() gates the output — the script
    refuses to write a corpus that violates any invariant.

Determinism
    Candidate order and any replacement draws are seeded by md5(deal_id), so the
    corpus is reproducible.

Inputs
    data/beautifulsoup/combined_fixed_final.csv   gold deals (acquirer text, dates)
    data/processed/deal_sample_clean.csv          CIQ target universe (pool + desc)
    data/hf_merge/hf_raw.parquet                  dated 10-K Item 1 per company

Outputs
    data/candidate_corpus/corpus_wide_final.json   (validated wide corpus)
    data/candidate_corpus/corpus_per_deal.csv      (long, 1 row per candidate)
    data/candidate_corpus/rebuild_report.csv       (per-deal build diagnostics)
    data/candidate_corpus/redaction_audit.csv      (per-candidate residual flags)
"""

from __future__ import annotations

import re
import json
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import entities as E
from lib import corpus_io as CIO

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_FILE = BASE_DIR / "data" / "beautifulsoup" / "combined_fixed_final.csv"
CIQ_FILE  = BASE_DIR / "data" / "processed" / "deal_sample_clean.csv"
HF_FILE   = BASE_DIR / "data" / "hf_merge" / "hf_raw.parquet"
OUT_DIR   = BASE_DIR / "data" / "candidate_corpus"

CORPUS_OUT   = OUT_DIR / "corpus_wide_final.json"
LONG_OUT     = OUT_DIR / "corpus_per_deal.csv"
REPORT_OUT   = OUT_DIR / "rebuild_report.csv"
REDACT_OUT   = OUT_DIR / "redaction_audit.csv"

# Deals excluded during manual finalization (wrong 10-K / SPAC / unresolvable),
# carried over from the original script 23 so the final sample stays at 286.
EXCLUDED_DEALS = {
    "SPTRD1880825", "SPTRD1939794", "SPTRD231102", "SPTRD1970157",
    "SPTRD1962598", "SPTRD1810033", "SPTRD1890129", "SPTRD1804516",
    "SPTRD1966655", "SPTRD1882956", "SPTRD1994676",
}

# ── Design constants (unchanged from the original design) ──────────────────
N_CANDIDATES = 50
N_SAME_SEC   = 22
N_ADJ_SEC    = 18
N_OTHER_SEC  = 9
SIZE_BUCKETS = [0, 100, 500, 2000, 10000, float("inf")]
SIZE_LABELS  = ["<$100M", "$100-500M", "$500M-2B", "$2-10B", ">$10B"]
DESC_CHARS   = 250

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Sector helpers (unchanged from 14) ─────────────────────────────────────
def sic_to_industry(sic_str):
    try:
        m = re.search(r"\((\d+)\)", str(sic_str))
        s = int(m.group(1)) if m else int(float(str(sic_str).strip()))
    except Exception:
        return "Unknown"
    if s == 9995: return "SPAC"
    if (7370 <= s <= 7379 or 3570 <= s <= 3579 or 3670 <= s <= 3679 or 3810 <= s <= 3829): return "Technology"
    if (2830 <= s <= 2836 or 8000 <= s <= 8099 or 3840 <= s <= 3859 or 8700 <= s <= 8734): return "Healthcare"
    if (1000 <= s <= 1499 or 1500 <= s <= 1799 or 2000 <= s <= 3569 or 3580 <= s <= 3809 or 3900 <= s <= 3999 or 4000 <= s <= 4799): return "Industrials"
    if (5000 <= s <= 5999 or 7000 <= s <= 7099 or 7200 <= s <= 7299 or 7800 <= s <= 7999): return "Consumer"
    if 4800 <= s <= 4899: return "Telecom"
    if 4900 <= s <= 4999: return "Utilities"
    if 6500 <= s <= 6799: return "Real Estate"
    if (6000 <= s <= 6499 or 6800 <= s <= 6999): return "Finance"
    if (7100 <= s <= 7199 or 7300 <= s <= 7799 or 8735 <= s <= 8999): return "Services"
    return "Other"

ADJACENT_SECTORS = {
    "Technology": ["Healthcare", "Services", "Industrials"],
    "Healthcare": ["Technology", "Services", "Consumer"],
    "Industrials": ["Technology", "Consumer", "Utilities"],
    "Consumer": ["Healthcare", "Industrials", "Services"],
    "Telecom": ["Technology", "Services", "Consumer"],
    "Utilities": ["Industrials", "Real Estate", "Finance"],
    "Real Estate": ["Finance", "Utilities", "Consumer"],
    "Finance": ["Real Estate", "Services", "Technology"],
    "Services": ["Technology", "Healthcare", "Finance"],
    "Other": ["Industrials", "Consumer", "Services"],
}

def get_size_bucket(mktcap):
    try:
        v = float(mktcap)
        for i in range(len(SIZE_BUCKETS) - 1):
            if SIZE_BUCKETS[i] <= v < SIZE_BUCKETS[i + 1]:
                return SIZE_LABELS[i]
    except Exception:
        pass
    return "Unknown"

def deal_seed(deal_id):
    return int(hashlib.md5(str(deal_id).encode()).hexdigest(), 16) % (2**32)

def _norm_join(s):
    """Normalized company key for joining CIQ names to HF `company`."""
    s = str(s).lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\b(inc|corp|ltd|llc|plc|co|the|group|holdings)\b", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ── Point-in-time description index ────────────────────────────────────────
def build_pit_index(hf: pd.DataFrame) -> dict:
    """
    {normalized_company -> [(date, item_1), ...] sorted ascending by date}.
    Enables selecting the last filing strictly before a given announced date.
    """
    hf = hf[["company", "date", "item_1"]].dropna(subset=["company", "date"]).copy()
    hf["date"] = pd.to_datetime(hf["date"], errors="coerce")
    hf = hf.dropna(subset=["date"])
    hf["key"] = hf["company"].apply(_norm_join)
    idx: dict[str, list] = {}
    for key, grp in hf.sort_values("date").groupby("key"):
        idx[key] = list(zip(grp["date"].tolist(), grp["item_1"].tolist()))
    return idx

def pit_description(name: str, announced, pit_idx: dict) -> str | None:
    """Last HF Item 1 strictly before `announced`, or None."""
    key = _norm_join(name)
    filings = pit_idx.get(key)
    if not filings or pd.isna(announced):
        return None
    chosen = None
    for date, item in filings:
        if date < announced:
            chosen = item
        else:
            break
    return chosen


# ── Candidate pool (target universe) ───────────────────────────────────────
def build_pool(ciq: pd.DataFrame) -> pd.DataFrame:
    pool = ciq[[
        "target_name", "target_ticker", "target_sic",
        "target_mktcap_usd_m", "target_revenue_ciq_000", "target_business_desc",
    ]].copy()
    pool.columns = ["company_name", "ticker", "sic", "mktcap_usd_m", "revenue_000", "ciq_desc"]
    pool = pool.drop_duplicates(subset=["ticker"])
    pool = pool[pool["ticker"].notna()].copy()
    pool["industry"]      = pool["sic"].apply(sic_to_industry)
    pool["size_bucket"]   = pool["mktcap_usd_m"].apply(get_size_bucket)
    pool = pool[pool["industry"] != "SPAC"].copy()
    pool["mktcap_bucket"]  = pool["mktcap_usd_m"].apply(get_size_bucket)
    pool["revenue_bucket"] = pool["revenue_000"].apply(
        lambda x: get_size_bucket(float(x) / 1000) if pd.notna(x) else "Unknown")
    return pool.reset_index(drop=True)


def select_distractors(target_ticker, target_industry, target_size, pool, rng):
    available = pool[pool["ticker"] != target_ticker].copy()
    size_idx = SIZE_LABELS.index(target_size) if target_size in SIZE_LABELS else 2
    valid = {SIZE_LABELS[i] for i in range(max(0, size_idx - 1), min(len(SIZE_LABELS), size_idx + 2))}
    valid.add(target_size)
    size_ok = available[available["size_bucket"].isin(valid)]

    def sample_from(df, n, used):
        df = df[~df["ticker"].isin(used)]
        n = min(n, len(df))
        return df.sample(n=n, random_state=rng) if n > 0 else pd.DataFrame()

    selected, used = pd.DataFrame(), {target_ticker}
    same = size_ok[size_ok["industry"] == target_industry]
    s1 = sample_from(same, N_SAME_SEC, used); selected = pd.concat([selected, s1]); used.update(s1.get("ticker", []))
    adj = ADJACENT_SECTORS.get(target_industry, [])
    s2 = sample_from(size_ok[size_ok["industry"].isin(adj)], N_ADJ_SEC, used); selected = pd.concat([selected, s2]); used.update(s2.get("ticker", []))
    s3 = sample_from(size_ok[~size_ok["industry"].isin([target_industry] + adj)], N_OTHER_SEC, used); selected = pd.concat([selected, s3]); used.update(s3.get("ticker", []))
    deficit = (N_CANDIDATES - 1) - len(selected)
    if deficit > 0:
        s4 = sample_from(available, deficit, used); selected = pd.concat([selected, s4]); used.update(s4.get("ticker", []))
    return selected.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def redacted_description(row, announced, pit_idx, deal_year=None) -> tuple[str, bool, str]:
    """
    Build a point-in-time, de-identified description for one candidate.
    Returns (description_text, residual_identity_found, source_tag).
    """
    name = str(row["company_name"])
    ticker = row.get("ticker")
    aliases = E.name_aliases(name)

    pit = pit_description(name, announced, pit_idx)
    if pit and len(str(pit)) > 60:
        raw, source = str(pit), "hf_pit"
    else:
        raw, source = str(row.get("ciq_desc") or ""), "ciq_fallback"

    # Drop any sentence dated after the deal year (defence against post-deal
    # CIQ text, e.g. "incorporated in 2020" for a 2019 deal).
    red, residual = E.redact_identity(raw, name, aliases, ticker, max_year=deal_year)
    red = red[:DESC_CHARS].strip()
    if len(red) < 40:
        red, source = "No description available.", (source + "_empty")
        residual = False
    return red, residual, source


def build_corpus(gold, pool, pit_idx):
    long_rows, wide_rows, report_rows, redact_rows = [], [], [], []

    for idx, deal in gold.iterrows():
        deal_id = deal["deal_id"]
        if deal_id in EXCLUDED_DEALS:
            report_rows.append({"deal_id": deal_id, "status": "excluded_manual", "n_candidates": 0})
            continue
        rng = np.random.default_rng(deal_seed(deal_id))
        announced = pd.to_datetime(deal.get("announced_date"), errors="coerce")

        target_ticker = deal["target_ticker"]
        target_industry = deal["industry"]
        target_size = deal["target_size_bucket"]

        # target entity row (from pool or gold)
        trow = pool[pool["ticker"] == target_ticker]
        if len(trow):
            target_df = trow.copy()
        else:
            target_df = pd.DataFrame([{
                "company_name": deal["target_name"], "ticker": target_ticker,
                "industry": target_industry, "size_bucket": target_size,
                "mktcap_bucket": get_size_bucket(deal.get("target_mktcap_usd_m")),
                "revenue_bucket": get_size_bucket(float(deal.get("target_revenue_ciq_000", 0) or 0) / 1000),
                "ciq_desc": deal.get("target_business_desc"),
            }])

        distractors = select_distractors(target_ticker, target_industry, target_size, pool, rng)
        all_cand = pd.concat([target_df, distractors], ignore_index=True)
        all_cand = all_cand.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

        # top up to exactly 50 unique entities (replacement draws, seeded)
        if len(all_cand) < N_CANDIDATES:
            used = set(all_cand["ticker"])
            extra = pool[~pool["ticker"].isin(used)]
            need = N_CANDIDATES - len(all_cand)
            if len(extra) >= need:
                topup = extra.sample(n=need, random_state=rng)
                all_cand = pd.concat([all_cand, topup], ignore_index=True)
        all_cand = all_cand.drop_duplicates(subset=["ticker"]).head(N_CANDIDATES).reset_index(drop=True)

        if len(all_cand) != N_CANDIDATES:
            report_rows.append({"deal_id": deal_id, "status": "dropped_insufficient_candidates",
                                "n_candidates": len(all_cand)})
            continue

        # deterministic shuffle, assign 1..50
        all_cand = all_cand.sample(frac=1, random_state=deal_seed(deal_id)).reset_index(drop=True)
        all_cand["candidate_id"] = [f"{i+1:02d}" for i in range(len(all_cand))]
        target_position = int(all_cand.index[all_cand["ticker"] == target_ticker][0]) + 1

        # per-candidate point-in-time, redacted description
        descs, residuals, sources = [], [], []
        deal_year_int = int(deal["deal_year"])
        for _, cand in all_cand.iterrows():
            d, res, src = redacted_description(cand, announced, pit_idx, deal_year=deal_year_int)
            descs.append(d); residuals.append(res); sources.append(src)
            redact_rows.append({
                "deal_id": deal_id, "candidate_id": cand["candidate_id"],
                "candidate_ticker": cand["ticker"], "is_target": cand["ticker"] == target_ticker,
                "source": src, "residual_identity": res,
            })
        all_cand["desc"] = descs
        all_cand["residual"] = residuals
        all_cand["pit_verified"] = [s == "hf_pit" for s in sources]

        # LEAKAGE GATE (audit Findings 2 & 3): any candidate whose description
        # still carries residual identity after redaction has its description
        # blanked. This is strict — a residual flag is a conservative detector,
        # so blanking guarantees Condition B never ranks on a leaked name. The
        # gold target is protected a fortiori.
        leak_mask = all_cand["residual"].astype(bool)
        gold_mask = all_cand["ticker"] == target_ticker
        gold_leaks = bool((leak_mask & gold_mask).any())
        all_cand.loc[leak_mask, "desc"] = "No description available."

        def line(r):
            return (f"Candidate #{r['candidate_id']} | "
                    f"Industry: {r.get('industry','Unknown')} | "
                    f"Market Cap: {r.get('mktcap_bucket','Unknown')} | "
                    f"Revenue: {r.get('revenue_bucket','Unknown')} | "
                    f"Description: {r['desc']}")
        all_cand["anon_description"] = all_cand.apply(line, axis=1)

        for _, c in all_cand.iterrows():
            long_rows.append({
                "deal_id": deal_id, "acquirer_ticker": deal["acquirer_ticker"],
                "acquirer_name": deal["acquirer_name"], "target_ticker": target_ticker,
                "target_name": deal["target_name"], "deal_year": deal["deal_year"],
                "candidate_id": c["candidate_id"], "candidate_ticker": c["ticker"],
                "candidate_name": c["company_name"], "candidate_industry": c.get("industry", "Unknown"),
                "candidate_size": c.get("size_bucket", "Unknown"),
                "is_target": c["ticker"] == target_ticker, "anon_description": c["anon_description"],
            })

        wide_rows.append({
            "deal_id": deal_id, "acquirer_ticker": deal["acquirer_ticker"],
            "acquirer_name": deal["acquirer_name"], "target_ticker": target_ticker,
            "target_name": deal["target_name"], "deal_year": int(deal["deal_year"]),
            "industry": deal["industry"], "target_position": target_position,
            "n_same_sector": int((all_cand["industry"] == deal["industry"]).sum()),
            "candidate_list": "\n".join(all_cand["anon_description"].tolist()),
            "candidate_tickers": all_cand["ticker"].tolist(),
            "candidate_pit_verified": all_cand["pit_verified"].astype(bool).tolist(),
            "target_pit_verified": bool(all_cand.loc[gold_mask, "pit_verified"].iloc[0]),
            "item1_text": deal.get("item1_text", ""), "mda_text": deal.get("mda_text", ""),
            "acquirer_revenue_m": deal.get("acquirer_revenue_m"),
            "acquirer_ebitda_margin_pct": deal.get("acquirer_ebitda_margin_pct"),
            # NOTE: target_mktcap_usd_m and deal_value_usd_m are intentionally NOT
            # carried into the corpus used for prompting (audit Finding 1). They
            # remain available in the gold table for outcome-side analysis only.
            "announced_date": str(deal.get("announced_date")),
        })

        report_rows.append({
            "deal_id": deal_id, "status": "ok", "n_candidates": len(all_cand),
            "target_position": target_position,
            "n_pit_desc": sources.count("hf_pit"),
            "n_ciq_fallback": sum(1 for s in sources if s.startswith("ciq")),
            "n_no_desc": sum(1 for s in sources if s.endswith("_empty")),
            "n_residual_identity": int(all_cand["residual"].sum()),
            "gold_leaked_pre_gate": gold_leaks,
        })

        if (idx + 1) % 50 == 0:
            log.info("processed %d deals", idx + 1)

    return long_rows, wide_rows, report_rows, redact_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("loading inputs")
    gold = pd.read_csv(GOLD_FILE)
    gold["industry"] = gold["target_sic"].apply(sic_to_industry)
    gold["target_size_bucket"] = gold["target_mktcap_usd_m"].apply(get_size_bucket)
    ciq = pd.read_csv(CIQ_FILE)
    pool = build_pool(ciq)
    log.info("pool: %d candidate entities", len(pool))

    log.info("building point-in-time HF index (this reads the parquet once)")
    hf = pd.read_parquet(HF_FILE, columns=["company", "date", "item_1"])
    pit_idx = build_pit_index(hf)
    log.info("pit index: %d companies", len(pit_idx))

    long_rows, wide_rows, report_rows, redact_rows = build_corpus(gold, pool, pit_idx)

    # HARD invariant gate BEFORE writing (audit Finding 4)
    log.info("asserting corpus invariants on %d deals", len(wide_rows))
    CIO.assert_invariants(wide_rows)
    log.info("invariants PASSED")

    # write wide (drop the helper candidate_tickers list is kept — used by audit)
    CORPUS_OUT.write_text(json.dumps(wide_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(long_rows).to_csv(LONG_OUT, index=False)
    pd.DataFrame(report_rows).to_csv(REPORT_OUT, index=False)
    pd.DataFrame(redact_rows).to_csv(REDACT_OUT, index=False)

    rep = pd.DataFrame(report_rows)
    ok = rep[rep["status"] == "ok"]
    log.info("wrote %s (%d deals)", CORPUS_OUT.name, len(wide_rows))
    print("\n=== REBUILD SUMMARY ===")
    print(f"deals written           : {len(wide_rows)}")
    print(f"point-in-time descs      : {int(ok['n_pit_desc'].sum())} candidates")
    print(f"ciq fallback descs       : {int(ok['n_ciq_fallback'].sum())} candidates")
    print(f"no-description candidates : {int(ok['n_no_desc'].sum())}")
    print(f"residual-identity flags   : {int(ok['n_residual_identity'].sum())} "
          f"({100*ok['n_residual_identity'].sum()/(len(ok)*N_CANDIDATES):.2f}% of candidates)")
    print(f"gold targets blanked      : {int(ok['gold_leaked_pre_gate'].sum())}")


if __name__ == "__main__":
    main()
