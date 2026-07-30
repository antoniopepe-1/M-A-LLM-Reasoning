"""
32_evaluate_named_probes

Standalone evaluation for the memorization probes produced by 30_probe_named_conditions.py (Conditions A_name and B_named).

CONDITIONS EVALUATED
---------------------
  A_name   (results/probe_a_name/)   — generation-type, evaluated like Condition A. Real acquirer identity (name + ticker), business
             description withheld. Metrics: Recall@1, Recall@5 via fuzzy name matching (match_target_name).

  B_named  (results/probe_b_named/)  — ranking-type, evaluated like Condition B. Acquirer stays masked; the 50 candidates carry their real name + ticker instead of anonymous IDs. Metrics:
             MRR, Recall@{1,5,10}, NDCG@{5,10}.
"""

import re
import json
import math
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Canonical library — SAME strict parser, entity matcher and MRR@10 as 17.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import parsing as _parsing
from lib import entities as _entities
from lib import metrics as _metrics
K = _metrics.DEFAULT_K
# Named probes are collected with the same structured prompts; keep False.
USE_LEGACY_PARSER = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
EVAL_DIR    = RESULTS_DIR / "evaluation"          # main pipeline output (read-only here)
PROBE_EVAL_DIR = RESULTS_DIR / "evaluation_probes"  # this script's own output
PROBE_EVAL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROBE_CONDITION_DIRS = {
    "A_name":  "probe_a_name",
    "B_named": "probe_b_named",
}

N_BOOTSTRAP  = 1000
BOOTSTRAP_CI = 0.95
RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# Schema normalization — bridges 29_probe_named_conditions.py output to the
# field names the metric/evaluator functions below expect.
# ---------------------------------------------------------------------------

def _normalize_probe_record(rec: dict) -> dict:
    """
    Script 29 field  -> expected field
      response        -> response_text
      tokens_in       -> input_tokens
      tokens_out      -> output_tokens
      model_used      -> model_id   (proxy; requested vs served not
                                      distinguished in the probe schema)
      (missing)       -> provider = "openrouter"
      (missing)       -> finish_reason = None, latency_s = 0.0
    """
    if "response_text" in rec:
        return rec  # already canonical, pass through unchanged

    rec = dict(rec)
    rec["response_text"] = rec.get("response", "")
    rec.setdefault("model_id",      rec.get("model_used"))
    rec.setdefault("provider",      "openrouter")
    rec.setdefault("finish_reason", None)
    rec.setdefault("input_tokens",  rec.get("tokens_in", 0))
    rec.setdefault("output_tokens", rec.get("tokens_out", 0))
    rec.setdefault("latency_s",     0.0)
    return rec


# ---------------------------------------------------------------------------
# Response parsing (identical logic to 17_evaluation.py)
# ---------------------------------------------------------------------------

def parse_ranking(response_text: str) -> tuple[list[int], bool]:
    """Canonical strict ranking parser (terminal block only; audit parser finding)."""
    if USE_LEGACY_PARSER:
        return _parsing.parse_ranking_legacy(response_text)
    return _parsing.parse_ranking(response_text)


def parse_generated_names(response_text: str) -> list[str]:
    """Canonical A/A_name name parser (FINAL TARGETS sentinel / numbered list)."""
    return _parsing.parse_generated_names(response_text)


def match_target_name(predicted_names: list[str], target_name: str, k: int,
                      target_ticker: str | None = None) -> bool:
    """Conservative entity-level match (audit Finding 5; no single-token matches)."""
    return _entities.match_target_in_topk(
        predicted_names, target_name, k,
        target_ticker=target_ticker,
        target_aliases=tuple(_entities.name_aliases(target_name)),
    )


# ---------------------------------------------------------------------------
# Metrics (identical logic to 17_evaluation.py)
# ---------------------------------------------------------------------------

# Canonical MRR@10 / Recall@k / NDCG@k (audit Finding 6).
def mrr(ranked: list[int], target: int) -> float:
    return _metrics.mrr_at_k(ranked, target, K)


recall_at_k = _metrics.recall_at_k
ndcg_at_k   = _metrics.ndcg_at_k


def bootstrap_ci(
    values: list[float],
    n_resamples: int = N_BOOTSTRAP,
    ci: float = BOOTSTRAP_CI,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng   = np.random.default_rng(seed)
    arr   = np.array(values, dtype=float)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    lower = float(np.quantile(means, (1 - ci) / 2))
    upper = float(np.quantile(means, (1 + ci) / 2))
    return (round(lower, 4), round(upper, 4))


# ---------------------------------------------------------------------------
# Per-deal evaluation
# ---------------------------------------------------------------------------

def evaluate_record_b_named(rec: dict) -> dict:
    """Compute ranking metrics for a single B_named record."""
    response   = rec.get("response_text", "")
    target_pos = rec.get("target_position")
    ranked, ok = parse_ranking(response)

    return {
        "deal_id":         rec.get("deal_id"),
        "acquirer_ticker": rec.get("acquirer_ticker"),
        "acquirer_name":   rec.get("acquirer_name"),
        "target_name":     rec.get("target_name"),
        "deal_year":       rec.get("deal_year"),
        "industry":        rec.get("industry"),
        "condition":       "B_named",
        "model_key":       rec.get("model_key"),
        "model_id":        rec.get("model_id"),
        "strategy":        rec.get("strategy"),
        "input_tokens":    rec.get("input_tokens", 0),
        "output_tokens":   rec.get("output_tokens", 0),
        "parse_success":   ok,
        "n_parsed":        len(ranked),
        "target_position": target_pos,
        "target_rank":     (ranked.index(target_pos) + 1)
                           if (ok and target_pos and target_pos in ranked) else None,
        "mrr":             mrr(ranked, target_pos),
        "recall_1":        recall_at_k(ranked, target_pos, 1),
        "recall_5":        recall_at_k(ranked, target_pos, 5),
        "recall_10":       recall_at_k(ranked, target_pos, 10),
        "ndcg_5":          ndcg_at_k(ranked, target_pos, 5),
        "ndcg_10":         ndcg_at_k(ranked, target_pos, 10),
    }


def evaluate_record_a_name(rec: dict) -> dict:
    """Compute generation metrics for a single A_name record."""
    response    = rec.get("response_text", "")
    target_name = rec.get("target_name", "")
    names       = parse_generated_names(response) if response else []
    ok          = len(names) > 0

    return {
        "deal_id":         rec.get("deal_id"),
        "acquirer_ticker": rec.get("acquirer_ticker"),
        "acquirer_name":   rec.get("acquirer_name"),
        "target_name":     target_name,
        "deal_year":       rec.get("deal_year"),
        "industry":        rec.get("industry"),
        "condition":       "A_name",
        "model_key":       rec.get("model_key"),
        "model_id":        rec.get("model_id"),
        "strategy":        rec.get("strategy"),
        "input_tokens":    rec.get("input_tokens", 0),
        "output_tokens":   rec.get("output_tokens", 0),
        "parse_success":   ok,
        "n_parsed":        len(names),
        "recall_1":        float(match_target_name(names, target_name, 1, rec.get("target_ticker"))),
        "recall_5":        float(match_target_name(names, target_name, 5, rec.get("target_ticker"))),
    }


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_probe_jsonl(jsonl_path: Path, condition: str) -> pd.DataFrame:
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(_normalize_probe_record(json.loads(line)))
            except json.JSONDecodeError:
                pass

    if not records:
        log.warning("Empty file: %s", jsonl_path)
        return pd.DataFrame()

    evaluator = evaluate_record_b_named if condition == "B_named" else evaluate_record_a_name
    df = pd.DataFrame([evaluator(r) for r in records])

    if len(df) < 286:
        log.warning(
            "Incomplete run: %d/286 deals found in %s",
            len(df), jsonl_path.relative_to(RESULTS_DIR),
        )

    fail_rate = 1.0 - df["parse_success"].mean()
    if fail_rate > 0.05:
        log.warning(
            "High parse failure rate %.1f%% in %s — manual inspection recommended",
            fail_rate * 100, jsonl_path.relative_to(RESULTS_DIR),
        )

    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

GROUP_COLS = ["condition", "model_key", "strategy"]


def aggregate_b_named(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(GROUP_COLS):
        mrr_ci      = bootstrap_ci(grp["mrr"].tolist())
        recall10_ci = bootstrap_ci(grp["recall_10"].tolist())

        row = dict(zip(GROUP_COLS, keys))
        row.update({
            "n_deals":           len(grp),
            "parse_success_pct": round(grp["parse_success"].mean(), 4),
            "mrr":               round(grp["mrr"].mean(), 4),
            "mrr_ci_lo":         mrr_ci[0],
            "mrr_ci_hi":         mrr_ci[1],
            "recall_1":          round(grp["recall_1"].mean(), 4),
            "recall_5":          round(grp["recall_5"].mean(), 4),
            "recall_10":         round(grp["recall_10"].mean(), 4),
            "recall_10_ci_lo":   recall10_ci[0],
            "recall_10_ci_hi":   recall10_ci[1],
            "ndcg_5":            round(grp["ndcg_5"].mean(), 4),
            "ndcg_10":           round(grp["ndcg_10"].mean(), 4),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP_COLS).reset_index(drop=True)


def aggregate_a_name(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(GROUP_COLS):
        recall1_ci = bootstrap_ci(grp["recall_1"].tolist())
        recall5_ci = bootstrap_ci(grp["recall_5"].tolist())

        row = dict(zip(GROUP_COLS, keys))
        row.update({
            "n_deals":           len(grp),
            "parse_success_pct": round(grp["parse_success"].mean(), 4),
            "recall_1":          round(grp["recall_1"].mean(), 4),
            "recall_1_ci_lo":    recall1_ci[0],
            "recall_1_ci_hi":    recall1_ci[1],
            "recall_5":          round(grp["recall_5"].mean(), 4),
            "recall_5_ci_lo":    recall5_ci[0],
            "recall_5_ci_hi":    recall5_ci[1],
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP_COLS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deltas vs. the primary A / B baselines (read from 17_evaluation.py output)
# ---------------------------------------------------------------------------

def compute_delta_a_name(df_agg_a_name: pd.DataFrame) -> pd.DataFrame | None:
    baseline_path = EVAL_DIR / "metrics_agg_A.csv"
    if not baseline_path.exists():
        log.warning(
            "Baseline not found: %s — run 17_evaluation.py --condition A first. "
            "Skipping delta(A_name - A).", baseline_path,
        )
        return None

    df_a = pd.read_csv(baseline_path)
    df_a = df_a[df_a["condition"] == "A"][["model_key", "strategy", "recall_1", "recall_5"]]
    df_a = df_a.rename(columns={"recall_1": "recall1_A", "recall_5": "recall5_A"})

    df_an = df_agg_a_name[["model_key", "strategy", "recall_1", "recall_5"]].rename(
        columns={"recall_1": "recall1_Aname", "recall_5": "recall5_Aname"}
    )

    delta = df_an.merge(df_a, on=["model_key", "strategy"], how="inner")
    if delta.empty:
        log.warning(
            "No matching (model_key, strategy) pairs between A_name and A baseline "
            "— check that model_key values are consistent across the two runs."
        )
        return None

    delta["delta_recall_1"] = (delta["recall1_Aname"] - delta["recall1_A"]).round(4)
    delta["delta_recall_5"] = (delta["recall5_Aname"] - delta["recall5_A"]).round(4)
    return delta.sort_values(["model_key", "strategy"]).reset_index(drop=True)


def compute_delta_b_named(df_agg_b_named: pd.DataFrame) -> pd.DataFrame | None:
    baseline_path = EVAL_DIR / "metrics_agg_BC.csv"
    if not baseline_path.exists():
        log.warning(
            "Baseline not found: %s — run 17_evaluation.py --condition B first. "
            "Skipping delta(B_named - B).", baseline_path,
        )
        return None

    df_b = pd.read_csv(baseline_path)
    df_b = df_b[df_b["condition"] == "B"][["model_key", "strategy", "mrr", "recall_10", "ndcg_10"]]
    df_b = df_b.rename(columns={"mrr": "mrr_B", "recall_10": "recall10_B", "ndcg_10": "ndcg10_B"})

    df_bn = df_agg_b_named[["model_key", "strategy", "mrr", "recall_10", "ndcg_10"]].rename(
        columns={"mrr": "mrr_Bnamed", "recall_10": "recall10_Bnamed", "ndcg_10": "ndcg10_Bnamed"}
    )

    delta = df_bn.merge(df_b, on=["model_key", "strategy"], how="inner")
    if delta.empty:
        log.warning(
            "No matching (model_key, strategy) pairs between B_named and B baseline "
            "— check that model_key values are consistent across the two runs."
        )
        return None

    delta["delta_mrr"]       = (delta["mrr_Bnamed"]      - delta["mrr_B"]).round(4)
    delta["delta_recall_10"] = (delta["recall10_Bnamed"] - delta["recall10_B"]).round(4)
    delta["delta_ndcg_10"]   = (delta["ndcg10_Bnamed"]   - delta["ndcg10_B"]).round(4)
    return delta.sort_values(["model_key", "strategy"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate memorization probes (29_probe_named_conditions.py output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--condition", choices=["A_name", "B_named", "all"], default="all",
        help="Which probe condition(s) to evaluate.",
    )
    parser.add_argument(
        "--model", default="all",
        help="Filter by model slug substring (e.g. 'deepseek'). Default: all.",
    )
    args = parser.parse_args()

    target_conditions = (
        list(PROBE_CONDITION_DIRS.keys()) if args.condition == "all" else [args.condition]
    )

    for condition in target_conditions:
        cond_dir  = PROBE_CONDITION_DIRS[condition]
        cond_path = RESULTS_DIR / cond_dir

        if not cond_path.exists():
            log.warning("Directory not found, skipping: %s", cond_path)
            continue

        frames = []
        for jsonl_path in sorted(cond_path.rglob("responses_raw.jsonl")):
            parts      = jsonl_path.parts
            model_slug = parts[-3] if len(parts) >= 3 else ""
            if args.model != "all" and args.model not in model_slug:
                continue
            log.info("Evaluating: %s", jsonl_path.relative_to(RESULTS_DIR))
            df = load_probe_jsonl(jsonl_path, condition)
            if not df.empty:
                frames.append(df)

        if not frames:
            log.warning("No data found for condition %s. Nothing written.", condition)
            continue

        df_raw = pd.concat(frames, ignore_index=True)
        raw_path = PROBE_EVAL_DIR / f"metrics_raw_{condition}.csv"
        df_raw.to_csv(raw_path, index=False)
        log.info("Raw metrics written: %s (%d rows)", raw_path, len(df_raw))

        if condition == "B_named":
            df_agg = aggregate_b_named(df_raw)
        else:
            df_agg = aggregate_a_name(df_raw)

        agg_path = PROBE_EVAL_DIR / f"metrics_agg_{condition}.csv"
        df_agg.to_csv(agg_path, index=False)
        log.info("Aggregated metrics written: %s (%d rows)", agg_path, len(df_agg))

        print("\n" + "=" * 70)
        print(f"RESULTS — Condition {condition}")
        print("=" * 70)
        print(df_agg.to_string(index=False))

        # Delta vs. primary A / B baseline (requires 17_evaluation.py output)
        if condition == "A_name":
            delta = compute_delta_a_name(df_agg)
            if delta is not None:
                delta_path = PROBE_EVAL_DIR / "delta_Aname_minus_A.csv"
                delta.to_csv(delta_path, index=False)
                print("\n" + "-" * 70)
                print("Delta(A_name - A) — acquirer-identity memorization probe")
                print("-" * 70)
                print(delta[["model_key", "strategy", "recall1_A", "recall1_Aname",
                              "delta_recall_1"]].to_string(index=False))

        elif condition == "B_named":
            delta = compute_delta_b_named(df_agg)
            if delta is not None:
                delta_path = PROBE_EVAL_DIR / "delta_Bnamed_minus_B.csv"
                delta.to_csv(delta_path, index=False)
                print("\n" + "-" * 70)
                print("Delta(B_named - B) — candidate-name memorization probe")
                print("-" * 70)
                print(delta[["model_key", "strategy", "mrr_B", "mrr_Bnamed",
                              "delta_mrr"]].to_string(index=False))

    print(f"\nAll probe outputs written to: {PROBE_EVAL_DIR}")


if __name__ == "__main__":
    main()