"""
Evaluation Pipeline
=========================================

Computes ranking metrics on JSONL outputs produced by 16c_v3_experiment_pipeline.py.

MODEL ROSTER - v4 final (6 models, corpus_wide_final.json - 286 deals)
-----------------------------------------------------------------------
  deepseek_v3   : DeepSeek V3     (685B MoE, 37B active)   — top anchor
  llama32_1b    : Llama 3.2 1B    (1B dense)                — lower bound
  llama33_70b   : Llama 3.3 70B   (70B dense)               — cross-family Meta
  phi4_14b      : Phi-4           (14B dense)                — Microsoft family
  qwen3_235b    : Qwen3-235B      (235B MoE, 22B active)    — large Qwen
  qwen3_8b      : Qwen3-8B        (8B dense)                — small Qwen

  Note: the slugs match the subdirectories under results/{open_ended,
  retrieval_aug, retrieval_aug_metadata}/ - the script discovers the models
  from the filesystem.

METRICS
-------
Conditions B and C (retrieval-augmented ranking, single relevant item per deal):

  MRR (Mean Reciprocal Rank)
    Primary metric. For each deal, computes 1/rank if the ground-truth target
    appears in the model's top-10 ranking, 0 otherwise. Averaged across deals.
    Directly answers: "How highly does the model prioritize the correct target?"
    With a single relevant item, MRR is the most natural ranking metric.

  Recall@k  (k = 1, 5, 10)
    Binary: 1 if the ground-truth target appears in the model's top-k, 0 otherwise.
    Averaged across deals. Answers: "Does the model include the target in a
    shortlist of size k?" Recall@1 = precision of the top recommendation.
    Note: with a single relevant item, Precision@k = Recall@k / k — Precision@k
    is therefore not reported separately as it adds no information.

  NDCG@k  (k = 5, 10)
    Normalized Discounted Cumulative Gain with binary relevance. Penalizes targets
    found at lower ranks within the top-k window using a log2 discount.
    With binary relevance and a single item, NDCG@k and MRR are highly correlated
    but NDCG@k is bounded to the top-k window — reported for alignment with the
    learning-to-rank literature.

  Random baseline
    Theoretical MRR and Recall@k expected under a random ranking, computed
    analytically from the empirical distribution of target positions in the corpus.
    Essential reference point: a model with MRR below the random baseline performs
    worse than chance.

  Bootstrap confidence intervals (95%, n=1000 resamples)
    Per-cell (model x strategy x condition) bootstrap CI on MRR and Recall@10.
    Required for significance claims in the paper (e.g., "Model A significantly
    outperforms Model B").

Condition A (unconstrained generation):
  Recall@k  (k = 1, 5)
    Whether the ground-truth target name appears in the model's top-k generated
    targets. Matching uses normalized fuzzy matching (see match_target_name).

MODERATOR ANALYSES
------------------
  By year    : MRR per deal_year — tests for training cutoff contamination.
               Models with cutoff close to 2021-2022 may show inflated MRR for
               recent deals due to memorized outcomes rather than genuine reasoning.
  By sector  : MRR per industry sector — tests for domain-specific reasoning ability.
  Delta B-C  : Per-model difference in MRR between Conditions B and C, quantifying
               the contribution of textual recognition to ranking performance.

PARSING
-------
  parse_ranking() uses a multi-pattern approach to extract candidate numbers from
  model responses, covering observed format variations across the five model families.
  Parsing failures (empty ranked list despite non-empty response) are counted and
  reported. A parse_failure_rate > 5% for any cell warrants manual inspection.

OUTPUT
------
  results/evaluation/
    metrics_raw_B.csv            per-deal metrics, Conditions B and C
    metrics_raw_A.csv            per-deal metrics, Condition A
    metrics_agg_B.csv            aggregated by model x strategy x condition
    metrics_agg_A.csv            aggregated by model x strategy
    summary_table.csv            paper-ready pivot table (main results)
    moderator_by_year.csv        MRR by deal year (training cutoff analysis)
    moderator_by_sector.csv      MRR by industry sector
    delta_B_minus_C.csv          textual recognition contribution per model

"""

import re
import json
import math
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Canonical library — the SAME parser, matcher, MRR@10 and chance baseline used
# by every other script (audit Findings 5, 6 and the paired-inference finding).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import parsing as _parsing
from lib import entities as _entities
from lib import metrics as _metrics

# Reporting window: MRR@10 for LLM AND baselines (audit Finding 6).
K = _metrics.DEFAULT_K
N_CANDIDATES = _metrics.N_CANDIDATES

# Whether historical (pre-structured-output) responses are being re-scored.
# When True, evaluation uses the legacy cross-response parser purely to salvage
# old runs; new runs use the strict terminal-block parser.
USE_LEGACY_PARSER = False


def _target_entity_dict():
    """Build the target entity dictionary once (names/aliases/tickers) for A."""
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "beautifulsoup" / "combined_fixed_final.csv"
    try:
        gold = pd.read_csv(corpus_path)
        rows = gold[["target_name", "target_ticker"]].to_dict("records")
        return _entities.build_entity_dict(rows)
    except Exception:
        return {}


_ENTITY_DICT = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
EVAL_DIR    = RESULTS_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Condition-to-directory mapping (mirrors 16c_experiment_pipeline.py)
CONDITION_DIRS = {
    "A": "open_ended",
    "B": "retrieval_aug",
    "C": "retrieval_aug_metadata",
}

# Bootstrap parameters
N_BOOTSTRAP   = 1000
BOOTSTRAP_CI  = 0.95
RANDOM_SEED   = 42


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_ranking(response_text: str) -> tuple[list[int], bool]:
    """
    Parse the ranked candidate IDs. Delegates to the canonical strict parser
    (terminal FINAL RANKING / JSON block only), so reasoning mentions are never
    mistaken for the ranking (audit parser finding). When re-scoring historical
    responses (USE_LEGACY_PARSER), falls back to the old cross-response parser.
    """
    if USE_LEGACY_PARSER:
        return _parsing.parse_ranking_legacy(response_text)
    return _parsing.parse_ranking(response_text)


def parse_generated_names(response_text: str) -> list[str]:
    """Canonical Condition-A name parser (FINAL TARGETS sentinel / numbered list)."""
    return _parsing.parse_generated_names(response_text)


def match_target_name(predicted_names: list[str], target_name: str, k: int,
                      target_ticker: str | None = None) -> bool:
    """
    Conservative entity-level match (audit Finding 5). Resolves each prediction
    against the target entity via normalized name, aliases and ticker with a HIGH
    fuzzy threshold — never on a single shared token.
    """
    global _ENTITY_DICT
    if _ENTITY_DICT is None:
        _ENTITY_DICT = _target_entity_dict()
    rec = _ENTITY_DICT.get(_entities.normalize_name(target_name), {})
    aliases = tuple(rec.get("aliases", set()))
    ticker = target_ticker or rec.get("ticker")
    return _entities.match_target_in_topk(predicted_names, target_name, k,
                                          target_ticker=ticker, target_aliases=aliases)


# ---------------------------------------------------------------------------
# Metrics — Conditions B and C
# ---------------------------------------------------------------------------

# Metric definitions are the canonical library ones so LLM and every baseline
# use identical MRR@10 / Recall@k / NDCG@k (audit Finding 6).
def mrr(ranked: list[int], target: int) -> float:
    """MRR@10: reciprocal rank only if target is in the top-K window, else 0."""
    return _metrics.mrr_at_k(ranked, target, K)


recall_at_k = _metrics.recall_at_k
ndcg_at_k   = _metrics.ndcg_at_k


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def compute_random_baseline(target_positions: list[int], n_candidates: int = 50) -> dict:
    """
    Chance expectation for the SAME @K window the models are scored on.

    Because LLMs (and truncated baselines) are scored with MRR@K, the correct
    random comparator is E[MRR@K] = H_K / n, NOT H_n / n (audit Finding 6). For
    K=10, n=50 this is ~0.0586, not ~0.090. Empirical simulation confirms it by
    placing the single relevant item uniformly over 1..n and truncating at K.
    """
    if not target_positions:
        return {}

    n = n_candidates
    base = _metrics.chance_baseline(K, n)

    # Empirical confirmation with the @K truncation applied.
    rng = np.random.default_rng(RANDOM_SEED)
    ranks = rng.integers(1, n + 1, size=N_BOOTSTRAP)
    empirical_mrr = float(np.mean([1.0 / r if r <= K else 0.0 for r in ranks]))

    return {
        "k":                      K,
        "random_mrr_at_k":        base["chance_mrr_at_k"],   # H_K / n ~ 0.0586
        "random_mrr_theoretical": base["chance_mrr_at_k"],   # kept for downstream readers
        "random_mrr_empirical":   round(empirical_mrr, 4),
        "random_recall_1":        base["chance_recall_1"],
        "random_recall_5":        base["chance_recall_5"],
        "random_recall_10":       base["chance_recall_10"],
        "n_candidates":           n,
        "n_deals":                len(target_positions),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_resamples: int = N_BOOTSTRAP,
    ci: float = BOOTSTRAP_CI,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for the mean.

    Resamples with replacement n_resamples times, computes the mean of each
    resample, and returns the (1-ci)/2 and (1+ci)/2 quantiles.
    Standard approach for IR metrics in NLP papers (Sakai 2014, SIGIR).
    """
    if not values:
        return (0.0, 0.0)
    rng    = np.random.default_rng(seed)
    arr    = np.array(values, dtype=float)
    means  = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    lower  = float(np.quantile(means, (1 - ci) / 2))
    upper  = float(np.quantile(means, (1 + ci) / 2))
    return (round(lower, 4), round(upper, 4))


# ---------------------------------------------------------------------------
# Per-deal evaluation
# ---------------------------------------------------------------------------

def evaluate_record_bc(rec: dict) -> dict:
    """Compute all B/C metrics for a single JSONL record."""
    response    = rec.get("response_text", "")
    target_pos  = rec.get("target_position")
    ranked, ok  = parse_ranking(response)

    return {
        # Identifiers
        "deal_id":         rec.get("deal_id"),
        "acquirer_ticker": rec.get("acquirer_ticker"),
        "acquirer_name":   rec.get("acquirer_name"),
        "target_name":     rec.get("target_name"),
        "deal_year":       rec.get("deal_year"),
        "industry":        rec.get("industry"),
        # Experimental metadata
        "condition":       rec.get("condition"),
        "model_key":       rec.get("model_key"),
        "model_id":        rec.get("model_id"),
        "model_used":      rec.get("model_used"),
        "provider":        rec.get("provider"),
        "strategy":        rec.get("strategy"),
        "finish_reason":   rec.get("finish_reason"),
        # Token usage
        "input_tokens":    rec.get("input_tokens", 0),
        "output_tokens":   rec.get("output_tokens", 0),
        "latency_s":       rec.get("latency_s", 0.0),
        # Parsing diagnostics
        "parse_success":   ok,
        "n_parsed":        len(ranked),
        "truncated":       rec.get("finish_reason") == "length",
        # Ground truth
        "target_position": target_pos,
        "target_rank":     (ranked.index(target_pos) + 1)
                           if (ok and target_pos and target_pos in ranked) else None,
        # Ranking metrics
        "mrr":             mrr(ranked, target_pos),
        "recall_1":        recall_at_k(ranked, target_pos, 1),
        "recall_5":        recall_at_k(ranked, target_pos, 5),
        "recall_10":       recall_at_k(ranked, target_pos, 10),
        "ndcg_5":          ndcg_at_k(ranked, target_pos, 5),
        "ndcg_10":         ndcg_at_k(ranked, target_pos, 10),
    }


def evaluate_record_a(rec: dict) -> dict:
    """Compute all Condition A metrics for a single JSONL record."""
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
        "condition":       rec.get("condition"),
        "model_key":       rec.get("model_key"),
        "model_id":        rec.get("model_id"),
        "model_used":      rec.get("model_used"),
        "provider":        rec.get("provider"),
        "strategy":        rec.get("strategy"),
        "finish_reason":   rec.get("finish_reason"),
        "input_tokens":    rec.get("input_tokens", 0),
        "output_tokens":   rec.get("output_tokens", 0),
        "latency_s":       rec.get("latency_s", 0.0),
        "parse_success":   ok,
        "n_parsed":        len(names),
        "truncated":       rec.get("finish_reason") == "length",
        "recall_1":        float(match_target_name(names, target_name, 1, rec.get("target_ticker"))),
        "recall_5":        float(match_target_name(names, target_name, 5, rec.get("target_ticker"))),
    }


# ---------------------------------------------------------------------------
# JSONL loading and evaluation
# ---------------------------------------------------------------------------

def load_and_evaluate(jsonl_path: Path, condition: str) -> pd.DataFrame:
    """Load a JSONL file and return a per-deal metrics DataFrame."""
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not records:
        log.warning("Empty file: %s", jsonl_path)
        return pd.DataFrame()

    if condition in ("B", "C"):
        rows = [evaluate_record_bc(r) for r in records]
    else:
        rows = [evaluate_record_a(r) for r in records]

    df = pd.DataFrame(rows)

    # Data integrity check: warn if n_deals < 286 (corpus_wide_final.json)
    if len(df) < 286:
        log.warning(
            "Incomplete run: %d/286 deals found in %s — "
            "metrics will be computed on partial data",
            len(df), jsonl_path.relative_to(RESULTS_DIR),
        )

    # Parsing diagnostics
    if condition in ("B", "C"):
        fail_rate = 1.0 - df["parse_success"].mean()
        trunc_rate = df["truncated"].mean()
        if fail_rate > 0.05:
            log.warning(
                "High parse failure rate %.1f%% in %s — "
                "manual inspection of response_text recommended",
                fail_rate * 100, jsonl_path.relative_to(RESULTS_DIR),
            )
        if trunc_rate > 0.05:
            log.warning(
                "High truncation rate %.1f%% (finish_reason=length) in %s — "
                "consider increasing MAX_TOKENS for CoT strategy",
                trunc_rate * 100, jsonl_path.relative_to(RESULTS_DIR),
            )

    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

GROUP_COLS = ["condition", "model_key", "model_id", "provider", "strategy"]


def aggregate_bc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-deal metrics to model x strategy x condition level.
    Includes bootstrap 95% CI on MRR and Recall@10.
    """
    rows = []
    for keys, grp in df.groupby(GROUP_COLS):
        mrr_vals      = grp["mrr"].tolist()
        recall10_vals = grp["recall_10"].tolist()
        mrr_ci        = bootstrap_ci(mrr_vals)
        recall10_ci   = bootstrap_ci(recall10_vals)

        row = dict(zip(GROUP_COLS, keys))
        row.update({
            "n_deals":           len(grp),
            "parse_success_pct": round(grp["parse_success"].mean(), 4),
            "truncation_pct":    round(grp["truncated"].mean(), 4),
            # Primary metric
            "mrr":               round(grp["mrr"].mean(), 4),
            "mrr_ci_lo":         mrr_ci[0],
            "mrr_ci_hi":         mrr_ci[1],
            # Recall
            "recall_1":          round(grp["recall_1"].mean(), 4),
            "recall_5":          round(grp["recall_5"].mean(), 4),
            "recall_10":         round(grp["recall_10"].mean(), 4),
            "recall_10_ci_lo":   recall10_ci[0],
            "recall_10_ci_hi":   recall10_ci[1],
            # NDCG
            "ndcg_5":            round(grp["ndcg_5"].mean(), 4),
            "ndcg_10":           round(grp["ndcg_10"].mean(), 4),
            # Rank distribution
            "rank_mean":         round(grp["target_rank"].mean(), 2),
            "rank_median":       round(grp["target_rank"].median(), 1),
            # Token efficiency
            "avg_input_tokens":  round(grp["input_tokens"].mean(), 0),
            "avg_output_tokens": round(grp["output_tokens"].mean(), 0),
            "avg_latency_s":     round(grp["latency_s"].mean(), 2),
        })
        rows.append(row)

    return pd.DataFrame(rows).sort_values(GROUP_COLS).reset_index(drop=True)


def aggregate_a(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Condition A per-deal metrics."""
    rows = []
    for keys, grp in df.groupby(GROUP_COLS):
        recall1_ci = bootstrap_ci(grp["recall_1"].tolist())
        recall5_ci = bootstrap_ci(grp["recall_5"].tolist())

        row = dict(zip(GROUP_COLS, keys))
        row.update({
            "n_deals":           len(grp),
            "parse_success_pct": round(grp["parse_success"].mean(), 4),
            "truncation_pct":    round(grp["truncated"].mean(), 4),
            "recall_1":          round(grp["recall_1"].mean(), 4),
            "recall_1_ci_lo":    recall1_ci[0],
            "recall_1_ci_hi":    recall1_ci[1],
            "recall_5":          round(grp["recall_5"].mean(), 4),
            "recall_5_ci_lo":    recall5_ci[0],
            "recall_5_ci_hi":    recall5_ci[1],
            "avg_input_tokens":  round(grp["input_tokens"].mean(), 0),
            "avg_output_tokens": round(grp["output_tokens"].mean(), 0),
            "avg_latency_s":     round(grp["latency_s"].mean(), 2),
        })
        rows.append(row)

    return pd.DataFrame(rows).sort_values(GROUP_COLS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Moderator analyses
# ---------------------------------------------------------------------------

def moderator_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    MRR by deal year, aggregated across all models and strategies.

    Primary use: detect training cutoff contamination. If MRR is systematically
    higher for deals close to the model's training cutoff (2021-2022), this
    suggests memory retrieval rather than genuine strategic reasoning, even
    under anonymization. This analysis should be reported per model in §4.
    """
    cols = ["condition", "model_key", "strategy", "deal_year"]
    return (
        df.groupby(cols)
        .agg(
            n_deals   = ("deal_id",  "count"),
            mrr       = ("mrr",      "mean"),
            recall_10 = ("recall_10","mean"),
        )
        .round(4)
        .reset_index()
        .sort_values(cols)
    )


def moderator_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    MRR by industry sector, aggregated across all models and strategies.

    Tests whether LLM strategic reasoning is domain-dependent. Technology and
    Healthcare sectors (comprising ~53% of deals) are expected to show higher
    performance if models have richer training data on tech M&A transactions.
    """
    cols = ["condition", "model_key", "strategy", "industry"]
    return (
        df.groupby(cols)
        .agg(
            n_deals   = ("deal_id",  "count"),
            mrr       = ("mrr",      "mean"),
            recall_10 = ("recall_10","mean"),
        )
        .round(4)
        .reset_index()
        .sort_values(cols)
    )


def compute_delta_bc(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Delta(B - C) with PAIRED, ACQUIRER-CLUSTERED inference (audit finding:
    "paired inference is missing").

    Rather than subtracting two aggregate means, this pairs Condition B and C on
    the SAME deal, forms the per-deal difference in MRR@10, and bootstraps that
    difference clustering by acquirer so repeated acquirers do not inflate
    significance. Reports the mean deal-level difference, its 95% CI and a
    bootstrap p-value for H0: mean difference = 0.
    """
    rows = []
    for (model_key, strategy), grp in df_raw.groupby(["model_key", "strategy"]):
        b = grp[grp["condition"] == "B"][["deal_id", "acquirer_ticker", "mrr", "recall_10", "ndcg_10"]]
        c = grp[grp["condition"] == "C"][["deal_id", "mrr", "recall_10", "ndcg_10"]]
        merged = b.merge(c, on="deal_id", suffixes=("_B", "_C"))
        if merged.empty:
            continue
        diff = (merged["mrr_B"] - merged["mrr_C"]).to_numpy()
        clusters = merged["acquirer_ticker"].fillna(merged["deal_id"]).to_numpy()
        boot = _metrics.paired_bootstrap(diff, clusters=clusters)
        rows.append({
            "model_key": model_key, "strategy": strategy,
            "n_pairs": boot["n"], "n_acquirers": boot.get("n_clusters"),
            "mrr_B": round(merged["mrr_B"].mean(), 4),
            "mrr_C": round(merged["mrr_C"].mean(), 4),
            "delta_mrr": boot["mean_diff"],
            "delta_mrr_ci_lo": boot["ci_lo"], "delta_mrr_ci_hi": boot["ci_hi"],
            "delta_mrr_p": boot["p_value"],
            "delta_recall_10": round((merged["recall_10_B"] - merged["recall_10_C"]).mean(), 4),
            "delta_ndcg_10": round((merged["ndcg_10_B"] - merged["ndcg_10_C"]).mean(), 4),
        })
    return pd.DataFrame(rows).sort_values(["model_key", "strategy"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paper-ready summary table
# ---------------------------------------------------------------------------

def build_summary_table(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Main results table for the paper.

    Pivot: rows = model x condition, columns = strategy.
    Values: MRR (primary), Recall@10, NDCG@10.
    Format matches Table 2 in the paper draft.
    """
    if df_agg.empty:
        return pd.DataFrame()

    pivot = df_agg.pivot_table(
        index   = ["condition", "model_key", "provider"],
        columns = "strategy",
        values  = ["mrr", "recall_10", "ndcg_10", "mrr_ci_lo", "mrr_ci_hi"],
        aggfunc = "first",
    )
    pivot.columns = ["_".join(str(c) for c in col) for col in pivot.columns]
    return pivot.reset_index().round(4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-as-M&A-Analyst experiment outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--condition",
        choices=["A", "B", "C", "BC", "all"],
        default="all",
        help="Which condition(s) to evaluate. 'BC' evaluates both retrieval conditions.",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Filter by model key (e.g., 'deepseek'). Default: all models.",
    )
    parser.add_argument(
        "--strategy",
        default="all",
        help="Filter by strategy (zeroshot, cot, fewshot). Default: all.",
    )
    args = parser.parse_args()

    # Resolve which conditions to evaluate
    if args.condition == "all":
        target_conditions = ["A", "B", "C"]
    elif args.condition == "BC":
        target_conditions = ["B", "C"]
    else:
        target_conditions = [args.condition]

    rows_bc: list[pd.DataFrame] = []
    rows_a:  list[pd.DataFrame] = []

    # Scan all available JSONL files
    for condition, cond_dir in CONDITION_DIRS.items():
        if condition not in target_conditions:
            continue

        cond_path = RESULTS_DIR / cond_dir
        if not cond_path.exists():
            log.info("Directory not found, skipping: %s", cond_path)
            continue

        for jsonl_path in sorted(cond_path.rglob("responses_raw.jsonl")):
            # Extract model slug and strategy from path structure:
            # results/{cond_dir}/{model_slug}/{strategy}/responses_raw.jsonl
            parts      = jsonl_path.parts
            model_slug = parts[-3] if len(parts) >= 3 else ""
            strategy   = parts[-2] if len(parts) >= 2 else ""

            if args.model    != "all" and args.model    not in model_slug:
                continue
            if args.strategy != "all" and args.strategy != strategy:
                continue

            log.info(
                "Evaluating: %s", jsonl_path.relative_to(RESULTS_DIR)
            )
            df = load_and_evaluate(jsonl_path, condition)

            if df.empty:
                continue

            if condition in ("B", "C"):
                rows_bc.append(df)
            else:
                rows_a.append(df)

    # ── Conditions B and C ────────────────────────────────────────────────────
    if rows_bc:
        df_bc_raw = pd.concat(rows_bc, ignore_index=True)
        df_bc_raw.to_csv(EVAL_DIR / "metrics_raw_BC.csv", index=False)
        log.info("Raw B/C: %d deal-level rows written", len(df_bc_raw))

        # Aggregate with bootstrap CIs
        df_bc_agg = aggregate_bc(df_bc_raw)
        df_bc_agg.to_csv(EVAL_DIR / "metrics_agg_BC.csv", index=False)
        log.info("Aggregated B/C: %d rows", len(df_bc_agg))

        # Random baseline
        target_positions = df_bc_raw["target_position"].dropna().astype(int).tolist()
        baseline = compute_random_baseline(target_positions)
        pd.DataFrame([baseline]).to_csv(EVAL_DIR / "random_baseline.csv", index=False)
        log.info(
            "Random baseline — MRR: %.4f | Recall@10: %.4f",
            baseline["random_mrr_theoretical"],
            baseline["random_recall_10"],
        )

        # Moderator analyses
        moderator_by_year(df_bc_raw).to_csv(
            EVAL_DIR / "moderator_by_year.csv", index=False
        )
        moderator_by_sector(df_bc_raw).to_csv(
            EVAL_DIR / "moderator_by_sector.csv", index=False
        )

        # Delta B-C (textual recognition contribution) — paired, acquirer-clustered
        if {"B", "C"}.issubset(set(df_bc_raw["condition"].unique())):
            delta = compute_delta_bc(df_bc_raw)
            delta.to_csv(EVAL_DIR / "delta_B_minus_C.csv", index=False)
            log.info("Delta B-C (paired/clustered) written: %d rows", len(delta))

        # Paper-ready summary table
        summary = build_summary_table(df_bc_agg)
        summary.to_csv(EVAL_DIR / "summary_table.csv", index=False)

        # Console output
        print("\n" + "=" * 70)
        print("RESULTS — CONDITIONS B and C (Retrieval-Augmented Ranking)")
        print("=" * 70)
        display_cols = [
            "condition", "model_key", "strategy", "n_deals",
            "mrr", "mrr_ci_lo", "mrr_ci_hi",
            "recall_1", "recall_10", "ndcg_10",
            "parse_success_pct", "truncation_pct",
        ]
        available = [c for c in display_cols if c in df_bc_agg.columns]
        print(df_bc_agg[available].to_string(index=False))

        if baseline:
            print(
                f"\nRandom baseline — "
                f"MRR: {baseline['random_mrr_theoretical']:.4f} | "
                f"Recall@1: {baseline['random_recall_1']:.4f} | "
                f"Recall@10: {baseline['random_recall_10']:.4f}"
            )

    # ── Condition A ───────────────────────────────────────────────────────────
    if rows_a:
        df_a_raw = pd.concat(rows_a, ignore_index=True)
        df_a_raw.to_csv(EVAL_DIR / "metrics_raw_A.csv", index=False)
        log.info("Raw A: %d deal-level rows written", len(df_a_raw))

        df_a_agg = aggregate_a(df_a_raw)
        df_a_agg.to_csv(EVAL_DIR / "metrics_agg_A.csv", index=False)

        print("\n" + "=" * 70)
        print("RESULTS — CONDITION A (Unconstrained Generation)")
        print("=" * 70)
        display_cols_a = [
            "model_key", "strategy", "n_deals",
            "recall_1", "recall_1_ci_lo", "recall_1_ci_hi",
            "recall_5", "recall_5_ci_lo", "recall_5_ci_hi",
            "parse_success_pct",
        ]
        available_a = [c for c in display_cols_a if c in df_a_agg.columns]
        print(df_a_agg[available_a].to_string(index=False))

    if not rows_bc and not rows_a:
        log.warning(
            "No JSONL files found under %s. "
            "Run 16c_experiment_pipeline.py first.",
            RESULTS_DIR,
        )
        return

    print(f"\nAll outputs written to: {EVAL_DIR}")


if __name__ == "__main__":
    main()