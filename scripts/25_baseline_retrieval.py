"""
25_baseline_retrieval.py
=========================

Implements BM25 and TF-IDF baselines on the same candidate ranking task performed by LLMs in Conditions B and C. Results are directly comparable
to 17_evaluation.py output (same metrics, same corpus, same deal sample).

RATIONALE
---------
Without a non-LLM baseline, it is impossible to quantify the marginal contribution of LLMs over classical information retrieval methods.
A reviewer of Management Science or ICAIF will almost certainly request this comparison. BM25 is the standard IR baseline; TF-IDF provides
a simpler comparison point.

DESIGN
------
Query  : anonymized acquirer profile (item1_text + mda_text, truncated)
         Same profile construction as build_acquirer_profile() in pipeline.

Corpus : 50 candidates per deal extracted from candidate_list field.

Condition B : query vs. full candidate text (Industry + Market Cap + Description)
Condition C : query vs. metadata-only (Industry + Market Cap, no Description)
              Mirrors strip_descriptions() in pipeline.

METRICS
-------
Same as 17_evaluation.py for Conditions B/C:
  MRR, Recall@1, Recall@5, Recall@10, NDCG@5, NDCG@10
  Bootstrap CI (95%, n=1000) on MRR and Recall@10.
  Random baseline included for comparison.

OUTPUT
------
  results/evaluation/
    baseline_metrics_raw.csv     per-deal metrics (BM25 and TF-IDF, B and C)
    baseline_metrics_agg.csv     aggregated by method × condition
    baseline_vs_llm.csv          side-by-side: baseline vs best LLM per condition
    latex/table_baseline.tex     LaTeX table for paper

"""

import re
import html
import json
import math
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Optional dependencies ──────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("WARNING: rank-bm25 not installed. Install with: pip install rank-bm25")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Canonical library — SAME query builder and MRR@10 as the LLM (audit Findings
# 6 and 7). Baselines must not build their own divergent query or metric.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import profile as _profile
from lib import metrics as _metrics
from lib import corpus_io as _corpus_io

K = _metrics.DEFAULT_K

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "candidate_corpus" / "corpus_wide_final.json"
EVAL_DIR    = BASE_DIR / "results" / "evaluation"
LLM_AGG_BC  = EVAL_DIR / "metrics_agg_BC.csv"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
N_BOOTSTRAP   = 1000
BOOTSTRAP_CI  = 0.95
RANDOM_SEED   = 42
EXPECTED_DEALS = 286


# ── Text preprocessing ─────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Basic text cleaning: decode HTML entities, lowercase, normalize whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return preprocess(text).split()


# ── Candidate parsing ──────────────────────────────────────────────────────

def parse_candidates(candidate_list: str, strip_desc: bool = False) -> list[str]:
    """
    Parse candidate_list string into list of 50 candidate text strings.

    Args:
        candidate_list : raw candidate_list field from corpus JSON
        strip_desc     : if True, remove Description field (Condition C)

    Returns:
        List of candidate text strings (one per candidate, index 0 = Candidate #01)
    """
    candidates = []
    for line in candidate_list.splitlines():
        line = line.strip()
        if not line or not line.startswith("Candidate"):
            continue
        if strip_desc:
            desc_idx = line.find("| Description:")
            if desc_idx != -1:
                line = line[:desc_idx].rstrip()
        candidates.append(line)
    return candidates


def get_target_position(candidate_list: str, target_pos: int) -> int:
    """
    Return 0-indexed position of target in candidate list.
    target_pos in corpus is 1-indexed (Candidate #target_pos).
    Returns -1 if not found.
    """
    return target_pos - 1  # corpus stores 1-indexed position


# ── Acquirer query construction ────────────────────────────────────────────

def build_query(deal: dict) -> str:
    """
    Canonical IR query — the SAME masked, truncated fields the LLM sees (audit
    Finding 7). Uses lib.profile.build_query_text so the baseline and the LLM are
    given identical input information.
    """
    return _profile.build_query_text(deal)


# ── Ranking ────────────────────────────────────────────────────────────────

def rank_bm25(query: str, candidates: list[str]) -> list[int]:
    """
    Rank candidates using BM25Okapi.
    Returns list of candidate indices sorted by score descending (rank 1 first).
    """
    tokenized_candidates = [tokenize(c) for c in candidates]
    tokenized_query      = tokenize(query)

    # Filter empty candidates to avoid BM25 crash
    non_empty = [(i, tc) for i, tc in enumerate(tokenized_candidates) if tc]
    if not non_empty:
        return list(range(len(candidates)))

    indices, tok_cands = zip(*non_empty)
    bm25   = BM25Okapi(list(tok_cands))
    scores = bm25.get_scores(tokenized_query)

    # Rebuild full ranking with empty candidates at the bottom
    scored = [(indices[i], scores[i]) for i in range(len(indices))]
    missing = [i for i in range(len(candidates)) if i not in indices]
    scored.extend([(i, -1.0) for i in missing])

    return [i for i, _ in sorted(scored, key=lambda x: x[1], reverse=True)]


def rank_tfidf(query: str, candidates: list[str]) -> list[int]:
    """
    Rank candidates using TF-IDF cosine similarity.
    Returns list of candidate indices sorted by similarity descending.
    """
    if not candidates:
        return []

    docs = [preprocess(c) for c in candidates]
    q    = preprocess(query)

    try:
        vectorizer = TfidfVectorizer(
            min_df=1,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        corpus_vec = vectorizer.fit_transform(docs)
        query_vec  = vectorizer.transform([q])
        sims       = cosine_similarity(query_vec, corpus_vec).flatten()
    except ValueError:
        # Empty vocabulary edge case
        sims = np.zeros(len(candidates))

    return list(np.argsort(sims)[::-1])


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(ranked_indices: list[int], target_idx: int, k_values: tuple = (1, 5, 10)) -> dict:
    """
    Compute MRR@10, Recall@k, NDCG@k for a single deal using the CANONICAL
    library metrics (audit Finding 6). The full 50-candidate ranking is
    truncated to the top-K window so the baseline MRR is directly comparable to
    the LLM's MRR@10 — NOT full-list MRR.
    """
    # +1 because library metrics use 1-indexed candidate IDs; here indices are
    # 0-indexed positions, so shift both ranking and target by 1.
    ranked = [i + 1 for i in ranked_indices]
    target = target_idx + 1

    metrics = {
        "mrr":      _metrics.mrr_at_k(ranked, target, K),
        "recall_1": _metrics.recall_at_k(ranked, target, 1),
        "recall_5": _metrics.recall_at_k(ranked, target, 5),
        "recall_10": _metrics.recall_at_k(ranked, target, 10),
        "ndcg_5":   _metrics.ndcg_at_k(ranked, target, 5),
        "ndcg_10":  _metrics.ndcg_at_k(ranked, target, 10),
    }
    metrics["rank"] = _metrics.target_rank(ranked, target, K)  # None if outside top-K
    return metrics


def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP,
                 ci: float = BOOTSTRAP_CI, seed: int = RANDOM_SEED) -> tuple[float, float]:
    """95% bootstrap CI on mean."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return round(lo, 4), round(hi, 4)


def random_baseline_mrr(n_candidates: int = 50) -> float:
    """Chance MRR@10 = H_10 / n (audit Finding 6), NOT full-list H_n / n."""
    return _metrics.chance_mrr_at_k(K, n_candidates)


# ── Main evaluation ────────────────────────────────────────────────────────

def evaluate_corpus(corpus: list[dict], conditions: list[str]) -> pd.DataFrame:
    """
    Run BM25 and TF-IDF on all deals for requested conditions.
    Returns raw per-deal metrics DataFrame.
    """
    rows = []

    for deal in corpus:
        deal_id    = deal["deal_id"]
        target_pos = deal.get("target_position")
        cand_list  = deal.get("candidate_list", "")

        if target_pos is None or not cand_list:
            log.warning(f"Skipping {deal_id}: missing target_position or candidate_list")
            continue

        target_idx = get_target_position(cand_list, target_pos)
        query      = build_query(deal)

        for cond in conditions:
            strip = (cond == "C")
            candidates = parse_candidates(cand_list, strip_desc=strip)

            if not candidates:
                log.warning(f"Skipping {deal_id} cond={cond}: no candidates parsed")
                continue

            for method, rank_fn in [("BM25", rank_bm25), ("TF-IDF", rank_tfidf)]:
                if method == "BM25" and not BM25_AVAILABLE:
                    continue

                ranked = rank_fn(query, candidates)
                m      = compute_metrics(ranked, target_idx)

                rows.append({
                    "deal_id":   deal_id,
                    "condition": cond,
                    "method":    method,
                    "industry":  deal.get("industry"),
                    "deal_year": deal.get("deal_year"),
                    "n_candidates": len(candidates),
                    **m,
                })

    return pd.DataFrame(rows)


def aggregate_metrics(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-deal metrics to per-method × condition level with CI."""
    rows = []
    for (method, cond), grp in df_raw.groupby(["method", "condition"]):
        mrr_arr    = grp["mrr"].values
        r10_arr    = grp["recall_10"].values
        mrr_ci_lo, mrr_ci_hi = bootstrap_ci(mrr_arr)
        r10_ci_lo, r10_ci_hi = bootstrap_ci(r10_arr)

        rows.append({
            "method":       method,
            "condition":    cond,
            "n_deals":      len(grp),
            "mrr":          round(mrr_arr.mean(), 4),
            "mrr_ci_lo":    mrr_ci_lo,
            "mrr_ci_hi":    mrr_ci_hi,
            "recall_1":     round(grp["recall_1"].mean(), 4),
            "recall_5":     round(grp["recall_5"].mean(), 4),
            "recall_10":    round(grp["recall_10"].mean(), 4),
            "recall_10_ci_lo": r10_ci_lo,
            "recall_10_ci_hi": r10_ci_hi,
            "ndcg_5":       round(grp["ndcg_5"].mean(), 4),
            "ndcg_10":      round(grp["ndcg_10"].mean(), 4),
        })

    return pd.DataFrame(rows)


def build_comparison_table(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Build side-by-side comparison: baselines vs best LLM results.
    Reads metrics_agg_BC.csv if available.
    """
    rows = list(df_agg.to_dict("records"))

    # Add random baseline (MRR@10 chance and analytic Recall@k = k/n)
    rnd_mrr = random_baseline_mrr(50)
    chance = _metrics.chance_baseline(K, 50)
    for cond in ["B", "C"]:
        rows.append({
            "method": "Random",
            "condition": cond,
            "n_deals": EXPECTED_DEALS,
            "mrr": round(rnd_mrr, 4),
            "mrr_ci_lo": None, "mrr_ci_hi": None,
            "recall_1": chance["chance_recall_1"], "recall_5": chance["chance_recall_5"],
            "recall_10": chance["chance_recall_10"],
            "recall_10_ci_lo": None, "recall_10_ci_hi": None,
            "ndcg_5": None, "ndcg_10": None,
        })

    # Add best LLM results if available
    if LLM_AGG_BC.exists():
        llm = pd.read_csv(LLM_AGG_BC)
        best_llm = (
            llm.sort_values("mrr", ascending=False)
               .groupby(["condition", "model_key"])
               .first()
               .reset_index()
               .sort_values("mrr", ascending=False)
               .groupby("condition")
               .first()
               .reset_index()
        )
        for _, row in best_llm.iterrows():
            rows.append({
                "method": f"LLM-best ({row['model_key']})",
                "condition": row["condition"],
                "n_deals": int(row.get("n_deals", EXPECTED_DEALS)),
                "mrr": round(row["mrr"], 4),
                "mrr_ci_lo": row.get("mrr_ci_lo"),
                "mrr_ci_hi": row.get("mrr_ci_hi"),
                "recall_1":  round(row["recall_1"], 4),
                "recall_5":  round(row["recall_5"], 4),
                "recall_10": round(row["recall_10"], 4),
                "recall_10_ci_lo": row.get("recall_10_ci_lo"),
                "recall_10_ci_hi": row.get("recall_10_ci_hi"),
                "ndcg_5":    round(row["ndcg_5"], 4),
                "ndcg_10":   round(row["ndcg_10"], 4),
            })

    return pd.DataFrame(rows)


# ── LaTeX export ───────────────────────────────────────────────────────────

def export_latex(df_cmp: pd.DataFrame, out_path: Path) -> None:
    """Export comparison table as LaTeX (for paper appendix)."""
    lines = []
    lines.append("% Table: Non-LLM baselines vs LLM best performance")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Non-LLM retrieval baselines vs. best LLM performance. "
                 r"BM25 and TF-IDF rank the 50 candidates using the acquirer profile as query. "
                 r"Condition B includes candidate descriptions; Condition C metadata only. "
                 r"Random baseline: theoretical MRR under uniform ranking ($n=50$ candidates).}")
    lines.append(r"\label{tab:baseline}")
    lines.append(r"\begin{tabular}{l l r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Cond.} & \textbf{MRR} & \textbf{95\% CI} "
                 r"& \textbf{R@1} & \textbf{R@10} & \textbf{NDCG@10} \\")
    lines.append(r"\midrule")

    method_order = ["Random", "TF-IDF", "BM25"]
    # Add LLM-best rows after
    llm_methods = [m for m in df_cmp["method"].unique() if m.startswith("LLM")]
    method_order += llm_methods

    for cond in ["B", "C"]:
        sub = df_cmp[df_cmp["condition"] == cond]
        for method in method_order:
            row = sub[sub["method"] == method]
            if row.empty:
                continue
            r = row.iloc[0]

            def fmt(v):
                return f"{v:.3f}" if pd.notna(v) and v is not None else "--"

            ci = (f"[{fmt(r.get('mrr_ci_lo'))}, {fmt(r.get('mrr_ci_hi'))}]"
                  if pd.notna(r.get("mrr_ci_lo")) else "--")

            lines.append(
                f"{method} & {cond} & {fmt(r['mrr'])} & {ci} "
                f"& {fmt(r.get('recall_1'))} & {fmt(r.get('recall_10'))} "
                f"& {fmt(r.get('ndcg_10'))} \\\\"
            )
        if cond == "B":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"LaTeX table saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BM25 + TF-IDF baseline for M&A ranking")
    parser.add_argument("--condition", nargs="+", choices=["B", "C"], default=["B", "C"],
                        help="Conditions to evaluate (default: B C)")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap CI (faster, for quick checks)")
    args = parser.parse_args()

    if not BM25_AVAILABLE:
        log.warning("BM25 not available. Install rank-bm25: pip install rank-bm25 --break-system-packages")

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    log.info(f"Corpus loaded: {len(corpus)} deals")

    if len(corpus) != EXPECTED_DEALS:
        log.warning(f"Expected {EXPECTED_DEALS} deals, found {len(corpus)}")

    log.info(f"Conditions: {args.condition}")
    log.info("Running BM25 and TF-IDF ranking...")

    df_raw = evaluate_corpus(corpus, args.condition)

    # Save raw
    raw_path = EVAL_DIR / "baseline_metrics_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    log.info(f"Raw metrics saved: {raw_path} ({len(df_raw)} rows)")

    # Aggregate
    df_agg = aggregate_metrics(df_raw)
    agg_path = EVAL_DIR / "baseline_metrics_agg.csv"
    df_agg.to_csv(agg_path, index=False)
    log.info(f"Aggregated metrics saved: {agg_path}")

    # Comparison table
    df_cmp = build_comparison_table(df_agg)
    cmp_path = EVAL_DIR / "baseline_vs_llm.csv"
    df_cmp.to_csv(cmp_path, index=False)
    log.info(f"Comparison table saved: {cmp_path}")

    # LaTeX
    latex_dir = EVAL_DIR / "latex"
    latex_dir.mkdir(exist_ok=True)
    export_latex(df_cmp, latex_dir / "table_baseline.tex")

    # Print summary
    print("\n" + "=" * 65)
    print("BASELINE RESULTS vs RANDOM BASELINE")
    print("=" * 65)
    rnd = random_baseline_mrr()
    print(f"{'Method':<20} {'Cond':>5} {'MRR':>8} {'vs Random':>10} {'R@1':>8} {'R@10':>8} {'NDCG@10':>9}")
    print("-" * 65)
    for _, row in df_agg.sort_values(["condition", "mrr"], ascending=[True, False]).iterrows():
        delta = row["mrr"] - rnd
        sign  = "+" if delta >= 0 else ""
        print(f"{row['method']:<20} {row['condition']:>5} {row['mrr']:>8.3f} "
              f"{sign}{delta:>8.3f}  {row['recall_1']:>8.3f} {row['recall_10']:>8.3f} "
              f"{row['ndcg_10']:>9.3f}")
    print("=" * 65)
    print(f"Random baseline MRR: {rnd:.3f}")
    print(f"\nOutput: {EVAL_DIR.resolve()}")
    print("Next step: inspect baseline_vs_llm.csv for LLM vs baseline comparison")


if __name__ == "__main__":
    main()
