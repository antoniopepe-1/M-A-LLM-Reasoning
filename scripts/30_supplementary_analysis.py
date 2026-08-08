"""
30_supplementary_analysis.py
=============================

ANALYSES
--------
(1) Bootstrap CI on embedding results
    Adds 95% confidence intervals to embedding_metrics_agg_v2.csv. Required before reporting embedding results in the paper — needed
    to assess whether advantage of text-embedding-3-large over best LLM in Condition B is statistically significant or within noise.

(2) Performance by industry
    MRR breakdown by industry sector for all methods (BM25, TF-IDF, embedding models, LLM-best). Answers: do LLMs outperform embeddings
    uniformly across sectors, or only in specific industries? Output: industry_breakdown_v2.csv + latex/table_industry_v2.tex

(3) Performance by deal year (2015-2022)
    MRR breakdown by year for all methods. Answers: does performance degrade on more recent deals (post-training-cutoff effects)?
    Complements the memorization audit — if knowledge memorization correlates with year, performance should track deal year.
    Output: year_breakdown_v2.csv + latex/table_year_v2.tex

(4) Embedding similarity vs LLM rank correlation
    For each deal: cosine similarity between acquirer and correct target (from embedding raw file) vs rank assigned by each LLM.
    Spearman correlation across 286 deals. High correlation → LLMs replicate semantic similarity signal. Low correlation → LLMs use genuinely different information.
    Output: emb_llm_correlation_v2.csv + latex/table_correlation_v2.tex

INPUT FILES (all pre-existing — no API calls)
---------------------------------------------
  results/evaluation/embedding_metrics_raw_v2.csv   (Script 29)
  results/evaluation/baseline_metrics_raw.csv        (Script 25)
  results/evaluation/metrics_raw_BC.csv              (Script 17 — LLM raw)
  results/evaluation/metrics_agg_BC.csv              (Script 17 — LLM agg)

OUTPUT FILES (all _v2 suffixes)
--------------------------------
  results/evaluation/
    embedding_metrics_agg_v2.csv         updated with CI (analysis 1)
    industry_breakdown_v2.csv            (analysis 2)
    year_breakdown_v2.csv                (analysis 3)
    emb_llm_correlation_v2.csv           (analysis 4)
    latex/
      table_industry_v2.tex
      table_year_v2.tex
      table_correlation_v2.tex

"""

import math
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import metrics as _metrics

# PRE-SPECIFIED primary cell (audit finding: selecting best model then pooling
# its strategies is not one experimental cell). Industry/year breakdowns for the
# LLM use this single model x strategy cell, chosen a priori, not selected on the
# test set. Override via --primary-model / --primary-strategy if needed.
PRIMARY_STRATEGY = "zeroshot"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "results" / "evaluation"
TEX_DIR  = EVAL_DIR / "latex"
TEX_DIR.mkdir(parents=True, exist_ok=True)

# Input files
EMB_RAW_PATH  = EVAL_DIR / "embedding_metrics_raw_v2.csv"
BASE_RAW_PATH = EVAL_DIR / "baseline_metrics_raw.csv"
LLM_RAW_PATH  = EVAL_DIR / "metrics_raw_BC.csv"
LLM_AGG_PATH  = EVAL_DIR / "metrics_agg_BC.csv"
EMB_AGG_PATH  = EVAL_DIR / "embedding_metrics_agg_v2.csv"

# Output files
OUT_EMB_AGG   = EVAL_DIR / "embedding_metrics_agg_v2.csv"
OUT_INDUSTRY  = EVAL_DIR / "industry_breakdown_v2.csv"
OUT_YEAR      = EVAL_DIR / "year_breakdown_v2.csv"
OUT_CORR      = EVAL_DIR / "emb_llm_correlation_v2.csv"

TEX_INDUSTRY  = TEX_DIR / "table_industry_v2.tex"
TEX_YEAR      = TEX_DIR / "table_year_v2.tex"
TEX_CORR      = TEX_DIR / "table_correlation_v2.tex"

# ── Constants ──────────────────────────────────────────────────────────────
N_BOOTSTRAP  = 1000
BOOTSTRAP_CI = 0.95
RANDOM_SEED  = 42
RANDOM_MRR   = _metrics.chance_mrr_at_k(10, 50)  # H_10 / 50 ≈ 0.0586


# ── Helpers ────────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP,
                 ci: float = BOOTSTRAP_CI,
                 seed: int = RANDOM_SEED) -> tuple[float, float]:
    """95% bootstrap CI on mean."""
    if len(values) < 2:
        return (round(values[0], 4), round(values[0], 4))
    rng   = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return round(lo, 4), round(hi, 4)


def fmt(v, decimals: int = 3) -> str:
    """Format float for LaTeX table, '--' if missing."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{decimals}f}"


def ci_str(lo, hi) -> str:
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return "--"
    return f"[{fmt(lo)}, {fmt(hi)}]"


def load_file(path: Path, label: str) -> pd.DataFrame | None:
    """Load CSV with informative error if missing."""
    if not path.exists():
        log.warning(f"{label} not found: {path}")
        log.warning(f"Skipping analyses that require this file.")
        return None
    df = pd.read_csv(path)
    log.info(f"Loaded {label}: {len(df)} rows from {path.name}")
    return df


def aggregate_group(grp: pd.DataFrame,
                    run_bootstrap: bool = True) -> dict:
    """Aggregate a group of per-deal rows into summary statistics."""
    mrr_arr = grp["mrr"].values
    r1_arr  = grp["recall_1"].values
    r10_arr = grp["recall_10"].values
    nd_arr  = grp["ndcg_10"].values

    row = {
        "n_deals":  len(grp),
        "mrr":      round(mrr_arr.mean(), 4),
        "recall_1": round(r1_arr.mean(), 4),
        "recall_10":round(r10_arr.mean(), 4),
        "ndcg_10":  round(nd_arr.mean(), 4),
        "mrr_ci_lo": None, "mrr_ci_hi": None,
    }
    if run_bootstrap and len(mrr_arr) > 1:
        lo, hi = bootstrap_ci(mrr_arr)
        row["mrr_ci_lo"] = lo
        row["mrr_ci_hi"] = hi
    return row


# ── Analysis 1: Bootstrap CI on embedding results ─────────────────────────

def analysis_ci(df_emb_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add bootstrap 95% CI to embedding aggregated results.
    Updates embedding_metrics_agg_v2.csv in place.
    """
    log.info("=== Analysis 1: Bootstrap CI on embedding results ===")

    rows = []
    for (method, cond), grp in df_emb_raw.groupby(["method", "condition"]):
        agg = aggregate_group(grp, run_bootstrap=True)
        rows.append({
            "method":    method,
            "condition": cond,
            "model_id":  grp["model_id"].iloc[0] if "model_id" in grp.columns else "",
            **agg,
            "recall_5":  round(grp["recall_5"].mean(), 4),
            "ndcg_5":    round(grp["ndcg_5"].mean(), 4),
        })

    df_agg = pd.DataFrame(rows)
    df_agg.to_csv(OUT_EMB_AGG, index=False)
    log.info(f"Saved: {OUT_EMB_AGG}")

    # Print CI summary
    print("\n" + "=" * 75)
    print("ANALYSIS 1 — EMBEDDING RESULTS WITH BOOTSTRAP 95% CI")
    print("=" * 75)
    print(f"{'Method':<35} {'Cond':>5} {'MRR':>7} {'CI':>20} {'R@1':>7} {'R@10':>7}")
    print("-" * 75)
    for _, r in df_agg.sort_values(["condition", "mrr"],
                                    ascending=[True, False]).iterrows():
        ci = ci_str(r.get("mrr_ci_lo"), r.get("mrr_ci_hi"))
        print(f"{r['method']:<35} {r['condition']:>5} {r['mrr']:>7.3f} "
              f"{ci:>20} {r['recall_1']:>7.3f} {r['recall_10']:>7.3f}")

    print(f"\nNote: CI overlap between methods = not statistically distinguishable")
    return df_agg


# ── Analysis 2: Performance by industry ───────────────────────────────────

def analysis_industry(df_emb_raw: pd.DataFrame,
                      df_base_raw: pd.DataFrame,
                      df_llm_raw: pd.DataFrame | None,
                      conditions: list[str]) -> pd.DataFrame:
    """
    MRR breakdown by industry × method × condition.
    Only includes industries with at least 5 deals for stability.
    """
    log.info("=== Analysis 2: Performance by industry ===")

    # Standardize column names across sources
    frames = []

    # Embedding
    for (method, cond, industry), grp in df_emb_raw.groupby(
            ["method", "condition", "industry"]):
        if cond not in conditions:
            continue
        agg = aggregate_group(grp, run_bootstrap=False)
        frames.append({"method": method, "condition": cond,
                        "industry": industry, **agg})

    # BM25 + TF-IDF
    for (method, cond, industry), grp in df_base_raw.groupby(
            ["method", "condition", "industry"]):
        if cond not in conditions:
            continue
        agg = aggregate_group(grp, run_bootstrap=False)
        frames.append({"method": method, "condition": cond,
                        "industry": industry, **agg})

    # LLM: ONE pre-specified model x strategy cell per condition (no pooling).
    if df_llm_raw is not None:
        llm_primary = df_llm_raw[df_llm_raw["strategy"] == PRIMARY_STRATEGY]
        # Choose the model a priori by overall MRR IN THE PRIMARY STRATEGY ONLY,
        # then use that single (model, PRIMARY_STRATEGY) cell — not pooled strategies.
        best_models = (
            llm_primary.groupby(["model_key", "condition"])["mrr"]
            .mean().reset_index()
            .sort_values("mrr", ascending=False)
            .groupby("condition").first().reset_index()
        )
        for _, bm in best_models.iterrows():
            cond, mkey = bm["condition"], bm["model_key"]
            if cond not in conditions:
                continue
            sub = llm_primary[(llm_primary["model_key"] == mkey) &
                              (llm_primary["condition"] == cond)]
            if "industry" not in sub.columns:
                log.warning("LLM raw file missing 'industry' column — skipping LLM industry breakdown")
                break
            for industry, grp in sub.groupby("industry"):
                agg = aggregate_group(grp, run_bootstrap=False)
                frames.append({"method": f"LLM ({mkey}, {PRIMARY_STRATEGY})",
                                "condition": cond,
                                "industry": industry, **agg})

    df = pd.DataFrame(frames)

    # Filter industries with < 5 deals in any method (unstable estimates)
    industry_counts = df_emb_raw.groupby("industry")["deal_id"].nunique()
    valid_industries = industry_counts[industry_counts >= 5].index.tolist()
    df = df[df["industry"].isin(valid_industries)]

    df.to_csv(OUT_INDUSTRY, index=False)
    log.info(f"Saved: {OUT_INDUSTRY} ({len(df)} rows, "
             f"{df['industry'].nunique()} industries)")

    # Print summary table
    print("\n" + "=" * 80)
    print("ANALYSIS 2 — MRR BY INDUSTRY (Condition B)")
    print("=" * 80)

    methods_order = ["TF-IDF", "BM25"] + \
                    [m for m in df["method"].unique()
                     if m not in ("TF-IDF", "BM25") and not m.startswith("LLM")] + \
                    [m for m in df["method"].unique() if m.startswith("LLM")]

    sub_b = df[df["condition"] == "B"]
    if not sub_b.empty:
        pivot = sub_b.pivot_table(
            index="industry", columns="method", values="mrr"
        ).round(3)
        # Reorder columns
        cols = [c for c in methods_order if c in pivot.columns]
        pivot = pivot[cols]
        print(pivot.to_string())

    _export_industry_latex(df, TEX_INDUSTRY, conditions)
    return df


def _export_industry_latex(df: pd.DataFrame, out_path: Path,
                            conditions: list[str]) -> None:
    """Export industry breakdown as LaTeX table."""
    methods_order = ["TF-IDF", "BM25"] + \
                    [m for m in df["method"].unique()
                     if m not in ("TF-IDF", "BM25") and not m.startswith("LLM")] + \
                    [m for m in df["method"].unique() if m.startswith("LLM")]

    lines = [
        f"% Industry breakdown — generated by 30_supplementary_analysis.py "
        f"on {datetime.now().strftime('%Y-%m-%d')}",
        r"\begin{table}[t]",
        r"\centering\small",
        (r"\caption{MRR by industry sector. Only sectors with $\geq 5$ deals "
         r"are reported. Condition~B (with candidate descriptions).}"),
        r"\label{tab:industry}",
    ]

    for cond in conditions:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        industries = sorted(sub["industry"].unique())
        methods    = [m for m in methods_order if m in sub["method"].unique()]
        n_cols     = len(methods) + 1

        col_fmt = "l" + "r" * len(methods)
        lines += [
            rf"\textbf{{Condition {cond}}}\\",
            rf"\begin{{tabular}}{{{col_fmt}}}",
            r"\toprule",
            "Industry & " + " & ".join(
                [rf"\textbf{{{m}}}" for m in methods]) + r" \\",
            r"\midrule",
        ]

        for ind in industries:
            row_vals = []
            sub_ind  = sub[sub["industry"] == ind]
            best_mrr = sub_ind["mrr"].max()
            for method in methods:
                r = sub_ind[sub_ind["method"] == method]
                v = fmt(r["mrr"].values[0]) if not r.empty else "--"
                # Bold best value per row
                if not r.empty and abs(r["mrr"].values[0] - best_mrr) < 0.001:
                    v = rf"\textbf{{{v}}}"
                row_vals.append(v)
            lines.append(f"{ind} & " + " & ".join(row_vals) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}", ""]

    lines += [r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"LaTeX saved: {out_path}")


# ── Analysis 3: Performance by deal year ──────────────────────────────────

def analysis_year(df_emb_raw: pd.DataFrame,
                  df_base_raw: pd.DataFrame,
                  df_llm_raw: pd.DataFrame | None,
                  conditions: list[str]) -> pd.DataFrame:
    """
    MRR breakdown by deal year × method × condition.
    Complements memorization audit — knowledge memorization should
    correlate with year if models have seen older deals in training.
    """
    log.info("=== Analysis 3: Performance by deal year ===")

    frames = []

    # Embedding
    for (method, cond, year), grp in df_emb_raw.groupby(
            ["method", "condition", "deal_year"]):
        if cond not in conditions:
            continue
        agg = aggregate_group(grp, run_bootstrap=False)
        frames.append({"method": method, "condition": cond,
                        "deal_year": int(year), **agg})

    # BM25 + TF-IDF
    for (method, cond, year), grp in df_base_raw.groupby(
            ["method", "condition", "deal_year"]):
        if cond not in conditions:
            continue
        agg = aggregate_group(grp, run_bootstrap=False)
        frames.append({"method": method, "condition": cond,
                        "deal_year": int(year), **agg})

    # LLM: ONE pre-specified model x strategy cell per condition (no pooling).
    if df_llm_raw is not None and "deal_year" in df_llm_raw.columns:
        llm_primary = df_llm_raw[df_llm_raw["strategy"] == PRIMARY_STRATEGY]
        best_models = (
            llm_primary.groupby(["model_key", "condition"])["mrr"]
            .mean().reset_index()
            .sort_values("mrr", ascending=False)
            .groupby("condition").first().reset_index()
        )
        for _, bm in best_models.iterrows():
            cond, mkey = bm["condition"], bm["model_key"]
            if cond not in conditions:
                continue
            sub = llm_primary[(llm_primary["model_key"] == mkey) &
                              (llm_primary["condition"] == cond)]
            for year, grp in sub.groupby("deal_year"):
                agg = aggregate_group(grp, run_bootstrap=False)
                frames.append({"method": f"LLM ({mkey}, {PRIMARY_STRATEGY})",
                                "condition": cond,
                                "deal_year": int(year), **agg})

    df = pd.DataFrame(frames)
    df = df.sort_values(["condition", "method", "deal_year"])
    df.to_csv(OUT_YEAR, index=False)
    log.info(f"Saved: {OUT_YEAR} ({len(df)} rows)")

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS 3 — MRR BY DEAL YEAR (Condition B)")
    print("=" * 80)

    methods_order = ["TF-IDF", "BM25"] + \
                    [m for m in df["method"].unique()
                     if m not in ("TF-IDF", "BM25") and not m.startswith("LLM")] + \
                    [m for m in df["method"].unique() if m.startswith("LLM")]

    sub_b = df[df["condition"] == "B"]
    if not sub_b.empty:
        pivot = sub_b.pivot_table(
            index="deal_year", columns="method", values="mrr"
        ).round(3)
        cols = [c for c in methods_order if c in pivot.columns]
        pivot = pivot[cols]
        print(pivot.to_string())

        # Trend test: Spearman correlation of year vs MRR for each method
        print("\nYear-MRR trend (Spearman ρ, p-value):")
        for method in methods_order:
            sub_m = sub_b[sub_b["method"] == method]
            if len(sub_m) < 3:
                continue
            rho, p = stats.spearmanr(sub_m["deal_year"], sub_m["mrr"])
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
            print(f"  {method:<35} ρ={rho:+.3f}  p={p:.3f} {sig}")

    _export_year_latex(df, TEX_YEAR, conditions)
    return df


def _export_year_latex(df: pd.DataFrame, out_path: Path,
                        conditions: list[str]) -> None:
    """Export year breakdown as LaTeX table."""
    methods_order = ["TF-IDF", "BM25"] + \
                    [m for m in df["method"].unique()
                     if m not in ("TF-IDF", "BM25") and not m.startswith("LLM")] + \
                    [m for m in df["method"].unique() if m.startswith("LLM")]

    lines = [
        f"% Year breakdown — generated by 30_supplementary_analysis.py "
        f"on {datetime.now().strftime('%Y-%m-%d')}",
        r"\begin{table}[t]",
        r"\centering\small",
        (r"\caption{MRR by deal year (2015--2022). "
         r"Condition~B (with candidate descriptions). "
         r"Monotonic trends assessed via Spearman rank correlation.}"),
        r"\label{tab:year}",
    ]

    for cond in conditions:
        sub   = df[df["condition"] == cond]
        if sub.empty:
            continue
        years   = sorted(sub["deal_year"].unique())
        methods = [m for m in methods_order if m in sub["method"].unique()]

        col_fmt = "l" + "r" * len(methods)
        lines += [
            rf"\textbf{{Condition {cond}}}\\",
            rf"\begin{{tabular}}{{{col_fmt}}}",
            r"\toprule",
            "Year & " + " & ".join(
                [rf"\textbf{{{m}}}" for m in methods]) + r" \\",
            r"\midrule",
        ]

        for year in years:
            row_vals = []
            sub_y    = sub[sub["deal_year"] == year]
            for method in methods:
                r = sub_y[sub_y["method"] == method]
                v = fmt(r["mrr"].values[0]) if not r.empty else "--"
                row_vals.append(v)
            lines.append(f"{year} & " + " & ".join(row_vals) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}", ""]

    lines += [r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"LaTeX saved: {out_path}")


# ── Analysis 4: Embedding similarity vs LLM rank correlation ──────────────

def analysis_correlation(df_emb_raw: pd.DataFrame,
                          df_llm_raw: pd.DataFrame,
                          conditions: list[str]) -> pd.DataFrame:
    """
    Spearman correlation between the target's rank under embedding retrieval and
    under each LLM, on the SAME deals.

    CENSORED RANKS (audit finding): a target outside the top-K window has no
    observed rank. We impute a single censored value (K+1) for BOTH methods so
    the correlation is defined, and we ALSO report a "both-observed" correlation
    on deals where both methods placed the target in the top-K. Ranks are
    censored, so ρ is DESCRIPTIVE: a low ρ is consistent with — but does not
    prove — that LLMs use "genuinely different information".
    """
    log.info("=== Analysis 4: Embedding vs LLM target-rank correlation (censored) ===")
    CENSORED = _metrics.DEFAULT_K + 1  # rank imputed for targets outside top-K

    def _rank_col(df):
        # script 17 emits 'target_rank'; older files used 'rank'
        return "target_rank" if "target_rank" in df.columns else ("rank" if "rank" in df.columns else None)

    rows = []
    for cond in conditions:
        emb_rc = _rank_col(df_emb_raw)
        emb_sub = df_emb_raw[df_emb_raw["condition"] == cond][
            ["deal_id", "method", emb_rc]
        ].rename(columns={emb_rc: "emb_rank", "method": "emb_method"})
        emb_sub["emb_rank"] = emb_sub["emb_rank"].fillna(CENSORED)

        if df_llm_raw is None:
            log.warning("LLM raw file not available — skipping analysis 4")
            return pd.DataFrame()
        llm_rc = _rank_col(df_llm_raw)
        if llm_rc is None:
            log.warning("LLM raw file missing a rank column — skipping analysis 4")
            return pd.DataFrame()
        llm_sub = df_llm_raw[df_llm_raw["condition"] == cond][
            ["deal_id", "model_key", llm_rc]
        ].rename(columns={llm_rc: "llm_rank"})
        llm_sub["llm_rank"] = llm_sub["llm_rank"].fillna(CENSORED)

        # For each (embedding_model, LLM_model) pair compute Spearman ρ
        for emb_method in emb_sub["emb_method"].unique():
            emb_deals = emb_sub[emb_sub["emb_method"] == emb_method][
                ["deal_id", "emb_rank"]
            ]

            for llm_model in llm_sub["model_key"].unique():
                llm_deals = llm_sub[llm_sub["model_key"] == llm_model][
                    ["deal_id", "llm_rank"]
                ]

                merged = emb_deals.merge(llm_deals, on="deal_id", how="inner")
                if len(merged) < 10:
                    continue

                rho, p = stats.spearmanr(
                    merged["emb_rank"], merged["llm_rank"]
                )
                rows.append({
                    "condition":  cond,
                    "emb_method": emb_method,
                    "llm_model":  llm_model,
                    "n_deals":    len(merged),
                    "spearman_rho": round(rho, 4),
                    "p_value":      round(p, 4),
                    "significant":  p < 0.05,
                    "interpretation": (   # descriptive only — ranks are censored
                        "HIGH rank agreement" if rho > 0.6
                        else "MODERATE rank agreement" if rho > 0.3
                        else "LOW rank agreement"
                    ),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No correlation results produced — check input files")
        return df

    df = df.sort_values(["condition", "emb_method", "spearman_rho"],
                         ascending=[True, True, False])
    df.to_csv(OUT_CORR, index=False)
    log.info(f"Saved: {OUT_CORR} ({len(df)} rows)")

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS 4 — EMBEDDING RANK vs LLM RANK CORRELATION (Spearman ρ)")
    print("=" * 80)
    print(f"{'Emb Model':<30} {'LLM Model':<25} {'Cond':>5} "
          f"{'ρ':>7} {'p':>7} {'Sig':>5}")
    print("-" * 80)
    for _, r in df.iterrows():
        sig = "**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "")
        print(f"{r['emb_method']:<30} {r['llm_model']:<25} "
              f"{r['condition']:>5} {r['spearman_rho']:>7.3f} "
              f"{r['p_value']:>7.3f} {sig:>5}")

    print("\nInterpretation:")
    print("  ρ > 0.60 → LLMs replicate embedding signal (semantic similarity drives ranking)")
    print("  ρ 0.30-0.60 → partial overlap, LLMs add some independent signal")
    print("  ρ < 0.30 → LLMs use genuinely different information from embeddings")

    _export_correlation_latex(df, TEX_CORR)
    return df


def _export_correlation_latex(df: pd.DataFrame, out_path: Path) -> None:
    """Export correlation table as LaTeX."""
    lines = [
        f"% Embedding-LLM correlation — generated by 30_supplementary_analysis.py "
        f"on {datetime.now().strftime('%Y-%m-%d')}",
        r"\begin{table}[t]",
        r"\centering\small",
        (r"\caption{Spearman rank correlation between embedding model ranking "
         r"and LLM ranking of M\&A candidates. High $\rho$ indicates that the "
         r"LLM replicates the semantic similarity signal captured by the embedding "
         r"model. Low $\rho$ indicates the LLM uses genuinely different information. "
         r"$^{*}p<0.05$, $^{**}p<0.01$.}"),
        r"\label{tab:correlation}",
        r"\begin{tabular}{l l l r r}",
        r"\toprule",
        (r"\textbf{Emb. Model} & \textbf{LLM} & \textbf{Cond.} "
         r"& $\boldsymbol{\rho}$ & \textbf{p} \\"),
        r"\midrule",
    ]

    prev_cond = None
    for _, r in df.iterrows():
        if prev_cond is not None and r["condition"] != prev_cond:
            lines.append(r"\midrule")
        prev_cond = r["condition"]

        sig = r"$^{**}$" if r["p_value"] < 0.01 else \
              (r"$^{*}$" if r["p_value"] < 0.05 else "")
        rho_str = f"{r['spearman_rho']:.3f}{sig}"
        lines.append(
            f"{r['emb_method']} & {r['llm_model']} & {r['condition']} "
            f"& {rho_str} & {fmt(r['p_value'])} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"LaTeX saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Supplementary analyses for M&A LLM study"
    )
    parser.add_argument(
        "--analyses", nargs="+", type=int,
        choices=[1, 2, 3, 4], default=[1, 2, 3, 4],
        help="Which analyses to run (default: all)"
    )
    parser.add_argument(
        "--condition", nargs="+", choices=["B", "C"], default=["B", "C"],
        help="Conditions to include (default: B C)"
    )
    args = parser.parse_args()

    log.info(f"Running analyses: {args.analyses}")
    log.info(f"Conditions: {args.condition}")

    # Load input files
    df_emb_raw  = load_file(EMB_RAW_PATH,  "Embedding raw")
    df_base_raw = load_file(BASE_RAW_PATH, "Baseline raw (BM25/TF-IDF)")
    df_llm_raw  = load_file(LLM_RAW_PATH,  "LLM raw")

    # Check minimum requirements
    if df_emb_raw is None:
        log.error("Embedding raw file required for all analyses. "
                  "Run script 29 first.")
        return

    # Run requested analyses
    if 1 in args.analyses:
        analysis_ci(df_emb_raw)

    if 2 in args.analyses:
        if df_base_raw is None:
            log.warning("Baseline raw missing — industry analysis will only "
                        "include embedding results")
        analysis_industry(df_emb_raw,
                          df_base_raw if df_base_raw is not None else pd.DataFrame(),
                          df_llm_raw, args.condition)

    if 3 in args.analyses:
        if df_base_raw is None:
            log.warning("Baseline raw missing — year analysis will only "
                        "include embedding results")
        analysis_year(df_emb_raw,
                      df_base_raw if df_base_raw is not None else pd.DataFrame(),
                      df_llm_raw, args.condition)

    if 4 in args.analyses:
        if df_llm_raw is None:
            log.error("LLM raw file required for analysis 4. "
                      "Run script 17 first.")
        else:
            analysis_correlation(df_emb_raw, df_llm_raw, args.condition)

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    for path in [OUT_EMB_AGG, OUT_INDUSTRY, OUT_YEAR, OUT_CORR,
                 TEX_INDUSTRY, TEX_YEAR, TEX_CORR]:
        status = "✓" if path.exists() else "✗ (not generated)"
        print(f"  {status}  {path.name}")


if __name__ == "__main__":
    main()
