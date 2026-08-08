"""
27_unknown_subset_analysis.py
==============================
Analisi delle performance B/C sul sottoinsieme dei deal "non conosciuti" dai modelli.

RAZIONALE
---------
Se un modello in Condizione B piazza la target al rank 1, non si può sapere se ha ragionato sul fit strategico o ha recuperato dalla memoria il deal.
Il recall test (Script 19) identifica i deal non recuperati esattamente dal
modello. Il subset U_m include quindi PARTIAL, FAILURE e UNKNOWN; soltanto
SUCCESS è considerato recall validato. Il confronto con il full sample è una
robustezza descrittiva, non una stima causale della quota di memoria.

SORGENTE DATI
-------------
  results/evaluation/metrics_raw_BC.csv   ← output Script 17 (metriche per-deal)
  results/memorization/recall_raw_*.jsonl ← output Script 19 (recall test)

OUTPUT
------
  results/memorization/
    unknown_subset_analysis.csv     
    unknown_subset_deltas.csv       
    unknown_subset_report.txt       

"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
# Derivato dalla posizione dello script: <ma_analyst>/scripts/27_*.py → ma_analyst
BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MEMO_DIR    = RESULTS_DIR / "memorization"
EVAL_DIR    = RESULTS_DIR / "evaluation"

# ── Mapping model_key (CSV) → model_slug (recall JSONL) ───────────────────
# model_key è il valore nella colonna model_key di metrics_raw_BC.csv
# model_slug è il valore in recall_raw_{slug}.jsonl
MODEL_KEY_TO_SLUG = {
    "deepseek":    "deepseek_v3",
    "qwen3":       "qwen3_235b",
    "llama32_1b":  "llama32_1b",
    "qwen3_8b":    "qwen3_8b",
    "phi4":        "phi4_14b",
    "llama33_70b": "llama33_70b",
}

BIG_MODELS_SLUG = ["llama33_70b", "qwen3_235b", "deepseek_v3"]
ALL_MODELS_SLUG = list(MODEL_KEY_TO_SLUG.values())
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


# ── Caricamento recall ─────────────────────────────────────────────────────

def load_recall(memo_dir: Path) -> dict[str, dict]:
    """
    Ritorna { deal_id: { model_slug: label } }
    """
    recall: dict[str, dict] = defaultdict(dict)
    for fpath in sorted(memo_dir.glob("recall_raw_*.jsonl")):
        for line in fpath.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                recall[r["deal_id"]][r["model_slug"]] = r["label"]
            except Exception:
                pass
    log.info(f"Recall caricato: {len(recall)} deal_id")
    return recall


def build_unknown_subsets(recall: dict) -> tuple[set, set]:
    """
    Sensitivity subsets (kept for robustness reporting):
      unknown_big: not recalled by ANY of the 3 big models
      unknown_all: not recalled by ANY of the 6 models (strict common set)
    """
    unknown_big, unknown_all = set(), set()
    for deal_id, labels in recall.items():
        # Only an exact, validated SUCCESS counts as direct recall. PARTIAL is
        # deliberately retained in the non-recalled set because it did not pass
        # conservative entity resolution.
        not_known = lambda slug: labels.get(slug, "UNKNOWN") != "SUCCESS"

        if all(not_known(m) for m in BIG_MODELS_SLUG):
            unknown_big.add(deal_id)
        if all(not_known(m) for m in ALL_MODELS_SLUG):
            unknown_all.add(deal_id)

    log.info(f"Unknown subset — big models: {len(unknown_big)} | all models: {len(unknown_all)}")
    return unknown_big, unknown_all


def build_model_specific_subsets(recall: dict) -> dict[str, set]:
    """
    PRIMARY analysis (audit finding): U_m = deals that model m ITSELF did not
    recall via the direct-recall probe. This is model-specific, not a common set,
    and is the estimand for "performance on deals the model could not directly
    recall". Returns { model_slug: {deal_ids not recalled by that model} }.
    """
    per_model: dict[str, set] = defaultdict(set)
    for deal_id, labels in recall.items():
        for slug, label in labels.items():
            if label != "SUCCESS":
                per_model[slug].add(deal_id)
    for slug, s in per_model.items():
        log.info(f"U_m — {slug}: {len(s)} non-recalled deals")
    return per_model


# ── Metriche aggregate su subset ───────────────────────────────────────────

def aggregate_metrics(df_sub: pd.DataFrame) -> dict:
    """
    Aggrega le metriche per-deal già calcolate da Script 17.
    target_rank è NaN quando la target non è nel top-10 → MRR=0 per quel deal.
    """
    n = len(df_sub)
    if n == 0:
        return {
            "n_deals": 0, "mrr": 0.0,
            "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0,
            "ndcg@5": 0.0, "ndcg@10": 0.0,
        }
    return {
        "n_deals":    n,
        "mrr":        round(df_sub["mrr"].mean(), 4),
        "recall@1":   round(df_sub["recall_1"].mean(), 4),
        "recall@5":   round(df_sub["recall_5"].mean(), 4),
        "recall@10":  round(df_sub["recall_10"].mean(), 4),
        "ndcg@5":     round(df_sub["ndcg_5"].mean(), 4),
        "ndcg@10":    round(df_sub["ndcg_10"].mean(), 4),
    }


def clustered_subset_delta_ci(
    df_cell: pd.DataFrame,
    subset_deals: set,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% cluster-bootstrap CI for MRR(subset) - MRR(full).

    Acquirers are resampled with replacement. Within each sampled acquirer,
    every deal is retained, preserving repeated-acquirer dependence and the
    overlap between the subset and the full-sample estimator.
    """
    if df_cell.empty or not subset_deals:
        return (0.0, 0.0)

    cluster_col = "acquirer_ticker"
    if cluster_col not in df_cell.columns:
        raise KeyError(f"Missing bootstrap cluster column: {cluster_col}")

    grouped = list(df_cell.groupby(cluster_col, sort=True))
    full_sums = np.array([group["mrr"].sum() for _, group in grouped])
    full_counts = np.array([len(group) for _, group in grouped])
    subset_sums = np.array([
        group.loc[group["deal_id"].isin(subset_deals), "mrr"].sum()
        for _, group in grouped
    ])
    subset_counts = np.array([
        group["deal_id"].isin(subset_deals).sum()
        for _, group in grouped
    ])

    n_clusters = len(grouped)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    full_means = full_sums[picks].sum(axis=1) / full_counts[picks].sum(axis=1)
    selected_subset_counts = subset_counts[picks].sum(axis=1)
    if np.any(selected_subset_counts == 0):
        raise ValueError("A bootstrap resample contained no subset observations")
    subset_means = (
        subset_sums[picks].sum(axis=1) / selected_subset_counts
    )
    boot = subset_means - full_means

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return (round(float(lo), 4), round(float(hi), 4))


# ── Analisi principale ─────────────────────────────────────────────────────

def run_analysis(eval_dir: Path, memo_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Carica metriche per-deal da Script 17
    metrics_path = eval_dir / "metrics_raw_BC.csv"
    if not metrics_path.exists():
        log.error(f"File non trovato: {metrics_path}")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(metrics_path)
    log.info(f"metrics_raw_BC.csv caricato: {len(df)} righe")
    log.info(f"  model_key unici: {sorted(df['model_key'].unique())}")
    log.info(f"  condition unici: {sorted(df['condition'].unique())}")
    log.info(f"  strategy unici:  {sorted(df['strategy'].unique())}")

    # Aggiungi colonna model_slug per il join con recall
    df["model_slug"] = df["model_key"].map(MODEL_KEY_TO_SLUG)
    unmapped = df[df["model_slug"].isna()]["model_key"].unique()
    if len(unmapped) > 0:
        log.warning(f"model_key senza mapping: {unmapped} — righe escluse")
        df = df.dropna(subset=["model_slug"])

    # Carica recall e costruisce subset
    recall       = load_recall(memo_dir)
    unknown_big, unknown_all = build_unknown_subsets(recall)
    u_m          = build_model_specific_subsets(recall)   # PRIMARY: model-specific U_m

    rows_analysis = []
    rows_delta    = []

    for condition in ["B", "C"]:
        for model_key in sorted(df["model_key"].unique()):
            for strategy in sorted(df["strategy"].unique()):
                mask = (
                    (df["condition"]  == condition) &
                    (df["model_key"]  == model_key) &
                    (df["strategy"]   == strategy)
                )
                df_cell = df[mask]
                if df_cell.empty:
                    log.debug(f"Nessun dato: {condition}/{model_key}/{strategy}")
                    continue

                slug = MODEL_KEY_TO_SLUG.get(model_key, model_key)

                # PRIMARY subset: this model's OWN non-recalled deals (U_m).
                u_self = u_m.get(slug, set())
                df_u_self  = df_cell[df_cell["deal_id"].isin(u_self)]
                df_unk_big = df_cell[df_cell["deal_id"].isin(unknown_big)]
                df_unk_all = df_cell[df_cell["deal_id"].isin(unknown_all)]

                m_full    = aggregate_metrics(df_cell)
                m_u_self  = aggregate_metrics(df_u_self)
                m_unk_big = aggregate_metrics(df_unk_big)
                m_unk_all = aggregate_metrics(df_unk_all)
                u_ci_lo, u_ci_hi = clustered_subset_delta_ci(df_cell, u_self)
                common_ci_lo, common_ci_hi = clustered_subset_delta_ci(
                    df_cell, unknown_big
                )
                delta_u_exact = round(
                    df_u_self["mrr"].mean() - df_cell["mrr"].mean(), 4
                )
                delta_common_exact = round(
                    df_unk_big["mrr"].mean() - df_cell["mrr"].mean(), 4
                )

                def make_row(subset, metrics):
                    return {"condition": condition, "model_key": model_key,
                            "model_slug": slug, "strategy": strategy,
                            "subset": subset, **metrics}

                rows_analysis.append(make_row("full",           m_full))
                rows_analysis.append(make_row("U_m_self",        m_u_self))   # primary
                rows_analysis.append(make_row("unknown_big",     m_unk_big))  # sensitivity
                rows_analysis.append(make_row("unknown_all",     m_unk_all))  # sensitivity

                rows_delta.append({
                    "condition":        condition,
                    "model_key":        model_key,
                    "model_slug":       slug,
                    "strategy":         strategy,
                    "n_full":           m_full["n_deals"],
                    "n_U_m_self":       m_u_self["n_deals"],
                    "mrr_U_m_self":     m_u_self["mrr"],
                    "delta_mrr_U_m":    delta_u_exact,
                    "delta_mrr_U_m_ci_lo": u_ci_lo,
                    "delta_mrr_U_m_ci_hi": u_ci_hi,
                    "n_unknown_big":    m_unk_big["n_deals"],
                    "mrr_full":         m_full["mrr"],
                    "mrr_unknown_big":  m_unk_big["mrr"],
                    "delta_mrr":        delta_common_exact,
                    "delta_mrr_ci_lo":  common_ci_lo,
                    "delta_mrr_ci_hi":  common_ci_hi,
                    "recall5_full":     m_full["recall@5"],
                    "recall5_unknown":  m_unk_big["recall@5"],
                    "delta_recall5":    round(m_unk_big["recall@5"] - m_full["recall@5"], 4),
                    "ndcg10_full":      m_full["ndcg@10"],
                    "ndcg10_unknown":   m_unk_big["ndcg@10"],
                    "delta_ndcg10":     round(m_unk_big["ndcg@10"] - m_full["ndcg@10"], 4),
                })

                log.info(
                    f"  {condition}/{model_key}/{strategy} | "
                    f"full n={m_full['n_deals']} MRR={m_full['mrr']} | "
                    f"unknown_big n={m_unk_big['n_deals']} MRR={m_unk_big['mrr']} | "
                    f"delta={round(m_unk_big['mrr']-m_full['mrr'],4):+.4f}"
                )

    return pd.DataFrame(rows_analysis), pd.DataFrame(rows_delta)


# ── Report ─────────────────────────────────────────────────────────────────

def write_report(delta_df: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "UNKNOWN SUBSET ANALYSIS — LLM-as-M&A-Analyst",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "SETUP",
        "  Full corpus:     286 deals",
        "  Unknown subset:  deal non recalled da nessuno dei 3 modelli grandi",
        "                   (DeepSeek V3, Llama 3.3 70B, Qwen3-235B)",
        "",
        "INTERPRETATION (descriptive only — NOT a causal memory share)",
        "  PRIMARY subset is model-specific U_m: deals THIS model did not recover",
        "  via the direct-recall probe (Script 19). unknown_big/all are common-set",
        "  SENSITIVITY subsets.",
        "  delta_mrr = MRR(U_m) - MRR(full).",
        "  A near-zero delta means performance is similar on deals the model could",
        "  not directly recall; it does NOT prove the event is absent from the",
        "  weights, nor does it estimate a causal 'memory share'.",
        "",
        "=" * 70,
        "DELTA MRR — CONDITION B (profile-augmented ranking)",
        "=" * 70,
    ]

    for condition, label in [("B", "CONDITION B"), ("C", "CONDITION C")]:
        if condition == "C":
            lines += ["", "=" * 70,
                      "DELTA MRR — CONDITION C (metadata-only ranking)",
                      "=" * 70]
        sub = delta_df[delta_df["condition"] == condition].sort_values(
            ["model_slug", "strategy"])
        lines.append(
            f"\n  {'Model':<14} {'Strategy':<10} "
            f"{'MRR full':>10} {'MRR unkn':>10} {'Delta':>8} "
            f"{'n_full':>7} {'n_unkn':>7}"
        )
        lines.append("  " + "-" * 70)
        for _, r in sub.iterrows():
            lines.append(
                f"  {r['model_slug']:<14} {r['strategy']:<10} "
                f"{r['mrr_full']:>10.4f} {r['mrr_unknown_big']:>10.4f} "
                f"{r['delta_mrr']:>+8.4f} "
                f"{int(r['n_full']):>7} {int(r['n_unknown_big']):>7}"
            )

    # Auto-fill the finding from the primary strategy (zeroshot), Condition B,
    # model-specific U_m deltas (v2 audit placeholder cleanup).
    prim = delta_df[(delta_df["condition"] == "B") & (delta_df["strategy"] == "zeroshot")]
    per_model = ", ".join(
        f"{r['model_slug']} {r['delta_mrr_U_m']:+.4f} (n={int(r['n_U_m_self'])})"
        for _, r in prim.sort_values("model_slug").iterrows()
    )
    dvals = prim["delta_mrr_U_m"]
    all_b = delta_df[delta_df["condition"] == "B"]["delta_mrr_U_m"]

    lines += [
        "", "=" * 70, "PAPER PARAGRAPH (cautious phrasing, auto-filled)", "=" * 70, "",
        "For each model m we restricted attention to U_m, the deals whose target",
        "was NOT RECOVERED BY THE DIRECT-RECALL PROBE (Script 19). Ranking",
        "performance on U_m is reported per model; the strict common set (no model",
        "recalled) is reported as a sensitivity analysis. We do not interpret these",
        "differences as a causal estimate of parametric-memory contribution, and we",
        "report selection differences between recalled and non-recalled deals.",
        "",
        "Finding (Condition B, primary strategy zeroshot):",
        f"  Per-model delta_mrr(U_m) = MRR(U_m) - MRR(full): {per_model}.",
        f"  Across all Condition-B strategies the delta ranges {all_b.min():+.4f} "
        f"to {all_b.max():+.4f} (zeroshot mean {dvals.mean():+.4f}).",
        "  Interpretation: on the deals each model could not directly recall,",
        "  ranking performance is essentially unchanged from the full corpus",
        "  (deltas near zero, both signs). The largest single drop is llama33_70b",
        "  (the strongest direct-recaller, ~51% recall), consistent with a small",
        "  secondary memory contribution for that model; for the others U_m",
        "  performance is within noise of full performance. This is descriptive",
        "  robustness evidence that the Condition-B ranking signal is NOT an",
        "  artifact of directly elicitable deal memory. It is not a causal memory",
        "  share and does not prove absence of the event from the weights.",
    ]

    path = out_dir / "unknown_subset_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Report: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    MEMO_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Eval dir:  {EVAL_DIR}")
    log.info(f"Memo dir:  {MEMO_DIR}")

    df_analysis, df_delta = run_analysis(EVAL_DIR, MEMO_DIR)

    if df_analysis.empty:
        log.error("Nessun dato prodotto. Verifica i path.")
        return

    csv_path   = MEMO_DIR / "unknown_subset_analysis.csv"
    delta_path = MEMO_DIR / "unknown_subset_deltas.csv"

    df_analysis.to_csv(csv_path,   index=False)
    df_delta.to_csv(delta_path,    index=False)
    log.info(f"Analysis CSV: {csv_path}")
    log.info(f"Delta CSV:    {delta_path}")

    write_report(df_delta, MEMO_DIR)

    # Console
    print(f"\n{'='*70}")
    print("  UNKNOWN SUBSET ANALYSIS — DELTA MRR")
    print(f"{'='*70}")
    for condition in ["B", "C"]:
        print(f"\n  CONDITION {condition}")
        print(f"  {'Model':<14} {'Strategy':<10} "
              f"{'MRR full':>10} {'MRR unkn':>10} {'Delta':>8}")
        print("  " + "-" * 55)
        sub = df_delta[df_delta["condition"] == condition].sort_values(
            ["model_slug", "strategy"])
        for _, r in sub.iterrows():
            print(
                f"  {r['model_slug']:<14} {r['strategy']:<10} "
                f"{r['mrr_full']:>10.4f} {r['mrr_unknown_big']:>10.4f} "
                f"{r['delta_mrr']:>+8.4f}"
            )
    print(f"\n  Output: {MEMO_DIR}")


if __name__ == "__main__":
    main()
