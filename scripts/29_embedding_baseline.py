"""
29_embedding_baseline.py
========================
Dense retrieval baseline via OpenRouter embedding API.

Extends the lexical baselines of Script 25 (BM25, TF-IDF) with two neural embedding baselines, forming a complete methodological panel:

  TF-IDF                  — bag-of-words, frequency weighting
  BM25                    — bag-of-words, length-normalized
  text-embedding-3-large  — dense retrieval, OpenAI standard baseline
  qwen3-embedding-8b      — dense retrieval, same family as LLM roster

The Qwen3-Embedding vs Qwen3-8B (LLM) comparison is a novel within-family decomposition: same model family, embedding mode vs. reasoning mode.
Delta between Qwen3-Embedding MRR and Qwen3-8B LLM MRR isolates the contribution of strategic reasoning beyond semantic similarity alone.

DESIGN
------
Query     : anonymized acquirer profile (industry + item1_text + mda_text)
            Identical to Script 25 / pipeline build_acquirer_profile()
Corpus    : 50 candidates per deal from candidate_list field
Condition B: full candidate text (Industry + Market Cap + Description)
Condition C: metadata only (Industry + Market Cap, no Description)

MODELS 
-----------------------
  openai/text-embedding-3-large   — 3072 dims, 
  qwen/qwen3-embedding-8b         — 4096 dims, 

METRICS
-------
Identical to Script 25 / Script 17:
  MRR (primary), Recall@1, Recall@5, Recall@10, NDCG@5, NDCG@10
  Bootstrap 95% CI (n=1000) on MRR and Recall@10

OUTPUT  (all _v2 suffixes — Script 25 originals never overwritten)
------
  results/evaluation/
    embedding_metrics_raw_v2.csv     per-deal metrics, both models × conditions
    embedding_metrics_agg_v2.csv     aggregated with CI
    baseline_vs_llm_v2.csv           full panel: Random+TF-IDF+BM25+Emb+LLM
    latex/table_baseline_v2.tex      LaTeX table for paper

"""

import os
import re
import html
import json
import math
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("ERROR: openai not installed. Run: pip install openai")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")

# Canonical library — SAME query builder, MRR@10 metric and provider routing.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import profile as _profile
from lib import metrics as _metrics
K = _metrics.DEFAULT_K

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "candidate_corpus" / "corpus_wide_final.json"
EVAL_DIR    = BASE_DIR / "results" / "evaluation"

# Script 25 outputs (read-only — merged into panel table)
S25_AGG_PATH    = EVAL_DIR / "baseline_metrics_agg.csv"
LLM_AGG_BC_PATH = EVAL_DIR / "metrics_agg_BC.csv"

# Script 29 outputs (_v2 — never overwrite Script 25 files)
RAW_OUT = EVAL_DIR / "embedding_metrics_raw_v2.csv"
AGG_OUT = EVAL_DIR / "embedding_metrics_agg_v2.csv"
CMP_OUT = EVAL_DIR / "baseline_vs_llm_v2.csv"
TEX_OUT = EVAL_DIR / "latex" / "table_baseline_v2.tex"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
(EVAL_DIR / "latex").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
EXPECTED_DEALS = 286
N_BOOTSTRAP    = 1000
BOOTSTRAP_CI   = 0.95
RANDOM_SEED    = 42
BATCH_SIZE     = 20   # texts per API call — conservative for rate limits
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5    # seconds between retries

# Model definitions
MODELS = {
    "openai": {
        "model_id":   "openai/text-embedding-3-large",
        "label":      "text-embedding-3-large",
        "short":      "OpenAI-Emb-3-Large",
        "dims":       3072,
        "cost_per_m": 0.13,
    },
    "qwen": {
        "model_id":   "qwen/qwen3-embedding-8b",
        "label":      "Qwen3-Embedding-8B",
        "short":      "Qwen3-Emb-8B",
        "dims":       4096,
        "cost_per_m": 0.01,
    },
    # Secondary dense baseline (different family from OpenAI) — replaces the
    # qwen3-embedding-8b endpoint, which responds at ~20s/call on OpenRouter and
    # is infeasible to run at corpus scale. BGE-M3 is a distinct open-source
    # architecture, so it tests whether the semantic-matching result generalizes
    # beyond a single provider. Re-run under Fix A (identical query fields).
    "bge": {
        "model_id":   "baai/bge-m3",
        "label":      "BGE-M3",
        "short":      "BGE-M3",
        "dims":       1024,
        "cost_per_m": 0.01,
    },
}


# ── OpenRouter client ──────────────────────────────────────────────────────

def get_client() -> "OpenAI":
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# ── Text preprocessing (identical to Script 25) ────────────────────────────

def preprocess(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Candidate parsing (identical to Script 25) ─────────────────────────────

def parse_candidates(candidate_list: str, strip_desc: bool = False) -> list[str]:
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


def build_query(deal: dict) -> str:
    """Canonical query — SAME masked fields the LLM sees (audit Finding 7)."""
    return _profile.build_query_text(deal)


# ── Metrics (CANONICAL library — MRR@10, identical to LLM; audit Finding 6) ──

def compute_metrics(ranked_indices: list[int], target_idx: int,
                    k_values: tuple = (1, 5, 10)) -> dict:
    ranked = [i + 1 for i in ranked_indices]   # 0-indexed positions -> 1-indexed IDs
    target = target_idx + 1
    metrics = {
        "mrr":      _metrics.mrr_at_k(ranked, target, K),
        "rank":     _metrics.target_rank(ranked, target, K),
        "recall_1": _metrics.recall_at_k(ranked, target, 1),
        "recall_5": _metrics.recall_at_k(ranked, target, 5),
        "recall_10": _metrics.recall_at_k(ranked, target, 10),
        "ndcg_5":   _metrics.ndcg_at_k(ranked, target, 5),
        "ndcg_10":  _metrics.ndcg_at_k(ranked, target, 10),
    }
    return metrics


def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP,
                 ci: float = BOOTSTRAP_CI, seed: int = RANDOM_SEED) -> tuple[float, float]:
    rng   = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    lo    = np.percentile(means, (1 - ci) / 2 * 100)
    hi    = np.percentile(means, (1 + ci) / 2 * 100)
    return round(lo, 4), round(hi, 4)


def random_baseline_mrr(n_candidates: int = 50) -> float:
    """Chance MRR@10 = H_10 / n (audit Finding 6), NOT full-list H_n / n."""
    return _metrics.chance_mrr_at_k(K, n_candidates)


# ── Embedding API calls ────────────────────────────────────────────────────

def embed_batch(client: "OpenAI", texts: list[str], model_id: str) -> list[list[float]]:
    """
    Embed a batch of texts via OpenRouter API with retry logic.
    Returns list of embedding vectors.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.embeddings.create(
                model=model_id,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                log.warning(f"Embedding API error (attempt {attempt+1}/{RETRY_ATTEMPTS}): {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


# ── Persistent embedding cache (resumability) ──────────────────────────────
# A run can be interrupted (long network job); a per-text on-disk cache keyed by
# (model_id, sha1(text)) lets a restart reuse already-fetched vectors and only
# request the missing ones. The cache is byte-for-byte reproducible and does not
# change any metric — it only avoids re-billing identical embedding requests.
import hashlib as _hashlib

_EMB_CACHE_DIR = BASE_DIR / "results" / "evaluation" / "_embedding_cache"


def _cache_path(model_id: str, text: str) -> Path:
    slug = model_id.replace("/", "_")
    h = _hashlib.sha1(text.encode("utf-8")).hexdigest()
    d = _EMB_CACHE_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.npy"


def embed_texts(client: "OpenAI", texts: list[str], model_id: str,
                batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Embed all texts in batches, using a persistent on-disk cache so the job is
    resumable across restarts. Returns (n_texts, dims) numpy array in input order.
    """
    out: list[np.ndarray | None] = [None] * len(texts)
    missing_idx: list[int] = []

    # 1. serve from cache
    for i, t in enumerate(texts):
        p = _cache_path(model_id, t)
        if p.exists():
            try:
                out[i] = np.load(p)
                continue
            except Exception:
                pass
        missing_idx.append(i)

    # 2. fetch the misses in batches and persist each vector
    for b in range(0, len(missing_idx), batch_size):
        idx_batch = missing_idx[b:b + batch_size]
        batch = [texts[i] for i in idx_batch]
        vecs  = embed_batch(client, batch, model_id)
        for i, v in zip(idx_batch, vecs):
            arr = np.asarray(v, dtype=np.float32)
            out[i] = arr
            try:
                np.save(_cache_path(model_id, texts[i]), arr)
            except Exception:
                pass
        if b + batch_size < len(missing_idx):
            time.sleep(0.1)

    return np.array([o for o in out])


# ── Ranking ────────────────────────────────────────────────────────────────

def rank_by_embedding(query_vec: np.ndarray,
                      cand_vecs: np.ndarray) -> list[int]:
    """
    Rank candidates by cosine similarity to query.
    Returns indices sorted by similarity descending (rank 1 first).
    """
    # query_vec: (1, dims) or (dims,) — reshape to (1, dims)
    q = query_vec.reshape(1, -1)
    sims = cosine_similarity(q, cand_vecs).flatten()
    return list(np.argsort(sims)[::-1])


# ── Per-deal evaluation ────────────────────────────────────────────────────

def evaluate_deal(client: "OpenAI", deal: dict, model_id: str,
                  conditions: list[str]) -> list[dict]:
    """
    Evaluate one deal for all requested conditions with a single model.

    Embeds query once, then embeds candidates for each condition separately
    (B has descriptions, C does not — different texts, different embeddings).

    Returns list of result dicts (one per condition).
    """
    deal_id    = deal["deal_id"]
    target_pos = deal.get("target_position")
    cand_list  = deal.get("candidate_list", "")

    if target_pos is None or not cand_list:
        log.warning(f"Skipping {deal_id}: missing target_position or candidate_list")
        return []

    target_idx = target_pos - 1  # corpus is 1-indexed
    query      = preprocess(build_query(deal))

    results = []

    for cond in conditions:
        strip      = (cond == "C")
        candidates = parse_candidates(cand_list, strip_desc=strip)

        if not candidates:
            log.warning(f"Skipping {deal_id} cond={cond}: no candidates parsed")
            continue

        clean_candidates = [preprocess(c) for c in candidates]

        # Embed query + all candidates in one batch call when possible
        all_texts = [query] + clean_candidates
        try:
            all_vecs = embed_texts(client, all_texts, model_id)
        except Exception as e:
            log.error(f"Embedding failed for {deal_id} cond={cond}: {e}")
            continue

        query_vec = all_vecs[0]
        cand_vecs = all_vecs[1:]

        ranked = rank_by_embedding(query_vec, cand_vecs)
        m      = compute_metrics(ranked, target_idx)

        results.append({
            "deal_id":      deal_id,
            "condition":    cond,
            "method":       model_id.split("/")[-1],   # short name for CSV
            "model_id":     model_id,
            "industry":     deal.get("industry"),
            "deal_year":    deal.get("deal_year"),
            "n_candidates": len(candidates),
            **m,
        })

    return results


# ── Corpus evaluation ──────────────────────────────────────────────────────

def evaluate_corpus(corpus: list[dict], model_cfg: dict,
                    conditions: list[str],
                    dry_run: bool = False) -> pd.DataFrame:
    """
    Run embedding baseline on all deals for one model.
    dry_run=True processes only first 3 deals for testing.
    """
    client   = get_client()
    model_id = model_cfg["model_id"]
    label    = model_cfg["label"]

    sample = corpus[:3] if dry_run else corpus
    n      = len(sample)

    log.info(f"Model: {label} ({model_id})")
    log.info(f"Deals: {n} | Conditions: {conditions}")
    if dry_run:
        log.info("DRY RUN — processing 3 deals only")

    rows = []
    for i, deal in enumerate(sample, 1):
        if i % 25 == 0 or i == n:
            log.info(f"  Progress: {i}/{n} deals")
        deal_rows = evaluate_deal(client, deal, model_id, conditions)
        rows.extend(deal_rows)

    return pd.DataFrame(rows)


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate_metrics(df_raw: pd.DataFrame,
                      run_bootstrap: bool = True) -> pd.DataFrame:
    rows = []
    for (method, cond), grp in df_raw.groupby(["method", "condition"]):
        mrr_arr = grp["mrr"].values
        r10_arr = grp["recall_10"].values

        if run_bootstrap and len(mrr_arr) > 1:
            mrr_ci_lo, mrr_ci_hi = bootstrap_ci(mrr_arr)
            r10_ci_lo, r10_ci_hi = bootstrap_ci(r10_arr)
        else:
            mrr_ci_lo = mrr_ci_hi = r10_ci_lo = r10_ci_hi = None

        rows.append({
            "method":          method,
            "condition":       cond,
            "model_id":        grp["model_id"].iloc[0],
            "n_deals":         len(grp),
            "mrr":             round(mrr_arr.mean(), 4),
            "mrr_ci_lo":       mrr_ci_lo,
            "mrr_ci_hi":       mrr_ci_hi,
            "recall_1":        round(grp["recall_1"].mean(), 4),
            "recall_5":        round(grp["recall_5"].mean(), 4),
            "recall_10":       round(r10_arr.mean(), 4),
            "recall_10_ci_lo": r10_ci_lo,
            "recall_10_ci_hi": r10_ci_hi,
            "ndcg_5":          round(grp["ndcg_5"].mean(), 4),
            "ndcg_10":         round(grp["ndcg_10"].mean(), 4),
        })

    return pd.DataFrame(rows)


# ── Full panel comparison table ────────────────────────────────────────────

def build_panel_table(df_emb_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all baselines into one panel DataFrame:
      Random | TF-IDF | BM25 | text-embedding-3-large | Qwen3-Emb-8B | LLM-best
    """
    rows = []

    # 1. Random baseline
    rnd_mrr = random_baseline_mrr(50)
    chance = _metrics.chance_baseline(K, 50)
    for cond in ["B", "C"]:
        rows.append({
            "method":          "Random",
            "condition":       cond,
            "n_deals":         EXPECTED_DEALS,
            "mrr":             round(rnd_mrr, 4),
            "mrr_ci_lo":       None, "mrr_ci_hi": None,
            "recall_1":        chance["chance_recall_1"], "recall_5": chance["chance_recall_5"],
            "recall_10":       chance["chance_recall_10"],
            "recall_10_ci_lo": None, "recall_10_ci_hi": None,
            "ndcg_5":          None, "ndcg_10": None,
        })

    # 2. TF-IDF and BM25 from Script 25
    if S25_AGG_PATH.exists():
        s25 = pd.read_csv(S25_AGG_PATH)
        for _, r in s25.iterrows():
            rows.append(r.to_dict())
        log.info(f"Loaded Script 25 baselines from {S25_AGG_PATH}")
    else:
        log.warning(f"Script 25 output not found: {S25_AGG_PATH}")
        log.warning("Run script 25 first to include BM25/TF-IDF in the panel.")

    # 3. Embedding baselines (this script)
    for _, r in df_emb_agg.iterrows():
        rows.append(r.to_dict())

    # 4. Best LLM per condition from Script 17
    if LLM_AGG_BC_PATH.exists():
        llm = pd.read_csv(LLM_AGG_BC_PATH)
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
        for _, r in best_llm.iterrows():
            rows.append({
                "method":          f"LLM-best ({r['model_key']})",
                "condition":       r["condition"],
                "n_deals":         int(r.get("n_deals", EXPECTED_DEALS)),
                "mrr":             round(r["mrr"], 4),
                "mrr_ci_lo":       r.get("mrr_ci_lo"),
                "mrr_ci_hi":       r.get("mrr_ci_hi"),
                "recall_1":        round(r["recall_1"], 4),
                "recall_5":        round(r["recall_5"], 4),
                "recall_10":       round(r["recall_10"], 4),
                "recall_10_ci_lo": r.get("recall_10_ci_lo"),
                "recall_10_ci_hi": r.get("recall_10_ci_hi"),
                "ndcg_5":          round(r["ndcg_5"], 4),
                "ndcg_10":         round(r["ndcg_10"], 4),
            })
        log.info(f"Loaded LLM results from {LLM_AGG_BC_PATH}")
    else:
        log.warning(f"LLM results not found: {LLM_AGG_BC_PATH}")

    return pd.DataFrame(rows)


# ── LaTeX export ───────────────────────────────────────────────────────────

def export_latex(df_panel: pd.DataFrame, out_path: Path) -> None:
    """
    Export full baseline panel as LaTeX table.

    Method order: Random → TF-IDF → BM25 → OpenAI-Emb → Qwen3-Emb → LLM-best
    Embedding rows highlighted with \textbf.
    """

    def fmt(v) -> str:
        return f"{v:.3f}" if pd.notna(v) and v is not None else "--"

    def ci_str(lo, hi) -> str:
        return f"[{fmt(lo)}, {fmt(hi)}]" if pd.notna(lo) and lo is not None else "--"

    # Build ordered method list
    emb_methods = [m for m in df_panel["method"].unique()
                   if m not in ("Random", "TF-IDF", "BM25")
                   and not m.startswith("LLM")]
    llm_methods = sorted(
        [m for m in df_panel["method"].unique() if m.startswith("LLM")],
        key=lambda x: -df_panel.loc[df_panel["method"] == x, "mrr"].max()
    )
    method_order = ["Random", "TF-IDF", "BM25"] + emb_methods + llm_methods

    lines = [
        f"% Full baseline panel — generated by 29_embedding_baseline.py on "
        f"{datetime.now().strftime('%Y-%m-%d')}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (r"\caption{Retrieval baseline panel. Lexical baselines (TF-IDF, BM25) "
         r"and dense embedding baselines (\textit{text-embedding-3-large}, "
         r"\textit{Qwen3-Embedding-8B}) are compared against the best LLM per condition. "
         r"Condition~B includes candidate descriptions; Condition~C uses metadata only. "
         r"Random: theoretical MRR for $n{=}50$ candidates. "
         r"Bootstrap 95\% CI on MRR.}"),
        r"\label{tab:baseline_panel}",
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        (r"\textbf{Method} & \textbf{Cond.} & \textbf{MRR} & \textbf{95\% CI} "
         r"& \textbf{R@1} & \textbf{R@10} & \textbf{NDCG@10} \\"),
        r"\midrule",
    ]

    embedding_methods = set(emb_methods)

    for cond in ["B", "C"]:
        sub = df_panel[df_panel["condition"] == cond]
        for method in method_order:
            row = sub[sub["method"] == method]
            if row.empty:
                continue
            r = row.iloc[0]

            mrr_s  = fmt(r["mrr"])
            ci_s   = ci_str(r.get("mrr_ci_lo"), r.get("mrr_ci_hi"))
            r1_s   = fmt(r.get("recall_1"))
            r10_s  = fmt(r.get("recall_10"))
            ndcg_s = fmt(r.get("ndcg_10"))

            # Bold embedding rows to highlight new neural baselines
            if method in embedding_methods:
                lines.append(
                    rf"\textbf{{{method}}} & {cond} & \textbf{{{mrr_s}}} & {ci_s} "
                    rf"& {r1_s} & {r10_s} & {ndcg_s} \\"
                )
            else:
                lines.append(
                    f"{method} & {cond} & {mrr_s} & {ci_s} "
                    f"& {r1_s} & {r10_s} & {ndcg_s} \\\\"
                )

        if cond == "B":
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"LaTeX table saved: {out_path}")


# ── Console summary ────────────────────────────────────────────────────────

def print_summary(df_emb_agg: pd.DataFrame, df_panel: pd.DataFrame) -> None:
    rnd = random_baseline_mrr()

    print("\n" + "=" * 75)
    print("EMBEDDING BASELINE RESULTS")
    print("=" * 75)
    print(f"{'Method':<35} {'Cond':>5} {'MRR':>8} {'vs Random':>10} "
          f"{'R@1':>8} {'R@10':>8} {'NDCG@10':>9}")
    print("-" * 75)
    for _, row in df_emb_agg.sort_values(["condition", "mrr"],
                                          ascending=[True, False]).iterrows():
        delta = row["mrr"] - rnd
        sign  = "+" if delta >= 0 else ""
        print(f"{row['method']:<35} {row['condition']:>5} {row['mrr']:>8.3f} "
              f"{sign}{delta:>8.3f}  {row['recall_1']:>8.3f} "
              f"{row['recall_10']:>8.3f} {row['ndcg_10']:>9.3f}")

    print("\n" + "=" * 75)
    print("FULL BASELINE PANEL")
    print("=" * 75)

    method_order = ["Random", "TF-IDF", "BM25"]
    emb_methods  = [m for m in df_panel["method"].unique()
                    if m not in ("Random", "TF-IDF", "BM25")
                    and not m.startswith("LLM")]
    llm_methods  = [m for m in df_panel["method"].unique() if m.startswith("LLM")]
    method_order += emb_methods + sorted(llm_methods)

    for cond in ["B", "C"]:
        sub = df_panel[df_panel["condition"] == cond]
        print(f"\n  --- Condition {cond} ---")
        for method in method_order:
            row = sub[sub["method"] == method]
            if row.empty:
                continue
            r    = row.iloc[0]
            lo   = f"{r['mrr_ci_lo']:.3f}" if pd.notna(r.get("mrr_ci_lo")) else "  --"
            hi   = f"{r['mrr_ci_hi']:.3f}" if pd.notna(r.get("mrr_ci_hi")) else "  --"
            r1   = f"{r['recall_1']:.3f}"  if pd.notna(r.get("recall_1"))  else "   --"
            nd   = f"{r['ndcg_10']:.3f}"   if pd.notna(r.get("ndcg_10"))   else "   --"
            star = " *" if method in emb_methods else "  "
            print(f"{star}{method:<33} {cond:>5} {r['mrr']:>8.3f} "
                  f"{lo:>7} {hi:>7} {r1:>8} {nd:>9}")

    print("\n" + "=" * 75)
    print(f"Random baseline MRR: {rnd:.4f}")
    print(f"\nOutput files:")
    for p in [RAW_OUT, AGG_OUT, CMP_OUT, TEX_OUT]:
        print(f"  {p}")


# ── Main ───────────────────────────────────────────────────────────────────

def _write_llm_minus_embedding(df_emb_raw: pd.DataFrame) -> None:
    """
    Paired LLM - embedding MRR@10 contrast on the SAME deals/candidates, per
    condition, clustered by acquirer (audit RQ4 finding: the embedding comparison
    must be paired and evaluated on identical deals, not aggregate-vs-aggregate).

    Reads the LLM per-deal metrics (metrics_raw_BC.csv) produced by script 17.
    For each (embedding model, condition) it inner-joins on deal_id with EACH LLM
    (model,strategy) cell and reports the paired mean difference, acquirer-
    clustered 95% CI and bootstrap p-value.
    """
    from lib import metrics as _m
    llm_path = EVAL_DIR / "metrics_raw_BC.csv"
    if not llm_path.exists():
        log.info("metrics_raw_BC.csv not found — run script 17 first for the paired contrast")
        return
    llm = pd.read_csv(llm_path)
    rows = []
    for (emb_model, cond), egrp in df_emb_raw.groupby(["method", "condition"]):
        e = egrp[["deal_id", "mrr"]].rename(columns={"mrr": "mrr_emb"})
        lc = llm[llm["condition"] == cond]
        for (lm, strat), lgrp in lc.groupby(["model_key", "strategy"]):
            merged = lgrp[["deal_id", "acquirer_ticker", "mrr"]].merge(e, on="deal_id")
            if merged.empty:
                continue
            diff = (merged["mrr"] - merged["mrr_emb"]).to_numpy()
            clusters = merged["acquirer_ticker"].fillna(merged["deal_id"]).to_numpy()
            b = _m.paired_bootstrap(diff, clusters=clusters)
            rows.append({
                "llm_model": lm, "llm_strategy": strat,
                "embedding_model": emb_model, "condition": cond,
                "n_pairs": b["n"], "n_acquirers": b.get("n_clusters"),
                "mrr_llm": round(merged["mrr"].mean(), 4),
                "mrr_embedding": round(merged["mrr_emb"].mean(), 4),
                "delta_mrr": b["mean_diff"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                "p_value": b["p_value"],
            })
    if rows:
        out = EVAL_DIR / "llm_minus_embedding_v2.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        log.info("Paired LLM-embedding contrast saved: %s (%d rows)", out, len(rows))


def main():
    parser = argparse.ArgumentParser(
        description="Dense embedding baseline for M&A candidate ranking via OpenRouter"
    )
    parser.add_argument(
        "--condition", nargs="+", choices=["B", "C"], default=["B", "C"],
        help="Conditions to evaluate (default: B C)"
    )
    parser.add_argument(
        "--models", nargs="+", choices=["openai", "qwen", "bge"], default=["openai", "bge"],
        help="Embedding models to run (default: openai + bge; qwen deprecated — ~20s/call)"
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip bootstrap CI (faster)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only 3 deals (API test, minimal cost)"
    )
    args = parser.parse_args()

    if not OPENAI_AVAILABLE:
        raise SystemExit("openai not installed. Run: pip install openai")
    if not SKLEARN_AVAILABLE:
        raise SystemExit("scikit-learn not installed. Run: pip install scikit-learn")

    # Load corpus
    log.info(f"Loading corpus: {CORPUS_PATH}")
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    log.info(f"Corpus loaded: {len(corpus)} deals")

    if len(corpus) != EXPECTED_DEALS and not args.dry_run:
        log.warning(f"Expected {EXPECTED_DEALS} deals, found {len(corpus)}")

    # Cost estimate
    n_deals  = 3 if args.dry_run else len(corpus)
    n_texts  = n_deals * len(args.condition) * 51  # 1 query + 50 candidates
    est_toks = n_texts * 200  # ~200 tokens per text
    log.info(f"Estimated API calls: {n_texts:,} texts (~{est_toks/1e6:.1f}M tokens)")
    for key in args.models:
        cfg  = MODELS[key]
        cost = (est_toks / 1e6) * cfg["cost_per_m"]
        log.info(f"  {cfg['label']}: ~${cost:.3f}")

    # Run each model
    all_raw = []
    for model_key in args.models:
        cfg = MODELS[model_key]
        log.info(f"\n{'='*60}")
        log.info(f"Running: {cfg['label']}")
        log.info(f"{'='*60}")
        df_raw = evaluate_corpus(corpus, cfg, args.condition, dry_run=args.dry_run)
        all_raw.append(df_raw)
        log.info(f"Completed: {cfg['label']} — {len(df_raw)} rows")

    # Combine raw results from all models
    df_all_raw = pd.concat(all_raw, ignore_index=True)
    df_all_raw.to_csv(RAW_OUT, index=False)
    log.info(f"\nRaw metrics saved: {RAW_OUT} ({len(df_all_raw)} rows)")

    # Paired, acquirer-clustered LLM - embedding contrast (RQ4; audit finding).
    try:
        _write_llm_minus_embedding(df_all_raw)
    except Exception as e:
        log.warning("LLM-minus-embedding contrast skipped: %s", e)

    # Aggregate
    run_bs = not args.no_bootstrap
    if run_bs:
        log.info("Computing bootstrap CIs (n=1000)...")
    df_agg = aggregate_metrics(df_all_raw, run_bootstrap=run_bs)
    df_agg.to_csv(AGG_OUT, index=False)
    log.info(f"Aggregated metrics saved: {AGG_OUT}")

    # Full panel
    df_panel = build_panel_table(df_agg)
    df_panel.to_csv(CMP_OUT, index=False)
    log.info(f"Full panel saved: {CMP_OUT}")

    # LaTeX
    export_latex(df_panel, TEX_OUT)

    # Summary
    print_summary(df_agg, df_panel)

    if args.dry_run:
        print("\nDRY RUN completed (3 deals). Re-run without --dry-run for full corpus.")


if __name__ == "__main__":
    main()