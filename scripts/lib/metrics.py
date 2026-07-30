"""
lib.metrics — one metric definition for every method.
=====================================================

Repairs audit Finding 6 (LLM used MRR@10 while baselines used full MRR, and the
random comparator was the wrong harmonic number) and the "paired inference is
missing" finding.

Key decisions
-------------
  * MRR@10 IS THE CONTRACT. Every method — LLM, BM25, TF-IDF, embeddings,
    random — is scored with `mrr_at_k(ranked, target, k=10)`: reciprocal rank
    only if the target is in the top-10, else 0. Baselines that internally rank
    all 50 candidates MUST truncate to 10 before scoring (see `truncate_at_k`).

  * CHANCE = H_10 / 50 ≈ 0.0586, not H_50 / 50 ≈ 0.090. `chance_mrr_at_k`
    computes the exact expectation of MRR@k for a single relevant item uniformly
    placed among n candidates:  (1/n) * sum_{r=1..k} 1/r  =  H_k / n.

  * PAIRED, ACQUIRER-CLUSTERED BOOTSTRAP for contrasts (B−C, LLM−embedding).
    Contrasts are computed per deal, then resampled by CLUSTER (acquirer) so
    repeated acquirers do not inflate significance.
"""

from __future__ import annotations

import math
import numpy as np

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
CI = 0.95
DEFAULT_K = 10          # the reporting window: MRR@10
N_CANDIDATES = 50


# ---------------------------------------------------------------------------
# Per-deal metrics — all @k
# ---------------------------------------------------------------------------

def truncate_at_k(ranked: list[int], k: int = DEFAULT_K) -> list[int]:
    """Truncate a full ranking to the top-k window used for scoring."""
    return list(ranked)[:k]


def mrr_at_k(ranked: list[int], target: int | None, k: int = DEFAULT_K) -> float:
    """Reciprocal rank if target is in the top-k, else 0. This is MRR@k."""
    if not ranked or target is None:
        return 0.0
    top = ranked[:k]
    try:
        return 1.0 / (top.index(target) + 1)
    except ValueError:
        return 0.0


def recall_at_k(ranked: list[int], target: int | None, k: int) -> float:
    if not ranked or target is None:
        return 0.0
    return float(target in ranked[:k])


def ndcg_at_k(ranked: list[int], target: int | None, k: int) -> float:
    """Binary-relevance NDCG@k with a single relevant item (IDCG = 1)."""
    if not ranked or target is None:
        return 0.0
    for i, cand in enumerate(ranked[:k]):
        if cand == target:
            return 1.0 / math.log2(i + 2)
    return 0.0


def target_rank(ranked: list[int], target: int | None, k: int = DEFAULT_K) -> int | None:
    """1-indexed rank within the top-k window, or None if outside (censored)."""
    if not ranked or target is None:
        return None
    top = ranked[:k]
    return top.index(target) + 1 if target in top else None


# ---------------------------------------------------------------------------
# Chance baseline — correct for the @k window
# ---------------------------------------------------------------------------

def harmonic(n: int) -> float:
    return sum(1.0 / i for i in range(1, n + 1))


def chance_mrr_at_k(k: int = DEFAULT_K, n: int = N_CANDIDATES) -> float:
    """
    Expected MRR@k for a single relevant item uniformly placed among n candidates.

      E[MRR@k] = sum_{r=1..k} P(rank=r) * (1/r) = (1/n) * H_k

    For k=10, n=50 this is ~0.0586 (audit Finding 6), NOT H_50/50 ~ 0.090.
    """
    return harmonic(k) / n


def chance_recall_at_k(k: int, n: int = N_CANDIDATES) -> float:
    return k / n


def chance_baseline(k: int = DEFAULT_K, n: int = N_CANDIDATES) -> dict:
    return {
        "k": k,
        "n_candidates": n,
        "chance_mrr_at_k": round(chance_mrr_at_k(k, n), 4),
        "chance_recall_1": round(chance_recall_at_k(1, n), 4),
        "chance_recall_5": round(chance_recall_at_k(5, n), 4),
        "chance_recall_10": round(chance_recall_at_k(10, n), 4),
    }


# ---------------------------------------------------------------------------
# Bootstrap — simple and paired/clustered
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_resamples: int = N_BOOTSTRAP,
                 ci: float = CI, seed: int = RANDOM_SEED) -> tuple[float, float]:
    """Non-parametric bootstrap CI for the mean of a single sample."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(n_resamples)]
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, (1 + ci) / 2))
    return (round(lo, 4), round(hi, 4))


def paired_bootstrap(diffs, clusters=None,
                     n_resamples: int = N_BOOTSTRAP, ci: float = CI,
                     seed: int = RANDOM_SEED) -> dict:
    """
    Paired bootstrap on per-deal differences (e.g. mrr_B - mrr_C for the same
    deal). If `clusters` (e.g. acquirer id per deal) is given, resamples CLUSTERS
    with replacement so repeated acquirers do not violate independence.

    Returns mean difference, CI, and a bootstrap two-sided p-value for H0: mean=0.
    """
    d = np.asarray(list(diffs), dtype=float)
    if d.size == 0:
        return {"mean_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_value": 1.0, "n": 0}

    rng = np.random.default_rng(seed)
    mean_diff = float(d.mean())

    if clusters is None:
        boot = np.array([d[rng.integers(0, d.size, d.size)].mean()
                         for _ in range(n_resamples)])
    else:
        clusters = np.asarray(list(clusters))
        uniq = np.unique(clusters)
        groups = {c: d[clusters == c] for c in uniq}
        boot = np.empty(n_resamples)
        for b in range(n_resamples):
            picks = uniq[rng.integers(0, uniq.size, uniq.size)]
            vals = np.concatenate([groups[c] for c in picks])
            boot[b] = vals.mean()

    lo = float(np.quantile(boot, (1 - ci) / 2))
    hi = float(np.quantile(boot, (1 + ci) / 2))
    # Two-sided bootstrap p-value: proportion of resamples on the far side of 0,
    # centred on the observed mean (basic bootstrap inversion).
    centered = boot - mean_diff
    p = 2.0 * min(np.mean(centered >= mean_diff), np.mean(centered <= mean_diff))
    p = float(min(1.0, p))
    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "p_value": round(p, 4),
        "n": int(d.size),
        "n_clusters": int(np.unique(clusters).size) if clusters is not None else int(d.size),
    }
