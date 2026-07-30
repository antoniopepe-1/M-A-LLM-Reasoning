"""
28_memorization_dataset_check_v2.py
=====================================
Dataset Memorization Test - v2.

SCOPE (audit finding): this is a SUPPLEMENTARY development-data contamination
check, NOT primary evidence. A negative result (probes fail) is WEAK evidence
that the assembled corpus was not memorized - it does not prove non-memorization.
Use multiple held-out records and negative controls; interpret only as "no
contamination detected by these probes". The archived Script 26 is obsolete and
must not be cited.

Output files:
  results/memorization/dataset_raw_v2.jsonl
  results/memorization/dataset_summary_v2.csv
  results/memorization/dataset_report_v2.txt

Historical Script 26 outputs, if available, can be compared with the v2 outputs
generated here.

CONCEPTUAL DISTINCTION
----------------------
Script 19, Section 2 (Deal Recall):
  Question: "Given acquirer + year, do you know the target?"
  Tests   : whether the model knows the deals as *facts about the world*
  Possible source: news, Bloomberg, Wikipedia, public SEC EDGAR filings
  -> Knowledge memorization

This script (Dataset Memorization):
  Question: "Have you seen the specific dataset assembled from Capital IQ?"
  Tests   : whether the corpus itself was in the model's training set
  Source  : direct exposure to the file only (not reconstructible from news)
  -> Dataset memorization

THREE PROBE TYPES
-----------------
P1 - Item recall probe  [updated: replaces the original count probe]
  Few-shot role-play places the model in the position of "being" the dataset.
  Given a DealID, it must return the corresponding acquirer name. Ground truth:
  acquirer_name of the record at ITEM_RECALL_QUERY_IDX.
  Scoring: token-F1 > 0.30 -> suspicious.

  Why this replaced the count probe: asking "how many items does this S&P 500
  dataset contain?" elicited the answer 500 by reasoning about the name (an
  upper-bound fallacy) rather than by corpus memorization, so the probe was not
  corpus-specific. The recall probe instead requires the exact company name at a
  specific position - information that cannot be inferred without having seen
  the corpus.

P2 - Completion probe
  "Given this dataset record [acquirer, year, target], what is the next record?"
  Ground truth: the actual record at position i+1 in the corpus.
  Scoring: token-F1 between the response and the gold string
  (acquirer + target + year). Threshold: F1 > 0.30 -> suspicious.

P3 - Enumeration probe
  "List the tickers of the S&P 500 acquirers in a Capital IQ M&A dataset,
   2015-2022."
  Ground truth: the set of 286 tickers in the corpus.
  Scoring: Jaccard similarity between predicted tickers and the gold set.
  Threshold: Jaccard > 0.15 -> suspicious (a low threshold, because the full
  list is long and even a partial match is informative).

INTERPRETING THE OUTCOMES
-------------------------
A model is flagged as "dataset-memorized" if at least 2 of 3 probes are
suspicious.

  Knowledge HIGH + Dataset LOW  -> defensible case
    The model knows the deals from news coverage, not from the corpus.
    Performance in B/C reflects reasoning rather than corpus recall.

  Knowledge HIGH + Dataset HIGH -> problematic case
    Possible corpus leakage; build a synthetic dataset and repeat the
    analysis on that subset.

  Knowledge LOW + Dataset LOW   -> clean case (small models)
    Results in B/C stem from reasoning alone.

OUTPUT
------
  results/memorization/
    dataset_raw_{slug}.jsonl     raw per-model probe results
    dataset_summary.csv          aggregated scores per model
    dataset_report.txt           paper-ready interpretation
"""

import re
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────
_env_path = next(
    (p / ".env"
     for p in [Path(__file__).resolve().parent,
               Path(__file__).resolve().parent.parent]
     if (p / ".env").exists()),
    None,
)
load_dotenv(dotenv_path=_env_path)

BASE_DIR    = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "candidate_corpus" / "corpus_wide_final.json"
OUT_DIR     = BASE_DIR / "results" / "memorization"
LOG_DIR     = BASE_DIR / "logs"

# ── v2 output filenames (never overwrite archived v1 results) ─────────────
JSONL_FILE   = "dataset_raw_v2.jsonl"
SUMMARY_FILE = "dataset_summary_v2.csv"
REPORT_FILE  = "dataset_report_v2.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"dataset_memo_{datetime.now():%Y%m%d_%H%M%S}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Model roster — identico a script 19 e 16c_v3 ─────────────────────────
MODELS = {
    "llama32_1b": {
        "model_id":     "meta-llama/llama-3.2-1b-instruct",
        "slug":         "llama32_1b",
        "extra_params": {},
    },
    "qwen3_8b": {
        "model_id":     "qwen/qwen3-8b-04-28",
        "slug":         "qwen3_8b",
        "extra_params": {"thinking": {"type": "disabled"}},
    },
    "phi4": {
        "model_id":     "microsoft/phi-4",
        "slug":         "phi4_14b",
        "extra_params": {},
    },
    "llama33_70b": {
        "model_id":     "meta-llama/llama-3.3-70b-instruct",
        "slug":         "llama33_70b",
        "extra_params": {},
    },
    "qwen3": {
        "model_id":     "qwen/qwen3-235b-a22b-2507",
        "slug":         "qwen3_235b",
        "extra_params": {"thinking": {"type": "disabled"}},
    },
    "deepseek": {
        "model_id":     "deepseek/deepseek-chat-v3-0324",
        "slug":         "deepseek_v3",
        "extra_params": {},
    },
}

MAX_TOKENS = 300

# ── Ground truth constants (corpus-specific, not derivable from news) ────
GROUND_TRUTH_N_ACQUIRERS  = 286
GROUND_TRUTH_YEAR_RANGE   = "2015 and 2022"
GROUND_TRUTH_SOURCE       = "Capital IQ"
ITEM_RECALL_N_SHOTS       = 3    # few-shot examples mostrati al modello (record 0..2)
ITEM_RECALL_QUERY_IDX     = 10   # record usato come query (fuori dai few-shot)
ITEM_RECALL_F1_THRESHOLD  = 0.30
COMPLETION_F1_THRESHOLD   = 0.30
ENUMERATION_JAC_THRESHOLD = 0.15  # low threshold: a partial match is still informative


# ── Prompt templates ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer based only on your training knowledge. Be concise and precise."
)

# P1 — Item recall probe (LLM-MemoryInspector style)
# ---------------------------------------------------------
# The model is placed in the position of "being" the Capital IQ dataset.
# The system prompt deliberately does NOT mention "S&P 500", to stop the model
# from answering 500 by reasoning about the name rather than from corpus
# memorization (the upper-bound fallacy seen in the original count probe).
#
# Pattern identico a LLM-MemoryInspector (Anelli et al., SIGIR 2024):
#   system: "You are the {dataset} dataset. Given a key, return the value."
#   few-shot: user → "Input: {id}::"  /  assistant → "{id}::{value}"
#   query:    user → "Input: {query_id}::"
#
# Scoring: token-F1 between the generated acquirer and the gold acquirer (threshold 0.30).

ITEM_RECALL_SYSTEM = (
    "You are the Capital IQ M&A transactions dataset. "
    "When given a DealID, you respond with the exact acquirer company name "
    "recorded in the dataset for that deal. "
    "Only respond with the acquirer name itself. "
    "If the DealID is unknown, respond with exactly: UNKNOWN. "
    "Follow this pattern strictly. Let's think step by step."
)


def build_item_recall_messages(corpus: list) -> tuple[list, str]:
    """
    Costruisce la sequenza messages per P1 nel formato few-shot di
    LLM-MemoryInspector.

    Usa ITEM_RECALL_N_SHOTS record come esempi (indici 0..N_SHOTS-1),
    poi interroga il record in posizione ITEM_RECALL_QUERY_IDX.
    Restituisce (messages, gold_acquirer_name).
    """
    messages = [{"role": "system", "content": ITEM_RECALL_SYSTEM}]

    for deal in corpus[:ITEM_RECALL_N_SHOTS]:
        deal_id  = str(deal.get("deal_id", ""))
        acquirer = deal.get("acquirer_name", "").split("(")[0].strip()
        messages.append({"role": "user",      "content": f"Input: {deal_id}::"})
        messages.append({"role": "assistant", "content": f"{deal_id}::{acquirer}"})

    query_deal  = corpus[ITEM_RECALL_QUERY_IDX]
    query_id    = str(query_deal.get("deal_id", ""))
    gold        = query_deal.get("acquirer_name", "").split("(")[0].strip()
    messages.append({"role": "user", "content": f"Input: {query_id}::"})

    return messages, gold


# P2 — Completion probe
# Supplies the first corpus record in its actual order and asks for the next one.
# The ordering is deterministic (deal_id seed in script 14b) but never published.
COMPLETION_PROMPT = """\
The following is the first entry from an M&A transactions dataset \
built from {source}, covering S&P 500 acquirers for deals announced \
between {year_range}:

  Acquirer: {acquirer_name} | Year: {deal_year} | Target: {target_name}

What is the next entry in this dataset?
Reply in exactly this format: "Acquirer: ... | Year: ... | Target: ..."
If you don't know, respond with exactly: UNKNOWN\
"""

# P3 — Enumeration probe
# The full list of 286 tickers reflects specific methodological choices
# (inclusion criteria, S&P 500 thresholds, time window) that cannot be
# reconstructed from external sources without knowing the exact parameters.
ENUMERATION_PROMPT = """\
List the stock ticker symbols of the S&P 500 companies that appear \
as acquirers in a {source} M&A transactions dataset covering deals \
announced between {year_range}.

Output format: comma-separated ticker symbols only. No words, no sentences, no explanation.
Example output: AAPL,MSFT,JPM,XOM
If you have no information, output exactly: UNKNOWN\
"""


# ── API client — identico a script 19 ─────────────────────────────────────

def get_client() -> OpenAI:
    import os
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer":     "https://github.com/llm-ma-analyst",
            "X-Title":          "LLM-as-MA-Analyst-DatasetMemo",
            "X-Provider-Order": "Together,DeepInfra,Fireworks",
        },
    )


def call_model(
    client: OpenAI,
    model_cfg: dict,
    prompt: str | None = None,
    messages: list | None = None,
) -> tuple[str, str]:
    """
    Chiama il modello via OpenRouter.

    Accetta o un singolo `prompt` (stringa, wrappato con SYSTEM_PROMPT)
    oppure una lista `messages` pre-costruita (usata per P1 few-shot).
    """
    try:
        extra_params = dict(model_cfg.get("extra_params", {}))
        extra_body = {}
        if "thinking" in extra_params:
            extra_body["thinking"] = extra_params.pop("thinking")

        if messages is None:
            if prompt is None:
                raise ValueError("call_model: devi fornire prompt o messages")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]

        params = {
            "model":       model_cfg["model_id"],
            "messages":    messages,
            "temperature": 0,
            "max_tokens":  MAX_TOKENS,
        }
        params.update(extra_params)
        if extra_body:
            params["extra_body"] = extra_body

        resp   = client.chat.completions.create(**params)
        text   = resp.choices[0].message.content or ""
        reason = resp.choices[0].finish_reason or "stop"
        return text.strip(), reason
    except Exception as e:
        log.error(f"API error: {e}")
        return "ERROR", "error"



# ── Scoring functions ──────────────────────────────────────────────────────

def score_item_recall(response: str, gold_acquirer: str) -> dict:
    """
    P1 scoring (item recall, LLM-MemoryInspector style):
    token-F1 tra acquirer generato e gold acquirer name.

    Il modello potrebbe ripetere il prefisso "deal_id::" prima del nome —
    lo strip() lo rimuove. Stessa logica token-F1 di P2 (ROUGE-1 style).
    """
    if not response or response.strip().upper() in ("UNKNOWN", "ERROR"):
        return {
            "p1_response":   response,
            "p1_gold":       gold_acquirer,
            "p1_f1":         0.0,
            "p1_suspicious": False,
            "p1_label":      "unknown",
        }

    # Strip a leading "deal_id::" prefix if the model echoes it back
    clean = re.sub(r"^\S+::", "", response).strip()
    f1    = _token_f1(clean, gold_acquirer)
    suspicious = f1 > ITEM_RECALL_F1_THRESHOLD

    return {
        "p1_response":   clean,
        "p1_gold":       gold_acquirer,
        "p1_f1":         round(f1, 4),
        "p1_suspicious": suspicious,
        "p1_label":      "suspicious" if suspicious else "clean",
    }


def _token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 (ROUGE-1 style, no stemming, no stopwords)."""
    stops = {"inc", "corp", "llc", "ltd", "the", "and", "co", "group",
             "holdings", "company", "international", "technologies"}
    pred_toks = {t for t in re.sub(r"[^\w\s]", " ", pred.lower()).split()
                 if t not in stops and len(t) > 1}
    gold_toks = {t for t in re.sub(r"[^\w\s]", " ", gold.lower()).split()
                 if t not in stops and len(t) > 1}
    if not pred_toks or not gold_toks:
        return 0.0
    tp   = len(pred_toks & gold_toks)
    prec = tp / len(pred_toks)
    rec  = tp / len(gold_toks)
    if prec + rec == 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 4)


def score_completion(response: str, gold_next: dict) -> dict:
    """
    P2 scoring: token-F1 tra risposta e gold record successivo.
    Gold string include acquirer_name, deal_year, target_name.
    """
    if not response or response.strip().upper() == "UNKNOWN":
        return {
            "p2_response":   response,
            "p2_f1":         0.0,
            "p2_suspicious": False,
            "p2_label":      "unknown",
        }

    gold_str = (
        f"{gold_next.get('acquirer_name', '')} "
        f"{gold_next.get('deal_year', '')} "
        f"{gold_next.get('target_name', '')}"
    )
    f1         = _token_f1(response, gold_str)
    suspicious = f1 > COMPLETION_F1_THRESHOLD

    return {
        "p2_response":   response,
        "p2_f1":         f1,
        "p2_gold":       gold_str,
        "p2_suspicious": suspicious,
        "p2_label":      "suspicious" if suspicious else "clean",
    }


def _parse_tickers(text: str) -> set[str]:
    """Estrae ticker da una stringa comma-separated (maiuscole, 2–5 char).

    Difese:
    - lunghezza minima 2 (evita token singoli come 'A', 'I')
    - stopwords inglesi comuni escluse (evita falsi positivi su risposte in prosa)
    """
    _STOPS = {
        "A", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN",
        "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP",
        "US", "WE", "AND", "ARE", "FOR", "HAS", "NOT", "THE", "WITH",
        "FROM", "THAT", "THIS", "HAVE", "WERE", "BEEN", "THEY", "ALSO",
        "EACH", "SUCH", "THAN", "THEN", "WHEN", "WILL", "YEAR", "LIST",
        "DEAL", "DEALS", "FIRM", "FIRMS", "ONLY", "SOME", "MANY", "MOST",
        "BOTH", "WELL", "HERE", "OVER", "INTO", "EVEN", "JUST", "BELOW",
        "ABOVE", "THESE", "THOSE", "WHERE", "WOULD", "COULD", "ABOUT",
        "OTHER", "AFTER", "UNDER", "WHICH", "BASED", "THEIR", "THERE",
        "SHOULD", "SURE", "IQ", "SP", "MA", "MB", "NA", "TA", "TE",
        "SA", "SE", "LA", "LE", "LI", "CO", "DE", "DI", "DA", "AL",
    }
    raw = re.sub(r"[^A-Z0-9,\s\.]", "", text.upper())
    candidates = {t.strip().rstrip(".") for t in re.split(r"[,\s]+", raw)}
    return {
        t for t in candidates
        if 2 <= len(t) <= 5
        and t.isalpha()
        and t not in _STOPS
    }


def _jaccard(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return round(len(pred & gold) / len(pred | gold), 4)


def score_enumeration(response: str, gold_tickers: set[str]) -> dict:
    """
    P3 scoring: Jaccard similarity tra ticker predetti e gold set.
    """
    if not response or response.strip().upper() == "UNKNOWN":
        return {
            "p3_response":    response[:200],
            "p3_jaccard":     0.0,
            "p3_pred_n":      0,
            "p3_overlap_n":   0,
            "p3_suspicious":  False,
            "p3_label":       "unknown",
        }

    pred_tickers = _parse_tickers(response)
    jac          = _jaccard(pred_tickers, gold_tickers)
    overlap      = len(pred_tickers & gold_tickers)
    suspicious   = jac > ENUMERATION_JAC_THRESHOLD

    return {
        "p3_response":   response[:200],
        "p3_jaccard":    jac,
        "p3_pred_n":     len(pred_tickers),
        "p3_overlap_n":  overlap,
        "p3_suspicious": suspicious,
        "p3_label":      "suspicious" if suspicious else "clean",
    }


def aggregate_flags(p1: dict, p2: dict, p3: dict) -> dict:
    """
    Aggrega i tre probe in un flag dataset_memorized.
    Criteri: >= 2 probe su 3 sono suspicious.
    """
    n_suspicious = sum([
        p1.get("p1_suspicious", False),
        p2.get("p2_suspicious", False),
        p3.get("p3_suspicious", False),
    ])
    return {
        "n_suspicious_probes":   n_suspicious,
        "dataset_memorized_flag": n_suspicious >= 2,
        "dataset_risk":          (
            "HIGH"   if n_suspicious >= 2 else
            "MEDIUM" if n_suspicious == 1 else
            "LOW"
        ),
    }


# ── Resume helper ──────────────────────────────────────────────────────────

def already_done(slug: str) -> bool:
    """Return True if a complete record already exists for this slug in the v2 file."""
    jsonl_path = OUT_DIR / JSONL_FILE
    if not jsonl_path.exists():
        return False
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("model_slug") == slug:
                    return True
            except Exception:
                pass
    return False


# ── Main probe runner ──────────────────────────────────────────────────────

def run_dataset_memo(
    model_key: str,
    corpus: list[dict],
    gold_tickers: set[str],
    pilot: bool = False,
) -> dict:
    """
    Esegue i tre probe per un singolo modello.
    Ogni modello riceve esattamente 3 chiamate API.
    """
    model_cfg  = MODELS[model_key]
    slug       = model_cfg["slug"]
    out_path   = OUT_DIR / JSONL_FILE

    if already_done(slug):
        log.info(f"[dataset/{slug}] Already completed in v2, skipping.")
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("model_slug") == slug:
                        return r
                except Exception:
                    pass

    client = get_client()
    log.info(f"[dataset/{slug}] Running 3 probes...")

    # Seed rows: the first N_SHOTS corpus records as few-shot examples, plus
    # the record at ITEM_RECALL_QUERY_IDX as the P1 query.
    # Per P2 usiamo i primi due record (seed + gold_next).
    seed_rows = corpus[:2] if not pilot else corpus[:2]
    seed      = seed_rows[0]
    gold_next = seed_rows[1] if len(seed_rows) > 1 else {}

    # ── P1: Item recall probe (LLM-MemoryInspector style) ────────
    p1_messages, p1_gold = build_item_recall_messages(corpus)
    p1_resp, p1_reason   = call_model(client, model_cfg, messages=p1_messages)
    p1_scores            = score_item_recall(p1_resp, p1_gold)
    log.info(f"  P1 item_recall: f1={p1_scores.get('p1_f1')}  "
             f"gold='{p1_gold}'  pred='{p1_scores.get('p1_response')}'  "
             f"suspicious={p1_scores.get('p1_suspicious')}")
    time.sleep(1.0)

    # ── P2: Completion probe ──────────────────────────────────────
    # Use real (non-anonymized) names: this probe tests knowledge of the raw
    # dataset, not reasoning ability
    acquirer_display = seed.get("acquirer_name", "").split("(")[0].strip()
    p2_prompt = COMPLETION_PROMPT.format(
        source=GROUND_TRUTH_SOURCE,
        year_range=GROUND_TRUTH_YEAR_RANGE,
        acquirer_name=acquirer_display,
        deal_year=seed.get("deal_year", ""),
        target_name=seed.get("target_name", ""),
    )
    p2_resp, p2_reason = call_model(client, model_cfg, p2_prompt)
    p2_scores = score_completion(p2_resp, gold_next)
    log.info(f"  P2 completion: f1={p2_scores.get('p2_f1')}  "
             f"suspicious={p2_scores.get('p2_suspicious')}")
    time.sleep(1.0)

    # ── P3: Enumeration probe ─────────────────────────────────────
    p3_prompt = ENUMERATION_PROMPT.format(
        source=GROUND_TRUTH_SOURCE,
        year_range=GROUND_TRUTH_YEAR_RANGE,
    )
    p3_resp, p3_reason = call_model(client, model_cfg, p3_prompt)
    p3_scores = score_enumeration(p3_resp, gold_tickers)
    log.info(f"  P3 enumeration: jaccard={p3_scores.get('p3_jaccard')}  "
             f"overlap={p3_scores.get('p3_overlap_n')}/{len(gold_tickers)}  "
             f"suspicious={p3_scores.get('p3_suspicious')}")
    time.sleep(1.0)

    # ── Aggregazione ──────────────────────────────────────────────
    flags = aggregate_flags(p1_scores, p2_scores, p3_scores)

    record = {
        "model_key":  model_key,
        "model_slug": slug,
        "model_id":   model_cfg["model_id"],
        "timestamp":  datetime.now().isoformat(),
        **p1_scores,
        **p2_scores,
        **p3_scores,
        **flags,
    }

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    log.info(f"[dataset/{slug}] Done. Risk={flags['dataset_risk']}  "
             f"Flag={flags['dataset_memorized_flag']}")
    return record


# ── Summary & report ───────────────────────────────────────────────────────

def load_dataset_results() -> list[dict]:
    path = OUT_DIR / JSONL_FILE
    if not path.exists():
        return []
    results = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass
    return results


def build_summary(results: list[dict]) -> pd.DataFrame:
    MODEL_ORDER = ["llama32_1b", "phi4_14b", "qwen3_8b",
                   "llama33_70b", "qwen3_235b", "deepseek_v3"]
    rows = []
    by_slug = {r["model_slug"]: r for r in results}
    for slug in MODEL_ORDER:
        r = by_slug.get(slug)
        if r is None:
            continue
        rows.append({
            "model_slug":             slug,
            "model_id":               r.get("model_id", ""),
            "p1_item_recall_f1":      r.get("p1_f1"),
            # keep the raw probe fields so the text report can read them (bug fix:
            # the report used to read p1_f1/p1_gold/p1_response off the RENAMED
            # summary frame and always got None).
            "p1_gold":                r.get("p1_gold"),
            "p1_response":            r.get("p1_response"),
            "p1_suspicious":          r.get("p1_suspicious"),
            "p2_completion_f1":       r.get("p2_f1"),
            "p2_suspicious":          r.get("p2_suspicious"),
            "p3_enumeration_jaccard": r.get("p3_jaccard"),
            "p3_overlap_n":           r.get("p3_overlap_n"),
            "p3_suspicious":          r.get("p3_suspicious"),
            "n_suspicious_probes":    r.get("n_suspicious_probes"),
            "dataset_memorized_flag": r.get("dataset_memorized_flag"),
            "dataset_risk":           r.get("dataset_risk"),
        })
    return pd.DataFrame(rows)


def cross_summary(
    dataset_df: pd.DataFrame,
    recall_csv: Path,
) -> pd.DataFrame:
    """
    Incrocia dataset memorization con knowledge memorization (deal recall
    da script 19) per produrre la classificazione finale a 4 quadranti.
    """
    if not recall_csv.exists():
        log.warning(f"memo_summary.csv non trovato: {recall_csv}")
        return dataset_df

    memo = pd.read_csv(recall_csv)
    recall_rows = memo[memo["test"] == "recall"][
        ["model_slug", "success_rate"]
    ].rename(columns={"success_rate": "knowledge_recall_rate"})

    merged = dataset_df.merge(recall_rows, on="model_slug", how="left")

    KNOWLEDGE_THRESHOLD = 0.10

    def classify(row):
        k_high = row.get("knowledge_recall_rate", 0) > KNOWLEDGE_THRESHOLD
        d_flag = row.get("dataset_memorized_flag", False)
        if k_high and d_flag:
            return "HIGH_BOTH — corpus leakage risk"
        elif k_high and not d_flag:
            return "HIGH_K / LOW_D — news knowledge, corpus clean"
        elif not k_high and d_flag:
            return "LOW_K / HIGH_D — unusual pattern, investigate"
        else:
            return "LOW_BOTH — clean baseline"

    merged["quadrant"] = merged.apply(classify, axis=1)
    return merged


def write_report(df: pd.DataFrame) -> None:
    MODEL_ORDER = ["llama32_1b", "phi4_14b", "qwen3_8b",
                   "llama33_70b", "qwen3_235b", "deepseek_v3"]

    lines = [
        "DATASET MEMORIZATION REPORT v2 — LLM-as-M&A-Analyst",
        "(P1: item recall probe — LLM-MemoryInspector style)",
        f"Generated:     {datetime.now():%Y-%m-%d %H:%M}",
        f"Corpus:        corpus_wide_final.json ({GROUND_TRUTH_N_ACQUIRERS} deals, 2015-2022)",
        f"Source:        {GROUND_TRUTH_SOURCE}",
        "",
        "PROBE THRESHOLDS",
        f"  P1 (item recall):  token-F1 > {ITEM_RECALL_F1_THRESHOLD} → suspicious",
        f"                     few-shot: {ITEM_RECALL_N_SHOTS} examples, query idx={ITEM_RECALL_QUERY_IDX}",
        f"  P2 (completion):   token-F1 > {COMPLETION_F1_THRESHOLD} → suspicious",
        f"  P3 (enumeration):  Jaccard > {ENUMERATION_JAC_THRESHOLD} → suspicious",
        f"  Dataset flag:      >= 2 suspicious probes → dataset_memorized = True",
        "",
        "=" * 65,
        "RESULTS PER MODEL",
        "=" * 65,
    ]

    by_slug = {row["model_slug"]: row for _, row in df.iterrows()}
    for slug in MODEL_ORDER:
        if slug not in by_slug:
            continue
        r = by_slug[slug]
        lines += [
            f"\n  {slug}",
            f"    P1 item_recall: F1={r.get('p1_item_recall_f1')}  "
            f"gold='{r.get('p1_gold')}'  pred='{r.get('p1_response')}'  "
            f"→ {r.get('p1_suspicious') and 'SUSPICIOUS' or 'clean'}",
            f"    P2 completion: F1={r.get('p2_completion_f1')}  "
            f"→ {r.get('p2_suspicious') and 'SUSPICIOUS' or 'clean'}",
            f"    P3 enumeration: Jaccard={r.get('p3_enumeration_jaccard')}  "
            f"overlap={r.get('p3_overlap_n')}/{GROUND_TRUTH_N_ACQUIRERS}  "
            f"→ {r.get('p3_suspicious') and 'SUSPICIOUS' or 'clean'}",
            f"    Suspicious probes: {r.get('n_suspicious_probes')}/3  "
            f"→ Dataset risk: {r.get('dataset_risk')}",
        ]
        if "quadrant" in r:
            lines.append(f"    Cross-check:   {r.get('quadrant')}")

    lines += [
        "",
        "=" * 65,
        "PAPER TEMPLATE (fill in results)",
        "=" * 65,
        "",
        "To assess whether models were exposed to the specific corpus used in",
        "this study during pre-training, we administered three dataset-specific",
        "probes adapted from Anelli et al. (2024): an item recall probe (asking",
        "the acquirer name for a given DealID, presented via few-shot role-play",
        "following the LLM-MemoryInspector protocol), a completion probe",
        "(predicting the next record given the first), and an enumeration probe",
        "(listing acquirer tickers). These probes target information that is an",
        "artefact of our specific corpus construction — not recoverable from",
        "public news sources.",
        "",
        "Results: [FILL IN]. No model scored above threshold on more than",
        "[N] of the three probes, yielding a dataset memorization flag of",
        "[False/True] for all models. Combined with the knowledge recall rates",
        "from Section 2 (Script 19), this pattern is consistent with [narrative:",
        "models retrieved deal outcomes from public news coverage rather than",
        "from the evaluation corpus itself / corpus leakage detected — see",
        "synthetic dataset analysis].",
    ]

    report_path = OUT_DIR / REPORT_FILE
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Report saved: {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Script 28 — Dataset memorization test v2 (P1 item-recall)"
    )
    parser.add_argument(
        "--model", nargs="+", metavar="MODEL_KEY",
        help="Model keys to run (default: all 6). "
             f"Valid: {list(MODELS.keys())}"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Use only first 2 corpus rows as seed (default: same, "
             "since probes are corpus-level, not per-deal)"
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip API calls, regenerate summary and report from existing JSONL"
    )
    args = parser.parse_args()

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    log.info(f"Corpus loaded: {len(corpus)} deals")

    gold_tickers: set[str] = set()
    for deal in corpus:
        t = (deal.get("acquirer_ticker") or "").strip().upper()
        if t and 1 <= len(t) <= 5:
            gold_tickers.add(t)
    log.info(f"Gold ticker set: {len(gold_tickers)} unique tickers")

    model_keys = args.model or list(MODELS.keys())
    invalid    = [k for k in model_keys if k not in MODELS]
    if invalid:
        parser.error(f"Unknown model keys: {invalid}. Valid: {list(MODELS.keys())}")

    all_results = []

    if not args.summary_only:
        for i, mk in enumerate(model_keys, 1):
            slug = MODELS[mk]["slug"]
            log.info(f"\n[{i}/{len(model_keys)}] {slug}")
            result = run_dataset_memo(mk, corpus, gold_tickers, pilot=args.pilot)
            all_results.append(result)
    else:
        all_results = load_dataset_results()
        if not all_results:
            log.warning("No results found. Run without --summary-only first.")
            return

    # ── Summary CSV ───────────────────────────────────────────────
    df = build_summary(all_results)

    # Cross-reference with the knowledge recall from script 19, when available
    memo_csv = OUT_DIR / "memo_summary.csv"
    df = cross_summary(df, memo_csv)

    summary_path = OUT_DIR / SUMMARY_FILE
    df.to_csv(summary_path, index=False)
    log.info(f"Summary saved: {summary_path}")

    # ── Report txt ────────────────────────────────────────────────
    write_report(df)

    # ── Console output ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  DATASET MEMORIZATION RESULTS")
    print(f"  Ground truth: {GROUND_TRUTH_N_ACQUIRERS} acquirers | "
          f"{len(gold_tickers)} tickers | source: {GROUND_TRUTH_SOURCE}")
    print(f"{'='*70}")
    print(f"\n  {'Model':<18} {'P1 F1':>8} {'P2 F1':>8} "
          f"{'P3 Jacc':>9} {'N susp':>7} {'Risk':>8}")
    print("  " + "-" * 65)
    for _, row in df.iterrows():
        print(
            f"  {row['model_slug']:<18} "
            f"{str(row.get('p1_item_recall_f1', '?')):>8} "
            f"{str(row.get('p2_completion_f1', '?')):>8} "
            f"{str(row.get('p3_enumeration_jaccard', '?')):>9} "
            f"{str(row.get('n_suspicious_probes', '?')):>7} "
            f"{str(row.get('dataset_risk', '?')):>8}"
        )
    if "quadrant" in df.columns:
        print(f"\n  {'Model':<18} {'Cross-check quadrant'}")
        print("  " + "-" * 65)
        for _, row in df.iterrows():
            print(f"  {row['model_slug']:<18} {row.get('quadrant', '?')}")

    print(f"\n  Output: {OUT_DIR.resolve()}")
    print(f"    {JSONL_FILE}      — raw probe responses (v2)")
    print(f"    {SUMMARY_FILE}    — aggregated scores (v2)")
    print(f"    {REPORT_FILE}     — paper-ready interpretation (v2)")


if __name__ == "__main__":
    main()
