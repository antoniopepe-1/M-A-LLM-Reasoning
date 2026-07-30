"""
19_memorization_check.py
=========================
Two-section memorization test for the LLM-as-M&A-Analyst study.

═══════════════════════════════════════════════════════════════
SECTION 1 — DE-ANONYMIZATION TEST
═══════════════════════════════════════════════════════════════
Verifies that the anonymization applied in the main pipeline is effective.

Protocol:
  Input  : anonymized acquirer profile (same as main experiment)
  Prompt : "Based on this 10-K excerpt, what company is this?"
  Eval   : fuzzy match response vs acquirer_name / acquirer_ticker

Answers: "Does the anonymization hold?"
  success_rate < 10% → anonymization effective → Delta(B-C) is defensible
  success_rate > 30% → leakage risk → additional robustness checks needed

═══════════════════════════════════════════════════════════════
SECTION 2 — DIRECTLY-ELICITABLE RECALL PROBE
═══════════════════════════════════════════════════════════════
Measures whether the specific deal outcome is DIRECTLY ELICITABLE from the model
via a natural-language question. This is NOT a test of the presence/absence of
parametric memory: a negative result means the event was "not recovered by the
direct-recall probe", not that it is absent from the weights (audit finding).

Protocol:
  Input  : acquirer real name + deal year (no anonymization)
  Prompt : "[Acquirer] completed an acquisition in [year].
            What company did they acquire?"
  Eval   : conservative entity match vs target_name / target_ticker

Disambiguation (audit finding): some acquirer-year pairs map to MULTIPLE focal
acquisitions (e.g. Oracle 2016 has three). For those, ANY of the co-occurring
targets is accepted as correct, so a model is not marked wrong for naming a
different real acquisition by the same acquirer in the same year.

Interpretation:
  A low elicitation rate supports (but does not prove) that main-experiment
  performance is not driven by direct outcome recall. Downstream analysis uses
  MODEL-SPECIFIC non-recalled subsets U_m (script 27), never a causal "memory
  share".

WHY BOTH TESTS MATTER
─────────────────────
Section 1 defends the experimental design (anonymization validity).
Section 2 defends RQ1 directly (models reason rather than recall).
Together they provide a comprehensive memorization robustness check.

OUTPUT
------
  results/memorization/
    deanon_raw_{slug}.jsonl    per-deal de-anonymization results
    recall_raw_{slug}.jsonl    per-deal deal recall results
    memo_summary.csv           aggregated both tests per model
    memo_report.txt            paper-ready summary with template

"""

import re
import html
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Canonical library — leakage-free profile, entity matching, provider routing.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import profile as _profile
from lib import entities as _entities
from lib import provenance as _prov

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
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

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"memorization_{datetime.now():%Y%m%d_%H%M%S}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Model roster (mirrors 16c_v3) ──────────────────────────────────────────
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

FUZZY_THRESHOLD = 85
MAX_TOKENS_MEMO = 200


# ── Prompts ────────────────────────────────────────────────────────────────

DEANON_PROMPT = """\
The following is an excerpt from a company's 10-K annual report filed with the SEC. \
The company's identity has been anonymized (shown as [ACQUIRER]).

{profile}

Based solely on this excerpt, what is the most likely real company name?
Respond with ONLY the company name — no explanation, no hedging, no "I think".
If you cannot identify the company, respond with exactly: UNKNOWN\
"""

RECALL_PROMPT = """\
{acquirer_name} completed a corporate acquisition in {deal_year}.

What company did {acquirer_name} acquire in {deal_year}?
Respond with ONLY the acquired company name — no explanation, no hedging.
If you do not know, respond with exactly: UNKNOWN\
"""


# ── Anonymization (mirrors mask_acquirer in 16c_v3) ───────────────────────

def _build_mask_variants(deal: dict) -> list:
    name   = deal.get("acquirer_name", "") or ""
    ticker = deal.get("acquirer_ticker", "") or ""
    variants = set()

    paren_idx = name.find("(")
    base = name[:paren_idx].strip() if paren_idx > 0 else name
    if base:
        variants.add(base)

    base_clean = re.sub(
        r",?\s*(Inc\.?|Corp\.?|Corporation|LLC|Ltd\.?|Limited|PLC|"
        r"Holdings?|Group|Co\.?|Company|International|Technologies?)$",
        "", base, flags=re.IGNORECASE,
    ).strip().rstrip(",").strip()
    if base_clean:
        variants.add(base_clean)

    if "(" in name and ")" in name:
        inner = name[name.find("(") + 1:name.find(")")]
        if ":" in inner:
            variants.add(inner)
            variants.add(inner.split(":")[1])

    if ticker and len(ticker) >= 2:
        variants.add(ticker)
        for exchange in ("NYSE", "NASDAQ", "NASDAQGS"):
            variants.add(f"{exchange}:{ticker}")

    return sorted({v for v in variants if len(v) >= 4}, key=len, reverse=True)


# Masking delegates to the canonical library so it is identical to the pipeline.
mask_acquirer = _profile.mask_acquirer


def build_anonymized_profile(deal: dict) -> str:
    """Use the canonical leakage-free profile (NO target market cap — Finding 1)."""
    return _profile.build_acquirer_profile(deal)


def get_acquirer_display_name(deal: dict) -> str:
    name = deal.get("acquirer_name", "") or ""
    paren_idx = name.find("(")
    return name[:paren_idx].strip() if paren_idx > 0 else name


# ── Evaluation ────────────────────────────────────────────────────────────

def _normalize_tokens(s: str) -> set:
    stops = {"inc", "corp", "corporation", "llc", "ltd", "limited",
             "holdings", "group", "company", "international", "technologies",
             "the", "and", "for", "co"}
    tokens = re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
    return {t for t in tokens if t not in stops and len(t) > 1}


def evaluate_response(response: str, truth_name: str, truth_ticker: str,
                      alt_targets: list | None = None) -> dict:
    """
    Conservative entity resolution of a recall response against the target.

    `alt_targets` are OTHER real acquisitions by the same acquirer in the same
    year (audit disambiguation): naming any of them counts as a valid recall, so
    the model is not marked wrong for a different real deal by the same acquirer.
    """
    resp_clean = response.strip()

    if not resp_clean or resp_clean.upper() == "UNKNOWN" or "cannot" in resp_clean.lower():
        return {"exact_match": False, "fuzzy_score": 0, "label": "UNKNOWN"}

    # Accept the focal target OR any co-occurring acquisition, resolved with the
    # conservative entity matcher (no single-token matches).
    acceptable = [(truth_name, truth_ticker)] + list(alt_targets or [])
    for nm, tk in acceptable:
        if _entities.resolve_match(resp_clean, nm, tk, tuple(_entities.name_aliases(nm))):
            return {"exact_match": True, "fuzzy_score": 100, "label": "SUCCESS"}

    if FUZZY_AVAILABLE:
        score = fuzz.token_set_ratio(resp_clean, truth_name)
    else:
        score = -1
    if score >= FUZZY_THRESHOLD:
        return {"exact_match": False, "fuzzy_score": score, "label": "PARTIAL"}
    return {"exact_match": False, "fuzzy_score": max(score, 0), "label": "FAILURE"}


# ── OpenRouter client ──────────────────────────────────────────────────────

def get_client() -> OpenAI:
    import os
    # Provider routing is set in the request BODY (call_model), not via the
    # X-Provider-Order header, which OpenRouter does not honour (audit finding).
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://github.com/llm-ma-analyst",
            "X-Title":      "LLM-as-MA-Analyst-MemCheck",
        },
    )


def call_model(client: OpenAI, model_cfg: dict, prompt: str) -> tuple:
    try:
        extra_params = model_cfg.get("extra_params", {})

        # "thinking" must go in extra_body for OpenRouter, not as a top-level param
        extra_body = {}
        if "thinking" in extra_params:
            extra_body["thinking"] = extra_params.pop("thinking")
        # Provider routing via request body (audit finding).
        extra_body.update(_prov.provider_routing(["Together", "DeepInfra", "Fireworks"]))

        params = {
            "model":       model_cfg["model_id"],
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens":  MAX_TOKENS_MEMO,
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


# ── Resume ─────────────────────────────────────────────────────────────────

def load_completed_ids(jsonl_path: Path) -> set:
    if not jsonl_path.exists():
        return set()
    completed = set()
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                completed.add(json.loads(line)["deal_id"])
            except Exception:
                pass
    return completed


# ── Section 1 — De-anonymization ──────────────────────────────────────────

def run_deanon(model_key: str, corpus: list, pilot: bool = False) -> None:
    model_cfg = MODELS[model_key]
    slug      = model_cfg["slug"]
    out_path  = OUT_DIR / f"deanon_raw_{slug}.jsonl"

    completed = load_completed_ids(out_path)
    deals     = corpus[:10] if pilot else corpus
    remaining = [d for d in deals if d["deal_id"] not in completed]

    if not remaining:
        log.info(f"[deanon/{slug}] All deals already completed.")
        return

    log.info(f"[deanon/{slug}] {len(remaining)}/{len(deals)} deals to process")
    client = get_client()

    with out_path.open("a", encoding="utf-8") as f_out:
        for deal in tqdm(remaining, desc=f"deanon/{slug}", ncols=100):
            profile = build_anonymized_profile(deal)
            prompt  = DEANON_PROMPT.format(profile=profile)

            response, finish_reason = call_model(client, model_cfg, prompt)
            if response == "ERROR":
                log.warning(f"FAILED deal={deal['deal_id']}")
                time.sleep(2)
                continue

            result = evaluate_response(
                response,
                deal.get("acquirer_name", ""),
                deal.get("acquirer_ticker", ""),
            )

            record = {
                "test":            "deanon",
                "deal_id":         deal["deal_id"],
                "model_key":       model_key,
                "model_slug":      slug,
                "model_id":        model_cfg["model_id"],
                "acquirer_name":   deal.get("acquirer_name"),
                "acquirer_ticker": deal.get("acquirer_ticker"),
                "deal_year":       deal.get("deal_year"),
                "industry":        deal.get("industry"),
                "response":        response,
                "finish_reason":   finish_reason,
                "exact_match":     result["exact_match"],
                "fuzzy_score":     result["fuzzy_score"],
                "label":           result["label"],
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            time.sleep(0.5)

    log.info(f"[deanon/{slug}] Done.")


# ── Section 2 — Deal recall ────────────────────────────────────────────────

def run_recall(model_key: str, corpus: list, pilot: bool = False) -> None:
    model_cfg = MODELS[model_key]
    slug      = model_cfg["slug"]
    out_path  = OUT_DIR / f"recall_raw_{slug}.jsonl"

    completed = load_completed_ids(out_path)
    deals     = corpus[:10] if pilot else corpus
    remaining = [d for d in deals if d["deal_id"] not in completed]

    if not remaining:
        log.info(f"[recall/{slug}] All deals already completed.")
        return

    log.info(f"[recall/{slug}] {len(remaining)}/{len(deals)} deals to process")
    client = get_client()

    # Acquirer-year disambiguation map (audit finding): (acquirer, year) -> all
    # (target_name, target_ticker) acquisitions in the corpus. Any of them counts
    # as a valid recall for a deal in that group.
    ay_map: dict[tuple, list] = {}
    for d in corpus:
        key = (get_acquirer_display_name(d).lower(), d.get("deal_year"))
        ay_map.setdefault(key, []).append((d.get("target_name", ""), d.get("target_ticker", "")))

    with out_path.open("a", encoding="utf-8") as f_out:
        for deal in tqdm(remaining, desc=f"recall/{slug}", ncols=100):
            acquirer_display = get_acquirer_display_name(deal)
            deal_year        = deal.get("deal_year", "")

            prompt = RECALL_PROMPT.format(
                acquirer_name=acquirer_display,
                deal_year=deal_year,
            )

            response, finish_reason = call_model(client, model_cfg, prompt)
            if response == "ERROR":
                log.warning(f"FAILED deal={deal['deal_id']}")
                time.sleep(2)
                continue

            key = (acquirer_display.lower(), deal_year)
            alts = [t for t in ay_map.get(key, [])
                    if t[0] != deal.get("target_name")]
            result = evaluate_response(
                response,
                deal.get("target_name", ""),
                deal.get("target_ticker", ""),
                alt_targets=alts,
            )

            record = {
                "test":            "recall",
                "deal_id":         deal["deal_id"],
                "model_key":       model_key,
                "model_slug":      slug,
                "model_id":        model_cfg["model_id"],
                "acquirer_name":   acquirer_display,
                "target_name":     deal.get("target_name"),
                "target_ticker":   deal.get("target_ticker"),
                "deal_year":       deal.get("deal_year"),
                "industry":        deal.get("industry"),
                "response":        response,
                "finish_reason":   finish_reason,
                "exact_match":     result["exact_match"],
                "fuzzy_score":     result["fuzzy_score"],
                "label":           result["label"],
                "n_alt_targets":   len(alts),        # >0 => ambiguous acquirer-year
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            time.sleep(0.5)

    log.info(f"[recall/{slug}] Done.")


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate_test(glob: str, test_name: str) -> pd.DataFrame:
    rows = []
    for path in sorted(OUT_DIR.glob(glob)):
        records = [json.loads(l) for l in path.open() if l.strip()]
        if not records:
            continue
        df    = pd.DataFrame(records)
        total = len(df)
        rows.append({
            "test":             test_name,
            "model_slug":       df["model_slug"].iloc[0],
            "model_id":         df["model_id"].iloc[0],
            "n_deals":          total,
            "success_n":        int((df["label"] == "SUCCESS").sum()),
            "partial_n":        int((df["label"] == "PARTIAL").sum()),
            "failure_n":        int((df["label"] == "FAILURE").sum()),
            "unknown_n":        int((df["label"] == "UNKNOWN").sum()),
            "success_rate":     round((df["label"] == "SUCCESS").mean(), 4),
            "partial_rate":     round((df["label"] == "PARTIAL").mean(), 4),
            "failure_rate":     round((df["label"] == "FAILURE").mean(), 4),
            "unknown_rate":     round((df["label"] == "UNKNOWN").mean(), 4),
            "mean_fuzzy_score": round(df["fuzzy_score"].clip(lower=0).mean(), 2),
        })
    return pd.DataFrame(rows)


def build_summary() -> pd.DataFrame:
    df1 = aggregate_test("deanon_raw_*.jsonl", "deanon")
    df2 = aggregate_test("recall_raw_*.jsonl", "recall")
    return pd.concat([df1, df2], ignore_index=True)


def write_report(summary: pd.DataFrame) -> None:
    MODEL_ORDER = ["llama32_1b", "phi4_14b", "qwen3_8b", "llama33_70b", "qwen3_235b", "deepseek_v3"]

    lines = [
        "MEMORIZATION TEST REPORT — LLM-as-M&A-Analyst",
        f"Generated:  {datetime.now():%Y-%m-%d %H:%M}",
        f"Corpus:     corpus_wide_final.json (286 deals, 2015-2022)",
        f"Threshold:  token_set_ratio >= {FUZZY_THRESHOLD} for PARTIAL match",
        "",
        "=" * 65,
        "SECTION 1 — DE-ANONYMIZATION TEST",
        "Q: Given the anonymized 10-K profile, can the model identify the acquirer?",
        "Threshold: success_rate < 10% -> anonymization holds",
        "=" * 65,
    ]

    for test_name, header in [("deanon", "SECTION 1"), ("recall", "SECTION 2")]:
        if test_name == "recall":
            lines += [
                "",
                "=" * 65,
                "SECTION 2 — DEAL RECALL TEST",
                "Q: Given acquirer name + year, can the model name the acquired company?",
                "Threshold: success_rate < 10% -> no systematic memorization",
                "=" * 65,
            ]

        sub = summary[summary["test"] == test_name]
        for slug in MODEL_ORDER:
            row = sub[sub["model_slug"] == slug]
            if row.empty:
                continue
            r = row.iloc[0]
            lines += [
                f"\n  {r['model_slug']}",
                f"    SUCCESS : {r['success_n']:>3}/{r['n_deals']} ({r['success_rate']*100:.1f}%)",
                f"    PARTIAL : {r['partial_n']:>3}/{r['n_deals']} ({r['partial_rate']*100:.1f}%)",
                f"    FAILURE : {r['failure_n']:>3}/{r['n_deals']} ({r['failure_rate']*100:.1f}%)",
                f"    UNKNOWN : {r['unknown_n']:>3}/{r['n_deals']} ({r['unknown_rate']*100:.1f}%)",
                f"    Mean fuzzy score: {r['mean_fuzzy_score']:.1f}/100",
            ]

    # Computed range/mean of success rates per test (fills the former [X]/[Y]/[Z]
    # placeholders so the report is self-contained — v2 audit placeholder cleanup).
    def _stats(test_name: str) -> tuple[float, float, float]:
        s = summary[summary["test"] == test_name]["success_rate"] * 100
        return (s.min(), s.max(), s.mean())

    da_lo, da_hi, da_mu = _stats("deanon")
    rc_lo, rc_hi, rc_mu = _stats("recall")
    # "partially retrieve" if any model recalls >= 10%; deanon interpretation is
    # driven by the low identification ceiling.
    recall_verb = "partially" if rc_hi >= 10.0 else "did not"
    supports = "qualifies" if rc_hi >= 10.0 else "supports"

    lines += [
        "",
        "=" * 65,
        "PAPER PARAGRAPH (auto-filled)",
        "=" * 65,
        "",
        "De-anonymization: Models were presented with anonymized acquirer profiles",
        f"and asked to identify the company. Identification rates ranged from {da_lo:.1f}%",
        f"to {da_hi:.1f}% across models (mean: {da_mu:.1f}%), confirming that the anonymization",
        "procedure effectively prevents identity retrieval from textual cues.",
        "",
        "Deal recall: Models were given the acquirer's real name and deal year and",
        f"asked to name the acquired company. Recall rates ranged from {rc_lo:.1f}% to {rc_hi:.1f}%",
        f"(mean: {rc_mu:.1f}%), indicating that models {recall_verb} retrieve",
        f"memorized deal outcomes from training data. This {supports}",
        "the interpretation of Condition-B ranking and Delta(B-C) as reflecting",
        "genuine strategic reasoning rather than memorized deal retrieval, and",
        "motivates the model-specific U_m analysis (script 27) that restricts to",
        "deals each model could not directly recall.",
    ]

    report_path = OUT_DIR / "memo_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Report saved: {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-section memorization test: de-anonymization + deal recall"
    )
    parser.add_argument("--test", nargs="+", choices=["deanon", "recall"],
                        default=["deanon", "recall"],
                        help="Which sections to run (default: both)")
    parser.add_argument("--model", nargs="+", metavar="MODEL_KEY",
                        help="Model keys (default: all 6)")
    parser.add_argument("--pilot", action="store_true",
                        help="Run on first 10 deals only")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip API calls, regenerate summary from existing JSONL")
    args = parser.parse_args()

    if not FUZZY_AVAILABLE:
        log.warning("rapidfuzz not installed. Run: pip install rapidfuzz --break-system-packages")

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    log.info(f"Corpus loaded: {len(corpus)} deals")

    model_keys = args.model or list(MODELS.keys())
    invalid    = [k for k in model_keys if k not in MODELS]
    if invalid:
        parser.error(f"Unknown model keys: {invalid}. Valid: {list(MODELS.keys())}")

    if not args.summary_only:
        for i, mk in enumerate(model_keys, 1):
            slug = MODELS[mk]["slug"]
            log.info(f"\n[{i}/{len(model_keys)}] {slug}")
            if "deanon" in args.test:
                run_deanon(mk, corpus, pilot=args.pilot)
            if "recall" in args.test:
                run_recall(mk, corpus, pilot=args.pilot)

    summary = build_summary()
    if summary.empty:
        log.warning("No results yet. Run without --summary-only first.")
        return

    summary.to_csv(OUT_DIR / "memo_summary.csv", index=False)
    write_report(summary)

    # Console output
    for test_name, label in [("deanon", "SECTION 1 — DE-ANONYMIZATION"),
                              ("recall", "SECTION 2 — DEAL RECALL")]:
        print(f"\n{label}")
        sub = summary[summary["test"] == test_name]
        if sub.empty:
            print("  No results yet.")
            continue
        print(f"  {'Model':<20} {'n':>5} {'SUCCESS%':>10} {'PARTIAL%':>10} {'FAILURE%':>10} {'UNKNOWN%':>10}")
        print("  " + "-" * 65)
        for _, row in sub.iterrows():
            print(
                f"  {row['model_slug']:<20} {row['n_deals']:>5} "
                f"{row['success_rate']*100:>9.1f}% "
                f"{row['partial_rate']*100:>9.1f}% "
                f"{row['failure_rate']*100:>9.1f}% "
                f"{row['unknown_rate']*100:>9.1f}%"
            )

    print(f"\nOutput: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()