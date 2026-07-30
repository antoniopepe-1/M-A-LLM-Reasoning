"""
31_probe_named_conditions.py
============================

CONDITION A_name — Identity-only generation (variante di A)
  Input : real acquirer name + ticker + fiscal year + financials (NO description)
  Task  : generate 5 acquisition targets from memory

CONDITION B_named — Named-candidates ranking (variante di B)
  Input : anonymized acquirer profile (same masking as B) +
          50 candidates with REAL names and tickers (deanonymized on candidate side)
INPUT
-----
  data/candidate_corpus/corpus_wide_final.json  — masked profile + metadata + anon candidate list
  data/candidate_corpus/corpus_per_deal.csv     — candidate_id → real name/ticker mapping
  prompts/A_name_zeroshot.txt
  prompts/B_named_zeroshot.txt
"""

import os
import re
import html
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import openai
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)
from tqdm import tqdm
from dotenv import load_dotenv

# Canonical library — leakage-free profile, masking, provider routing.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import profile as _profile
from lib import provenance as _prov

mask_acquirer = _profile.mask_acquirer  # keep name; delegate to canonical masker

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

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
CORPUS_LONG = BASE_DIR / "data" / "candidate_corpus" / "corpus_per_deal.csv"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR     = BASE_DIR / "logs"
PROMPTS_DIR = BASE_DIR / "prompts"

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"probe_{datetime.now():%Y%m%d_%H%M%S}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model roster — identical to 16c_v3
# ─────────────────────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "llama32_1b": {
        "model_id":     "meta-llama/llama-3.2-1b-instruct",
        "provider":     "openrouter",
        "slug":         "llama32_1b",
        "extra_params": {},
    },
    "qwen3_8b": {
        "model_id":     "qwen/qwen3-8b-04-28",
        "provider":     "openrouter",
        "slug":         "qwen3_8b",
        "extra_params": {"thinking": {"type": "disabled"}},
    },
    "phi4": {
        "model_id":     "microsoft/phi-4",
        "provider":     "openrouter",
        "slug":         "phi4_14b",
        "extra_params": {},
    },
    "llama33_70b": {
        "model_id":     "meta-llama/llama-3.3-70b-instruct",
        "provider":     "openrouter",
        "slug":         "llama33_70b",
        "extra_params": {},
    },
    "qwen3": {
        "model_id":     "qwen/qwen3-235b-a22b-2507",
        "provider":     "openrouter",
        "slug":         "qwen3_235b",
        "extra_params": {"thinking": {"type": "disabled"}},
    },
    "deepseek": {
        "model_id":     "deepseek/deepseek-chat-v3-0324",
        "provider":     "openrouter",
        "slug":         "deepseek_v3",
        "extra_params": {},
    },
}

CONDITIONS   = ["A_name", "B_named"]
STRATEGIES   = ["zeroshot"]        # only zeroshot for these probes
MAX_TOKENS   = 1500

# ─────────────────────────────────────────────────────────────────────────────
# Acquirer masking (subset from 16c_v3, needed for Condition B_named)
# ─────────────────────────────────────────────────────────────────────────────

# mask_acquirer is delegated to lib.profile at import time (see top of file).
# The local copies were removed so masking is identical to the pipeline.




# ─────────────────────────────────────────────────────────────────────────────
# Profile builders
# ─────────────────────────────────────────────────────────────────────────────

def build_masked_profile_with_description(deal: dict) -> str:
    """
    Standard masked profile — canonical, leakage-free (NO target market cap;
    audit Finding 1). Used for Condition B_named.
    """
    return _profile.build_acquirer_profile(deal)


def build_identity_only_profile(deal: dict) -> str:
    """
    Identity-only profile for Condition A_name: acquirer name + ticker + FY +
    ACQUIRER-only financials. The realized target's market cap is removed (audit
    Finding 1) so A_name-minus-A isolates identity, not outcome size.
    """
    name   = deal.get("acquirer_name", "").strip()
    ticker = deal.get("acquirer_ticker", "N/A")
    return (
        f"COMPANY: {name} ({ticker})\n"
        f"FISCAL YEAR: FY{deal['deal_year'] - 1}\n"
        f"KEY FINANCIALS: "
        f"Revenue ${deal.get('acquirer_revenue_m', 'N/A')}M | "
        f"EBITDA margin {deal.get('acquirer_ebitda_margin_pct', 'N/A')}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Named candidate list reconstruction (Condition B_named)
# ─────────────────────────────────────────────────────────────────────────────

def build_named_candidate_list(deal_id: str, corpus_long: pd.DataFrame) -> str:
    """
    Reconstruct candidate list with real company names for Condition B_named.
    Preserves the same ordering (by candidate_id) as the anonymized version,
    so target_position from corpus_wide_final.json remains valid.

    Each line format:
      Candidate #NN: <Real Name> (<TICKER>) | Industry: X | Market Cap: Y | Revenue: Z | Description: ...
    """
    rows = corpus_long[corpus_long["deal_id"] == deal_id].copy()
    if rows.empty:
        raise ValueError(f"No candidates found for deal_id={deal_id}")

    # Ensure candidate_id is zero-padded 2-digit string for correct ordering
    rows["candidate_id"] = rows["candidate_id"].astype(str).str.zfill(2)
    rows = rows.sort_values("candidate_id")

    lines = []
    for _, r in rows.iterrows():
        anon = str(r["anon_description"])
        # Drop the leading "Candidate #NN | " to keep metadata + description only
        m = re.match(r"Candidate #\d+\s*\|\s*(.*)", anon, flags=re.DOTALL)
        rest = m.group(1) if m else anon
        name   = r["candidate_name"]
        ticker = r["candidate_ticker"]
        lines.append(f"Candidate #{r['candidate_id']}: {name} ({ticker}) | {rest}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt loading
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_FILES: dict[tuple[str, str], str] = {
    ("A_name",  "zeroshot"): "A_name_zeroshot.txt",
    ("B_named", "zeroshot"): "B_named_zeroshot.txt",
}

SYSTEM_PROMPTS: dict[str, str] = {
    "A_name": (
        "You are a senior strategic M&A advisor with deep expertise in "
        "corporate strategy, industry dynamics, and transaction structuring. "
        "You are given only the identity and financial context of a company — "
        "reason from what you know about this company and its market position."
    ),
    "B_named": (
        "You are a senior strategic M&A advisor. Rank the acquisition "
        "candidates by their strategic fit with the given acquirer profile. "
        "The acquirer profile has been anonymized, but candidates are "
        "identified by their real names and tickers."
    ),
}

_prompt_cache: dict[str, str] = {}


def load_prompt_template(condition: str, strategy: str) -> str:
    filename = PROMPT_FILES.get((condition, strategy))
    if not filename:
        raise ValueError(f"No prompt file for condition={condition} strategy={strategy}")
    if filename in _prompt_cache:
        return _prompt_cache[filename]

    candidates = [
        PROMPTS_DIR / filename,
        BASE_DIR / "prompts" / filename,
        Path(__file__).resolve().parent / "prompts" / filename,
        Path(__file__).resolve().parent.parent / "prompts" / filename,
    ]
    for path in candidates:
        if path.exists():
            template = path.read_text(encoding="utf-8")
            _prompt_cache[filename] = template
            log.info("Loaded prompt template: %s", path)
            return template
    raise FileNotFoundError(f"Prompt file '{filename}' not found in prompts/")


def build_prompt(
    deal: dict, condition: str, strategy: str, corpus_long: pd.DataFrame | None
) -> tuple[str, str]:
    template = load_prompt_template(condition, strategy)
    system   = SYSTEM_PROMPTS[condition]

    if condition == "A_name":
        profile = build_identity_only_profile(deal)
        user = template.format_map({"acquirer_profile": profile})
    elif condition == "B_named":
        if corpus_long is None:
            raise ValueError("corpus_long required for Condition B_named")
        profile        = build_masked_profile_with_description(deal)
        candidate_list = build_named_candidate_list(deal["deal_id"], corpus_long)
        user = template.format_map({
            "acquirer_profile": profile,
            "candidate_list":   candidate_list,
        })
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# API client + call
# ─────────────────────────────────────────────────────────────────────────────

def get_client(model_cfg: dict) -> openai.OpenAI:
    if model_cfg["provider"] != "openrouter":
        raise ValueError(f"Unsupported provider: {model_cfg['provider']}")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type((
        openai.APIConnectionError, openai.APIStatusError, openai.RateLimitError,
    )),
    reraise=True,
)
def call_api(
    client: openai.OpenAI, model_cfg: dict, system: str, user: str
) -> tuple[str, int, int]:
    kwargs = {
        "model":       model_cfg["model_id"],
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0,
        "max_tokens":  MAX_TOKENS,
    }
    extra = dict(model_cfg.get("extra_params", {}) or {})
    # Provider routing via request body (audit finding), same as the pipeline.
    extra.update(_prov.provider_routing(["Together", "DeepInfra", "Fireworks"]))
    if extra:
        kwargs["extra_body"] = extra

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    n_in  = getattr(usage, "prompt_tokens",     0) if usage else 0
    n_out = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, n_in, n_out


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_output_path(condition: str, model_cfg: dict, strategy: str) -> Path:
    dir_map = {"A_name": "probe_a_name", "B_named": "probe_b_named"}
    out_dir = RESULTS_DIR / dir_map[condition] / model_cfg["slug"] / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "responses_raw.jsonl"


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.open(encoding="utf-8"):
        try:
            done.add(json.loads(line)["deal_id"])
        except Exception:
            pass
    return done


def append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model_key: str, condition: str, strategy: str,
    corpus: list[dict], corpus_long: pd.DataFrame | None,
) -> None:
    model_cfg = MODELS[model_key]
    out_path  = get_output_path(condition, model_cfg, strategy)
    completed = load_completed_ids(out_path)

    log.info(
        "▶ model=%s | condition=%s | strategy=%s | to_do=%d (done=%d)",
        model_cfg["slug"], condition, strategy,
        len(corpus) - len(completed), len(completed),
    )

    client = get_client(model_cfg)
    n_ok, n_fail = 0, 0
    tot_in, tot_out = 0, 0
    t0 = time.time()

    for deal in tqdm(corpus, desc=f"{model_cfg['slug']}/{condition}"):
        did = deal["deal_id"]
        if did in completed:
            continue

        try:
            system, user = build_prompt(deal, condition, strategy, corpus_long)
        except Exception as e:
            log.error("Prompt build failed for %s: %s", did, e)
            n_fail += 1
            continue

        try:
            text, n_in, n_out = call_api(client, model_cfg, system, user)
            tot_in  += n_in
            tot_out += n_out
            n_ok += 1
        except Exception as e:
            log.error("API failed for %s: %s", did, e)
            n_fail += 1
            continue

        rec = {
            "deal_id":         did,
            "acquirer_name":   deal.get("acquirer_name"),
            "acquirer_ticker": deal.get("acquirer_ticker"),
            "target_name":     deal.get("target_name"),
            "target_ticker":   deal.get("target_ticker"),
            "target_position": deal.get("target_position"),
            "deal_year":       deal.get("deal_year"),
            "industry":        deal.get("industry"),
            "condition":       condition,
            "strategy":        strategy,
            "model_key":       model_key,
            "model_slug":      model_cfg["slug"],
            "model_used":      model_cfg["model_id"],
            "prompt_system":   system,
            "prompt_user":     user,
            "response":        text,
            "tokens_in":       n_in,
            "tokens_out":      n_out,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        append_record(out_path, rec)

    dt = time.time() - t0
    log.info(
        "[DONE] model=%s cond=%s | ok=%d fail=%d | tokens in/out=%d/%d | elapsed=%.1fs",
        model_cfg["slug"], condition, n_ok, n_fail, tot_in, tot_out, dt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Named-condition memorization probes")
    parser.add_argument(
        "--model", default="all",
        help=f"Model key or 'all'. Available: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--condition", default="all", choices=CONDITIONS + ["all"],
        help="Condition to run",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")
    if not CORPUS_LONG.exists():
        raise FileNotFoundError(f"Long-format corpus not found: {CORPUS_LONG}")

    corpus      = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus_long = pd.read_csv(CORPUS_LONG)
    log.info("Loaded corpus: %d deals, %d candidate rows",
             len(corpus), len(corpus_long))

    models     = list(MODELS.keys()) if args.model     == "all" else [args.model]
    conditions = CONDITIONS           if args.condition == "all" else [args.condition]

    for mk in models:
        if mk not in MODELS:
            log.error("Unknown model: %s", mk); continue
        for cond in conditions:
            long_arg = corpus_long if cond == "B_named" else None
            run_experiment(mk, cond, "zeroshot", corpus, long_arg)


if __name__ == "__main__":
    main()