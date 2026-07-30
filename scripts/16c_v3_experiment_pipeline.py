"""
EXPERIMENTAL DESIGN
-------------------
Three conditions decompose LLM performance into distinct cognitive components:

  Condition A — Unconstrained Generation
    Input : anonymized acquirer profile (Item 1 + MD&A + key financials)
    Task  : generate 5 acquisition targets from memory
    Measures: memory retrieval from training data
    Metrics : Recall@1, Recall@5

  Condition B — Profile-Augmented Ranking
    Input : acquirer profile + 50 anonymous candidates (sector, size, full description)
    Task  : rank candidates by strategic fit
    Measures: reasoning + textual recognition
    Metrics : MRR, NDCG@10, Recall@10

  Condition C — Metadata-Only Ranking
    Input : acquirer profile + 50 anonymous candidates (sector, size only — no description)
    Task  : rank candidates by strategic fit
    Measures: structural reasoning without textual cues
    Metrics : MRR, NDCG@10, Recall@10

  Delta(B - C) isolates the contribution of textual recognition to ranking performance, which is the primary methodological contribution of this study.

ANONYMIZATION
-------------
Acquirer identity is fully masked before prompt construction to prevent models from retrieving memorized deal outcomes rather than reasoning about strategic fit.
Masking covers: ticker symbols, full legal names, HTML-encoded variants, exchange prefixes, and all significant name tokens appearing in the 10-K text.

MODEL ROSTER - v4 final (OpenRouter cloud only)
-----------------------------------------------
  Scaling curve, by increasing active parameters:

    llama32_1b   : Llama 3.2 1B    (1B dense)               - extreme lower bound (Meta)
    qwen3_8b     : Qwen3-8B        (8B dense)               - small Qwen
    phi4         : Phi-4           (14B dense)              - large Phi
    llama33_70b  : Llama 3.3 70B   (70B dense)              - cross-family dense (Meta)
    qwen3        : Qwen3-235B      (235B MoE, 22B active)   - large Qwen
    deepseek     : DeepSeek V3     (685B MoE, 37B active)   - top anchor

  Comparisons this roster supports:
    RQ3 scaling:      1B -> 8B -> 14B -> 70B -> 22B(MoE) -> 37B(MoE)
    Intra-family Phi: phi4 (14B) is the sole Microsoft entry (no 3.8B slot available)
    Intra-family Qwen: qwen3_8b (8B) vs qwen3 (22B active) - scale + architecture
    Dense vs MoE:     phi4 (14B) + llama33_70b (70B) dense vs qwen3/deepseek (MoE)

  All models are served through OpenRouter; requires OPENROUTER_API_KEY in .env.
  Qwen3 models run with non-thinking mode forced via extra_params.

temperature=0 across all models.


PROMPTING STRATEGIES
--------------------
  zeroshot : direct instruction, no examples
  cot      : explicit 4-step chain-of-thought before answer
  fewshot  : 3 out-of-sample demonstrations (pre-2010 deals, outside evaluation window)

OUTPUT
------
  results/open_ended/{model}/{strategy}/responses_raw.jsonl         (Condition A)
  results/retrieval_aug/{model}/{strategy}/responses_raw.jsonl      (Condition B)
  results/retrieval_aug_metadata/{model}/{strategy}/responses_raw.jsonl (Condition C)
  logs/pipeline_{timestamp}.log

------------
  pip install openai tenacity tqdm python-dotenv
  OPENROUTER_API_KEY in .env file at project root  (for DeepSeek)
  Ollama running on localhost:11434                 (for local models)

REPRODUCIBILITY
---------------
  Model versions are logged in the `model_used` field of every output record.
  Candidate list ordering is deterministically seeded by deal_id (see script 14b).
  Few-shot examples are restricted to deals announced before 2005 to prevent temporal contamination with the evaluation window (2015-2022).
"""

import os
import re
import html
import json
import time
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm
from dotenv import load_dotenv

# Canonical shared library — single source of truth for the acquirer profile,
# candidate stripping, provider routing and provenance. See scripts/lib/.
sys_path = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(sys_path))
from lib import profile as _profile
from lib import provenance as _prov
from lib import corpus_io as _corpus_io


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Locate .env file traversing from script directory to project root
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
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR     = BASE_DIR / "logs"
PROMPTS_DIR = BASE_DIR / "prompts"

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model roster
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Final roster - scaling curve from 1B to 37B active params (all OpenRouter)
    # Ordered by increasing active parameters
    # ------------------------------------------------------------------

    # Llama 3.2 1B - 1B dense - extreme lower bound (Meta family)
    "llama32_1b": {
        "model_id":     "meta-llama/llama-3.2-1b-instruct",
        "provider":     "openrouter",
        "slug":         "llama32_1b",
        "extra_params": {},
    },

    # Qwen3-8B - 8B dense - small Qwen (Qwen family)
    # Pinned to the 04-28 served version for reproducibility
    "qwen3_8b": {
        "model_id":     "qwen/qwen3-8b-04-28",
        "provider":     "openrouter",
        "slug":         "qwen3_8b",
        "extra_params": {"thinking": {"type": "disabled"}},
    },

    # Phi-4 — 14B dense — large Phi | confronto dense vs MoE
    "phi4": {
        "model_id":     "microsoft/phi-4",
        "provider":     "openrouter",
        "slug":         "phi4_14b",
        "extra_params": {},
    },

    # Llama 3.3 70B — 70B dense — cross-family dense (Meta)
    # Sostituisce Llama 4 Maverick (v3): endpoint stabile, costo ~$2.50 vs $2.71, Meta mantenuta
    "llama33_70b": {
        "model_id":     "meta-llama/llama-3.3-70b-instruct",
        "provider":     "openrouter",
        "slug":         "llama33_70b",
        "extra_params": {},
    },

    # Qwen3-235B — 235B MoE, 22B active — large Qwen
    "qwen3": {
        "model_id":     "qwen/qwen3-235b-a22b-2507",
        "provider":     "openrouter",
        "slug":         "qwen3_235b",
        "extra_params": {"thinking": {"type": "disabled"}},
    },

    # DeepSeek V3 — 685B MoE, 37B active — top anchor
    "deepseek": {
        "model_id":     "deepseek/deepseek-chat-v3-0324",
        "provider":     "openrouter",
        "slug":         "deepseek_v3",
        "extra_params": {},
    },
}

CONDITIONS   = ["A", "B", "C"]
STRATEGIES   = ["zeroshot", "cot", "fewshot"]
PILOT_N      = 10    # deals used in pilot validation runs
CHECKPOINT_N = 50    # log progress every N deals
MAX_TOKENS   = 2000  # raised to further reduce CoT truncation (audit: Qwen/Phi C-CoT)

# Concurrency for API calls WITHIN a single cell. This affects ONLY execution
# speed, not results: each call keeps the identical payload (temperature=0,
# top_p=1.0, prompt, route), so responses are unchanged vs. serial execution.
# The per-call retry/backoff (see call_api) absorbs the occasional 429 that a
# higher concurrency may trigger. Default kept conservative because qwen3_8b has
# hit upstream rate limits; lower via --workers for rate-limited providers.
DEFAULT_WORKERS = 4

# OpenRouter provider routing preference. Applied via the request BODY
# (extra_body["provider"]) per current OpenRouter docs — NOT the old
# X-Provider-Order header, which is not honoured (audit finding).
PROVIDER_ROUTE = ["Together", "DeepInfra", "Fireworks"]

# Derived acquirer-only budget proxy in the profile. OFF by default: the
# realized target market cap was removed entirely (audit Finding 1); enabling
# this substitutes an ex-ante, acquirer-only proxy instead.
INCLUDE_BUDGET = False


# ---------------------------------------------------------------------------
# Acquirer anonymization + profile + candidate stripping
# ---------------------------------------------------------------------------
# These now delegate to the canonical shared library (scripts/lib/profile.py)
# so the acquirer query is IDENTICAL across the pipeline and every baseline,
# and so the realized target's market cap can never re-enter the profile
# (audit Findings 1 and 7). Do not reintroduce local copies.

mask_acquirer = _profile.mask_acquirer


def build_acquirer_profile(deal: dict) -> str:
    """Leakage-free acquirer profile (no outcome-derived fields). See lib.profile."""
    return _profile.build_acquirer_profile(deal, include_budget=INCLUDE_BUDGET)


strip_descriptions = _profile.strip_descriptions


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
# Prompt templates are stored as .txt files in the prompts/ directory.
# Each file corresponds to one condition x strategy combination.
# The {acquirer_profile} and {candidate_list} placeholders are injected
# at runtime via str.format_map(), keeping prompt text fully editable
# without touching pipeline code.
#
# Expected filenames (configurable via PROMPT_FILES below):
#   prompts/A_zeroshot.txt   prompts/B_zeroshot.txt
#   prompts/A_cot.txt        prompts/B_cot.txt
#   prompts/A_fewshot.txt    prompts/B_fewshot.txt
#
# Condition C reuses the B templates — strip_descriptions() removes the
# Description field from the candidate list before injection, so no
# separate C template is needed.

PROMPT_FILES: dict[tuple[str, str], str] = {
    ("A", "zeroshot"): "A_zeroshot.txt",
    ("A", "cot"):      "A_cot.txt",
    ("A", "fewshot"):  "A_fewshot.txt",
    ("B", "zeroshot"): "B_zeroshot.txt",
    ("B", "cot"):      "B_cot.txt",
    ("B", "fewshot"):  "B_fewshot.txt",
    # Condition C reuses B templates (candidate list is stripped at runtime)
    ("C", "zeroshot"): "B_zeroshot.txt",
    ("C", "cot"):      "B_cot.txt",
    ("C", "fewshot"):  "B_fewshot.txt",
}

SYSTEM_PROMPTS: dict[str, str] = {
    "A": (
        "You are a senior strategic M&A advisor with deep expertise in "
        "corporate strategy, industry dynamics, and transaction structuring."
    ),
    "B": (
        "You are a senior strategic M&A advisor. Rank the anonymous acquisition "
        "candidates by their strategic fit with the given acquirer profile. "
        "Candidates are identified only by number — do not attempt to infer "
        "their real identities."
    ),
    "C": (
        "You are a senior strategic M&A advisor. Rank the anonymous acquisition "
        "candidates by their strategic fit with the given acquirer profile. "
        "Candidates are identified only by number — do not attempt to infer "
        "their real identities. Only structural metadata is available (sector, "
        "market cap, revenue) — base your ranking on these attributes alone."
    ),
}

# Cache loaded templates to avoid repeated disk reads across 297 deals
_prompt_cache: dict[str, str] = {}


def load_prompt_template(condition: str, strategy: str) -> str:
    """
    Load prompt template from file, with caching and clear error on missing file.

    Templates use {acquirer_profile} and {candidate_list} as placeholders.
    The file is read once per (condition, strategy) pair and cached for the
    remainder of the run.
    """
    filename = PROMPT_FILES.get((condition, strategy))
    if not filename:
        raise ValueError(f"No prompt file configured for condition={condition} strategy={strategy}")

    if filename in _prompt_cache:
        return _prompt_cache[filename]

    # Search in prompts/ relative to project root or script directory
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

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Prompt file '{filename}' not found. Searched:\n  {searched}\n"
        f"Create the file or check PROMPTS_DIR in configuration."
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(deal: dict, condition: str, strategy: str) -> tuple[str, str]:
    """
    Construct (system_prompt, user_prompt) for a given deal, condition, strategy.

    Loads the appropriate template from prompts/ directory and injects:
      - {acquirer_profile}  : anonymized acquirer profile (Item 1 + MD&A + financials)
      - {candidate_list}    : 50 anonymous candidates (B: full, C: metadata only)

    Condition C reuses the B template — candidate descriptions are stripped
    by strip_descriptions() before injection, so no separate template is needed.
    The system prompt for C explicitly notes the metadata-only constraint.
    """
    profile    = build_acquirer_profile(deal)
    system     = SYSTEM_PROMPTS[condition]
    template   = load_prompt_template(condition, strategy)

    if condition in ("B", "C"):
        raw_candidates = deal.get("candidate_list", "")
        candidate_list = (
            strip_descriptions(raw_candidates) if condition == "C" else raw_candidates
        )
        user = template.format_map({
            "acquirer_profile": profile,
            "candidate_list":   candidate_list,
        })
    else:  # A
        user = template.format_map({
            "acquirer_profile": profile,
        })

    return system, user


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def get_client(model_cfg: dict) -> openai.OpenAI:
    """
    Initialize the appropriate API client based on model provider.

    - "openrouter" : cloud inference via OpenRouter (requires OPENROUTER_API_KEY)
    - "ollama"     : local inference via Ollama (requires ollama serve on localhost:11434)

    Both return an openai.OpenAI client with compatible interfaces — the rest
    of the pipeline is provider-agnostic.
    """
    provider = model_cfg.get("provider", "openrouter")

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible endpoint on localhost
        # No real API key needed — "ollama" is a placeholder
        log.info("Using Ollama local inference for model: %s", model_cfg["model_id"])
        return openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    else:
        # OpenRouter cloud inference
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not found. Add it to .env at the project root."
            )
        # NB: provider ordering is NOT set here via X-Provider-Order (that header
        # is not honoured by OpenRouter). It is passed in the request body in
        # call_api() via extra_body["provider"] (audit finding).
        return openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/llm-ma-analyst",
                "X-Title":      "LLM-as-MA-Analyst",
            },
        )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
    ),
    reraise=True,
)
def call_api(
    client: openai.OpenAI,
    model_cfg: dict,
    system: str,
    user: str,
) -> dict:
    """
    Call OpenRouter API with automatic retry on transient failures.

    Retry policy: up to 5 attempts with exponential backoff (4s to 60s).
    Covers rate limit errors, connection errors, and API timeouts.
    The `model_used` field in the response captures the exact model version
    served, which may differ from `model_id` during provider-side updates.
    """
    start = time.time()

    kwargs: dict = dict(
        model=model_cfg["model_id"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,      # deterministic output; see paper §3.4 for caveats
        max_tokens=MAX_TOKENS,
        top_p=1.0,
    )

    # Build extra_body: model-specific params (e.g. disable Qwen thinking) PLUS
    # OpenRouter provider routing in the request body (audit finding). Ollama
    # does not support extra_body — skip for local models.
    if model_cfg.get("provider") != "ollama":
        extra_body = dict(model_cfg.get("extra_params") or {})
        extra_body.update(_prov.provider_routing(PROVIDER_ROUTE))
        if extra_body:
            kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)

    # Capture the provider actually used, when OpenRouter reports it.
    served_provider = None
    try:
        served_provider = getattr(response, "provider", None)
    except Exception:
        served_provider = None

    return {
        "response_text":  response.choices[0].message.content,
        "finish_reason":  response.choices[0].finish_reason,
        "model_used":     response.model,          # served alias (may differ from requested)
        "served_provider": served_provider,        # underlying provider, if reported
        "input_tokens":   response.usage.prompt_tokens,
        "output_tokens":  response.usage.completion_tokens,
        "latency_s":      round(time.time() - start, 2),
    }


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def get_output_path(condition: str, model_cfg: dict, strategy: str) -> Path:
    """Map (condition, model, strategy) to output JSONL path."""
    cond_dir = {
        "A": "open_ended",
        "B": "retrieval_aug",
        "C": "retrieval_aug_metadata",
    }[condition]
    out_dir = RESULTS_DIR / cond_dir / model_cfg["slug"] / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "responses_raw.jsonl"


def load_completed_ids(path: Path, fingerprint: str | None = None) -> set[str]:
    """
    Return deal_ids already completed FOR THIS EXACT EXPERIMENT.

    A deal counts as complete only if its record carries the current
    `experiment_fingerprint` (prompt+corpus+model+route). A changed prompt or
    corpus produces a different fingerprint, so stale outputs from an earlier
    version are NOT reused (audit: resume logic could mix experiment versions).
    Records without a fingerprint (legacy) are ignored when a fingerprint is
    supplied, forcing a clean re-run onto the corrected corpus.
    """
    if not path.exists():
        return set()
    completed: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if fingerprint is not None and rec.get("experiment_fingerprint") != fingerprint:
                continue
            if "deal_id" in rec:
                completed.add(rec["deal_id"])
    return completed


def append_record(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Cost tracking (informational — does not affect results)
# ---------------------------------------------------------------------------

class CostTracker:
    """
    Approximate cost estimation based on OpenRouter pricing (April 2026).
    Used for monitoring during run; not reported in paper results.
    """

    # USD per 1M tokens (input, output) — OpenRouter pricing April 2026
    _PRICES: dict[str, tuple[float, float]] = {
        "llama32_1b":  (0.027, 0.20),
        "qwen3_8b":    (0.03,  0.05),
        "phi4_14b":    (0.07,  0.14),
        "llama33_70b": (0.10,  0.32),   # llama-3.3-70b — sostituisce llama-4-maverick
        "qwen3_235b":  (0.07,  0.10),
        "deepseek_v3": (0.20,  0.77),
    }

    def __init__(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0
        self.cost = 0.0

    def add(self, slug: str, n_in: int, n_out: int) -> None:
        self.calls += 1
        self.input_tokens  += n_in
        self.output_tokens += n_out
        p_in, p_out = self._PRICES.get(slug, (1.0, 3.0))
        self.cost += n_in / 1e6 * p_in + n_out / 1e6 * p_out

    def summary(self) -> str:
        return (
            f"calls={self.calls} | "
            f"tokens_in={self.input_tokens:,} | "
            f"tokens_out={self.output_tokens:,} | "
            f"est_cost=${self.cost:.4f}"
        )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    condition: str,
    model_key: str,
    strategy:  str,
    pilot:     bool = False,
    dry_run:   bool = False,
    workers:   int  = DEFAULT_WORKERS,
) -> None:
    """
    Execute one cell of the experimental matrix: condition × model × strategy.

    Supports incremental execution: deals already present in the output JSONL
    are skipped, allowing safe resumption after interruption. Each output
    record contains the full experimental context needed for downstream
    evaluation and reproducibility verification.

    The `finish_reason` field is logged and monitored: a high rate of
    "length" truncations for CoT strategies would indicate MAX_TOKENS is
    insufficient and requires adjustment before reporting results.
    """
    model_cfg   = MODELS[model_key]
    output_path = get_output_path(condition, model_cfg, strategy)

    # Fingerprint this exact experiment (schema+parser+system+template+corpus+
    # model+route). Resume reuses a deal ONLY if its record shares this
    # fingerprint, so a changed prompt/corpus cannot silently reuse stale
    # outputs (audit: provenance/resume can mix experiment versions).
    _sys = SYSTEM_PROMPTS[condition]
    _tmpl = load_prompt_template(condition, strategy)
    fingerprint = _prov.experiment_fingerprint(
        system_prompt=_sys, user_template=_tmpl, corpus_path=CORPUS_PATH,
        model_id=model_cfg["model_id"], provider_route=PROVIDER_ROUTE,
    )
    completed   = load_completed_ids(output_path, fingerprint=fingerprint)

    # For local models, verify Ollama is reachable before starting
    if model_cfg.get("provider") == "ollama":
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:11434/v1/models", timeout=5)
            log.info("Ollama server reachable at localhost:11434")
        except Exception:
            raise RuntimeError(
                f"Ollama server not reachable at localhost:11434.\n"
                f"Start it with: ollama serve\n"
                f"Then pull the model: ollama pull {model_cfg['model_id']}"
            )

    log.info(
        "START | condition=%s model=%s strategy=%s workers=%d output=%s",
        condition, model_key, strategy, workers, output_path,
    )

    # Load corpus WITH hard invariant checks (audit Finding 4). A malformed
    # corpus aborts the run before any paid API call.
    corpus: list[dict] = _corpus_io.load_corpus(CORPUS_PATH, strict=True)

    deals      = corpus[:PILOT_N] if pilot else corpus
    deals_todo = [d for d in deals if d["deal_id"] not in completed]

    log.info(
        "corpus=%d deals | completed=%d | remaining=%d%s",
        len(corpus), len(completed), len(deals_todo),
        " [PILOT]" if pilot else "",
    )

    if not deals_todo:
        log.info("All deals already completed. Nothing to do.")
        return

    if dry_run:
        # Display prompt for first pending deal without making an API call
        system, user = build_prompt(deals_todo[0], condition, strategy)
        log.info("DRY RUN — prompt preview for deal %s", deals_todo[0]["deal_id"])
        print("\n[SYSTEM]\n" + system)
        print("\n[USER — first 1000 chars]\n" + user[:1000] + " ...")
        return

    client  = get_client(model_cfg)
    tracker = CostTracker()
    errors: list[dict] = []
    truncations = 0  # count of finish_reason == "length" (CoT risk)

    # One deal → one output record. This is a PURE per-deal unit of work: it
    # performs the identical prompt build + API call as the former serial loop
    # (same payload, temperature=0, top_p=1.0, route), so parallelizing it does
    # not alter any result — only the wall-clock time to produce it.
    def process_deal(deal: dict) -> dict:
        deal_id      = deal["deal_id"]
        system, user = build_prompt(deal, condition, strategy)
        result       = call_api(client, model_cfg, system, user)
        return {
            # Experimental metadata — needed for evaluation and reproducibility
            "deal_id":          deal_id,
            "acquirer_ticker":  deal.get("acquirer_ticker"),
            "acquirer_name":    deal.get("acquirer_name"),
            "target_name":      deal.get("target_name"),
            "target_ticker":    deal.get("target_ticker"),
            "deal_year":        deal.get("deal_year"),
            "industry":         deal.get("industry"),
            "condition":        condition,
            "model_key":        model_key,
            "model_id":         model_cfg["model_id"],
            "model_used":       result["model_used"],       # served alias
            "served_provider":  result.get("served_provider"),  # underlying provider
            "provider":         model_cfg["provider"],
            "strategy":         strategy,
            "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
            # Ground truth — used by evaluation script (17_evaluation.py)
            "target_position":  deal.get("target_position"),
            "n_candidates":     len((deal.get("candidate_list") or "").splitlines()),
            # Model response
            "response_text":    result["response_text"],
            "finish_reason":    result["finish_reason"],
            # Token counts and latency — for cost and efficiency analysis
            "input_tokens":     result["input_tokens"],
            "output_tokens":    result["output_tokens"],
            "latency_s":        result["latency_s"],
            # Full provenance block (prompt/corpus/route hashes, fingerprint)
            **_prov.record_provenance(
                fingerprint=fingerprint, system_prompt=system, user_prompt=user,
                corpus_path=CORPUS_PATH, model_id=model_cfg["model_id"],
                provider_route=PROVIDER_ROUTE,
            ),
        }

    # Concurrency changes only speed. The write to the JSONL and the shared
    # counters must be serialized, so they are guarded by a lock. `max_workers`
    # caps in-flight requests; that ceiling (plus call_api's retry/backoff) is
    # the rate-limit control that replaces the old serial time.sleep(0.5).
    write_lock = threading.Lock()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_deal, deal): deal for deal in deals_todo}
        for fut in tqdm(
            as_completed(futures),
            total=len(deals_todo),
            desc=f"cond={condition} model={model_key} strategy={strategy}",
        ):
            deal    = futures[fut]
            deal_id = deal["deal_id"]
            try:
                record = fut.result()
            except Exception as exc:
                log.error("FAILED deal=%s error=%s", deal_id, exc)
                with write_lock:
                    errors.append({"deal_id": deal_id, "error": str(exc)})
                continue

            # Warn if response was truncated — relevant for CoT strategy
            if record["finish_reason"] == "length":
                log.warning(
                    "Response truncated (finish_reason=length): deal=%s | "
                    "output_tokens=%d — consider increasing MAX_TOKENS for CoT",
                    deal_id, record["output_tokens"],
                )

            # Warn if model version served differs from requested
            if record["model_used"] != model_cfg["model_id"]:
                log.warning(
                    "Model version mismatch: requested=%s served=%s deal=%s",
                    model_cfg["model_id"], record["model_used"], deal_id,
                )

            with write_lock:
                if record["finish_reason"] == "length":
                    truncations += 1
                append_record(output_path, record)
                tracker.add(
                    model_cfg["slug"], record["input_tokens"], record["output_tokens"]
                )
                done += 1
                if done % CHECKPOINT_N == 0:
                    log.info(
                        "checkpoint %d/%d | %s", done, len(deals_todo), tracker.summary()
                    )

    if truncations:
        log.warning(
            "%d/%d responses truncated (finish_reason=length)",
            truncations, len(deals_todo),
        )

    # Persist error log for inspection
    if errors:
        err_path = output_path.parent / "errors.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        log.warning("%d errors written to %s", len(errors), err_path)

    if truncations > 0:
        log.warning(
            "%d/%d responses truncated (finish_reason=length) for "
            "condition=%s strategy=%s — verify CoT completeness before evaluation",
            truncations, len(deals_todo), condition, strategy,
        )

    log.info("DONE | %s", tracker.summary())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-M&A-Analyst experiment pipeline. "
            "Runs one or more cells of the condition × model × strategy matrix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Pilot: 10 deals, DeepSeek, Condition B, zero-shot\n"
            "  python 16c_experiment_pipeline.py --condition B --model deepseek "
            "--strategy zeroshot --pilot\n\n"
            "  # Full run: all conditions and strategies for one model\n"
            "  python 16c_experiment_pipeline.py --condition all --model deepseek "
            "--strategy all\n\n"
            "  # Complete experiment matrix (all models)\n"
            "  python 16c_experiment_pipeline.py --condition all --model all "
            "--strategy all"
        ),
    )

    parser.add_argument(
        "--condition",
        choices=CONDITIONS + ["all"],
        default="B",
        help=(
            "Experimental condition: "
            "A=unconstrained generation (memory), "
            "B=profile-augmented ranking (reasoning+recognition), "
            "C=metadata-only ranking (pure structural reasoning), "
            "all=run all three"
        ),
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="deepseek",
        help="Model key or 'all' to run the full model roster.",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES + ["all"],
        default="zeroshot",
        help="Prompting strategy or 'all' to run all three.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=f"Run on first {PILOT_N} deals only (validation before full run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt for first pending deal without calling the API.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"Concurrent API calls within a cell (default {DEFAULT_WORKERS}). "
            "Affects speed only, not results. Lower it for rate-limited "
            "providers (e.g. qwen3_8b); raise it for high-throughput ones."
        ),
    )

    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    conditions = CONDITIONS          if args.condition == "all" else [args.condition]
    models     = list(MODELS.keys()) if args.model    == "all" else [args.model]
    strategies = STRATEGIES          if args.strategy == "all" else [args.strategy]

    n_runs = len(conditions) * len(models) * len(strategies)
    log.info(
        "Experiment matrix: %d runs | conditions=%s models=%s strategies=%s | "
        "%s | %s | workers=%d",
        n_runs,
        conditions, models, strategies,
        "PILOT" if args.pilot else "FULL",
        "DRY_RUN" if args.dry_run else "LIVE",
        args.workers,
    )

    for cond in conditions:
        for model_key in models:
            for strategy in strategies:
                run_experiment(
                    condition=cond,
                    model_key=model_key,
                    strategy=strategy,
                    pilot=args.pilot,
                    dry_run=args.dry_run,
                    workers=args.workers,
                )


if __name__ == "__main__":
    main()
