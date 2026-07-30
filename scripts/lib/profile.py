"""
lib.profile — canonical, leakage-free acquirer profile / IR query.
==================================================================

Repairs audit Findings 1 (outcome leakage) and 7 (baselines received a
different query than the LLM).

Two rules are enforced here so they cannot drift between scripts:

  1. NO OUTCOME-DERIVED FIELDS. The realized target's market cap
     (`target_mktcap_usd_m`) and any other property of the realized deal
     (deal value, target revenue, target name/ticker) are NEVER placed in the
     acquirer query. They are properties of the outcome, not pre-deal acquirer
     information. Their presence in `build_acquirer_profile()` was Finding 1.

     If an ex-ante notion of "how big a target can this acquirer afford" is
     wanted, `estimate_acquisition_budget()` derives it from ACQUIRER-ONLY
     variables and is clearly labelled as a derived predictor. It is OFF by
     default and must be opted into explicitly.

  2. ONE QUERY, EVERY METHOD. `build_query_text()` returns the exact bag of
     text handed to lexical / embedding baselines, derived from the SAME
     masked, truncated fields as the LLM profile. LLM, BM25, TF-IDF and the
     embedding models must all call this function.

Acquirer identity is masked by `mask_acquirer()` (moved here unchanged in
behaviour from the original pipeline) so Condition A/B/C never leak the
acquirer name.
"""

from __future__ import annotations

import re
import html

# Truncation budget (chars). Kept identical to the historical pipeline so that
# corrected results remain comparable in token budget to the original design.
ITEM1_CHARS = 400
MDA_CHARS   = 300


# ---------------------------------------------------------------------------
# Acquirer masking (behaviour preserved from 16c_v3_experiment_pipeline.py)
# ---------------------------------------------------------------------------

_LEGAL_SUFFIX = (
    r",?\s*(Inc\.?|Corp\.?|Corporation|LLC|Ltd\.?|Limited|PLC|"
    r"Holdings?|Group|Co\.?|Company|International|Technologies?)$"
)


def _build_mask_variants(deal: dict) -> list[str]:
    """Ordered acquirer name variants to mask, longest first."""
    raw_name = (deal.get("acquirer_name") or "").strip()
    ticker   = (deal.get("acquirer_ticker") or "").strip()
    variants: set[str] = set()

    if raw_name:
        variants.add(raw_name)

    paren_idx = raw_name.find("(")
    if paren_idx > 0:
        base_with_suffix = raw_name[:paren_idx].strip()
        variants.add(base_with_suffix)
        base_clean = re.sub(_LEGAL_SUFFIX, "", base_with_suffix,
                            flags=re.IGNORECASE).strip().rstrip(",").strip()
        if base_clean and base_clean != base_with_suffix:
            variants.add(base_clean)

    m = re.search(r"\(([A-Z]+:[A-Z.]+)\)", raw_name)
    if m:
        variants.add(f"({m.group(1)})")
        variants.add(m.group(1))

    if ticker and len(ticker) >= 2:
        variants.add(ticker)
        if ":" not in ticker:
            for exchange in ("NYSE", "NASDAQ", "NASDAQGS"):
                variants.add(f"{exchange}:{ticker}")

    variants = {v for v in variants if len(v) >= 4}
    return sorted(variants, key=len, reverse=True)


def mask_acquirer(text: str, deal: dict, placeholder: str = "[ACQUIRER]") -> str:
    """Replace all identifiable acquirer name variants with a placeholder."""
    if not text:
        return text

    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("​", "").replace("﻿", "")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')

    variants   = _build_mask_variants(deal)
    multiword  = [v for v in variants if " " in v]
    singleword = [v for v in variants if " " not in v]

    for v in multiword:
        text = re.sub(re.escape(v), placeholder, text, flags=re.IGNORECASE)
    for v in singleword:
        text = re.sub(r"\b" + re.escape(v) + r"\b", placeholder, text, flags=re.IGNORECASE)

    raw_name  = deal.get("acquirer_name") or ""
    paren_idx = raw_name.find("(")
    base      = raw_name[:paren_idx].strip() if paren_idx > 0 else raw_name
    base_clean = re.sub(_LEGAL_SUFFIX, "", base, flags=re.IGNORECASE).strip().rstrip(",").strip()

    stopwords = {"and", "the", "for", "with", "from", "into", "that", "this", "its",
                 "company", "companies", "group", "holdings", "corporation", "soup"}
    # Strip possessive 's so "Campbell's" also masks "Campbell".
    sig_words = [re.sub(r"['’]s?$", "", w) for w in base_clean.split()]
    sig_words = [w for w in sig_words if len(w) >= 5 and w.lower() not in stopwords]
    if len(sig_words) >= 2:
        for w in sig_words:
            text = re.sub(r"\b" + re.escape(w) + r"\b", placeholder, text, flags=re.IGNORECASE)
    elif len(sig_words) == 1 and len(sig_words[0]) >= 6:
        # Single distinctive token (e.g. "Southern", "Campbell"): still mask it —
        # a ≥6-char proper-noun acquirer identifier is not a common English word
        # in this context, and leaving it un-masked leaks the acquirer identity.
        text = re.sub(r"\b" + re.escape(sig_words[0]) + r"\b", placeholder, text, flags=re.IGNORECASE)

    text = re.sub(r"(\[ACQUIRER\]\s*)+", "[ACQUIRER] ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Ex-ante acquisition budget (derived predictor — OFF by default)
# ---------------------------------------------------------------------------

def estimate_acquisition_budget(deal: dict) -> str | None:
    """
    Ex-ante, ACQUIRER-ONLY proxy for feasible target size.

    Uses only pre-deal acquirer variables (revenue). NEVER touches
    target_mktcap_usd_m or any realized-deal quantity. Returned as a coarse
    bucket string and explicitly labelled 'derived' in the profile.

    Heuristic: acquirers typically pursue targets whose enterprise value is a
    fraction of the acquirer's own scale. We bucket acquirer revenue and expose
    the bucket only; the exact number is withheld to avoid fingerprinting.
    """
    rev = deal.get("acquirer_revenue_m")
    try:
        rev = float(rev)
    except (TypeError, ValueError):
        return None
    if rev < 500:
        return "small (<$0.5B revenue acquirer)"
    if rev < 5000:
        return "mid ($0.5-5B revenue acquirer)"
    if rev < 20000:
        return "large ($5-20B revenue acquirer)"
    return "mega (>$20B revenue acquirer)"


# ---------------------------------------------------------------------------
# Canonical profile and query
# ---------------------------------------------------------------------------

def build_acquirer_profile(deal: dict, include_budget: bool = False) -> str:
    """
    Canonical anonymized acquirer profile injected into the LLM prompt.

    Contains ONLY pre-deal, acquirer-side information:
      - masked 10-K Item 1 (core business)
      - masked 10-K MD&A (stated strategy)
      - acquirer revenue and EBITDA margin

    The realized target's market cap is intentionally ABSENT (audit Finding 1).
    If `include_budget` is True, an ex-ante, acquirer-only budget proxy is added
    and labelled 'DERIVED (acquirer-only)' so it can never be mistaken for the
    realized target size.
    """
    item1 = mask_acquirer((deal.get("item1_text") or "")[:500], deal)[:ITEM1_CHARS]
    mda   = mask_acquirer((deal.get("mda_text")   or "")[:400], deal)[:MDA_CHARS]

    lines = [
        f"ACQUIRER PROFILE -- Company A, FY{deal['deal_year'] - 1}",
        f"CORE BUSINESS: {item1}",
        f"STATED STRATEGY: {mda}",
        f"KEY FINANCIALS: "
        f"Revenue ${deal.get('acquirer_revenue_m', 'N/A')}M | "
        f"EBITDA margin {deal.get('acquirer_ebitda_margin_pct', 'N/A')}%",
    ]

    if include_budget:
        budget = estimate_acquisition_budget(deal)
        if budget:
            lines.append(f"DERIVED (acquirer-only) ACQUISITION BUDGET PROXY: {budget}")

    return "\n".join(lines)


def build_query_text(deal: dict, include_budget: bool = False) -> str:
    """
    Canonical IR query for the lexical and embedding baselines.

    Field-IDENTICAL to `build_acquirer_profile()` so that RQ4 compares generative
    ranking against semantic similarity on the SAME input information (audit
    Findings 7 and v2-A). Baselines MUST call this and must NOT re-read raw
    unmasked item1/mda.

    v2 audit Finding A: the acquirer `industry` string was previously prepended
    here but is ABSENT from the LLM profile — the model must infer the sector
    from prose. Because candidate lines begin `Industry: <sector>`, an explicit
    industry token handed a directly-matchable signal to BM25/TF-IDF/embeddings
    that the LLM never received, biasing RQ4 in the baselines' favour. The
    industry field is therefore removed so both sides serialize {item1, mda,
    financials-free prose} identically.
    """
    item1 = mask_acquirer((deal.get("item1_text") or "")[:500], deal)[:ITEM1_CHARS]
    mda   = mask_acquirer((deal.get("mda_text")   or "")[:400], deal)[:MDA_CHARS]
    parts = [item1, mda]
    if include_budget:
        budget = estimate_acquisition_budget(deal)
        if budget:
            parts.append(budget)
    return " ".join(p for p in parts if p)


def strip_descriptions(candidate_list: str) -> str:
    """
    Remove the Description field from each candidate line (Condition C).

    Canonical version shared by the pipeline and every baseline so B and C use
    identical candidate serializations.
    """
    cleaned = []
    for line in candidate_list.splitlines():
        idx = line.find("| Description:")
        cleaned.append(line[:idx].rstrip() if idx != -1 else line)
    return "\n".join(cleaned)
