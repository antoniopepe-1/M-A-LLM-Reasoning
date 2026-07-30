"""
lib.entities — entity resolution and identity redaction.
========================================================

Repairs audit Findings 2 (Condition B not anonymous), 3 (post-deal candidate
text) and 5 (Condition A false-positive matcher).

Three responsibilities:

  1. normalize_name() / build_entity_dict()
     Deterministic normalization and an auto-built dictionary mapping every
     target (and every candidate) to its canonical name, aliases and ticker.

  2. redact_identity()
     Entity-aware redaction of a business description: legal names, aliases,
     tickers, "d/b/a"/"formerly" constructions, URLs. Also strips POST-DEAL
     language ("was acquired by", "reverse merger", "SPAC") so descriptions are
     ex-ante (Finding 3 is handled at the data level too, this is defence in
     depth). Returns (redacted_text, residual_identity_found).

  3. resolve_match()
     Conservative Condition-A matcher. Declares a hit ONLY when a predicted name
     resolves to the target entity via exact normalized name, alias, ticker, or
     a HIGH fuzzy threshold on the FULL normalized string — never on a single
     shared token (the original single-token rule produced ~570 false positives,
     Finding 5).
"""

from __future__ import annotations

import re
from functools import lru_cache

try:
    from rapidfuzz import fuzz
    _HAVE_FUZZ = True
except ImportError:  # pragma: no cover
    _HAVE_FUZZ = False


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_LEGAL_SUFFIX_RE = re.compile(
    r"\b(incorporated|inc|corp|corporation|llc|l\.l\.c|ltd|limited|plc|"
    r"holdings?|holding|group|co|company|international|technologies|technology|"
    r"sa|nv|ag|se|lp|l\.p|trust|bancorp|financial|solutions)\b\.?",
    re.IGNORECASE,
)
_PAREN_RE  = re.compile(r"\([^)]*\)")
_DBA_RE    = re.compile(r"\b(d/b/a|dba|formerly|f/k/a|fka|n/k/a|nka)\b.*$", re.IGNORECASE)
_PUNCT_RE  = re.compile(r"[^\w\s]")
_WS_RE     = re.compile(r"\s+")

# Generic corporate words that must never, on their own, license a match.
STOP_TOKENS = {
    "the", "and", "of", "for", "group", "holdings", "holding", "company",
    "corp", "inc", "co", "international", "systems", "technologies",
    "technology", "solutions", "services", "capital", "partners", "global",
    "industries", "networks", "communications", "financial", "pharmaceuticals",
    "energy", "resources", "healthcare", "health", "media", "digital",
}


def normalize_name(name: str) -> str:
    """Canonical lowercase form: drop parentheticals, d/b/a tails, legal suffixes, punctuation."""
    if not name:
        return ""
    s = str(name).lower()
    s = _DBA_RE.sub("", s)
    s = _PAREN_RE.sub(" ", s)
    s = _LEGAL_SUFFIX_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def name_aliases(name: str) -> set[str]:
    """
    Alias surface forms for masking / matching. Includes the raw d/b/a and
    'formerly' operands, which are explicit alternate identities.
    """
    aliases: set[str] = set()
    if not name:
        return aliases
    raw = str(name).strip()
    aliases.add(raw)

    # d/b/a "One Medical" -> capture "One Medical"
    m = re.search(r"\b(?:d/b/a|dba|formerly|f/k/a|fka|n/k/a|nka)\b\s+(.+)$", raw, re.IGNORECASE)
    if m:
        aliases.add(m.group(1).strip().rstrip("."))

    # Base name before any comma / paren / d/b/a
    base = _DBA_RE.sub("", raw)
    base = _PAREN_RE.sub("", base).split(",")[0].strip()
    if base:
        aliases.add(base)
        no_suffix = _LEGAL_SUFFIX_RE.sub("", base).strip().rstrip(",").strip()
        if no_suffix and len(no_suffix) >= 3:
            aliases.add(no_suffix)

    return {a for a in aliases if a and len(a) >= 3}


# ---------------------------------------------------------------------------
# Entity dictionary
# ---------------------------------------------------------------------------

def build_entity_dict(rows: list[dict],
                      name_key: str = "target_name",
                      ticker_key: str = "target_ticker") -> dict:
    """
    Build {normalized_name -> {"canonical", "ticker", "aliases"}} from data rows.

    Auto-built from the gold/CIQ table (per the user's choice). Borderline / very
    short names are still resolvable but flagged by `ambiguous_entities()`.
    """
    ent: dict[str, dict] = {}
    for r in rows:
        name = r.get(name_key)
        ticker = (r.get(ticker_key) or "").strip().upper() or None
        if not name:
            continue
        norm = normalize_name(name)
        if not norm:
            continue
        rec = ent.setdefault(norm, {"canonical": name, "ticker": ticker, "aliases": set()})
        rec["aliases"].update(name_aliases(name))
        if ticker:
            rec["ticker"] = ticker
    return ent


def ambiguous_entities(ent: dict) -> list[str]:
    """Normalized names that are too short / generic to resolve safely — for review."""
    out = []
    for norm in ent:
        toks = [t for t in norm.split() if t not in STOP_TOKENS]
        if not toks or len(norm) < 4:
            out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Identity redaction  (Findings 2 and 3)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|io|co)\b", re.IGNORECASE)
_TICKER_PAREN_RE = re.compile(r"\((?:NYSE|NASDAQ|NASDAQGS|NYSEMKT|AMEX|OTC)[:\s]*[A-Z.]{1,6}\)", re.IGNORECASE)
# v2 audit Finding E1: bare exchange names ("listed on NASDAQ") survived the
# parenthetical rule. An exchange name is not an identity, but it is a redaction
# residual, so mask the bare token as well.
_BARE_EXCHANGE_RE = re.compile(
    r"\b(?:NYSE|NASDAQ|NASDAQGS|NASDAQGM|NASDAQCM|NYSEMKT|NYSEArca|AMEX|OTCQB|OTCQX|OTC)\b",
    re.IGNORECASE,
)

# Post-deal / non-point-in-time language. Whole sentences containing these are
# dropped so a description can never announce the realized outcome.
_POSTDEAL_RE = re.compile(
    r"(was acquired by|has been acquired|to be acquired|reverse merger|"
    r"business combination|special purpose acquisition|\bSPAC\b|"
    r"completed (?:its|the) (?:merger|acquisition)|following the (?:merger|acquisition)|"
    r"as of \w+ \d{1,2},? \d{4})",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def redact_identity(text: str,
                    canonical_name: str,
                    aliases: set[str] | None = None,
                    ticker: str | None = None,
                    placeholder: str = "[COMPANY]",
                    max_year: int | None = None) -> tuple[str, bool]:
    """
    Entity-aware redaction of a candidate business description.

    Returns (redacted_text, residual_identity_found).

    Steps:
      1. Drop sentences containing post-deal language (ex-ante requirement), and
         — when `max_year` is given (the deal year) — sentences asserting a year
         STRICTLY AFTER the deal (e.g. "incorporated in 2021" for a 2019 deal):
         such text cannot be point-in-time and would leak the future.
      2. Mask URLs and (EXCHANGE:TICKER) constructions.
      3. Mask every alias surface form and the ticker as a whole token.
      4. Mask 'd/b/a'/'formerly' operands.
      5. Re-scan for residual distinctive tokens.
    """
    if not text:
        return "", False

    aliases = set(aliases or set())
    aliases |= name_aliases(canonical_name)

    def _future_year(s: str) -> bool:
        if max_year is None:
            return False
        return any(int(y) > max_year for y in re.findall(r"\b(20\d{2})\b", s))

    # 1. Drop post-deal AND future-dated sentences
    kept = [s for s in _split_sentences(text)
            if not _POSTDEAL_RE.search(s) and not _future_year(s)]
    red = " ".join(kept).strip()
    if not red:
        # Whole description was post-deal; nothing safe remains.
        return "", False

    # 2. URLs, exchange:ticker, bare exchange names (Finding E1)
    red = _URL_RE.sub(placeholder, red)
    red = _TICKER_PAREN_RE.sub(placeholder, red)
    red = _BARE_EXCHANGE_RE.sub(placeholder, red)

    # 3. alias surface forms, longest first
    for surface in sorted(aliases, key=len, reverse=True):
        if len(surface) < 3:
            continue
        red = re.sub(re.escape(surface), placeholder, red, flags=re.IGNORECASE)

    # ticker as standalone token
    if ticker and len(ticker) >= 2:
        red = re.sub(r"\b" + re.escape(ticker) + r"\b", placeholder, red, flags=re.IGNORECASE)

    # 4. residual d/b/a operands
    red = _DBA_RE.sub(placeholder, red)

    # absorb a legal suffix left dangling right after a placeholder
    # e.g. "[COMPANY] , Inc." / "[COMPANY] Inc." -> "[COMPANY]"
    red = re.sub(
        r"\[COMPANY\]\s*,?\s*(?:Inc|Corp|Corporation|LLC|Ltd|Limited|PLC|"
        r"Holdings?|Group|Co|Company|International|Technologies?)\.?",
        placeholder, red, flags=re.IGNORECASE,
    )

    # collapse placeholders / whitespace
    red = re.sub(r"(\[COMPANY\]\s*)+", "[COMPANY] ", red)
    red = re.sub(r"\s+([,.;])", r"\1", red)
    red = _WS_RE.sub(" ", red).strip()

    # 5. residual identity scan
    distinctive = [t for t in normalize_name(canonical_name).split() if t not in STOP_TOKENS and len(t) >= 4]
    residual = any(re.search(r"\b" + re.escape(t) + r"\b", red, re.IGNORECASE) for t in distinctive)

    return red, residual


# ---------------------------------------------------------------------------
# Conservative Condition-A matcher  (Finding 5)
# ---------------------------------------------------------------------------

FUZZY_THRESHOLD = 90  # token_sort_ratio on the FULL normalized string


@lru_cache(maxsize=4096)
def _norm_cached(s: str) -> str:
    return normalize_name(s)


def resolve_match(predicted_name: str,
                  target_name: str,
                  target_ticker: str | None = None,
                  target_aliases: tuple[str, ...] = ()) -> bool:
    """
    Does a single predicted name resolve to the target ENTITY?

    Conservative resolution — a hit requires ONE of:
      - exact normalized-name equality with the target or any alias,
      - the predicted string contains the target ticker as a standalone token,
      - a fuzzy score >= FUZZY_THRESHOLD on the FULL normalized string,
      - the target's distinctive multi-token core is fully contained in the
        prediction (e.g. "aerohive networks" vs "aerohive networks inc").

    NEVER matches on a single shared generic token. This is the fix for the
    single-token rule that scored Aryaka==Aerohive, Main Street==American, etc.
    """
    if not predicted_name or not target_name:
        return False

    pred_norm = _norm_cached(predicted_name)
    tgt_norm  = _norm_cached(target_name)
    if not pred_norm or not tgt_norm:
        return False

    # exact normalized match against target + aliases
    candidates = {tgt_norm}
    candidates.update(_norm_cached(a) for a in target_aliases)
    candidates.discard("")
    if pred_norm in candidates:
        return True

    # ticker as standalone token in the raw prediction.
    # v2 audit Finding E2: a 2-char ticker that is also an English word (e.g.
    # "IT", "AI", "SO") could false-positive inside prose. Harden by requiring
    # the ticker to appear in its UPPERCASE surface form (tickers are cased;
    # prose words are not), which keeps every real ticker hit while rejecting
    # lowercase common-word coincidences. Tickers of length >= 3 are distinctive
    # enough to match case-insensitively.
    if target_ticker and len(target_ticker) >= 2:
        tk = re.escape(target_ticker)
        if len(target_ticker) >= 3:
            if re.search(r"\b" + tk + r"\b", predicted_name, re.IGNORECASE):
                return True
        else:
            # 2-char ticker: require exact-case (uppercase) standalone token
            if re.search(r"\b" + tk + r"\b", predicted_name):
                return True

    # distinctive multi-token core containment
    tgt_core = [t for t in tgt_norm.split() if t not in STOP_TOKENS]
    if len(tgt_core) >= 2:
        core = " ".join(tgt_core)
        if core and core in pred_norm:
            return True

    # high fuzzy on full string
    if _HAVE_FUZZ:
        for cand in candidates:
            if fuzz.token_sort_ratio(pred_norm, cand) >= FUZZY_THRESHOLD:
                # require at least one distinctive shared token to avoid
                # generic-word-driven high scores
                shared = (set(pred_norm.split()) & set(cand.split())) - STOP_TOKENS
                if any(len(w) >= 4 for w in shared):
                    return True
    return False


def match_target_in_topk(predicted_names: list[str],
                         target_name: str,
                         k: int,
                         target_ticker: str | None = None,
                         target_aliases: tuple[str, ...] = ()) -> bool:
    """True if the target entity resolves within the top-k predictions."""
    for name in predicted_names[:k]:
        if resolve_match(name, target_name, target_ticker, target_aliases):
            return True
    return False
