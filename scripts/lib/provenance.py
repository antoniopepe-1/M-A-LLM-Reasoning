"""
lib.provenance — hashing and OpenRouter provider routing.
=========================================================

Repairs the "provenance and resume logic can mix experiment versions" and
"X-Provider-Order header is not the documented routing mechanism" findings.

  * hash_text() / experiment_fingerprint() give every run a stable fingerprint
    over {system prompt, user prompt template, corpus, parser version, model,
    provider route}. The fingerprint is stored on every record and used as part
    of the resume key so a changed prompt or corpus can never silently reuse a
    stale output.

  * provider_routing() returns the request-BODY `provider` object
    (order / allow_fallbacks) per current OpenRouter docs — NOT the legacy
    `X-Provider-Order` header, which is not honoured.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Bump when the parsing contract or prompt schema changes so old outputs are
# not silently reused by a resume.
PARSER_VERSION = "strict-v1"
PROMPT_SCHEMA_VERSION = "structured-v1"


def hash_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def hash_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def experiment_fingerprint(*, system_prompt: str, user_template: str,
                           corpus_path: str | Path, model_id: str,
                           provider_route: list[str] | None) -> str:
    """Stable fingerprint identifying exactly which experiment produced a record."""
    return hash_text(
        PROMPT_SCHEMA_VERSION,
        PARSER_VERSION,
        system_prompt,
        user_template,
        hash_file(corpus_path),
        model_id,
        ",".join(provider_route or []),
    )


def provider_routing(order: list[str] | None = None,
                     allow_fallbacks: bool = True) -> dict:
    """
    OpenRouter provider routing via the REQUEST BODY (current API).

    Returns a dict suitable for `extra_body={"provider": ...}`. The legacy
    `X-Provider-Order` header did not enforce ordering and must not be used.
    """
    if not order:
        return {}
    return {"provider": {"order": list(order), "allow_fallbacks": allow_fallbacks}}


def record_provenance(*, fingerprint: str, system_prompt: str, user_prompt: str,
                      corpus_path: str | Path, model_id: str,
                      provider_route: list[str] | None, seed: int = 0) -> dict:
    """Provenance block written into every JSONL record."""
    return {
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "experiment_fingerprint": fingerprint,
        "system_prompt_hash": hash_text(system_prompt),
        "user_prompt_hash": hash_text(user_prompt),
        "corpus_hash": hash_file(corpus_path),
        "requested_model_id": model_id,
        "provider_route": provider_route or [],
        "sampling_seed": seed,
    }
