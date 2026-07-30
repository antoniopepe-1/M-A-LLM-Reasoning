"""
Canonical shared library for the LLM-as-M&A-Analyst experiment.
=================================================================

This package is the single source of truth for every component that must be
IDENTICAL across the experiment pipeline, the evaluation, the baselines, and the
audits. It was introduced to repair the validity defects documented in
`experiment_validity_audit_16c_onward.md`.

Design principle
----------------
Any quantity that is compared across methods (LLM vs BM25 vs embedding vs random)
must be produced by the SAME code path. Divergent, copy-pasted implementations of
the acquirer query, the ranking metric, or the target matcher are what caused
Findings 6 and 7 in the audit. Import them from here instead.

Modules
-------
  profile        build_acquirer_profile()  — canonical, leakage-free acquirer query
  entities       resolve target / candidate identities; redaction; entity dictionary
  metrics        MRR@10, Recall@k, NDCG@k, chance baseline, paired/clustered bootstrap
  parsing        strict ranking parser (terminal block / unique-ID) and name parser
  provenance     prompt/corpus/code hashing and OpenRouter provider routing
  corpus_io      load corpus + hard invariant assertions (50 unique IDs, one target)
"""

from . import profile, entities, metrics, parsing, provenance, corpus_io  # noqa: F401
