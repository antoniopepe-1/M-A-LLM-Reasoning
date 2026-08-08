# LLM-as-M&A-Analyst — Code and Prompts

This repository provides the complete code and prompt templates used in the paper. Licensed input data and generated experimental outputs are not redistributed. DATA_ACQUISITION.md documents the filters, required fields, expected sample counts, and reconstruction procedure.

The experiment compares six models under three conditions:

- **Condition A:** unconstrained target generation from an anonymized acquirer
  profile;
- **Condition B:** ranking of 50 anonymized candidates with business
  descriptions;
- **Condition C:** ranking of the same candidates using metadata only.

Each condition is evaluated with zero-shot, chain-of-thought, and few-shot
prompting.

## Repository contents

```text
prompts/       Exact prompt templates
scripts/       Dataset construction, experiments, evaluation, and robustness tests
scripts/lib/   Shared parsing, masking, metrics, and provenance utilities
scripts/archive/  Legacy scripts retained only for development provenance
```

## Data availability

The study combines:

1. **S&P Capital IQ** transaction data;
2. public **SEC EDGAR 10-K filings**;
3. the public HuggingFace dataset `jlohding/sp500-edgar-10k`.

Capital IQ data are licensed and cannot be redistributed. Consequently, the
`data/` directory is not included in the repository.

To reproduce the final sample, the following author-provided artifact is also
required:

```text
data/beautifulsoup/combined_fixed_final.csv
```

The current repository does not contain the intermediate manual-validation step
that generates this file. This limitation prevents a fully automated
reconstruction of the final 286-deal sample from the raw export alone.

## Installation

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install datasets==2.18.0 pyarrow==15.0.0 \
  rank-bm25==0.2.2 rapidfuzz==3.6.1 scipy==1.12.0 httpx==0.27.2
cp .env.example .env
```

Add the OpenRouter API key to `.env`:

```dotenv
OPENROUTER_API_KEY=...
```

`SEC_CONTACT_EMAIL` is needed only by the archived alternative EDGAR ingestion
script and is not required by the canonical workflow below.

## Reproduction workflow

All commands must be run from the repository root.

### 1. Prepare the data

Place the Capital IQ export at:

```text
data/raw/SPGlobal_Export_3-27-2026.csv
```

Despite its extension, the file is read as the original Excel workbook.

```bash
python scripts/01_clean_deals.py
python scripts/06_00_huggingface_merge.py
```

After adding `data/beautifulsoup/combined_fixed_final.csv`, construct the
canonical point-in-time candidate corpus:

```bash
python scripts/14b_rebuild_pointintime_corpus.py
python scripts/18_audit_leakage.py
```

The resulting experimental corpus is:

```text
data/candidate_corpus/corpus_wide_final.json
```

### 2. Run the experiments

A pilot run can be used to validate the configuration:

```bash
python scripts/16c_v3_experiment_pipeline.py \
  --condition B --model llama32_1b --strategy zeroshot --pilot
```

Run the complete condition × model × strategy grid with:

```bash
python scripts/16c_v3_experiment_pipeline.py \
  --condition all --model all --strategy all
```

Raw responses are written under `results/open_ended/`,
`results/retrieval_aug/`, and `results/retrieval_aug_metadata/`. Interrupted
runs can be resumed; stored experiment fingerprints prevent responses generated
with different prompts or corpora from being silently reused.

### 3. Evaluate the results

```bash
python scripts/17_evaluation.py --condition all
python scripts/25_baseline_retrieval.py
python scripts/29_embedding_baseline.py
python scripts/30_supplementary_analysis.py
```

Evaluation tables and deal-level metrics are written to
`results/evaluation/`.

### 4. Run robustness checks

```bash
python scripts/19_memorization_check.py
python scripts/27_unknown_subset_analysis.py
python scripts/28_memorization_dataset_check_v2.py
python scripts/31_probe_named_conditions.py
python scripts/32_evaluate_named_probes.py
```

Legacy and superseded scripts are retained under `scripts/archive/` for
development provenance and are not part of the replication workflow.

## Reproducibility notes

Candidate sampling is deterministic and seeded by deal identifier. Model calls
use `temperature=0`, while each raw record stores model, provider, timestamp,
and hashes of the corpus and prompts. Exact bitwise reproduction cannot be
guaranteed because inference is performed through remote model providers whose
backends may change over time.

## License

The code is released under the MIT License. This license does not cover
third-party data, including S&P Capital IQ data, which remain subject to their
original terms of use.
