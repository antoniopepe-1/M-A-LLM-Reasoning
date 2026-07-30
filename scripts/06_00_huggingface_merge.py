"""
06_00_huggingface_merge.py
==========================
Enriches the cleaned Capital IQ sample with pre-extracted 10-K text from the
public HuggingFace dataset `jlohding/sp500-edgar-10k`.

This is an alternative source to the EDGAR scrape in script 03: where a match is
found, Item 1 and MD&A come already extracted and cleaned, which yields better
text quality than the regex extraction on raw filings.

Join logic:
  For each Capital IQ deal, look for the HuggingFace row with
    - the same normalized acquirer name, and
    - filing year = announced_year - 1 (10k_year), allowing +/-1 year
  On a match, take Item 1 (item_1) and MD&A (item_7).

Output in data/hf_merge/:
  hf_raw.parquet              full HuggingFace dataset, cached locally
  hf_merge_results.csv        all deals with their join outcome
  hf_merge_matched.csv        matched deals only (with Item 1 + MD&A)
  hf_merge_unmatched.csv      deals with no HuggingFace match

"""

import pandas as pd
import re
import os

# ── Configuration ─────────────────────────────────────────────────────────
CIQ_FILE    = "data/processed/deal_sample_clean.csv"
OUT_DIR     = "data/hf_merge"
HF_PARQUET  = os.path.join(OUT_DIR, "hf_raw.parquet")
RESULTS_CSV = os.path.join(OUT_DIR, "hf_merge_results.csv")
MATCHED_CSV = os.path.join(OUT_DIR, "hf_merge_matched.csv")
UNMATCHED_CSV = os.path.join(OUT_DIR, "hf_merge_unmatched.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

# ══════════════════════════════════════════════════════════════════
# STEP 1 - Load the HuggingFace dataset
# ══════════════════════════════════════════════════════════════════
def load_huggingface():
    log("STEP 1 - Loading HuggingFace dataset")

    # Reuse the local cache when available
    if os.path.exists(HF_PARQUET):
        print(f"Loading from local cache: {HF_PARQUET}")
        df_hf = pd.read_parquet(HF_PARQUET)
        print(f"{len(df_hf):,} rows loaded")
        return df_hf

    print("Downloading from HuggingFace...")
    from datasets import load_dataset
    ds    = load_dataset("jlohding/sp500-edgar-10k")
    df_hf = ds["train"].to_pandas()

    # Cache locally for subsequent runs
    df_hf.to_parquet(HF_PARQUET, index=False)
    print(f"{len(df_hf):,} rows downloaded and cached in {HF_PARQUET}")

    return df_hf

# ══════════════════════════════════════════════════════════════════
# STEP 2 - Prepare both datasets for the join
# ══════════════════════════════════════════════════════════════════
def normalize_name(name):
    """
    Normalize a company name for matching: lowercase it, strip the exchange /
    ticker in parentheses, drop legal-form and generic tokens (Inc., Corp.,
    Group, Holdings, ...), and remove punctuation and extra whitespace.
    """
    name = str(name).lower()
    name = re.sub(r'\([^)]*\)', '', name)          # drop e.g. (NASDAQ:AMZN)
    name = re.sub(r'\b(inc|corp|ltd|llc|plc|co|the|group|holdings|company|international|technologies|technology|solutions|services|systems)\b', '', name)
    name = re.sub(r'[^\w\s]', '', name)            # drop punctuation
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def prepare_datasets(df_ciq, df_hf):
    log("STEP 2 - Preparing datasets for the join")

    # Capital IQ
    df_ciq = df_ciq.copy()
    df_ciq["announced_date"] = pd.to_datetime(df_ciq["announced_date"])
    df_ciq["deal_year"]      = df_ciq["announced_date"].dt.year
    df_ciq["10k_year"]       = df_ciq["deal_year"] - 1
    df_ciq["name_norm"]      = df_ciq["acquirer_name"].apply(normalize_name)

    # HuggingFace
    df_hf = df_hf.copy()
    df_hf["date"]      = pd.to_datetime(df_hf["date"])
    df_hf["hf_year"]   = df_hf["date"].dt.year
    df_hf["name_norm"] = df_hf["company"].apply(normalize_name)

    print(f"  Capital IQ deals:      {len(df_ciq)}")
    print(f"  HuggingFace rows:      {len(df_hf)}")
    print(f"  HuggingFace years:     {df_hf['hf_year'].min()} - {df_hf['hf_year'].max()}")
    print(f"  HuggingFace companies: {df_hf['name_norm'].nunique()}")

    return df_ciq, df_hf

# ══════════════════════════════════════════════════════════════════
# STEP 3 - Join on normalized name + year
# ══════════════════════════════════════════════════════════════════
def do_merge(df_ciq, df_hf):
    log("STEP 3 - Joining Capital IQ x HuggingFace")

    # Index the HF rows by (name_norm, hf_year) for O(1) lookup
    hf_index = {}
    for _, row in df_hf.iterrows():
        key = (row["name_norm"], int(row["hf_year"]))
        hf_index[key] = row

    results = []
    matched = unmatched = 0

    for _, ciq_row in df_ciq.iterrows():
        name  = ciq_row["name_norm"]
        year  = int(ciq_row["10k_year"])
        result = ciq_row.to_dict()

        # Try the exact year first, then +/-1
        hf_row = None
        for yr in [year, year - 1, year + 1]:
            key = (name, yr)
            if key in hf_index:
                hf_row = hf_index[key]
                break

        if hf_row is not None:
            # Match found - attach the HuggingFace fields
            result["hf_match"]        = True
            result["hf_company"]      = hf_row["company"]
            result["hf_cik"]          = hf_row["cik"]
            result["hf_sic"]          = hf_row["sic"]
            result["hf_filing_date"]  = hf_row["date"]
            result["hf_filing_year"]  = int(hf_row["hf_year"])
            result["hf_mktcap"]       = hf_row.get("mkt_cap")
            result["item1_text"]      = hf_row.get("item_1", "")
            result["mda_text"]        = hf_row.get("item_7", "")
            result["item1_ok"]        = len(str(hf_row.get("item_1", ""))) > 200
            result["mda_ok"]          = len(str(hf_row.get("item_7", ""))) > 200
            result["hf_match_method"] = f"name_norm+year_{yr}"
            matched += 1
            print(f"{ciq_row['acquirer_ticker']:<8} -> {hf_row['company']} ({yr})")
        else:
            result["hf_match"]        = False
            result["hf_company"]      = None
            result["hf_cik"]          = None
            result["hf_filing_date"]  = None
            result["hf_filing_year"]  = None
            result["hf_mktcap"]       = None
            result["item1_text"]      = ""
            result["mda_text"]        = ""
            result["item1_ok"]        = False
            result["mda_ok"]          = False
            result["hf_match_method"] = "no_match"
            unmatched += 1
            print(f"{ciq_row['acquirer_ticker']:<8} -> no HuggingFace match (searched: '{name}', year {year})")

        results.append(result)

    df_results = pd.DataFrame(results)

    print(f"\n  Matched:          {matched} / {len(df_ciq)}")
    print(f"  Unmatched:        {unmatched} / {len(df_ciq)}")
    print(f"  Match rate:       {matched/len(df_ciq)*100:.1f}%")

    return df_results

# ══════════════════════════════════════════════════════════════════
# STEP 4 - Write output and summary
# ══════════════════════════════════════════════════════════════════
def save_results(df_results):
    log("STEP 4 - Writing results")

    df_matched   = df_results[df_results["hf_match"] == True].copy()
    df_unmatched = df_results[df_results["hf_match"] == False].copy()

    # Final column order
    final_cols = [
        # Deal identifiers
        "deal_id", "acquirer_name", "acquirer_ticker",
        "target_name", "target_ticker",
        # Dates
        "announced_date", "closed_date", "deal_year", "10k_year",
        # Deal info
        "transaction_type", "status", "deal_value_usd_m",
        "target_sic", "acquirer_country", "target_country",
        # Acquirer financials (Capital IQ)
        "acquirer_revenue_ciq_000", "acquirer_ebitda_ciq_000",
        "acquirer_revenue_000", "acquirer_ebitda_000", "acquirer_equity_000",
        # Target financials (Capital IQ)
        "target_revenue_ciq_000", "target_ebitda_ciq_000", "target_mktcap_usd_m",
        # Premiums
        "premium_1day_pct", "premium_1week_pct", "premium_1year_pct",
        # Consideration
        "consideration_cash_m", "consideration_stock_m",
        # HuggingFace metadata
        "hf_match", "hf_match_method", "hf_company", "hf_cik",
        "hf_sic", "hf_filing_date", "hf_filing_year", "hf_mktcap",
        # 10-K text
        "item1_ok", "mda_ok", "item1_text", "mda_text",
    ]
    final_cols = [c for c in final_cols if c in df_results.columns]

    df_results[final_cols].to_csv(RESULTS_CSV, index=False)
    df_matched[final_cols].to_csv(MATCHED_CSV, index=False)
    df_unmatched[final_cols].to_csv(UNMATCHED_CSV, index=False)

    log("FINAL SUMMARY")
    print(f"  Capital IQ deals:                {len(df_results)}")
    print(f"  Matched in HuggingFace:          {len(df_matched)}")
    print(f"  Unmatched:                       {len(df_unmatched)}")
    print(f"  With Item 1 extracted:           {df_matched['item1_ok'].sum()}")
    print(f"  With MD&A extracted:             {df_matched['mda_ok'].sum()}")
    print(f"  With both (complete deals):      {(df_matched['item1_ok'] & df_matched['mda_ok']).sum()}")

    print(f"\n  Files written to {OUT_DIR}/")
    print(f"    hf_raw.parquet           full HuggingFace dataset (cache)")
    print(f"    hf_merge_results.csv     all deals with join outcome")
    print(f"    hf_merge_matched.csv     matched deals only")
    print(f"    hf_merge_unmatched.csv   unmatched deals")

    # Year distribution of matched deals
    print(f"\n  Distribution by year (matched deals):")
    print(df_matched["deal_year"].value_counts().sort_index().to_string())

    # Print a few merged profiles as a sanity check
    log("SAMPLE: 3 PROFILES FROM THE MERGED DATASET")
    sample = df_matched[df_matched["item1_ok"] & df_matched["mda_ok"]].head(3)
    for _, row in sample.iterrows():
        print(f"""
  ────────────────────────────────────────────────
  ACQUIRER: {row['acquirer_name']} ({row['acquirer_ticker']})
  TARGET:   {row['target_name']}
  YEAR:     {row['deal_year']} | 10-K: FY{row['hf_filing_year']}
  HF match: {row['hf_company']}

  ITEM 1 (first 300 chars):
  {str(row['item1_text'])[:300]}...

  MD&A (first 200 chars):
  {str(row['mda_text'])[:200]}...
  ────────────────────────────────────────────────""")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Step 1: load the HuggingFace dataset
    df_hf = load_huggingface()

    # Step 2: load Capital IQ and prepare both sides
    df_ciq_raw       = pd.read_csv(CIQ_FILE)
    df_ciq, df_hf_prep = prepare_datasets(df_ciq_raw, df_hf)

    # Step 3: join
    df_results = do_merge(df_ciq, df_hf_prep)

    # Step 4: write results
    save_results(df_results)