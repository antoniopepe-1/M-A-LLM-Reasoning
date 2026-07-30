"""
01_clean_deals.py
=================
Cleans the raw Capital IQ export into the deal sample used by the pipeline.

    RAW -> [step 1] rename columns
        -> [step 2] drop duplicate deal_id
        -> [step 3] drop rows with NA in critical columns
        -> CLEAN

Each step writes a backup CSV to data/processed/backups/.
Final output: data/processed/deal_sample_clean.csv
"""

import pandas as pd
import os

# ── Paths ────────────────────────────────────────────────────────────────
INPUT_FILE   = "data/raw/SPGlobal_Export_3-27-2026.csv"
BACKUP_DIR   = "data/processed/backups"
OUTPUT_FILE  = "data/processed/deal_sample_clean.csv"

os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Helper ───────────────────────────────────────────────────────────────────
def log(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def snapshot(df, step_name):
    """Write a backup CSV for the step and print its row/column counts."""
    path = os.path.join(BACKUP_DIR, f"{step_name}.csv")
    df.to_csv(path, index=False)
    print(f"  [OK] Backup saved: {path}")
    print(f"  [INFO] Rows: {len(df)} | Columns: {len(df.columns)}")
    return df

# ════════════════════════════════════════════════════════════════════════════
# LOAD RAW FILE
# ════════════════════════════════════════════════════════════════════════════
log("LOADING RAW FILE")

# The export is an Excel file despite its .csv extension; the real header
# sits on row index 2, so data starts on the following row.
df = pd.read_excel(INPUT_FILE, header=2)

# Drop the empty leading column (Unnamed: 0 is all-NA)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Drop the duplicated name column (identical to Target/Issuer Name)
df = df.drop(columns=["Target/Issuer Name.1"], errors="ignore")

print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 - RENAME COLUMNS
# ════════════════════════════════════════════════════════════════════════════
log("STEP 1 - Rename columns")

RENAME_MAP = {
    "Target/Issuer Name"                                        : "target_name",
    "MI Transaction ID"                                         : "deal_id",
    "Announced Date\nMM/dd/yyyy"                               : "announced_date",
    "Completion Date\nMM/dd/yyyy"                              : "closed_date",
    "Definitive Agreement Date\nMM/dd/yyyy"                    : "def_agreement_date",
    "Transaction Type"                                          : "transaction_type",
    "Transaction Status"                                        : "status",
    "Description of Consideration"                             : "consideration_desc",
    "Deal Summary"                                             : "deal_summary",
    "Buyers/Investors Name"                                    : "acquirer_name",
    "Buyer Ticker(s)"                                          : "acquirer_ticker",
    "Actual Acquirer Country/Region"                           : "acquirer_country",
    "Target/Issuer Ticker"                                     : "target_ticker",
    "SIC Code\n(Target/Issuer)"                               : "target_sic",
    "Target Country/ Region"                                   : "target_country",
    "Target/Issuer Business Description"                       : "target_business_desc",
    "Target/Issuer Long Business Description"                  : "target_long_desc",
    "Total Transaction Value\n($M)"                            : "deal_value_usd_m",
    "Buyer: CIQ Total Revenue\n($000)"                        : "acquirer_revenue_ciq_000",
    "Buyer: CIQ EBITDA Incl EQ Income from Affiliates\n($000)": "acquirer_ebitda_ciq_000",
    "Buyer: Total Revenue\n($000)"                             : "acquirer_revenue_000",
    "Buyer: EBITDA\n($000)"                                    : "acquirer_ebitda_000",
    "Buyer: Total Equity\n($000)"                              : "acquirer_equity_000",
    "Target: Period Ended\nMM/dd/yyyy"                        : "target_period_end",
    "Target: CIQ Total Revenue\n($000)"                       : "target_revenue_ciq_000",
    "Target: CIQ EBITDA Incl EQ Income from Affiliates\n($000)": "target_ebitda_ciq_000",
    "Target: Market Capitalization\n($M)"                     : "target_mktcap_usd_m",
    "Premium Paid\n($000)"                                     : "premium_paid_000",
    "Deal Premium 1 Day Before\n(%)"                          : "premium_1day_pct",
    "Deal Premium 1 Week Before\n(%)"                         : "premium_1week_pct",
    "Deal Premium 1 Year Before\n(%)"                         : "premium_1year_pct",
    "Consideration: Cash\n($M)"                               : "consideration_cash_m",
    "Consideration: Common Stock\n($M)"                       : "consideration_stock_m",
    "Consideration: Earnout/ Contingent Payments\n($M)"       : "consideration_earnout_m",
    "Adviser Name\n(All Roles Target/Issuer)"                 : "adviser_target",
    "Adviser Name\n(All Roles Buyer/Investor)"                : "adviser_acquirer",
}

df = df.rename(columns=RENAME_MAP)

print("  Renamed columns:")
for col in df.columns:
    print(f"    {col}")

snapshot(df, "step1_renamed")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 - DROP DUPLICATES
# ════════════════════════════════════════════════════════════════════════════
log("STEP 2 - Drop duplicates")

before = len(df)

# deal_id is the unique transaction identifier
df = df.drop_duplicates(subset=["deal_id"], keep="first")

after = len(df)
print(f"  Duplicates removed: {before - after}")
print(f"  Rows remaining: {after}")

snapshot(df, "step2_no_duplicates")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 - DROP ROWS WITH NA IN CRITICAL COLUMNS
# ════════════════════════════════════════════════════════════════════════════
log("STEP 3 - Drop NA in critical columns")

# Without these fields a deal cannot be used in the study
CRITICAL_COLS = [
    "target_name",        # ground truth
    "deal_id",            # unique deal identifier
    "announced_date",     # needed to match the correct 10-K
    "closed_date",        # confirms the deal completed
    "acquirer_name",      # acquiring company
    "acquirer_ticker",    # used to look up the 10-K on EDGAR
    "target_ticker",      # used for ground truth and anonymization
    "target_sic",         # used for industry stratification
    "deal_value_usd_m",   # used for size stratification
    "target_mktcap_usd_m",# used to build the candidate pool
]

before = len(df)

# Report NA counts before dropping so the loss is auditable
print("\n  NA count per critical column (before removal):")
for col in CRITICAL_COLS:
    na_count = df[col].isna().sum()
    print(f"    {col}: {na_count} NA")

# Drop only rows missing a critical field
df = df.dropna(subset=CRITICAL_COLS)

after = len(df)
print(f"\n  Rows dropped for critical NA: {before - after}")
print(f"  Rows remaining: {after}")

snapshot(df, "step3_no_critical_na")

# ════════════════════════════════════════════════════════════════════════════
# WRITE FINAL OUTPUT
# ════════════════════════════════════════════════════════════════════════════
log("WRITING FINAL FILE")

df.to_csv(OUTPUT_FILE, index=False)
print(f"Clean file saved: {OUTPUT_FILE}")
print(f"Final size: {len(df)} rows x {len(df.columns)} columns")

# ── Final summary ─────────────────────────────────────────────────────────
log("PIPELINE SUMMARY")
print(f"  Raw file:              1130 rows")
print(f"  After step 1 (rename): {len(pd.read_csv(BACKUP_DIR+'/step1_renamed.csv'))} rows")
print(f"  After step 2 (dedup):  {len(pd.read_csv(BACKUP_DIR+'/step2_no_duplicates.csv'))} rows")
print(f"  After step 3 (NA):     {len(df)} rows")
print(f"\n  Backups saved in:      {BACKUP_DIR}/")
print(f"  Final file:            {OUTPUT_FILE}")