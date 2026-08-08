# Capital IQ Screening Filters

Filters applied in Capital IQ's **Transaction Screener** to build the deal sample used in this project.

## Transaction type
- Transaction Type: `Merger/Acquisition`
- Transaction Status: `Completed`
- Announced Date: `01/01/2015 – 31/12/2022`

## Geography
- Acquirer Nation: `United States`
- Target Nation: `United States`

## Public status
- Acquirer Public Status: `Public Company`
- Target Public Status: `Public Company`

(Required so the acquirer's 10-K is available on SEC EDGAR and the target's ticker can serve as ground truth.)

## Deal size
- Transaction Value (USD): `Min 50,000,000` (no maximum)

## Sector exclusions
- Target Primary SIC Code: excluded `6000–6999` (financial sector)

## Quality filters
- Excluded deals where Acquirer = Target (self-tenders / buybacks)
- Deal Type: excluded `Buyback`, `Recapitalization`, `Spin-off`

## Notes
- These filters returned an initial pool of candidate deals, from which the final sample of **286 S&P 500 acquisitions (2015–2022)** was drawn.
- Capital IQ data is not redistributable; see `PIPELINE_EXECUTION_ORDER.md` for data access instructions.
