# Congressional Disclosure Backtest (Prototype)

## Overview

This project parses U.S. congressional financial disclosure PDFs (Periodic Transaction Reports and annual reports),
extracts transaction records, and constructs a disclosures-only signal suitable for slow-turnover analysis and backtesting.

**Compliance posture (hard boundary):**
- Public disclosures only
- No pre-disclosure inference
- Signal availability is anchored to `filing_datetime` (public availability), not transaction date

This is an early-stage prototype intended for hypothesis testing and data/ETL hardening — not production use.

---

## Repository Layout
ppf/
├── unify_ppf.py
├── ppf_pipeline.py
├── ppf_make_mapping_skeleton.py
├── ppf_backtest_simple.py
├── ppf_master.py
├── data/
│ ├── raw/ # input PDFs live here (Senate eFD PTRs, annuals, House PDFs, etc.)
│ ├── processed/ # normalized intermediate datasets
│ └── reference/ # deterministic mapping tables
├── outputs/ # final outputs + manifests
└── logs/ # run logs


---

## Key Data Products

### A) Unified Transactions (raw-ish, normalized columns)
**File:** `outputs/ppf_transactions_unified.csv`  
**Produced by:** `unify_ppf.py`

Core fields include:
- `ticker` (raw extracted ticker; may be `--` for non-ticker instruments)
- `asset_name`
- `asset_type`
- `transaction_type` (BUY/SELL)
- `amount_bucket_raw`
- `official_name`
- `filing_datetime` (availability anchor)

### B) Enriched Transactions (analysis-ready)
**File:** `data/processed/transactions_enriched.csv`  
**Produced by:** `ppf_pipeline.py`

Ticker handling (this answers “are tickers present”):
- `ticker`: carried from unified input (if present)
- `ticker_norm`: cleaned/validated ticker string
- `asset_id`: **primary identifier**; equals `ticker_norm` when a ticker exists, else `issuer_norm`

The pipeline never discards tickers; it prefers them as the join key for equities.

### C) Mapping Tables (deterministic + auditable)
**File:** `data/reference/asset_to_sector_mapping.csv`  
**Produced by:** `ppf_make_mapping_skeleton.py` (skeleton) + user edits and/or optional auto-map

Used to map `asset_id` → `sector/subsector`. For equities, `asset_id` is normally the ticker.

**Optional:** `data/reference/sector_to_etf.csv`  
Used to express sectors as liquid ETF proxies for portfolio simulation.

---

## How To Run (Minimal)

From `PROJECT_ROOT`:

### Step 1 — Parse PDFs into unified transactions
Place disclosure PDFs in:

Outputs:

outputs/ppf_transactions_unified.csv

outputs/ppf_superset_all_records.csv

outputs/extraction_manifest.json

data/processed/pdf_pages.csv

data/processed/annual_table_rows.csv


Then run:

```bash
python unify_ppf.py --project-root . --verbose

python ppf_make_mapping_skeleton.py --project-root . --verbose

Outputs:

data/reference/asset_to_sector_mapping.csv

Populate sector/subsector columns (or use optional auto-map if enabled in your pipeline workflow).

Step 3 — Enrich + apply mapping + compute impulses/scores
python ppf_pipeline.py --project-root . --verbose --auto-map-yfinance


Outputs:

data/processed/transactions_enriched.csv

outputs/ppf_asset_universe.csv

outputs/ppf_assets_unmapped.csv

outputs/ppf_impulses_sector_daily.csv

outputs/ppf_scores_sector_daily.csv

outputs/ppf_scores_sector_snapshot_latest.csv

outputs/ppf_pipeline_manifest.json

Step 4 — Backtest scaffold
python ppf_backtest_simple.py --project-root . --verbose


Notes:

If sector_to_etf.csv is empty, the backtest will exit cleanly with warnings and a summary JSON.

Outputs (when enough mappings exist):

outputs/ppf_backtest_quarterly_portfolios.csv

outputs/ppf_backtest_summary.json

One Command (Master Orchestration)

If your environment and flags are configured:

python ppf_master.py --project-root . --verbose


This should execute staged runs (crawl → unify → pipeline → analysis) depending on enabled stages.

# ----------------------------------------------------------------------------------------------
# Old version below here
# Congressional Disclosure Backtest (Prototype)

## Overview

This project parses U.S. congressional financial disclosure PDFs (Periodic Transaction Reports),
extracts transaction data, and simulates a simple strategy that follows disclosed BUY trades.

The objective is to test a hypothesis:

> Does following disclosed congressional BUY transactions outperform SPY?

This repository currently contains:

- A PDF ingestion and normalization pipeline
- A simple quarterly backtest engine
- A basic SPY benchmark comparison

This is an early-stage prototype intended for hypothesis testing — not production use.

---

# What Has Been Built

## 1. PDF Ingestion Pipeline

`ppf_pipeline.py`

Extracts from disclosure PDFs:

- Official name
- Filing date (public availability anchor)
- Transaction date
- Ticker symbol
- Asset name
- Transaction type (BUY / SELL)
- Amount bucket
- Derived min/max USD bucket
- Normalized ticker (`asset_id`)
- Signed transaction impulse

ppf/
├── ppf_pipeline.py
├── ppf_backtest_simple.py
├── README.md
├── data/
│ ├── raw/ # Place disclosure PDFs here
│ ├── processed/
│ │ └── transactions_enriched.csv
│ └── reference/
│ └── asset_to_sector_mapping.csv
├── outputs/
│ ├── ppf_impulses_sector_daily.csv
│ ├── ppf_scores_sector_daily.csv
│ ├── ppf_scores_sector_snapshot_latest.csv
│ └── ppf_pipeline_manifes

Do everything:
python ppf_master.py \
  --project-root . \
  --do-house --house-start-year 2015 --house-end-year 2026 --house-headless --house-download-pdfs \
  --do-senate --senate-since 2012-01-01 --senate-download \
  --do-unify \
  --do-pipeline --auto-map-yfinance \
  --do-analysis \
  --verbose



Get Senate data:
python ppf_crawl_efd.py --project-root . --since 2012-01-01 --max-pages 0 --download

Get Congress data:
python ppf_house_crawler.py \
  --project-root . \
  --filing-year 2026 \
  --prefix-range A Z \
  --download-pdfs \
  --headless


How To Run
Step 1 – Parse Disclosure PDFs

Place disclosure PDFs into:

data/raw/


Then run:

python ppf_pipeline.py --project-root . --verbose


This generates:

data/processed/transactions_enriched.csv


and several output diagnostic files.

Step 2 – Run Backtest
python ppf_backtest_simple.py


This will:

Construct quarterly portfolios

Pull historical prices via yfinance

Simulate portfolio returns

Compare to SPY

Display a performance plot

Generated Files
From Pipeline
data/processed/transactions_enriched.csv


Clean normalized transaction dataset.

outputs/ppf_impulses_sector_daily.csv


Signed sector impulses (if sector mapping used).

outputs/ppf_scores_sector_daily.csv


Daily PPF scores.

outputs/ppf_scores_sector

