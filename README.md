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

