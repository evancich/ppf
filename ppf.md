Gemini said
Congressional Alpha: PPF Engine Documentation
This document provides the complete setup, execution logic, and analytical framework for the Politician Portfolio Factor (PPF) Engine. This system is designed to audit congressional stock disclosures and calculate "Alpha" (excess return) relative to the S&P 500.

1. Project Overview
The PPF Engine quantifies the "Information Advantage" of congressional traders. It processes raw disclosure data from the House and Senate, synchronizes it with historical market data via Yahoo Finance, and calculates performance across 5, 20, and 60-day horizons.

The Alpha Formula
The engine calculates alpha as the true excess return over the benchmark (SPY), adjusted for the trade direction (Impulse):

Alpha=(( 
P 
entry
​
 
P 
exit
​
 
​
 −1)−( 
S 
entry
​
 
S 
exit
​
 
​
 −1))×Impulse
Where:

P: Price of the asset.

S: Price of the SPY benchmark.

Impulse: +1 for Buys/Activity, −1 for Sells.

2. Directory Structure
Ensure your workspace is organized as follows:

Plaintext
.
├── ppf_setup.py            # Environment initialization
├── ppf_full_stack2.py      # Master Audit Engine
├── outputs/
│   ├── ppf_final_signal_debug_rows.csv   # Senate Source
│   └── house/
│       └── parsed_2024.csv               # House Source
└── data/                   # Auto-generated market data cache
3. The Setup Script (ppf_setup.py)
Run this script first to initialize folders and verify that your source data is in the correct location.

Python
import os
from pathlib import Path

def setup_ppf_environment():
    print("═══ PPF ENGINE SETUP ═══")
    
    # 1. Define required directory structure
    required_dirs = ["outputs/house", "data/reference", "logs"]
    
    print("\n[1/3] Creating directories...")
    for d in required_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  + Created: {d}")

    # 2. Check for dependencies
    print("\n[2/3] Checking dependencies...")
    try:
        import pandas
        import yfinance
        print("  - Pandas: FOUND")
        print("  - YFinance: FOUND")
    except ImportError:
        print("  ! MISSING: Run 'pip install pandas yfinance'")

    # 3. Verify Data Presence
    files_to_check = {
        "Senate Data": "outputs/ppf_final_signal_debug_rows.csv",
        "House Data": "outputs/house/parsed_2024.csv"
    }

    all_ready = True
    for label, path in files_to_check.items():
        if Path(path).exists():
            print(f"  - {label}: FOUND")
        else:
            print(f"  ! {label}: MISSING at {path}")
            all_ready = False

    if all_ready:
        print("\n✅ SETUP COMPLETE: Ready for audit.")

if __name__ == "__main__":
    setup_ppf_environment()
4. The Master Audit Engine (ppf_full_stack2.py)
This is the core processor. It handles identity resolution, market data ingestion, and alpha computation.

Python
import pandas as pd
import yfinance as yf
from pathlib import Path
import re
import sys

# --- CONFIG ---
SUMMARY_FILE = "congressional_performance_report.csv"
RAW_AUDIT_FILE = "congress_raw_audit_data.csv"
SENATE_PATH = Path("outputs/ppf_final_signal_debug_rows.csv")
HOUSE_PATH = Path("outputs/house/parsed_2024.csv")

ID_MAP = {
    'B195126E-7BB2-4D54-BAF5-E6FC8E7A0165': 'WARNER',
    'D7CAE837-F73C-4EB1-B9FB-E510F53D65DE': 'TUBERVILLE',
    '98B3317B-2632-48C3-B636-D5555C4680CA': 'GRASSLEY',
    '7AB0D2FD-19EA-459D-A1D9-3701E2CC8E93': 'THUNE',
    'BD886067-927F-48A8-9D43-8AB6FA713F98': 'CARPER',
    '996378A6-200B-48F6-96F0-A3A1F45E445E': 'MORAN'
}

def clean_name(name_str):
    if not isinstance(name_str, str): return "UNKNOWN"
    s = name_str.upper()
    s = re.sub(r'\b(HON|MRS|MR|DR|MS|REP|SEN|SENATOR|REPRESENTATIVE)\b\.?', '', s)
    s = re.sub(r'[^A-Z\s\-]', '', s)
    return s.strip()

def get_price_on_or_after(price_series, target_date, max_date):
    target = pd.to_datetime(target_date).tz_localize(None)
    if target > max_date: return None, None
    available_dates = price_series.index.tz_localize(None)
    for date in available_dates:
        if date >= target: return price_series.loc[date], date
    return None, None

def load_trades(path, chamber):
    if not path.exists(): return []
    raw = pd.read_csv(path)
    trades = []
    for _, row in raw.iterrows():
        t = re.sub(r"[^A-Z\-]", "", str(row.get('ticker', '')).upper())
        if not (2 <= len(t) <= 5): continue
        
        if chamber == "HOUSE":
            name = (row.get('filer') or row.get('representative') or 
                    row.get('name') or row.get('politician') or "UNKNOWN_HOUSE")
        else:
            name = row.get('filer')
            if pd.isna(name) or str(name).upper() in ["UNKNOWN", "UNKNOWN_SENATE"]:
                m = re.search(r"\{?([A-F0-9-]{36})\}?", str(row.get("source_file", "")), flags=re.I)
                name = ID_MAP.get(m.group(1).upper() if m else "", "UNKNOWN_SENATE")
        
        name = clean_name(name)
        dt_val = row.get('filing_datetime') or row.get('disclosure_date') or row.get('filing_date')
        if pd.isna(dt_val): continue
        dt = pd.to_datetime(dt_val).tz_localize(None)
        
        impulse = 1 if chamber == "HOUSE" else (-1 if str(row.get('impulse')) == '-1' else 1)
        trades.append({'ticker': t, 'chamber': chamber, 'filer': name, 'date': dt, 'impulse': impulse})
    return trades

def main():
    all_trades = load_trades(SENATE_PATH, "SENATE") + load_trades(HOUSE_PATH, "HOUSE")
    tickers = list(set([t['ticker'] for t in all_trades])) + ['SPY']
    print(f"Fetching market data for {len(tickers)} symbols...")
    market_data = yf.download(tickers, start="2023-01-01", auto_adjust=True, progress=False)['Close']
    max_market_date = market_data.index.max().tz_localize(None)

    raw_audit_log = []
    for i, trade in enumerate(all_trades):
        ticker, impulse = trade['ticker'], trade['impulse']
        sys.stdout.write(f"\rAuditing trade {i+1}/{len(all_trades)}...")
        
        if ticker not in market_data.columns: continue
        p_series, spy_series = market_data[ticker].dropna(), market_data['SPY'].dropna()
        p_entry, dt_entry = get_price_on_or_after(p_series, trade['date'] + pd.Timedelta(days=1), max_market_date)
        s_entry, _ = get_price_on_or_after(spy_series, trade['date'] + pd.Timedelta(days=1), max_market_date)
        
        if p_entry is None or s_entry is None: continue

        for h in [5, 20, 60]:
            exit_target = dt_entry + pd.Timedelta(days=h)
            p_exit, dt_exit = get_price_on_or_after(p_series, exit_target, max_market_date)
            s_exit, _ = get_price_on_or_after(spy_series, exit_target, max_market_date)
            
            record = {
                'Filer': trade['filer'], 'Chamber': trade['chamber'], 'Ticker': ticker,
                'Filing_Date': trade['date'].date(), 'Horizon': h,
                'Entry_Price': round(p_entry, 2), 'Exit_Date': dt_exit.date() if dt_exit else "PENDING",
                'Alpha_BPS': None
            }
            if p_exit and s_exit:
                alpha = ((p_exit/p_entry - 1) - (s_exit/s_entry - 1)) * impulse
                record['Alpha_BPS'] = int(alpha * 10000)
            raw_audit_log.append(record)

    df_raw = pd.DataFrame(raw_audit_log)
    df_raw.to_csv(RAW_AUDIT_FILE, index=False)
    summary = df_raw.groupby(['Filer', 'Chamber', 'Horizon'])['Alpha_BPS'].agg(['mean', 'count']).unstack()
    summary.to_csv(SUMMARY_FILE)
    print(f"\n\n[SUCCESS] Reports saved to {SUMMARY_FILE} and {RAW_AUDIT_FILE}")

if __name__ == "__main__":
    main()
5. Interpreting the Output
congressional_performance_report.csv
This is your executive summary.

Alpha_BPS (mean): The average outperformance in basis points. (100 BPS=1%).

Count: The number of trades analyzed for that person/horizon.

congress_raw_audit_data.csv
The granular log. Use this to verify individual trades.

PENDING: Indicates the trade occurred too recently for the 20 or 60-day window to have closed yet.

Alpha_BPS: The specific "win" or "loss" for that ticker.

Statistical Context
Senate 5-Day Alpha: High positive values (e.g., >100 BPS) suggest a "Disclosure Pop" where the market reacts to the news of the Senator's trade.

The Fade: If 5-day Alpha is positive but 60-day Alpha is negative, the "information edge" is short-lived and purely based on the filing event itself.
