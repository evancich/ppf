import pandas as pd
import yfinance as yf
from pathlib import Path
import re
import sys

# --- CONFIG ---
SENATE_PATH = Path("outputs/ppf_final_signal_debug_rows.csv")

ID_MAP = {
    'B195126E-7BB2-4D54-BAF5-E6FC8E7A0165': 'WARNER',
    'D7CAE837-F73C-4EB1-B9FB-E510F53D65DE': 'TUBERVILLE',
    '98B3317B-2632-48C3-B636-D5555C4680CA': 'GRASSLEY',
    '7AB0D2FD-19EA-459D-A1D9-3701E2CC8E93': 'THUNE',
    'BD886067-927F-48A8-9D43-8AB6FA713F98': 'CARPER',
    '996378A6-200B-48F5-96F0-A3A1F45E445E': 'MORAN'
}

def normalize_ticker(t):
    if not isinstance(t, str): return ""
    t = re.sub(r"[^A-Z\-]", "", t.strip().upper())
    return t if 2 <= len(t) <= 5 else ""

def get_price_on_or_after(price_series, target_date):
    target = pd.to_datetime(target_date).tz_localize(None)
    available_dates = price_series.index.tz_localize(None)
    for date in available_dates:
        if date >= target:
            return price_series.loc[date], date
    return None, None

def run_senator_audit():
    print("--- STARTING SENATOR PERFORMANCE AUDIT ---")
    
    if not SENATE_PATH.exists():
        print(f"Error: {SENATE_PATH} not found.")
        return
    
    raw_df = pd.read_csv(SENATE_PATH)
    trades = []
    
    for _, row in raw_df.iterrows():
        t = normalize_ticker(row.get('ticker'))
        if t:
            # Resolve Senator Name
            m = re.search(r"\{?([A-F0-9-]{36})\}?", str(row.get("source_file", "")), flags=re.I)
            filer = ID_MAP.get(m.group(1).upper(), "UNKNOWN") if m else "UNKNOWN"
            
            trades.append({
                'ticker': t,
                'filer': filer,
                'date': pd.to_datetime(row['filing_datetime']).tz_localize(None)
            })
    
    unique_tickers = list(set([t['ticker'] for t in trades])) + ['SPY']
    print(f"Downloading data for {len(unique_tickers)} tickers...")
    market_data = yf.download(unique_tickers, start="2023-01-01", auto_adjust=True, progress=False)['Close']

    results = []
    total = len(trades)
    
    for i, trade in enumerate(trades):
        ticker = trade['ticker']
        entry_target = trade['date'] + pd.Timedelta(days=1)
        
        sys.stdout.write(f"\rProcessing {i+1}/{total} | {trade['filer']} - {ticker}")
        sys.stdout.flush()

        if ticker not in market_data.columns: continue
            
        p_series = market_data[ticker].dropna()
        spy_series = market_data['SPY'].dropna()
        
        price_entry, actual_entry_dt = get_price_on_or_after(p_series, entry_target)
        spy_entry, _ = get_price_on_or_after(spy_series, entry_target)
        
        if price_entry is None or spy_entry is None: continue

        for h in [5, 20, 60]:
            exit_target = actual_entry_dt + pd.Timedelta(days=h)
            price_exit, _ = get_price_on_or_after(p_series, exit_target)
            spy_exit, _ = get_price_on_or_after(spy_series, exit_target)
            
            if price_exit is not None and spy_exit is not None:
                alpha = (price_exit / price_entry) - (spy_exit / spy_entry)
                cagr = ((1 + alpha) ** (365 / h)) - 1
                
                results.append({
                    'filer': trade['filer'],
                    'horizon': h,
                    'alpha': alpha,
                    'cagr': cagr
                })

    print("\n\n" + "═"*65 + "\n  SENATOR PERFORMANCE LEADERBOARD (5-DAY HORIZON)\n" + "═"*65)
    res_df = pd.DataFrame(results)
    
    # Filter for just the 5-day pop to find the leaders
    h5 = res_df[res_df['horizon'] == 5].copy()
    leaderboard = h5.groupby('filer').agg({
        'alpha': ['count', 'mean'],
        'cagr': 'mean'
    }).sort_values(('alpha', 'mean'), ascending=False)
    
    print(leaderboard)
    
    print("\n" + "═"*65 + "\n  FULL SUMMARY BY HORIZON\n" + "═"*65)
    print(res_df.groupby(['horizon']).agg({'alpha': 'mean', 'cagr': 'mean'}))

if __name__ == "__main__":
    run_senator_audit()
