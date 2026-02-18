import pandas as pd
import yfinance as yf
from pathlib import Path
import re
import sys

# --- CONFIG ---
SENATE_PATH = Path("outputs/ppf_final_signal_debug_rows.csv")
HOUSE_PATH = Path("outputs/house/parsed_2024.csv")

# Senate ID Mapping (Essential to resolve UNKNOWN)
ID_MAP = {
    'B195126E-7BB2-4D54-BAF5-E6FC8E7A0165': 'WARNER',
    'D7CAE837-F73C-4EB1-B9FB-E510F53D65DE': 'TUBERVILLE',
    '98B3317B-2632-48C3-B636-D5555C4680CA': 'GRASSLEY',
    '7AB0D2FD-19EA-459D-A1D9-3701E2CC8E93': 'THUNE',
    'BD886067-927F-48A8-9D43-8AB6FA713F98': 'CARPER',
    '996378A6-200B-48F6-96F0-A3A1F45E445E': 'MORAN'
}

def normalize_ticker(t):
    if not isinstance(t, str): return ""
    t = re.sub(r"[^A-Z\-]", "", str(t).strip().upper())
    return t if 2 <= len(t) <= 5 else ""

def get_price_on_or_after(price_series, target_date):
    target = pd.to_datetime(target_date).tz_localize(None)
    available_dates = price_series.index.tz_localize(None)
    for date in available_dates:
        if date >= target:
            return price_series.loc[date], date
    return None, None

def load_trades(path, chamber_label):
    if not path.exists(): return []
    raw = pd.read_csv(path)
    trades = []
    
    for _, row in raw.iterrows():
        ticker = normalize_ticker(row.get('ticker'))
        if not ticker: continue
        
        # 1. ENFORCED IDENTITY RESOLUTION
        if chamber_label == "SENATE":
            # Extract UUID from source_file if name isn't present
            m = re.search(r"\{?([A-F0-9-]{36})\}?", str(row.get("source_file", "")), flags=re.I)
            name = ID_MAP.get(m.group(1).upper(), "UNKNOWN_SENATE") if m else "UNKNOWN_SENATE"
        else:
            # Clean "HON." prefix from House names
            raw_name = str(row.get('representative') or row.get('filer') or "UNKNOWN_HOUSE")
            name = raw_name.replace("Hon. ", "").replace("Hon.", "").strip().upper()

        # 2. IMPULSE (Buy/Sell)
        # Assuming 1 for Buy, -1 for Sell. If missing, we default to 1.
        impulse = 1
        if 'impulse' in row:
            impulse = -1 if str(row['impulse']).strip() in ['-1', 'Sell', 'S'] else 1

        date_val = row.get('filing_datetime') or row.get('disclosure_date')
        if not date_val: continue

        trades.append({
            'ticker': ticker,
            'chamber': chamber_label,
            'filer': name,
            'impulse': impulse,
            'date': pd.to_datetime(date_val).tz_localize(None)
        })
    return trades

def run_hardened_audit():
    all_trades = load_trades(SENATE_PATH, "SENATE")
    all_trades += load_trades(HOUSE_PATH, "HOUSE")
    
    unique_tickers = list(set([t['ticker'] for t in all_trades])) + ['SPY']
    market_data = yf.download(unique_tickers, start="2023-01-01", auto_adjust=True, progress=False)['Close']

    results = []
    for i, trade in enumerate(all_trades):
        ticker, impulse = trade['ticker'], trade['impulse']
        entry_target = trade['date'] + pd.Timedelta(days=1)
        
        sys.stdout.write(f"\rAudit: {i+1}/{len(all_trades)} | {trade['filer'][:15]}")
        sys.stdout.flush()

        if ticker not in market_data.columns: continue
        p_series, spy_series = market_data[ticker].dropna(), market_data['SPY'].dropna()
        p_entry, dt_entry = get_price_on_or_after(p_series, entry_target)
        s_entry, _ = get_price_on_or_after(spy_series, entry_target)
        
        if p_entry is None or s_entry is None: continue

        for h in [5, 20, 60]:
            exit_t = dt_entry + pd.Timedelta(days=h)
            p_exit, _ = get_price_on_or_after(p_series, exit_t)
            s_exit, _ = get_price_on_or_after(spy_series, exit_t)
            
            if p_exit is not None and s_exit is not None:
                # 3. CORRECT ALPHA MATH (Excess Return)
                asset_ret = (p_exit / p_entry) - 1
                spy_ret = (s_exit / s_entry) - 1
                
                # Apply impulse: if it's a sell (-1), we profit if the stock drops
                alpha = (asset_ret - spy_ret) * impulse
                
                results.append({'chamber': trade['chamber'], 'filer': trade['filer'], 'horizon': h, 'alpha': alpha})

    res_df = pd.DataFrame(results)
    print("\n\n" + "═"*55 + "\n AUDIT COMPLETE: EXCESS RETURN (ALPHA) SUMMARY\n" + "═"*55)
    if not res_df.empty:
        print(res_df.groupby(['chamber', 'horizon'])['alpha'].mean().unstack())
        print("\nTOP PERFORMERS (5-Day Alpha)")
        print(res_df[res_df['horizon']==5].groupby('filer')['alpha'].mean().sort_values(ascending=False).head(10))

if __name__ == "__main__":
    run_hardened_audit()
