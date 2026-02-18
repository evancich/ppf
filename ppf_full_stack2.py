import pandas as pd
import yfinance as yf
from pathlib import Path
import re
import sys

# --- CONFIG & PATHS ---
SUMMARY_FILE = "congressional_performance_report.csv"
RAW_AUDIT_FILE = "congress_raw_audit_data.csv"
SENATE_PATH = Path("outputs/ppf_final_signal_debug_rows.csv")
HOUSE_PATH = Path("outputs/house/parsed_2024.csv")

# Known Senate UUID Mapping
ID_MAP = {
    'B195126E-7BB2-4D54-BAF5-E6FC8E7A0165': 'WARNER',
    'D7CAE837-F73C-4EB1-B9FB-E510F53D65DE': 'TUBERVILLE',
    '98B3317B-2632-48C3-B636-D5555C4680CA': 'GRASSLEY',
    '7AB0D2FD-19EA-459D-A1D9-3701E2CC8E93': 'THUNE',
    'BD886067-927F-48A8-9D43-8AB6FA713F98': 'CARPER',
    '996378A6-200B-48F6-96F0-A3A1F45E445E': 'MORAN'
}

def clean_name(name_str):
    if not isinstance(name_str, str) or pd.isna(name_str): return "UNKNOWN"
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
        # 1. Ticker Cleaning
        t = re.sub(r"[^A-Z\-]", "", str(row.get('ticker', '')).upper())
        if not (2 <= len(t) <= 5): continue
        
        # 2. Name / Filer Logic (Fixing House Data Parsing)
        if chamber == "HOUSE":
            # Search all potential name columns emitted by different scraper versions
            name = (row.get('filer') or row.get('representative') or 
                    row.get('last_name') or row.get('official_name') or 
                    row.get('name') or "UNKNOWN_HOUSE")
        else:
            name = row.get('filer')
            if pd.isna(name) or str(name).upper() in ["UNKNOWN", "UNKNOWN_SENATE"]:
                m = re.search(r"\{?([A-F0-9-]{36})\}?", str(row.get("source_file", "")), flags=re.I)
                name = ID_MAP.get(m.group(1).upper() if m else "", "UNKNOWN_SENATE")
        
        name = clean_name(name)
        
        # 3. Date Parsing
        dt_val = row.get('filing_datetime') or row.get('disclosure_date') or row.get('filing_date')
        if pd.isna(dt_val): continue
        dt = pd.to_datetime(dt_val).tz_localize(None)
        
        # 4. Buy/Sell Direction (Impulse) logic
        raw_type = str(row.get('transaction_type', '')).upper()
        raw_impulse = row.get('impulse')
        
        if not pd.isna(raw_impulse):
            impulse = int(raw_impulse)
        else:
            # Detect direction from text if 'impulse' column is missing
            if any(x in raw_type for x in ["PURCHASE", "BUY"]): impulse = 1
            elif any(x in raw_type for x in ["SALE", "SELL"]): impulse = -1
            else: impulse = 1 # Default to Buy
            
        trades.append({
            'ticker': t, 
            'chamber': chamber, 
            'filer': name, 
            'date': dt, 
            'impulse': impulse,
            'type': "BUY" if impulse == 1 else "SELL"
        })
    return trades

def main():
    all_trades = load_trades(SENATE_PATH, "SENATE") + load_trades(HOUSE_PATH, "HOUSE")
    if not all_trades:
        print("No trades found. Check your input paths.")
        return

    tickers = list(set([t['ticker'] for t in all_trades])) + ['SPY']
    print(f"Fetching data for {len(tickers)} symbols...")
    market_data = yf.download(tickers, start="2023-01-01", auto_adjust=True, progress=False)['Close']
    max_market_date = market_data.index.max().tz_localize(None)

    raw_audit_log = []
    
    for i, trade in enumerate(all_trades):
        ticker, impulse = trade['ticker'], trade['impulse']
        sys.stdout.write(f"\rAuditing {i+1}/{len(all_trades)}...")
        
        if ticker not in market_data.columns: continue
        p_series, spy_series = market_data[ticker].dropna(), market_data['SPY'].dropna()
        
        # Entry Prices (T+1 day after filing)
        p_entry, dt_entry = get_price_on_or_after(p_series, trade['date'] + pd.Timedelta(days=1), max_market_date)
        s_entry, _ = get_price_on_or_after(spy_series, trade['date'] + pd.Timedelta(days=1), max_market_date)
        
        if p_entry is None or s_entry is None: continue

        for h in [5, 20, 60]:
            exit_target = dt_entry + pd.Timedelta(days=h)
            p_exit, dt_exit = get_price_on_or_after(p_series, exit_target, max_market_date)
            s_exit, _ = get_price_on_or_after(spy_series, exit_target, max_market_date)
            
            record = {
                'Filer': trade['filer'], 
                'Chamber': trade['chamber'], 
                'Ticker': ticker,
                'Transaction': trade['type'], # <--- NEW BUY/SELL COLUMN
                'Filing_Date': trade['date'].date(), 
                'Horizon': h,
                'Entry_Price': round(float(p_entry), 2), 
                'Exit_Date': dt_exit.date() if dt_exit else "PENDING",
                'Alpha_BPS': None
            }
            
            if p_exit and s_exit:
                # Alpha accounts for direction (Shorting a 'SELL' ticker)
                alpha = ((float(p_exit)/float(p_entry) - 1) - (float(s_exit)/float(s_entry) - 1)) * impulse
                record['Alpha_BPS'] = int(alpha * 10000)
            
            raw_audit_log.append(record)

    # OUTPUTS
    df_raw = pd.DataFrame(raw_audit_log)
    df_raw.to_csv(RAW_AUDIT_FILE, index=False)
    
    # Improved Summary Table
    summary = df_raw.groupby(['Filer', 'Chamber', 'Horizon'])['Alpha_BPS'].agg(['mean', 'count']).unstack()
    summary.to_csv(SUMMARY_FILE)
    
    print(f"\n\n[COMPLETE]\n1. Raw Log: {RAW_AUDIT_FILE} (Includes Transaction column)\n2. Summary: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
