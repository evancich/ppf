import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np
import os

RAW_AUDIT_FILE = "congress_raw_audit_data.csv"
UNIFIED_PATH = Path("outputs/ppf_transactions_unified.csv")

ID_MAP = {
    'B195126E-7BB2-4D54-BAF5-E6FC8E7A0165': 'WARNER',
    'D7CAE837-F73C-4EB1-B9FB-E510F53D65DE': 'TUBERVILLE',
    '98B3317B-2632-48C3-B636-D5555C4680CA': 'GRASSLEY',
    '7AB0D2FD-19EA-459D-A1D9-3701E2CC8E93': 'THUNE',
    'BD886067-927F-48A8-9D43-8AB6FA713F98': 'CARPER',
    '996378A6-200B-48F6-96F0-A3A1F45E445E': 'MORAN'
}

def clean_name(name):
    if not isinstance(name, str): return "UNKNOWN"
    name = name.upper().replace("HON.", "").replace("REP.", "").replace("SEN.", "").strip()
    return ID_MAP.get(name, name)

def get_first_scalar(val):
    if isinstance(val, (pd.Series, np.ndarray, list)):
        return val[0]
    return val

def run_performance_audit():
    if not UNIFIED_PATH.exists():
        print(f"Error: {UNIFIED_PATH} not found.")
        return

    df = pd.read_csv(UNIFIED_PATH)
    df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    df = df.dropna(subset=['filing_date', 'ticker'])
    df['filer'] = df['filer'].apply(clean_name)
    
    # Clean tickers
    tickers = sorted([str(t).replace('.', '-') for t in df['ticker'].unique() if t != "UNKNOWN"])
    
    if not tickers:
        print("No tickers found to process.")
        return
    
    print(f"Downloading data for {len(tickers)} tickers + SPY...")
    # Fetch SPY first
    spy = yf.download("SPY", start="2022-01-01", progress=False)['Close']
    if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0] # Ensure it's a series
    spy = spy[~spy.index.duplicated(keep='first')]

    # Fetch tickers
    prices_df = yf.download(tickers, start="2022-01-01", progress=False)['Close']
    
    # If only one ticker, yfinance might return a Series. Force to DataFrame
    if isinstance(prices_df, pd.Series):
        prices_df = prices_df.to_frame(name=tickers[0])

    raw_audit_log = []
    
    for _, trade in df.iterrows():
        ticker = str(trade['ticker']).replace('.', '-')
        if ticker not in prices_df.columns:
            continue
        
        # Get the individual series and remove duplicates/NaNs
        p_series = prices_df[ticker].dropna()
        p_series = p_series[~p_series.index.duplicated(keep='first')]
        
        if p_series.empty:
            continue
            
        impulse = 1 if str(trade['transaction_type']).upper() == "BUY" else -1
        
        # Entry calculation: find the first trading day on or after filing
        entry_mask = p_series.index >= trade['filing_date']
        if not entry_mask.any():
            continue
        
        dt_entry = p_series.index[entry_mask][0]
        p_idx = p_series.index.get_loc(dt_entry)
        
        # Need to find corresponding index in SPY
        spy_mask = spy.index >= dt_entry
        if not spy_mask.any():
            continue
        dt_spy_entry = spy.index[spy_mask][0]
        s_idx = spy.index.get_loc(dt_spy_entry)
        
        p_entry = get_first_scalar(p_series.iloc[p_idx])
        s_entry = get_first_scalar(spy.iloc[s_idx])

        # Horizons: 5, 20, 60 trading days
        for h in [5, 20, 60]:
            exit_p_idx = p_idx + h
            exit_s_idx = s_idx + h
            
            record = {
                'Filer': trade['filer'], 
                'Ticker': ticker, 
                'Type': trade['transaction_type'],
                'Filing_Date': trade['filing_date'].date(), 
                'Horizon': h, 
                'Alpha_BPS': None
            }
            
            if exit_p_idx < len(p_series) and exit_s_idx < len(spy):
                p_exit = get_first_scalar(p_series.iloc[exit_p_idx])
                s_exit = get_first_scalar(spy.iloc[exit_s_idx])
                
                if pd.notnull(p_exit) and pd.notnull(s_exit) and p_entry > 0 and s_entry > 0:
                    stock_ret = (float(p_exit) / float(p_entry)) - 1
                    spy_ret = (float(s_exit) / float(s_entry)) - 1
                    alpha = (stock_ret - spy_ret) * impulse
                    record['Alpha_BPS'] = int(alpha * 10000)
            
            raw_audit_log.append(record)

    out_df = pd.DataFrame(raw_audit_log)
    if not out_df.empty:
        out_df.to_csv(RAW_AUDIT_FILE, index=False)
        print(f"Audit complete. Results saved to {RAW_AUDIT_FILE}")
    else:
        print("No valid trade horizons found to audit.")

if __name__ == "__main__":
    run_performance_audit()
