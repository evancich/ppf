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
    
    # Traceability: Initial Row Count
    initial_rows = len(df)
    df = df.dropna(subset=['filing_date', 'ticker'])
    after_cleaning = len(df)
    
    df['filer'] = df['filer'].apply(clean_name)
    tickers = sorted([str(t).replace('.', '-') for t in df['ticker'].unique() if t != "UNKNOWN"])
    
    print(f"Downloaded {len(tickers)} tickers + SPY (using auto_adjust=True)...")
    
    # DATA QUALITY FIX: Using auto_adjust=True for accurate split/div accounting
    spy = yf.download("SPY", start="2022-01-01", progress=False, auto_adjust=True)['Close']
    if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
    spy = spy[~spy.index.duplicated(keep='first')]

    prices_df = yf.download(tickers, start="2022-01-01", progress=False, auto_adjust=True)['Close']
    if isinstance(prices_df, pd.Series):
        prices_df = prices_df.to_frame(name=tickers[0])

    audit_log = []
    
    for _, trade in df.iterrows():
        ticker = str(trade['ticker']).replace('.', '-')
        skip_reason = None
        
        if ticker not in prices_df.columns:
            skip_reason = "Ticker missing in download"
        else:
            p_series = prices_df[ticker].dropna()
            p_series = p_series[~p_series.index.duplicated(keep='first')]
            
            if p_series.empty:
                skip_reason = "No price data available"
            else:
                impulse = 1 if str(trade['transaction_type']).upper() == "BUY" else -1
                
                entry_mask = p_series.index >= trade['filing_date']
                spy_mask = spy.index >= trade['filing_date']
                
                if not entry_mask.any() or not spy_mask.any():
                    skip_reason = "Filing date outside price range"
                else:
                    dt_entry = p_series.index[entry_mask][0]
                    dt_spy_entry = spy.index[spy_mask][0]
                    
                    p_idx = p_series.index.get_loc(dt_entry)
                    s_idx = spy.index.get_loc(dt_spy_entry)
                    
                    p_entry = float(get_first_scalar(p_series.iloc[p_idx]))
                    s_entry = float(get_first_scalar(spy.iloc[s_idx]))

                    for h in [5, 20, 60]:
                        exit_p_idx = p_idx + h
                        exit_s_idx = s_idx + h
                        
                        # Full traceability record
                        record = {
                            'Filer': trade['filer'], 
                            'Ticker': ticker, 
                            'Type': trade['transaction_type'],
                            'Filing_Date': trade['filing_date'].date(),
                            'Horizon': h,
                            'Entry_Date': dt_entry.date(),
                            'Entry_Price': p_entry,
                            'SPY_Entry_Price': s_entry,
                            'Exit_Date': None,
                            'Exit_Price': None,
                            'SPY_Exit_Price': None,
                            'Stock_Return': None,
                            'SPY_Return': None,
                            'Alpha_BPS': None,
                            'Status': 'Incomplete'
                        }
                        
                        if exit_p_idx < len(p_series) and exit_s_idx < len(spy):
                            dt_exit = p_series.index[exit_p_idx]
                            dt_spy_exit = spy.index[exit_s_idx]
                            
                            p_exit = float(get_first_scalar(p_series.iloc[exit_p_idx]))
                            s_exit = float(get_first_scalar(spy.iloc[exit_s_idx]))
                            
                            stock_ret = (p_exit / p_entry) - 1
                            spy_ret = (s_exit / s_entry) - 1
                            alpha = (stock_ret - spy_ret) * impulse
                            
                            record.update({
                                'Exit_Date': dt_exit.date(),
                                'Exit_Price': p_exit,
                                'SPY_Exit_Price': s_exit,
                                'Stock_Return': stock_ret,
                                'SPY_Return': spy_ret,
                                'Alpha_BPS': int(alpha * 10000),
                                'Status': 'Success'
                            })
                        else:
                            record['Status'] = 'Future Trade / No Exit'
                        
                        audit_log.append(record)
        
        if skip_reason:
            audit_log.append({
                'Filer': trade['filer'], 'Ticker': ticker, 'Status': f"Skipped: {skip_reason}"
            })

    out_df = pd.DataFrame(audit_log)
    out_df.to_csv(RAW_AUDIT_FILE, index=False)
    print(f"\nHardened Audit Complete.")
    print(f"Initial Rows: {initial_rows} -> Rows Processed: {after_cleaning}")
    print(f"Results with traceability saved to {RAW_AUDIT_FILE}")

if __name__ == "__main__":
    run_performance_audit()
