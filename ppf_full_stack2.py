import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

INPUT = Path("outputs/ppf_transactions_unified.csv")
AUDIT = Path("outputs/congress_raw_audit_data.csv")

def run_audit():
    df = pd.read_csv(INPUT)
    # Ensure filing date is real
    df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    
    # Download Tickers + SPY
    tickers = [t for t in df['ticker'].unique() if pd.notna(t)]
    data = yf.download(tickers + ["SPY"], start="2024-01-01", progress=True, auto_adjust=True)['Close']

    results = []
    for row in tqdm(df.to_dict('records'), desc="Auditing"):
        ticker = row.get('ticker')
        tx_type = str(row.get('type')).upper()
        
        # CRITICAL FIX 2.0: Stop Sign Inversion
        if tx_type not in ["BUY", "SELL"]:
            row["Status"] = f"Skipped: Invalid Direction ({tx_type})"
            results.append(row)
            continue
            
        if not ticker or ticker not in data.columns:
            row["Status"] = "Skipped: Missing Market Data"
            results.append(row)
            continue

        # Math Logic
        try:
            f_date = row['filing_date']
            # Get first available trading day on or after filing
            trading_days = data.index[data.index >= f_date]
            if trading_days.empty:
                row["Status"] = "Skipped: Date out of range"
            else:
                entry_date = trading_days[0]
                row["entry_date_actual"] = entry_date
                
                for h in [5, 20, 60]:
                    idx = data.index.get_loc(entry_date)
                    if idx + h < len(data):
                        exit_date = data.index[idx + h]
                        asset_ret = (data.iloc[idx+h][ticker] - data.iloc[idx][ticker]) / data.iloc[idx][ticker]
                        spy_ret = (data.iloc[idx+h]["SPY"] - data.iloc[idx]["SPY"]) / data.iloc[idx]["SPY"]
                        
                        # Directional Alpha
                        alpha = (asset_ret - spy_ret) if tx_type == "BUY" else (spy_ret - asset_ret)
                        row[f"Alpha_{h}d_BPS"] = round(alpha * 10000, 2)
                        row[f"Exit_{h}d"] = exit_date
                
                row["Status"] = "Success"
        except Exception as e:
            row["Status"] = f"Error: {str(e)}"
            
        results.append(row)

    pd.DataFrame(results).to_csv(AUDIT, index=False)

if __name__ == "__main__":
    run_audit()
