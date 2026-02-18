import argparse
from pathlib import Path
import pandas as pd
import re

def extract_ticker(text):
    if not isinstance(text, str): return "UNKNOWN"
    # Matches (AAPL) or (NYSE: AAPL)
    m = re.search(r'\((?:[A-Z]+:\s*)?([A-Z]{1,5})\)', text.upper())
    return m.group(1).strip().upper() if m else "UNKNOWN"

def unify(project_root):
    idx_path = project_root / "outputs" / "efd_reports_index.csv"
    all_tx = []
    
    if not idx_path.exists():
        return pd.DataFrame()

    index_df = pd.read_csv(idx_path)
    
    for _, row in index_df.iterrows():
        l_path = row.get('local_path')
        if not l_path or pd.isna(l_path) or not Path(l_path).exists():
            continue
        
        # We only process HTML in this step; PDFs require a different tool
        if str(l_path).endswith('.html'):
            try:
                # Read all tables in the file
                dfs = pd.read_html(str(l_path))
                for df in dfs:
                    # Look for the table that actually contains trade data
                    # Senate trade tables usually have 'Ticker' or 'Asset'
                    content_str = df.to_string().upper()
                    if 'TICKER' in content_str or 'ASSET' in content_str:
                        for _, tx_row in df.iterrows():
                            # Try to find a ticker in the 'Asset' column (usually col index 1 or 2)
                            ticker = "UNKNOWN"
                            for cell in tx_row:
                                found = extract_ticker(str(cell))
                                if found != "UNKNOWN":
                                    ticker = found
                                    break
                            
                            if ticker == "UNKNOWN": continue
                            
                            all_tx.append({
                                "ticker": ticker,
                                "filer": f"{row['filer_first']} {row['filer_last']}",
                                "chamber": "SENATE",
                                "transaction_type": "BUY" if "PURCHASE" in str(tx_row).upper() else "SELL",
                                "filing_date": row['date_received_raw'],
                            })
            except Exception:
                continue
            
    return pd.DataFrame(all_tx)

if __name__ == "__main__":
    root = Path(".").resolve()
    df = unify(root)
    if not df.empty:
        out_path = root / "outputs" / "ppf_transactions_unified.csv"
        df.to_csv(out_path, index=False)
        print(f"Success! Extracted {len(df)} transactions.")
    else:
        print("Still no transactions found. Note: PDF reports are currently skipped.")
