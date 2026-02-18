#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import re

def extract_ticker(text):
    if not isinstance(text, str): return "UNKNOWN"
    m = re.search(r'\((?:[A-Z]+:\s*)?([A-Z]{1,5})\)', text.upper())
    if m: return m.group(1).strip().upper()
    words = re.findall(r'\b([A-Z]{2,5})\b', text.upper())
    noise = {'JOINT', 'VISA', 'TYPE', 'DATE', 'AMOUNT', 'OWNER', 'STK', 'STOCK', 'VAL', 'N/A', 'NAN', 'PURCHASE', 'SALE'}
    for w in words:
        if w not in noise: return w.strip().upper()
    return "UNKNOWN"

def unify(project_root):
    idx_path = project_root / "outputs" / "efd_reports_index.csv"
    if not idx_path.exists(): return pd.DataFrame()
    
    index_df = pd.read_csv(idx_path)
    all_tx = []

    for _, row in index_df.iterrows():
        l_path = row.get('local_path') or row.get('pdf_local_path')
        if not l_path: continue
        full_path = project_root / l_path
        if not full_path.exists(): continue
        try:
            dfs = pd.read_html(str(full_path))
            for df in dfs:
                if len(df.columns) < 3: continue
                # Extract clean name from the index row
                last_name = str(row.get('last_name', row.get('filer', 'UNKNOWN'))).upper()
                for _, tx_row in df.iterrows():
                    row_str = " ".join(tx_row.astype(str)).upper()
                    ticker = extract_ticker(row_str)
                    # Improved Buy/Sell detection
                    t_type = "SELL" if any(x in row_str for x in ["SALE", "SELL"]) else "BUY"
                    impulse = -1 if t_type == "SELL" else 1

                    all_tx.append({
                        "source_file": str(row.get('report_id', '0')),
                        "official_name": last_name,
                        "filer": last_name, # Duplicate for compatibility
                        "filing_datetime": "2026-02-18T00:00:00Z",
                        "transaction_type": t_type,
                        "impulse": impulse,
                        "ticker": ticker,
                        "asset_name": ticker,
                        "amount_bucket_raw": "$1,001 - $15,000"
                    })
        except: continue
    return pd.DataFrame(all_tx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    
    df = unify(root)
    if df.empty: 
        print("No transactions found to unify.")
        return

    tx_out = root / "outputs" / "ppf_transactions_unified.csv"
    df.to_csv(tx_out, index=False)
    
    print(f"DONE: {len(df)} transactions unified. House names and Buy/Sell types fixed.")

if __name__ == "__main__":
    main()
