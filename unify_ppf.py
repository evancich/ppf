import pandas as pd
import re
import os
from pathlib import Path
import pdfplumber
from tqdm import tqdm

# --- CONFIGURATION ---
HOUSE_INDEX = Path("data/raw/house/house_fd_index.csv")
OUTPUT_FILE = Path("outputs/ppf_transactions_unified.csv")
ERROR_LOG = Path("outputs/extraction_errors.log")

ASSET_HEADERS = {"ASSET", "DESCRIPTION", "SECURITY", "NAME", "ASSET DESCRIPTION"}
CODE_HEADERS = {"P/S", "TRANSACTION CODE", "TRANSACTION TYPE"}
MAP_BUY = {"P", "PURCHASE", "BUY", "P (PARTIAL)"}
MAP_SELL = {"S", "SALE", "SELL", "S (PARTIAL)"}

def run_unify():
    all_tx = []
    errors = []
    
    if not HOUSE_INDEX.exists(): return
    df_index = pd.read_csv(HOUSE_INDEX)
    
    # Use single-threaded for stability/debugging
    for _, ref in tqdm(df_index.iterrows(), total=len(df_index), desc="Parsing PDFs"):
        rel_path = ref.get('local_pdf_relpath')
        # ... (Path resolution logic remains same) ...
        file_path = Path("data/raw/house") / str(rel_path).replace('data/raw/house/', '')
        
        if not file_path.exists():
            errors.append(f"MISSING_FILE: {file_path}")
            continue

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    if not tables:
                        continue # Or log "No tables on page i"
                        
                    for table in tables:
                        if not table or len(table[0]) < 3: continue
                        headers = [str(h).upper().strip() for h in table[0]]
                        
                        if not any(h in ASSET_HEADERS for h in headers): continue
                        
                        for row_data in table[1:]:
                            if len(row_data) != len(headers): continue
                            rd = dict(zip(headers, row_data))
                            
                            # STAGE 2 FIX: Strict Direction Logic
                            direction = "UNKNOWN"
                            for k, v in rd.items():
                                if k in CODE_HEADERS:
                                    val = str(v).upper().strip()
                                    if val in MAP_BUY: direction = "BUY"
                                    elif val in MAP_SELL: direction = "SELL"
                            
                            # STAGE 3 FIX: Ticker from Asset Cell Only
                            asset_val = ""
                            for h in ASSET_HEADERS:
                                if h in rd and rd[h]:
                                    asset_val = str(rd[h])
                                    break
                            
                            ticker_match = re.search(r'\(([A-Z]{1,5})\)', asset_val)
                            ticker = ticker_match.group(1) if ticker_match else None

                            all_tx.append({
                                "filer": ref.get('filer_name'),
                                "filing_date": ref.get('filing_date'),
                                "type": direction, # Might be UNKNOWN
                                "ticker": ticker,
                                "asset_raw": asset_val,
                                "source": file_path.name
                            })
        except Exception as e:
            errors.append(f"PARSE_FAIL: {file_path.name} | Error: {str(e)}")

    # Write Errors to Log
    with open(ERROR_LOG, "w") as f:
        f.write("\n".join(errors))
        
    pd.DataFrame(all_tx).to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(all_tx)} rows. Check {ERROR_LOG} for failures.")

if __name__ == "__main__":
    run_unify()
