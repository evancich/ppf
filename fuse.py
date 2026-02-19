import csv
import logging
import os
import re
import sys

# 1. SET UP LOGGING
# We log to both a file and the console for maximum visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fusion_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def clean_name(name):
    """Standardizes messy Senator/Filer names using RegEx."""
    if not name: return "Unknown"
    try:
        # Remove newlines and collapse whitespace
        name = re.sub(r'\s+', ' ', str(name)).strip()
        # Remove formal prefixes
        name = re.sub(r'^(The Honorable|Mr\.|Ms\.|Mrs\.)\s+', '', name, flags=re.IGNORECASE)
        # Remove bracketed/parenthetical suffixes (e.g., '(McConnell, Mitch)')
        name = re.sub(r'\(.*?\)', '', name).strip()
        return name
    except Exception as e:
        logging.warning(f"Could not clean name '{name}': {e}")
        return name

def normalize_amount(amount_str):
    """Parses range strings into numeric Low/High values."""
    if not amount_str: return "", ""
    try:
        # Remove currency symbols and separators
        clean = amount_str.replace('$', '').replace(',', '').strip()
        
        # Case: "$1,001 - $15,000"
        if ' - ' in clean:
            parts = clean.split(' - ')
            return float(parts[0]), float(parts[1])
        
        # Case: "Over $1,000,000"
        nums = re.findall(r'\d+', clean)
        if nums:
            val = float(nums[0])
            if 'Over' in amount_str:
                return val, "" # High is infinite/unknown
            return val, val
    except Exception:
        pass
    return "", ""

def fuse_data(transaction_file, results_file, output_file):
    print(f"--- FUSION INITIALIZED ---")
    
    # 2. FILE SYSTEM CHECK
    if not os.path.exists(transaction_file):
        logging.error(f"Critical Error: {transaction_file} not found.")
        return
    if not os.path.exists(results_file):
        logging.error(f"Critical Error: {results_file} not found.")
        return

    # 3. LOAD AUDIT DATA INTO MEMORY (Index by Filename)
    # This allows O(1) lookup during the fusion pass
    audit_map = {}
    print(f"STATUS: Indexing audit results from {results_file}...")
    try:
        with open(results_file, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get('filename', '').strip()
                if fname:
                    audit_map[fname] = {
                        'Extraction_Status': row.get('status', 'Unknown'),
                        'Report_Type': row.get('doc_type', 'Unknown')
                    }
        print(f"SUCCESS: Indexed {len(audit_map)} audit records.")
    except Exception as e:
        logging.error(f"Failed to read audit file: {e}")
        return

    # 4. FUSION AND CLEANING PASS
    fused_rows = []
    print(f"STATUS: Fusing transactions from {transaction_file}...")
    
    try:
        with open(transaction_file, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for line_no, row in enumerate(reader, 1):
                try:
                    # Resolve File Link
                    src_file = row.get('Source File', '').strip()
                    audit_meta = audit_map.get(src_file, {'Extraction_Status': 'Not Found', 'Report_Type': 'Unknown'})
                    
                    # Normalize Amounts
                    amt_low, amt_high = normalize_amount(row.get('Amount', ''))
                    
                    # Construct Fused Record
                    fused_record = {
                        'Filer': clean_name(row.get('Senator', row.get('Filer', 'Unknown'))),
                        'Transaction_Date': row.get('Transaction Date', ''),
                        'Ticker': row.get('Ticker', ''),
                        'Asset_Name': row.get('Asset Name', ''),
                        'Type': row.get('Type', ''),
                        'Amount_Raw': row.get('Amount', ''),
                        'Amount_Low': amt_low,
                        'Amount_High': amt_high,
                        'Extraction_Status': audit_meta['Extraction_Status'],
                        'Report_Type': audit_meta['Report_Type'],
                        'Source_File': src_file
                    }
                    fused_rows.append(fused_record)
                    
                    if line_no % 5000 == 0:
                        print(f"PROGRESS: Processed {line_no} rows...")
                        
                except Exception as e:
                    logging.warning(f"Skipping malformed row at line {line_no}: {e}")
                    continue
                    
    except Exception as e:
        logging.error(f"Failed to process transactions: {e}")
        return

    # 5. SECURE WRITE
    if not fused_rows:
        print("WARNING: No data fused. Output file will not be created.")
        return

    print(f"STATUS: Writing {len(fused_rows)} fused records to {output_file}...")
    try:
        fieldnames = fused_rows[0].keys()
        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fused_rows)
        print(f"--- FUSION COMPLETE: {output_file} created ---")
    except PermissionError:
        logging.error(f"Could not write to {output_file}. Ensure the file is not open in Excel.")
    except Exception as e:
        logging.error(f"Error during file write: {e}")

if __name__ == "__main__":
    fuse_data(
        transaction_file='senate_transactions.csv',
        results_file='hardened_results.csv',
        output_file='fused_congressional_ledger.csv'
    )
