import os
import csv
import logging
from bs4 import BeautifulSoup

# Setup logging to show progress in the terminal
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_senate_transactions(directory='senate_data/reports'):
    all_transactions = []

    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"ERROR: The directory '{directory}' does not exist.")
        print(f"Current Working Directory: {os.getcwd()}")
        return []

    files = [f for f in os.listdir(directory) if f.endswith(".html")]
    print(f"--- Found {len(files)} HTML files in {directory} ---")

    for filename in sorted(files):
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
                soup = BeautifulSoup(f, 'html.parser')
        except Exception as e:
            logging.error(f"Could not read {filename}: {e}")
            continue

        # Extract Senator Name
        name_header = soup.find('h2', class_='filedReport')
        senator_name = name_header.get_text(strip=True) if name_header else "Unknown Senator"

        # Find the specific transaction table
        table = None
        for t in soup.find_all('table'):
            if 'Transaction Date' in t.get_text():
                table = t
                break
        
        if not table:
            # Files like Merkley or Blumenthal are 'Paper' reports and have no <table>
            print(f"SKIPPED: {filename} (No digital transaction table found - likely a scanned paper report)")
            continue

        # Map columns dynamically
        header_map = {}
        headers = table.find_all('th')
        for i, th in enumerate(headers):
            header_map[th.get_text(strip=True).lower()] = i

        cols_to_find = {
            'Date': 'transaction date',
            'Owner': 'owner',
            'Ticker': 'ticker',
            'Asset': 'asset name',
            'Type': 'type',
            'Amount': 'amount',
            'Comment': 'comment'
        }

        tbody = table.find('tbody')
        rows = tbody.find_all('tr') if tbody else table.find_all('tr')[1:]

        file_transaction_count = 0
        for row in rows:
            tds = row.find_all('td')
            if not tds: continue

            def get_cell(key):
                idx = header_map.get(cols_to_find[key])
                if idx is not None and idx < len(tds):
                    return tds[idx].get_text(strip=True).replace('--', '').strip()
                return ""

            transaction = {
                'Senator': senator_name,
                'Transaction Date': get_cell('Date'),
                'Owner': get_cell('Owner'),
                'Ticker': get_cell('Ticker'),
                'Asset Name': get_cell('Asset'),
                'Type': get_cell('Type'),
                'Amount': get_cell('Amount'),
                'Comment': get_cell('Comment'),
                'Source File': filename
            }
            all_transactions.append(transaction)
            file_transaction_count += 1

        print(f"SUCCESS: Parsed {file_transaction_count} transactions from {filename} ({senator_name})")

    return all_transactions

def save_to_csv(data, output_file='senate_transactions.csv'):
    if not data:
        print("--- NO DATA FOUND. CSV NOT CREATED ---")
        return

    # Use absolute path for the output file so we know exactly where it goes
    abs_path = os.path.abspath(output_file)
    fieldnames = list(data[0].keys())
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\n--- FINAL REPORT ---")
    print(f"Total Transactions: {len(data)}")
    print(f"File Saved To: {abs_path}")

if __name__ == "__main__":
    transactions = extract_senate_transactions('senate_data/reports')
    save_to_csv(transactions)
