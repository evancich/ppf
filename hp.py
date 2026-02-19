import re
import os
import csv
import logging
import pdfplumber
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class HardenedHouseParser:
    def __init__(self, source_dir, output_file):
        self.source_dir = source_dir
        self.output_file = output_file
        self.headers = ['filename', 'status', 'filer_name', 'doc_type', 'error_detail']
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.headers).writeheader()

    def get_raw_text(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pdf':
            with pdfplumber.open(path) as pdf:
                content = " ".join([page.extract_text() or "" for page in pdf.pages])
                if not content.strip():
                    raise ValueError("OCR_REQUIRED: No machine-readable text found.")
                return content
        elif ext in ['.html', '.htm']:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return BeautifulSoup(f, 'html.parser').get_text(separator=' ')
        raise TypeError(f"UNSUPPORTED_FORMAT: {ext}")

    def parse_hardened(self, text):
        data = {'filer_name': 'Not Found', 'doc_type': 'Unknown'}
        if not text: return data

        # 1. Normalize for classification
        text_upper = text.upper()
        if "FINANCIAL DISCLOSURE REPORT" in text_upper:
            data['doc_type'] = "Full Report"
        elif "EXTENSION REQUEST" in text_upper:
            data['doc_type'] = "Extension"
        elif "CAMPAIGN NOTICE" in text_upper:
            data['doc_type'] = "Campaign Notice"

        # 2. Priority-based Regex Extraction
        # Patterns handle: Label, optional quotes, optional spaces
        patterns = [
            r"Name\s*of\s*Requestor:?\s*[\"']?([^\"\n\r]+)",
            r"Name:?\s*[\"']?([^\"\n\r]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Clean up artifacts like commas or trailing quotes
                name = match.group(1).split('\n')[0].strip().strip('",')
                data['filer_name'] = name
                break
        
        return data

    def run(self):
        for filename in os.listdir(self.source_dir):
            path = os.path.join(self.source_dir, filename)
            row = {'filename': filename, 'status': 'Failed', 'filer_name': 'N/A', 'doc_type': 'N/A', 'error_detail': ''}
            
            try:
                text = self.get_raw_text(path)
                info = self.parse_hardened(text)
                row.update(info)
                row['status'] = 'Success'
                print(f"Processed: {filename}")
            except Exception as e:
                row['error_detail'] = str(e)
                print(f"Skipped: {filename} ({str(e)})")

            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=self.headers).writerow(row)

if __name__ == "__main__":
    HardenedHouseParser('house_data', 'hardened_results.csv').run()
