import os
import zipfile
import requests
import xml.etree.ElementTree as ET
import time

# Constants
BASE_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/"
INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
SAVE_DIR = "house_data"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def download_actual_disclosures(years):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for year in years:
        print(f"--- Processing {year} ---")
        zip_path = f"{year}.zip"
        
        # 1. Get the Index
        r = requests.get(INDEX_URL.format(year=year), headers=HEADERS)
        if r.status_code != 200:
            print(f"Index for {year} not found.")
            continue
            
        with open(zip_path, 'wb') as f:
            f.write(r.content)

        # 2. Extract and Parse the XML
        # The internal file is usually named FinancialDisclosure.xml
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Find the xml file regardless of exact name
            xml_name = [name for name in z.namelist() if name.endswith('.xml')][0]
            z.extract(xml_name)
            
            tree = ET.parse(xml_name)
            root = tree.getroot()

            # 3. Pull the PDFs
            for member in root.findall('Member'):
                last_name = member.find('Last').text or "Unknown"
                doc_id = member.find('DocID').text
                
                if not doc_id:
                    continue

                file_url = f"{BASE_URL}{year}/{doc_id}.pdf"
                dest_path = os.path.join(SAVE_DIR, f"{year}_{last_name}_{doc_id}.pdf")

                if os.path.exists(dest_path):
                    continue

                print(f"Fetching PDF: {year} - {last_name} ({doc_id})")
                
                try:
                    res = requests.get(file_url, headers=HEADERS, timeout=30)
                    if res.status_code == 200 and b'%PDF' in res.content[:4]:
                        with open(dest_path, 'wb') as f:
                            f.write(res.content)
                    else:
                        print(f"Failed or Invalid PDF for {doc_id}")
                    
                    # Be slightly polite to avoid a 403/IP block
                    time.sleep(0.5) 
                except Exception as e:
                    print(f"Error on {doc_id}: {e}")

            os.remove(xml_name)
        os.remove(zip_path)

if __name__ == "__main__":
    # Start with a small range to test
    download_actual_disclosures([2023, 2024])
