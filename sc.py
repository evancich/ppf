import asyncio
import os
from playwright.async_api import async_playwright

async def run_senate_final_fix():
    output_dir = "senate_data"
    os.makedirs(output_dir, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            print("[-] Loading Home...")
            await page.goto("https://efdsearch.senate.gov/search/home/", wait_until="load")

            # 1. TRAP: The checkbox is often inside a label. 
            # Instead of ID, we find the label that CONTAINS the text.
            print("[-] Searching for Agreement text...")
            
            # This finds the checkbox regardless of whether the ID is 
            # 'agree_statement' or 'id_acknowledgement'
            agreement_checkbox = page.locator("input[type='checkbox']").first
            agreement_label = page.get_by_text("I understand", exact=False)

            # 2. BRUTAL WAIT: Force the modal to exist
            await page.wait_for_selector("input[type='checkbox']", state="attached", timeout=15000)

            # 3. CLICK THE LABEL (Safe for VBox)
            await agreement_label.click(force=True)
            print("[+] Clicked agreement label.")

            # 4. CLICK AGREE
            await page.get_by_role("button", name="Agree").click(force=True)
            
            # 5. VERIFY SEARCH PAGE
            await page.wait_for_url("**/search/", timeout=10000)
            print("[+ SUCCESS] Reached: " + page.url)

        except Exception as e:
            print(f"[X] Failed: {e}")
            await page.screenshot(path=f"{output_dir}/selector_check.png")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(run_senate_final_fix())import csv
import datetime as dt
import os
import time
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

# --- SETUP ---
OUT_DIR = Path("senate_data")
REPORT_DIR = OUT_DIR / "reports"
INDEX_CSV = OUT_DIR / "efd_master_index.csv"

def log(msg):
    print(f"{dt.datetime.now().strftime('%H:%M:%S')} | {msg}")

def safe_fn(text):
    # Cleans filenames for Linux/Windows
    return re.sub(r"[^\w\-.]+", "_", text)[:150]

def scrape_and_download(page, context, writer):
    # Find all report links on the current page
    links = page.query_selector_all("a[href*='/search/view/']")
    count = 0
    
    for link in links:
        href = link.get_attribute("href")
        url = f"https://efdsearch.senate.gov{href}"
        
        # Pull data from the table row
        try:
            row = page.evaluate_handle("(el) => el.closest('tr')", link)
            cells = row.query_selector_all("td")
            last = cells[0].inner_text().strip()
            first = cells[1].inner_text().strip()
            date = cells[4].inner_text().strip().replace("/", "-")
        except:
            last, first, date = "Unknown", "Unknown", "00-00-0000"

        # Create a unique filename
        report_id = href.split("/")[-2]
        filename = safe_fn(f"{last}_{first}_{date}_{report_id}") + ".html"
        filepath = REPORT_DIR / filename
        
        if not filepath.exists():
            try:
                # Use the browser's active session to download the content
                resp = context.request.get(url)
                with open(filepath, "wb") as f:
                    f.write(resp.body())
                log(f"   [OK] Saved: {filename}")
            except Exception as e:
                log(f"   [ERR] {filename}: {e}")
        
        writer.writerow([last, first, date, url, filename])
        count += 1
    return count

def main():
    OUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0")
        page = context.new_page()

        log("Opening Senate site...")
        page.goto("https://efdsearch.senate.gov/search/home/")

        print("\n" + "!"*60)
        print("MANUAL STEPS REQUIRED:")
        print("1. Click 'Agree' in the browser.")
        print("2. Type 01/01/2015 in 'From' and today's date in 'To'.")
        print("3. Check the 'Annual Report' box.")
        print("4. Click the blue 'Search' button.")
        print("5. Once you see the table of names, PRESS ENTER HERE.")
        print("!"*60 + "\n")

        input("===> PRESS ENTER HERE TO START DOWNLOADS <===")

        with open(INDEX_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Last", "First", "Date", "URL", "Filename"])

            page_num = 1
            while True:
                log(f"Processing Page {page_num}...")
                found = scrape_and_download(page, context, writer)
                log(f"Finished Page {page_num}. Found {found} reports.")

                # Try to find the 'Next' button to automate the rest
                next_btn = page.query_selector("a.next.paginate_button:not(.disabled)")
                if next_btn:
                    log("Clicking 'Next' page automatically...")
                    next_btn.click()
                    time.sleep(3) # Wait for table to refresh
                    page_num += 1
                else:
                    log("No more pages found.")
                    break

        log(f"ALL DONE. Files are in {REPORT_DIR}")
        input("Press Enter to close the browser...")
        browser.close()

if __name__ == "__main__":
    main()
