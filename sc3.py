import asyncio
import os
import csv
import requests
from playwright.async_api import async_playwright

async def run_combined_system():
    base_dir = "senate_data"
    report_dir = os.path.join(base_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    index_file = os.path.join(base_dir, "efd_master_index.csv")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        dl_session = requests.Session()
        dl_session.headers.update({"User-Agent": "Mozilla/5.0"})

        print("\n" + "="*60)
        print("1. Click 'Agree' in the browser.")
        print("2. Run your search (e.g. 2015).")
        print("3. When you see names, press ENTER in this terminal.")
        print("="*60)

        await page.goto("https://efdsearch.senate.gov/search/home/")

        while True:
            cmd = input("\n[READY] Press ENTER to scrape (or 'exit'): ").strip().lower()
            if cmd == 'exit': break

            # Sync cookies for the downloader
            cookies = await page.context.cookies()
            for c in cookies:
                dl_session.cookies.set(c['name'], c['value'], domain=c['domain'])

            try:
                # NEW SELECTOR: Look for any row inside the specific result table ID
                rows = page.locator("#searchReports tbody tr")
                count = await rows.count()
                
                # If that fails, try a generic table row scrape
                if count == 0:
                    rows = page.locator("table tbody tr")
                    count = await rows.count()

                first_row_text = await rows.first.inner_text() if count > 0 else "EMPTY"
                
                if count == 0 or "No data available" in first_row_text:
                    print(f"!! Nothing found. Script sees: {first_row_text[:50]}...")
                    continue

                print(f"[*] Found {count} records. Downloading files...")

                batch = []
                for i in range(count):
                    cells = rows.nth(i).locator("td")
                    if await cells.count() < 4: continue
                    
                    last = (await cells.nth(0).inner_text()).strip()
                    first = (await cells.nth(1).inner_text()).strip()
                    date = (await cells.nth(4).inner_text()).strip().replace("/", "-")
                    
                    # Find the link specifically
                    link_el = rows.nth(i).locator("a")
                    href = await link_el.first.get_attribute("href")
                    if not href: continue
                    
                    full_url = f"https://efdsearch.senate.gov{href}"
                    filename = f"{last}_{first}_{date}.html"
                    filepath = os.path.join(report_dir, filename)

                    # Download
                    if not os.path.exists(filepath):
                        r = dl_session.get(full_url)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(r.text)
                        print(f"    [OK] {filename}")
                    
                    batch.append([last, first, date, full_url, filename])

                # Save the batch
                with open(index_file, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(batch)
                
                print(f"\n[SUCCESS] Saved {len(batch)} items. Flip page and hit ENTER again.")

            except Exception as e:
                print(f"!! Error: {e}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_combined_system())
