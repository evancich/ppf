import asyncio
import os
import csv
from playwright.async_api import async_playwright

async def run_scraper():
    output_file = "senate_data/efd_master_index.csv"
    os.makedirs("senate_data", exist_ok=True)

    async with async_playwright() as p:
        # slow_mo adds a 500ms delay to EVERY action to look more human
        browser = await p.chromium.launch(headless=False, args=["--no-sandbox"], slow_mo=500)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
        page = await context.new_page()

        print("\n[!] STEP 1: HANDSHAKE")
        await page.goto("https://efdsearch.senate.gov/search/home/")
        print(">> Manually click 'Agree'. Once the search form appears, the script will take over.")
        
        # We wait for you to click Agree
        await page.wait_for_selector("#fromDate", timeout=120000)

        for year in range(2015, 2027):
            print(f"\n--- Starting Year {year} ---")
            
            # Use 'fill' but follow it with a physical click to trigger the internal JS
            await page.fill("#fromDate", f"01/01/{year}")
            await page.fill("#toDate", f"12/31/{year}")
            
            # Check 'Annual' and 'Candidate' to ensure we get results
            await page.check("input[value='1']") # Annual
            await page.check("input[value='4']") # Candidate
            
            print("[-] Clicking Search...")
            await page.click("button[type='submit']")

            # WAIT FOR THE TABLE
            try:
                # We wait for the 'Processing' box to disappear and data to appear
                await page.wait_for_selector("table#searchReports tbody tr", timeout=20000)
                
                rows = page.locator("table#searchReports tbody tr")
                count = await rows.count()
                
                # Check if it's just the 'No data' row
                first_text = await rows.first.inner_text()
                if "No data available" in first_text:
                    print(f"[!] No data for {year}")
                else:
                    print(f"[+] SUCCESS: Found {count} rows. Saving...")
                    
                    data = []
                    for i in range(count):
                        cells = rows.nth(i).locator("td")
                        if await cells.count() >= 4:
                            last = await cells.nth(0).inner_text()
                            first = await cells.nth(1).inner_text()
                            url = await rows.nth(i).locator("a").first.get_attribute("href")
                            data.append([year, last.strip(), first.strip(), f"https://efdsearch.senate.gov{url}"])
                    
                    with open(output_file, 'a', newline='') as f:
                        csv.writer(f).writerows(data)
                        
            except Exception:
                print(f"[!] Year {year} timed out. The server isn't responding to the click.")

            # IMPORTANT: Click the 'Search Again' button to reset the form for the next year
            # This is better than reloading the whole page
            print("[-] Resetting form for next year...")
            search_again = page.locator("button:has-text('Search Again')")
            if await search_again.is_visible():
                await search_again.click()
            else:
                await page.goto("https://efdsearch.senate.gov/search/")
            
            await page.wait_for_selector("#fromDate")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_scraper())
