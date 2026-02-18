#!/usr/bin/env python3
import sys
import subprocess
import logging
import time
import argparse
import traceback
from pathlib import Path

# --- AXIOMATIC PATHING ---
# Ensures the script works regardless of where you call it from
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
DATA_DIR = BASE_DIR / "data" / "raw"

# --- HARDENED LOGGING ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ppf_system_master.log")
    ]
)
logger = logging.getLogger("ppf_system")

def preflight_checks(year, headless):
    """Deep verification of the environment before starting long-running tasks."""
    logger.info("="*70)
    logger.info(f" BEGINNING SYSTEM PREFLIGHT (Target Year: {year})")
    logger.info("="*70)
    
    # 1. Dependency Checks
    try:
        from playwright.sync_api import sync_playwright
        import bs4
        import pandas
        logger.info("[PASS] Core libraries found.")
    except ImportError as e:
        logger.error(f"[FAIL] Missing library: {e}")
        print("\nFIX: pip install beautifulsoup4 playwright pandas lxml\n")
        sys.exit(1)

    # 2. Browser Engine Validation
    logger.info(f"Validating Playwright browser engine...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        logger.info("[PASS] Playwright browser is ready.")
    except Exception as e:
        logger.error(f"[FAIL] Playwright browser binaries missing.")
        print("\nFIX: playwright install chromium\n")
        sys.exit(1)

    # 3. Worker Script Verification
    scripts = ["ppf_crawl_efd.py", "ppf_house_crawler.py", "unify_ppf.py", "ppf_pipeline.py"]
    for s in scripts:
        if not (BASE_DIR / s).exists():
            logger.error(f"[FAIL] Worker script not found: {s}")
            sys.exit(1)
        logger.info(f"[PASS] Script verified: {s}")

    # 4. Directory Provisioning
    for p in [DATA_DIR / "house", DATA_DIR / "senate", OUTPUT_DIR, LOG_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    
    logger.info(">>> PREFLIGHT SUCCESSFUL. Ready for launch.\n")

def run_step_streaming(step_name, cmd_args):
    """
    Runs a worker script and streams output directly to the terminal.
    Allows for real-time monitoring of progress and errors.
    """
    logger.info(f"--- [STARTING STAGE: {step_name}] ---")
    start_ts = time.time()
    
    try:
        # We run the command and allow it to write directly to stdout/stderr
        # This prevents the 'hang' feeling during long downloads.
        subprocess.run(
            cmd_args,
            cwd=BASE_DIR,
            check=True, # Abort immediately if the script returns an error
            text=True
        )
        
        elapsed = time.time() - start_ts
        logger.info(f"--- [SUCCESS: {step_name} completed in {elapsed:.1f}s] ---\n")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"!!! CRITICAL FAILURE in {step_name} !!!")
        logger.error(f"The script exited with code {e.returncode}. Review logs above.")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"!!! UNEXPECTED SYSTEM ERROR in {step_name} !!!")
        logger.error(traceback.format_exc())
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="PPF Factor System Orchestrator")
    parser.add_argument("--year", type=int, default=2024, help="Target year for House crawl")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Watch browser UI")
    parser.add_argument("--fast-test", action="store_true", help="Run only the last 7 days of Senate data")
    parser.set_defaults(headless=True)
    args = parser.parse_args()

    preflight_checks(args.year, args.headless)

    # --- STAGE 1: SENATE ACQUISITION ---
    senate_args = [
        sys.executable, "ppf_crawl_efd.py",
        "--project-root", ".",
        "--download"
    ]
    if args.fast_test:
        # MM/DD/YYYY format as expected by ppf_crawl_efd.py
        recent_date = (time.strftime("%m/%d/%Y", time.localtime(time.time() - 7*86400)))
        senate_args.extend(["--since", recent_date])
        logger.info(f"FAST TEST MODE: Limiting Senate crawl to since {recent_date}")

    run_step_streaming("Senate Crawler (eFD)", senate_args)

    # --- STAGE 2: HOUSE ACQUISITION ---
    house_cmd = [
        sys.executable, "ppf_house_crawler.py",
        "--project-root", ".",
        "--filing-year", str(args.year)
    ]
    if args.headless:
        house_cmd.append("--headless")
    
    run_step_streaming("House Crawler (Playwright)", house_cmd)

    # --- STAGE 3: UNIFICATION ---
    run_step_streaming("Data Unification", [
        sys.executable, "unify_ppf.py",
        "--project-root", "."
    ])

    # --- STAGE 4: FACTOR PIPELINE ---
    run_step_streaming("PPF Factor Scoring", [
        sys.executable, "ppf_pipeline.py",
        "--project-root", "."
    ])

    # --- FINAL SYSTEM AUDIT ---
    logger.info("="*70)
    logger.info(" FINAL SYSTEM INTEGRITY AUDIT")
    logger.info("="*70)
    
    final_csv = OUTPUT_DIR / "ppf_scores_sector_snapshot_latest.csv"
    
    if final_csv.exists() and final_csv.stat().st_size > 1000:
        logger.info(f"[SUCCESS] Factor Signal generated: {final_csv.name}")
        logger.info(f"Signal size: {final_csv.stat().st_size / 1024:.2f} KB")
        logger.info("SYSTEM RUN COMPLETE.")
    else:
        logger.error("[CRITICAL] Final output file is missing or empty. Check for mapping errors.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] USER ABORT (Ctrl+C). Cleaning up and exiting...")
        sys.exit(1)
