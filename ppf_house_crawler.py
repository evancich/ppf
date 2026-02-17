#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ppf_house_crawler.py

PURPOSE
-------
Crawl the U.S. House Clerk Financial Disclosure search UI:
  https://disclosures-clerk.house.gov/FinancialDisclosure

Collect an index of disclosure report PDF URLs (and optionally download the PDFs),
organized for later ingestion by your existing parsing pipeline.

WHY PLAYWRIGHT
--------------
The House site is UI-driven and can be dynamic. Playwright lets us:
  - drive the search form
  - wait for the results to render
  - extract PDF URLs reliably from the rendered DOM

WHAT THIS DOES (HIGH LEVEL)
---------------------------
For each last-name prefix (A..Z) and the selected filing year:
  1) Load the search page
  2) Fill Last Name prefix
  3) Choose Filing Year
  4) Submit Search
  5) Extract all links containing ".pdf"
  6) Parse "nearby row text" (best-effort metadata)
  7) Append rows to CSV index
  8) Optionally download PDFs (requests, using Playwright cookies)

OUTPUTS
-------
Index:
  data/raw/house/house_fd_index.csv

PDF cache:
  data/raw/house/pdfs/<filing_year>/*.pdf

Logs:
  logs/ppf_house_crawler.log  (very verbose)
"""

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import pandas as pd
import requests

# Playwright is used for UI automation
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError


# -----------------------------
# CONSTANTS / CONFIG DEFAULTS
# -----------------------------

HOUSE_SEARCH_URL = "https://disclosures-clerk.house.gov/FinancialDisclosure/ViewSearch"

DEFAULT_INDEX_REL = Path("data/raw/house/house_fd_index.csv")
DEFAULT_PDF_DIR_REL = Path("data/raw/house/pdfs")
DEFAULT_LOG_REL = Path("logs/ppf_house_crawler.log")

# Polite jitter between requests (seconds).
# Keep this non-zero to avoid hammering the site.
MIN_DELAY = 0.8
MAX_DELAY = 2.0

# Retry controls
HTTP_RETRIES = 3     # downloading PDFs
UI_RETRIES = 2       # rerun the UI search if it times out

# If the UI returns a *lot* of anchors, we may be pulling duplicates.
# We dedupe by URL in-memory + against the existing index file.
# That keeps the crawler idempotent enough for repeated runs.


# -----------------------------
# LOGGING (simple, explicit)
# -----------------------------

import logging

LOGGER = logging.getLogger("ppf_house_crawler")


def setup_logging(project_root: Path, verbose: bool) -> None:
    """
    Configure logging:
      - Console handler (INFO by default, DEBUG if --verbose)
      - File handler (DEBUG always)
    """
    log_path = project_root / DEFAULT_LOG_REL
    log_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.setLevel(logging.DEBUG)

    # File handler: everything
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # Console handler: less noise unless --verbose
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)sZ | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    # Avoid duplicate handlers if re-imported / re-run in same interpreter
    LOGGER.handlers.clear()
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)

    LOGGER.info("Logging initialized")
    LOGGER.info("log_file=%s", str(log_path))


# -----------------------------
# UTILS
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jitter_sleep(min_s: float = MIN_DELAY, max_s: float = MAX_DELAY) -> None:
    """Polite randomized delay to reduce load and avoid looking like a bot."""
    time.sleep(random.uniform(min_s, max_s))


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def normalize_ws(s: str) -> str:
    """Collapse whitespace to a single space, strip."""
    return re.sub(r"\s+", " ", (s or "").strip())


def safe_filename(name: str) -> str:
    """Conservative filename sanitizer."""
    name = normalize_ws(name)
    name = re.sub(r"[^\w\-. ]+", "", name)
    name = name.replace(" ", "_")
    return name[:180] if len(name) > 180 else name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def absolutize(base: str, href: str) -> str:
    """
    Convert relative hrefs (common on the House site) to absolute URLs.
    """
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return "https://disclosures-clerk.house.gov" + href
    if base.endswith("/"):
        return base + href
    return base + "/" + href


def iter_prefixes(a: str, b: str) -> Iterable[str]:
    """
    Generate A..Z-like ranges; accepts reverse too (Z..A) if you want.
    """
    a = (a or "").strip().upper()
    b = (b or "").strip().upper()
    if not (len(a) == len(b) == 1 and a.isalpha() and b.isalpha()):
        raise ValueError("--prefix-range must be single letters like: A Z")
    start = ord(a)
    end = ord(b)
    step = 1 if start <= end else -1
    for code in range(start, end + step, step):
        yield chr(code)


def extract_report_id_from_url(url: str) -> str:
    """
    Try to extract a numeric ID from a URL like .../2026/1234567.pdf
    If not present, return "" and we'll fall back to a derived ID.
    """
    m = re.search(r"/(\d{6,})\.pdf(?:\?|$)", url)
    return m.group(1) if m else ""


# -----------------------------
# DATA MODEL
# -----------------------------

@dataclass(frozen=True)
class HouseFDRow:
    # Query provenance
    filing_year: int
    last_name_prefix: str

    # Best-effort metadata inferred from the UI row text
    filer_name: str
    state: str
    district: str
    office: str
    report_type: str
    filing_date: str

    # Core link info
    pdf_url: str
    report_id: str
    retrieved_utc: str

    # Local materialization (optional)
    local_pdf_relpath: str = ""
    local_pdf_sha256: str = ""


# -----------------------------
# INDEX FILE I/O
# -----------------------------

def read_existing_index(index_path: Path) -> pd.DataFrame:
    """
    Load the existing index (if present) so we can:
      - avoid re-downloading PDFs
      - support resumability
    """
    if not index_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(index_path, dtype=str).fillna("")
        LOGGER.info("Loaded existing index: %s rows=%d", index_path, len(df))
        return df
    except Exception:
        LOGGER.exception("Failed to read existing index at %s", index_path)
        return pd.DataFrame()


def append_rows_to_index(index_path: Path, rows: List[HouseFDRow]) -> None:
    """
    Append new rows to the index CSV incrementally (so crashes don’t lose progress).
    """
    if not rows:
        return

    ensure_parent(index_path)
    file_exists = index_path.exists()

    fieldnames = list(asdict(rows[0]).keys())

    try:
        with index_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        LOGGER.debug("Appended %d rows to %s", len(rows), index_path)
    except Exception:
        LOGGER.exception("Failed writing index rows to %s", index_path)
        raise


# -----------------------------
# PDF DOWNLOAD
# -----------------------------

def build_requests_session_from_playwright_context(context) -> requests.Session:
    """
    Transfer cookies from Playwright into requests.Session.
    This helps if the PDF endpoints rely on session cookies.
    """
    s = requests.Session()

    try:
        cookies = context.cookies()
        for c in cookies:
            s.cookies.set(c.get("name"), c.get("value"), domain=c.get("domain"), path=c.get("path"))
    except Exception:
        # Non-fatal: downloads may still work without cookies
        LOGGER.exception("Failed copying cookies from Playwright context into requests session.")

    s.headers.update(
        {
            "User-Agent": "ppf-house-crawler/1.0 (+research; polite rate limiting)",
            "Accept": "application/pdf,*/*;q=0.8",
            "Referer": HOUSE_SEARCH_URL,
        }
    )
    return s


def download_pdf(session: requests.Session, url: str, out_path: Path) -> Tuple[bool, str]:
    """
    Download a PDF with retries, write atomically, verify PDF signature.
    Returns (ok, sha256).
    """
    ensure_parent(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    # Idempotency / resume semantics:
    # If we already have a valid PDF at out_path, do NOT re-download.
    # This is the primary 'resume' mechanism for interrupted runs.
    try:
        if out_path.exists() and out_path.is_file() and out_path.stat().st_size > 0:
            with out_path.open('rb') as f:
                sig = f.read(5)
            if sig == b"%PDF-":
                digest = sha256_file(out_path)
                LOGGER.info("Skip download (exists) bytes=%d sha256=%s path=%s", out_path.stat().st_size, digest[:12], out_path)
                return True, digest
            else:
                LOGGER.warning("Existing file is not a PDF; will re-download path=%s", out_path)
    except Exception as e:
        LOGGER.warning("Skip-existing check failed; will attempt download path=%s err=%s", out_path, repr(e))


    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            LOGGER.debug("Downloading PDF attempt=%d/%d url=%s -> %s", attempt, HTTP_RETRIES, url, out_path)
            resp = session.get(url, timeout=60, stream=True)

            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")

            with tmp.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)

            # Verify PDF signature
            with tmp.open("rb") as f:
                sig = f.read(5)
            if sig != b"%PDF-":
                raise RuntimeError("Downloaded file is not a PDF (missing %PDF- signature).")

            tmp.replace(out_path)
            digest = sha256_file(out_path)
            LOGGER.info("Downloaded PDF ok bytes=%d sha256=%s path=%s", out_path.stat().st_size, digest[:12], out_path)
            return True, digest

        except Exception as e:
            LOGGER.warning("PDF download failed attempt=%d/%d url=%s err=%s", attempt, HTTP_RETRIES, url, repr(e))
            LOGGER.debug("Trace:\n%s", traceback.format_exc())

            # Cleanup partial file
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

            if attempt == HTTP_RETRIES:
                return False, ""

            # backoff + jitter
            time.sleep(1.0 * attempt)
            jitter_sleep(1.0, 3.0)

    return False, ""


# -----------------------------
# DOM EXTRACTION
# -----------------------------

def scrape_results_table(page) -> List[Dict[str, Any]]:
    """
    Extract candidate PDF links from results.

    We deliberately do NOT assume a stable table structure.
    We simply look for <a href="...pdf">, then attempt to capture
    nearby row text (ancestor tr/div) for best-effort metadata.
    """
    page.wait_for_timeout(250)

    anchors = page.locator("a[href*='.pdf']")
    n = anchors.count()

    LOGGER.debug("Found %d PDF anchors on page", n)

    results: List[Dict[str, Any]] = []
    for i in range(n):
        a = anchors.nth(i)
        href = a.get_attribute("href") or ""
        pdf_url = absolutize("https://disclosures-clerk.house.gov", href)

        row_text = ""
        try:
            tr = a.locator("xpath=ancestor::tr[1]")
            if tr.count() > 0:
                row_text = normalize_ws(tr.first.inner_text())
            else:
                div = a.locator("xpath=ancestor::div[1]")
                if div.count() > 0:
                    row_text = normalize_ws(div.first.inner_text())
        except Exception:
            # non-fatal
            row_text = ""

        results.append({"pdf_url": pdf_url, "row_text": row_text})

    return results


def parse_row_text(row_text: str) -> Dict[str, str]:
    """
    Heuristic parsing of the row text.
    This is intentionally best-effort because the House UI can change.

    Fields we attempt:
      - filing_date (MM/DD/YYYY)
      - state (2-letter token, weak)
      - district (a number, weak)
      - report_type (PTR if the token is present)
      - filer_name (leading chunk before date/report tokens)
    """
    row_text = normalize_ws(row_text)

    filing_date = ""
    mdate = re.search(r"(\b\d{1,2}/\d{1,2}/\d{4}\b)", row_text)
    if mdate:
        filing_date = mdate.group(1)

    # Very weak state heuristic; you may replace this with deterministic parsing
    state = ""
    mstate = re.search(r"\b([A-Z]{2})\b", row_text)
    if mstate:
        state = mstate.group(1)

    district = ""
    mdist = re.search(r"\b(?:District\s*)?(\d{1,2})\b", row_text)
    if mdist:
        district = mdist.group(1)

    report_type = "PTR" if re.search(r"\bPTR\b", row_text, re.IGNORECASE) else ""

    filer_name = ""
    if row_text:
        cleaned = row_text
        if filing_date:
            cleaned = cleaned.split(filing_date)[0]
        cleaned = re.sub(r"\bPTR\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = normalize_ws(cleaned)
        if len(cleaned) >= 3:
            filer_name = cleaned[:120]

    return {
        "filer_name": filer_name,
        "state": state,
        "district": district,
        "office": "",
        "report_type": report_type,
        "filing_date": filing_date,
    }


# -----------------------------
# UI DRIVING
# -----------------------------

def run_search(page, filing_year: int, last_name_prefix: str) -> None:
    """
    Drive the ViewSearch form.

    This function is the most likely to break if the House changes DOM IDs.
    If it breaks, update the selector lists.
    """
    LOGGER.debug("Loading search page: %s", HOUSE_SEARCH_URL)
    page.goto(HOUSE_SEARCH_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(250)

    # Try multiple selectors to find the Last Name input
    last_name_selectors = [
        "input[name='LastName']",
        "input#LastName",
        "input[aria-label*='LAST NAME' i]",
        "xpath=//label[contains(., 'LAST NAME')]/following::input[1]",
        "xpath=//input[contains(@placeholder,'LAST NAME') or contains(@aria-label,'LAST NAME')]",
    ]

    last_input = None
    for sel in last_name_selectors:
        loc = page.locator(sel)
        if loc.count() > 0:
            last_input = loc.first
            LOGGER.debug("Found last-name input via selector: %s", sel)
            break
    if last_input is None:
        raise RuntimeError("Could not locate LAST NAME input. Site UI changed.")

    # Filing year dropdown
    year_selectors = [
        "select[name='FilingYear']",
        "select#FilingYear",
        "xpath=//label[contains(., 'Filing Year')]/following::select[1]",
        "xpath=//select[contains(@aria-label,'Filing Year')]",
    ]
    year_sel = None
    for sel in year_selectors:
        loc = page.locator(sel)
        if loc.count() > 0:
            year_sel = loc.first
            LOGGER.debug("Found filing-year select via selector: %s", sel)
            break
    if year_sel is None:
        raise RuntimeError("Could not locate Filing Year <select>. Site UI changed.")

    # Search button
    search_selectors = [
        "button:has-text('Search')",
        "input[type='submit'][value*='Search' i]",
        "xpath=//button[contains(., 'Search')]",
        "xpath=//input[@type='submit' and contains(@value,'Search')]",
    ]
    search_btn = None
    for sel in search_selectors:
        loc = page.locator(sel)
        if loc.count() > 0:
            search_btn = loc.first
            LOGGER.debug("Found search button via selector: %s", sel)
            break
    if search_btn is None:
        raise RuntimeError("Could not locate Search button. Site UI changed.")

    # Fill and submit
    LOGGER.debug("Submitting search: year=%s prefix=%s", filing_year, last_name_prefix)
    last_input.fill(last_name_prefix)
    year_sel.select_option(str(filing_year))

    search_btn.click()

    # Important: wait for the results to load
    page.wait_for_load_state("networkidle", timeout=60000)
    LOGGER.debug("Search completed: year=%s prefix=%s", filing_year, last_name_prefix)


# -----------------------------
# MAIN CRAWL LOOP
# -----------------------------

def crawl_house_disclosures(
    project_root: Path,
    filing_year: int,
    prefixes: Iterable[str],
    download_pdfs: bool,
    headless: bool,
    verbose: bool,
) -> int:
    """
    Crawl all prefixes, write an index, optionally download PDFs.

    The crawler:
      - loads existing index to dedupe
      - persists progress incrementally
      - continues on errors (logs them loudly)
    """
    out_index = project_root / DEFAULT_INDEX_REL
    pdf_root = project_root / DEFAULT_PDF_DIR_REL / str(filing_year)
    pdf_root.mkdir(parents=True, exist_ok=True)

    existing = read_existing_index(out_index)
    already = set(existing["pdf_url"].tolist()) if not existing.empty and "pdf_url" in existing.columns else set()

    LOGGER.info("Crawler start: filing_year=%d prefixes=%d download_pdfs=%s headless=%s",
                filing_year, len(list(prefixes)), download_pdfs, headless)

    # Re-materialize prefixes iterable (we may have iterated it above)
    prefixes = list(prefixes)

    collected_batch: List[HouseFDRow] = []
    total_new_rows = 0
    total_downloaded = 0
    total_failed_downloads = 0

    # Launch Playwright
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=headless)
            except Exception as e:
                # This is the "playwright install chromium" failure mode
                LOGGER.error("Failed to launch Chromium via Playwright. "
                             "You probably need: python -m playwright install chromium")
                LOGGER.error("Launch error: %s", repr(e))
                LOGGER.debug("Trace:\n%s", traceback.format_exc())
                return 2

            context = browser.new_context()
            page = context.new_page()

            # Requests session for downloads (reuse cookies)
            req_session = build_requests_session_from_playwright_context(context)

            for idx, prefix in enumerate(prefixes, start=1):
                LOGGER.info("Prefix [%d/%d] prefix=%s year=%d", idx, len(prefixes), prefix, filing_year)
                jitter_sleep()

                # Run the UI search with retries
                ui_ok = False
                for attempt in range(1, UI_RETRIES + 1):
                    try:
                        run_search(page, filing_year=filing_year, last_name_prefix=prefix)
                        ui_ok = True
                        break
                    except PWTimeoutError:
                        LOGGER.warning("UI timeout prefix=%s attempt=%d/%d", prefix, attempt, UI_RETRIES)
                        LOGGER.debug("Trace:\n%s", traceback.format_exc())
                        jitter_sleep(2.0, 5.0)
                    except Exception as e:
                        LOGGER.warning("UI error prefix=%s attempt=%d/%d err=%s",
                                       prefix, attempt, UI_RETRIES, repr(e))
                        LOGGER.debug("Trace:\n%s", traceback.format_exc())
                        jitter_sleep(2.0, 5.0)

                if not ui_ok:
                    LOGGER.error("Skipping prefix=%s after %d UI attempts", prefix, UI_RETRIES)
                    continue

                # Extract pdf links from results
                try:
                    raw_rows = scrape_results_table(page)
                except Exception as e:
                    LOGGER.error("Failed extracting results prefix=%s err=%s", prefix, repr(e))
                    LOGGER.debug("Trace:\n%s", traceback.format_exc())
                    continue

                LOGGER.info("Results extracted: prefix=%s pdf_links=%d", prefix, len(raw_rows))

                # Transform to index rows
                new_rows_this_prefix = 0
                downloaded_this_prefix = 0

                for rr in raw_rows:
                    pdf_url = rr.get("pdf_url", "")
                    if not pdf_url:
                        continue

                    # Dedup across runs
                    if pdf_url in already:
                        LOGGER.debug("Skip existing url=%s", pdf_url)
                        continue

                    fields = parse_row_text(rr.get("row_text", ""))
                    report_id = extract_report_id_from_url(pdf_url) or safe_filename(fields.get("filer_name", "")) or safe_filename(prefix)

                    row = HouseFDRow(
                        filing_year=filing_year,
                        last_name_prefix=prefix,
                        filer_name=fields.get("filer_name", ""),
                        state=fields.get("state", ""),
                        district=fields.get("district", ""),
                        office=fields.get("office", ""),
                        report_type=fields.get("report_type", ""),
                        filing_date=fields.get("filing_date", ""),
                        pdf_url=pdf_url,
                        report_id=report_id,
                        retrieved_utc=utc_now_iso(),
                    )

                    # Download PDF (optional)
                    if download_pdfs:
                        fname = safe_filename(f"{report_id}_{row.filer_name or 'unknown'}.pdf")
                        out_pdf = pdf_root / fname

                        ok, digest = download_pdf(req_session, pdf_url, out_pdf)
                        if ok:
                            row = HouseFDRow(**{**asdict(row),
                                                "local_pdf_relpath": str(out_pdf.relative_to(project_root)),
                                                "local_pdf_sha256": digest})
                            total_downloaded += 1
                            downloaded_this_prefix += 1
                        else:
                            total_failed_downloads += 1
                            LOGGER.warning("Download failed (kept index row) url=%s", pdf_url)

                    collected_batch.append(row)
                    already.add(pdf_url)
                    total_new_rows += 1
                    new_rows_this_prefix += 1

                    # Periodic flush so we never lose much progress
                    if len(collected_batch) >= 50:
                        append_rows_to_index(out_index, collected_batch)
                        collected_batch.clear()

                # Flush end of prefix
                if collected_batch:
                    append_rows_to_index(out_index, collected_batch)
                    collected_batch.clear()

                LOGGER.info("Prefix summary prefix=%s new_rows=%d downloaded=%d failed_downloads=%d",
                            prefix, new_rows_this_prefix, downloaded_this_prefix, total_failed_downloads)

            # Close browser cleanly
            try:
                browser.close()
            except Exception:
                pass

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user (Ctrl+C). Index should be mostly preserved.")
        return 130
    except Exception as e:
        LOGGER.error("Fatal crawler error: %s", repr(e))
        LOGGER.debug("Trace:\n%s", traceback.format_exc())
        return 1

    LOGGER.info("Crawler done: new_rows=%d downloaded=%d failed_downloads=%d index=%s pdf_root=%s",
                total_new_rows, total_downloaded, total_failed_downloads, out_index, pdf_root)
    return 0


# -----------------------------
# CLI
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True, help="Repo root (where ppf_pipeline.py lives).")
    ap.add_argument("--filing-year", type=int, required=True, help="Filing year dropdown value (e.g., 2026).")
    ap.add_argument("--prefix-range", nargs=2, metavar=("A", "Z"), help="Last-name prefix range, e.g., A Z.")
    ap.add_argument("--download-pdfs", action="store_true", help="Download PDFs into data/raw/house/pdfs/<year>/")
    ap.add_argument("--headless", action="store_true", help="Run browser headless.")
    ap.add_argument("--verbose", action="store_true", help="Verbose console logging (DEBUG).")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    setup_logging(project_root, verbose=args.verbose)

    # Determine prefixes
    if args.prefix_range:
        prefixes = list(iter_prefixes(args.prefix_range[0], args.prefix_range[1]))
    else:
        prefixes = list(iter_prefixes("A", "Z"))

    LOGGER.info("CLI args: project_root=%s filing_year=%d prefixes=%s..%s download_pdfs=%s headless=%s verbose=%s",
                project_root, args.filing_year, prefixes[0], prefixes[-1],
                args.download_pdfs, args.headless, args.verbose)

    return crawl_house_disclosures(
        project_root=project_root,
        filing_year=args.filing_year,
        prefixes=prefixes,
        download_pdfs=args.download_pdfs,
        headless=args.headless,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())