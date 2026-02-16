#!/usr/bin/env python3
"""
ppf_crawl_efd.py

Crawler/downloader for U.S. Senate eFD PTR reports (2012–present).

What it does:
- Creates a session
- Fetches CSRF token from /search/home/
- Posts prohibition_agreement=1 to unlock search
- Pages through /search/report/data/ for report_types [11] (PTR)
- Extracts report links and metadata
- Downloads HTML PTRs and PDF/paper PTRs to data/raw/efd/{html,pdf}/
- Writes outputs/efd_reports_index.csv with audit metadata (sha256, status, etc.)

This is intended as a data acquisition component for your existing pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


ROOT = "https://efdsearch.senate.gov"
LANDING_PAGE_URL = f"{ROOT}/search/home/"
SEARCH_PAGE_URL = f"{ROOT}/search/"
REPORTS_URL = f"{ROOT}/search/report/data/"

# Empirically used by scrapers: report_types [11] corresponds to PTRs.
REPORT_TYPES_PTR = "[11]"

PDF_PREFIX = "/search/view/paper/"
HTML_PREFIX = "/search/view/ptr/"

DEFAULT_BATCH_SIZE = 100
DEFAULT_RATE_LIMIT_SECS = 1.2
DEFAULT_TIMEOUT_SECS = 30


@dataclass
class ReportRow:
    report_id: str
    filer_first: str
    filer_last: str
    date_received_raw: str
    url_path: str
    url_full: str
    is_pdf: bool
    is_html: bool

    # download/audit fields
    downloaded: bool = False
    local_path: str = ""
    http_status: int = 0
    content_type: str = ""
    sha256: str = ""
    bytes: int = 0
    downloaded_utc: str = ""
    error: str = ""


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rate_limit_sleep(secs: float) -> None:
    if secs > 0:
        time.sleep(secs)


def parse_report_id_from_path(url_path: str) -> str:
    # e.g. /search/view/ptr/446c7588-... or /search/view/paper/446c...
    parts = url_path.strip("/").split("/")
    return parts[-1] if parts else ""


def get_csrf_and_accept_terms(session: requests.Session, rate_limit: float) -> str:
    """
    Loads landing page, extracts csrfmiddlewaretoken, posts prohibition_agreement,
    returns CSRF token stored in cookies.
    """
    rate_limit_sleep(rate_limit)
    r = session.get(LANDING_PAGE_URL, timeout=DEFAULT_TIMEOUT_SECS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    token_el = soup.find(attrs={"name": "csrfmiddlewaretoken"})
    if not token_el or not token_el.get("value"):
        # The landing page is partly dynamic; in practice the token is present in the HTML.
        raise RuntimeError("Unable to locate csrfmiddlewaretoken on landing page.")

    form_csrf = token_el["value"]
    payload = {"csrfmiddlewaretoken": form_csrf, "prohibition_agreement": "1"}

    rate_limit_sleep(rate_limit)
    pr = session.post(
        LANDING_PAGE_URL,
        data=payload,
        headers={"Referer": LANDING_PAGE_URL},
        timeout=DEFAULT_TIMEOUT_SECS,
    )
    pr.raise_for_status()

    # Django uses csrftoken commonly; some deployments use csrf
    if "csrftoken" in session.cookies:
        return session.cookies["csrftoken"]
    if "csrf" in session.cookies:
        return session.cookies["csrf"]
    # Fallback: use form token
    return form_csrf


def post_reports_page(
    session: requests.Session,
    csrf_token: str,
    offset: int,
    length: int,
    submitted_start_date: str,
    submitted_end_date: str,
    rate_limit: float,
) -> Dict[str, Any]:
    """
    Calls /search/report/data/ to return paged report rows.
    """
    payload = {
        "start": str(offset),
        "length": str(length),
        "report_types": REPORT_TYPES_PTR,  # PTR
        "filer_types": "[]",
        "submitted_start_date": submitted_start_date,  # "01/01/2012 00:00:00"
        "submitted_end_date": submitted_end_date,      # "" means 'up to now'
        "candidate_state": "",
        "senator_state": "",
        "office_id": "",
        "first_name": "",
        "last_name": "",
        "csrfmiddlewaretoken": csrf_token,
    }

    rate_limit_sleep(rate_limit)
    r = session.post(
        REPORTS_URL,
        data=payload,
        headers={"Referer": SEARCH_PAGE_URL},
        timeout=DEFAULT_TIMEOUT_SECS,
    )
    r.raise_for_status()
    return r.json()


def extract_rows_to_reports(rows: List[List[Any]]) -> List[ReportRow]:
    """
    The API returns a table-like row structure. A common observed pattern:
      [first, last, _, link_html, date_received]
    We defensively parse.
    """
    out: List[ReportRow] = []
    for row in rows:
        try:
            # defensively index
            first = str(row[0]).strip() if len(row) > 0 else ""
            last = str(row[1]).strip() if len(row) > 1 else ""
            link_html = str(row[3]) if len(row) > 3 else ""
            date_received = str(row[4]).strip() if len(row) > 4 else ""

            link_soup = BeautifulSoup(link_html, "lxml")
            a = link_soup.find("a")
            href = a.get("href") if a else ""
            href = href.strip() if href else ""
            if not href.startswith("/search/view/"):
                continue

            report_id = parse_report_id_from_path(href)
            url_full = f"{ROOT}{href}"
            is_pdf = href.startswith(PDF_PREFIX)
            is_html = href.startswith(HTML_PREFIX)

            out.append(
                ReportRow(
                    report_id=report_id,
                    filer_first=first,
                    filer_last=last,
                    date_received_raw=date_received,
                    url_path=href,
                    url_full=url_full,
                    is_pdf=is_pdf,
                    is_html=is_html,
                )
            )
        except Exception:
            # Skip malformed rows
            continue
    return out


def download_report(
    session: requests.Session,
    report: ReportRow,
    out_dir_html: Path,
    out_dir_pdf: Path,
    rate_limit: float,
    overwrite: bool,
) -> ReportRow:
    """
    Downloads HTML or PDF for a given report row, records audit metadata.
    """
    try:
        target_dir = out_dir_pdf if report.is_pdf else out_dir_html
        ext = ".pdf" if report.is_pdf else ".html"
        local_path = target_dir / f"{report.report_id}{ext}"

        if local_path.exists() and not overwrite:
            report.downloaded = True
            report.local_path = str(local_path)
            report.http_status = 200
            report.content_type = "cached"
            report.bytes = local_path.stat().st_size
            report.sha256 = sha256_file(local_path)
            report.downloaded_utc = utc_now_iso()
            return report

        rate_limit_sleep(rate_limit)
        r = session.get(report.url_full, headers={"Referer": SEARCH_PAGE_URL}, timeout=DEFAULT_TIMEOUT_SECS, stream=True)

        report.http_status = r.status_code
        report.content_type = r.headers.get("Content-Type", "")

        # Session expiry detection: sometimes redirects back to landing page
        if r.url.rstrip("/") == LANDING_PAGE_URL.rstrip("/"):
            # refresh agreement/token and retry once
            csrf = get_csrf_and_accept_terms(session, rate_limit=rate_limit)
            rate_limit_sleep(rate_limit)
            r = session.get(report.url_full, headers={"Referer": SEARCH_PAGE_URL}, timeout=DEFAULT_TIMEOUT_SECS, stream=True)
            report.http_status = r.status_code
            report.content_type = r.headers.get("Content-Type", "")

        r.raise_for_status()

        safe_mkdir(target_dir)

        # write file
        with local_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)

        report.downloaded = True
        report.local_path = str(local_path)
        report.bytes = local_path.stat().st_size
        report.sha256 = sha256_file(local_path)
        report.downloaded_utc = utc_now_iso()
        return report

    except Exception as e:
        report.error = repr(e)
        report.downloaded = False
        return report


def write_index_csv(path: Path, reports: List[ReportRow]) -> None:
    safe_mkdir(path.parent)
    fieldnames = list(asdict(reports[0]).keys()) if reports else [
        "report_id","filer_first","filer_last","date_received_raw","url_path","url_full",
        "is_pdf","is_html","downloaded","local_path","http_status","content_type",
        "sha256","bytes","downloaded_utc","error"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in reports:
            w.writerow(asdict(r))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True, help="Project root (same as used by ppf_pipeline.py)")
    p.add_argument("--since", default="2012-01-01", help="Start date (YYYY-MM-DD) for submitted_start_date")
    p.add_argument("--until", default="", help="End date (YYYY-MM-DD) for submitted_end_date; blank = now")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--rate-limit-secs", type=float, default=DEFAULT_RATE_LIMIT_SECS)
    p.add_argument("--max-pages", type=int, default=0, help="0 = no limit; otherwise number of pages")
    p.add_argument("--download", action="store_true", help="If set, downloads report files")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads")
    return p.parse_args()


def to_efd_datetime_str(d: str, end_of_day: bool) -> str:
    """
    eFD expects 'MM/DD/YYYY HH:MM:SS'. We use midnight for since,
    and end-of-day for until when provided.
    """
    if not d:
        return ""
    yyyy, mm, dd = d.split("-")
    if end_of_day:
        return f"{mm}/{dd}/{yyyy} 23:59:59"
    return f"{mm}/{dd}/{yyyy} 00:00:00"


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    out_dir = project_root / "data" / "raw" / "efd"
    out_dir_html = out_dir / "html"
    out_dir_pdf = out_dir / "pdf"

    outputs_dir = project_root / "outputs"
    index_csv = outputs_dir / "efd_reports_index.csv"

    submitted_start = to_efd_datetime_str(args.since, end_of_day=False)
    submitted_end = to_efd_datetime_str(args.until, end_of_day=True) if args.until else ""

    session = requests.Session()

    # 1) Accept terms + get CSRF
    csrf_token = get_csrf_and_accept_terms(session, rate_limit=args.rate_limit_secs)

    # 2) Page through report rows
    all_reports: List[ReportRow] = []
    offset = 0
    page = 0

    while True:
        if args.max_pages and page >= args.max_pages:
            break

        payload = post_reports_page(
            session=session,
            csrf_token=csrf_token,
            offset=offset,
            length=args.batch_size,
            submitted_start_date=submitted_start,
            submitted_end_date=submitted_end,
            rate_limit=args.rate_limit_secs,
        )

        rows = payload.get("data", [])
        if not rows:
            break

        reports = extract_rows_to_reports(rows)
        all_reports.extend(reports)

        offset += args.batch_size
        page += 1

        # progress
        print(f"[{utc_now_iso()}] fetched page={page} rows={len(rows)} total_reports={len(all_reports)}")

    # 3) Download (optional)
    if args.download:
        safe_mkdir(out_dir_html)
        safe_mkdir(out_dir_pdf)

        for i, rep in enumerate(all_reports, start=1):
            rep = download_report(
                session=session,
                report=rep,
                out_dir_html=out_dir_html,
                out_dir_pdf=out_dir_pdf,
                rate_limit=args.rate_limit_secs,
                overwrite=args.overwrite,
            )
            all_reports[i - 1] = rep
            if i % 50 == 0:
                print(f"[{utc_now_iso()}] downloaded {i}/{len(all_reports)}")

    # 4) Write index
    write_index_csv(index_csv, all_reports)
    print(f"[{utc_now_iso()}] wrote index: {index_csv} rows={len(all_reports)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
