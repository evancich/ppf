#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ppf_master.py

MASTER ORCHESTRATOR
-------------------
One command to:
  1) Crawl House disclosures (2015..current year) into data/raw/house/...
  2) Crawl Senate eFD PTR reports (2012..today) into data/raw/efd/...
  3) Run the normalization pipeline (ppf_pipeline.py)
  4) Run the simple backtest / analysis (ppf_backtest_simple.py)

ASSUMPTIONS (reuses what you've already built)
----------------------------------------------
Expected sibling scripts in the same directory:
  - ppf_house_crawler.py
  - ppf_crawl_efd.py
  - ppf_pipeline.py
  - ppf_backtest_simple.py

Expected directories (created if missing):
  - data/raw/house/pdfs/<year>/
  - data/raw/efd/pdf/ and data/raw/efd/html/
  - data/processed/
  - outputs/
  - logs/

USAGE
-----
# Full end-to-end:
python ppf_master.py --project-root . --do-house --do-senate --do-pipeline --do-analysis

# Crawl only:
python ppf_master.py --project-root . --do-house --do-senate

# Pipeline + analysis only (assuming PDFs already present):
python ppf_master.py --project-root . --do-pipeline --do-analysis

NOTES
-----
- This orchestrator runs subprocesses so you don't have to manually invoke each stage.
- It logs verbosely to logs/ppf_master.log and mirrors key progress to console.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# -----------------------------
# Logging
# -----------------------------

LOG_REL = Path("logs/ppf_master.log")
LOGGER = logging.getLogger("ppf_master")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(project_root: Path, verbose: bool) -> None:
    log_path = project_root / LOG_REL
    log_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)sZ | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    LOGGER.handlers.clear()
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)

    LOGGER.info("Logging initialized")
    LOGGER.info("log_file=%s", str(log_path))


# -----------------------------
# Subprocess helpers
# -----------------------------

@dataclass
class CmdResult:
    cmd: List[str]
    returncode: int
    elapsed_s: float


def run_cmd(
    cmd: List[str],
    cwd: Path,
    env: Optional[dict] = None,
    check: bool = True,
) -> CmdResult:
    """
    Run a subprocess with streaming output (so you see progress in real time),
    while also keeping structured logging of start/end.
    """
    start = time.time()
    LOGGER.info("RUN: %s", " ".join(cmd))

    # Stream output line-by-line to console; capture return code
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\n")
        if line:
            LOGGER.info("[subprocess] %s", line)

    p.wait()
    elapsed = time.time() - start

    LOGGER.info("DONE: rc=%d elapsed=%.2fs", p.returncode, elapsed)

    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed rc={p.returncode}: {' '.join(cmd)}")

    return CmdResult(cmd=cmd, returncode=p.returncode, elapsed_s=elapsed)


# -----------------------------
# Stage runners
# -----------------------------

def ensure_expected_scripts(project_root: Path) -> None:
    required = [
        "ppf_house_crawler.py",
        "ppf_crawl_efd.py",
        "ppf_pipeline.py",
        "ppf_backtest_simple.py",
    ]
    missing = [s for s in required if not (project_root / s).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required scripts in project root:\n  " + "\n  ".join(missing)
        )


def stage_house_crawl(
    project_root: Path,
    year_start: int,
    year_end: int,
    headless: bool,
    download_pdfs: bool,
    verbose: bool,
    sleep_between_years_s: float,
) -> None:
    """
    Crawl House filings for each year in [year_start..year_end].
    """
    LOGGER.info("=== Stage: House crawl ===")
    for year in range(year_start, year_end + 1):
        LOGGER.info("House year=%d", year)
        cmd = [
            sys.executable,
            "ppf_house_crawler.py",
            "--project-root",
            str(project_root),
            "--filing-year",
            str(year),
            "--prefix-range",
            "A",
            "Z",
        ]
        if download_pdfs:
            cmd.append("--download-pdfs")
        if headless:
            cmd.append("--headless")
        if verbose:
            cmd.append("--verbose")

        run_cmd(cmd, cwd=project_root, check=True)

        LOGGER.info("Sleep between years: %.1fs", sleep_between_years_s)
        time.sleep(max(0.0, sleep_between_years_s))

    LOGGER.info("=== Stage: House crawl complete ===")


def stage_senate_crawl(
    project_root: Path,
    since: str,
    max_pages: int,
    download: bool,
    overwrite: bool,
) -> None:
    """
    Crawl Senate eFD PTRs using the existing ppf_crawl_efd.py.
    """
    LOGGER.info("=== Stage: Senate crawl ===")
    cmd = [
        sys.executable,
        "ppf_crawl_efd.py",
        "--project-root",
        str(project_root),
        "--since",
        since,
        "--max-pages",
        str(max_pages),
    ]
    if download:
        cmd.append("--download")
    if overwrite:
        cmd.append("--overwrite")

    run_cmd(cmd, cwd=project_root, check=True)
    LOGGER.info("=== Stage: Senate crawl complete ===")


def stage_pipeline(project_root: Path, verbose: bool) -> None:
    LOGGER.info("=== Stage: Pipeline ===")
    cmd = [sys.executable, "ppf_pipeline.py", "--project-root", str(project_root)]
    if verbose:
        cmd.append("--verbose")
    run_cmd(cmd, cwd=project_root, check=True)
    LOGGER.info("=== Stage: Pipeline complete ===")


def stage_analysis(project_root: Path) -> None:
    """
    Runs the simple backtest. This script currently prints to console and plots.
    """
    LOGGER.info("=== Stage: Analysis (simple backtest) ===")
    cmd = [sys.executable, "ppf_backtest_simple.py"]
    run_cmd(cmd, cwd=project_root, check=True)
    LOGGER.info("=== Stage: Analysis complete ===")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--project-root", required=True, help="Repo root (directory containing the ppf_*.py scripts).")

    # Which stages to run
    p.add_argument("--do-house", action="store_true", help="Crawl House filings.")
    p.add_argument("--do-senate", action="store_true", help="Crawl Senate eFD PTR filings.")
    p.add_argument("--do-pipeline", action="store_true", help="Run ppf_pipeline.py normalization/enrichment.")
    p.add_argument("--do-analysis", action="store_true", help="Run ppf_backtest_simple.py analysis.")

    # House crawl options
    p.add_argument("--house-start-year", type=int, default=2015, help="House crawl start year (default 2015).")
    p.add_argument("--house-end-year", type=int, default=datetime.now().year, help="House crawl end year (default current year).")
    p.add_argument("--house-headless", action="store_true", help="Run House crawler headless (recommended).")
    p.add_argument("--house-download-pdfs", action="store_true", help="Actually download PDFs during House crawl.")
    p.add_argument("--house-sleep-between-years", type=float, default=5.0, help="Seconds to sleep between years.")

    # Senate crawl options
    p.add_argument("--senate-since", default="2012-01-01", help="Start date for Senate crawl (YYYY-MM-DD).")
    p.add_argument("--senate-max-pages", type=int, default=0, help="0 = no limit; else number of pages.")
    p.add_argument("--senate-download", action="store_true", help="Download Senate reports (HTML/PDF).")
    p.add_argument("--senate-overwrite", action="store_true", help="Overwrite existing Senate downloads.")

    # General
    p.add_argument("--verbose", action="store_true", help="Verbose master logging and pass through where supported.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    setup_logging(project_root, verbose=args.verbose)

    LOGGER.info("=== PPF Master start ===")
    LOGGER.info("project_root=%s", str(project_root))
    LOGGER.info("argv=%s", " ".join(sys.argv))

    ensure_expected_scripts(project_root)

    # If user runs with no stages, fail fast with actionable message
    if not (args.do_house or args.do_senate or args.do_pipeline or args.do_analysis):
        LOGGER.error("No stages selected. Use one or more: --do-house --do-senate --do-pipeline --do-analysis")
        return 2

    try:
        if args.do_house:
            stage_house_crawl(
                project_root=project_root,
                year_start=args.house_start_year,
                year_end=args.house_end_year,
                headless=args.house_headless,
                download_pdfs=args.house_download_pdfs,
                verbose=args.verbose,
                sleep_between_years_s=args.house_sleep_between_years,
            )

        if args.do_senate:
            stage_senate_crawl(
                project_root=project_root,
                since=args.senate_since,
                max_pages=args.senate_max_pages,
                download=args.senate_download,
                overwrite=args.senate_overwrite,
            )

        if args.do_pipeline:
            stage_pipeline(project_root=project_root, verbose=args.verbose)

        if args.do_analysis:
            stage_analysis(project_root=project_root)

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user (Ctrl+C).")
        return 130
    except Exception as e:
        LOGGER.exception("Master failed: %s", repr(e))
        return 1

    LOGGER.info("=== PPF Master end (success) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
