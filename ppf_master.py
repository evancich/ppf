#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ppf_master.py

MASTER ORCHESTRATOR
-------------------
One command to:
  1) Crawl House disclosures (2015..current year) into data/raw/house/...
  2) Crawl Senate eFD PTR reports (2012..today) into data/raw/efd/...
  3) Run unified extraction/parsing over PDFs (unify_ppf.py)
  4) Run the normalization pipeline + PPF scoring (ppf_pipeline.py)
  5) Run the simple backtest / analysis (ppf_backtest_simple.py)

Why unify_ppf.py is required:
  - It is the PDF table extractor that creates outputs/ppf_transactions_unified.csv
  - ppf_pipeline.py consumes that unified CSV.

USAGE
-----
python ppf_master.py --project-root . --do-house --do-senate --do-unify --do-pipeline --do-analysis --verbose

Recommended end-to-end (House 2015..2026):
python ppf_master.py \
  --project-root . \
  --do-house --house-start-year 2015 --house-end-year 2026 --house-headless --house-download-pdfs \
  --do-senate --senate-since 2012-01-01 --senate-download \
  --do-unify \
  --do-pipeline --auto-map-yfinance \
  --do-analysis \
  --verbose
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


LOG_REL = Path("logs/ppf_master.log")
LOGGER = logging.getLogger("ppf_master")


def setup_logging(project_root: Path, verbose: bool) -> None:
    log_path = project_root / LOG_REL
    log_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()

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

    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)

    LOGGER.info("Logging initialized")
    LOGGER.info("log_file=%s", str(log_path))


@dataclass
class CmdResult:
    cmd: List[str]
    returncode: int
    elapsed_s: float


def run_cmd(cmd: List[str], cwd: Path, check: bool = True) -> CmdResult:
    start = time.time()
    LOGGER.info("RUN: %s", " ".join(cmd))

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
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


def ensure_expected_scripts(project_root: Path) -> None:
    required = [
        "ppf_house_crawler.py",
        "ppf_crawl_efd.py",
        "unify_ppf.py",
        "ppf_pipeline.py",
        "ppf_backtest_simple.py",
    ]
    missing = [s for s in required if not (project_root / s).exists()]
    if missing:
        raise FileNotFoundError("Missing required scripts:\n  " + "\n  ".join(missing))


def stage_house_crawl(
    project_root: Path,
    year_start: int,
    year_end: int,
    headless: bool,
    download_pdfs: bool,
    verbose: bool,
    sleep_between_years_s: float,
) -> None:
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
        time.sleep(max(0.0, sleep_between_years_s))
    LOGGER.info("=== Stage: House crawl complete ===")


def stage_senate_crawl(
    project_root: Path,
    since: str,
    max_pages: int,
    download: bool,
    overwrite: bool,
) -> None:
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


def stage_unify(project_root: Path, verbose: bool) -> None:
    LOGGER.info("=== Stage: unify_ppf (parse PDFs) ===")
    cmd = [sys.executable, "unify_ppf.py", "--project-root", str(project_root)]
    if verbose:
        cmd.append("--verbose")
    run_cmd(cmd, cwd=project_root, check=True)
    LOGGER.info("=== Stage: unify_ppf complete ===")


def stage_pipeline(project_root: Path, verbose: bool, auto_map_yfinance: bool) -> None:
    LOGGER.info("=== Stage: pipeline ===")
    cmd = [sys.executable, "ppf_pipeline.py", "--project-root", str(project_root)]
    if verbose:
        cmd.append("--verbose")
    if auto_map_yfinance:
        cmd.append("--auto-map-yfinance")
    run_cmd(cmd, cwd=project_root, check=True)
    LOGGER.info("=== Stage: pipeline complete ===")


def stage_analysis(project_root: Path, verbose: bool) -> None:
    LOGGER.info("=== Stage: analysis ===")
    cmd = [sys.executable, "ppf_backtest_simple.py"]
    if verbose:
        cmd.append("--verbose")
    run_cmd(cmd, cwd=project_root, check=True)
    LOGGER.info("=== Stage: analysis complete ===")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)

    p.add_argument("--do-house", action="store_true")
    p.add_argument("--do-senate", action="store_true")
    p.add_argument("--do-unify", action="store_true")
    p.add_argument("--do-pipeline", action="store_true")
    p.add_argument("--do-analysis", action="store_true")

    p.add_argument("--house-start-year", type=int, default=2015)
    p.add_argument("--house-end-year", type=int, default=datetime.now().year)
    p.add_argument("--house-headless", action="store_true")
    p.add_argument("--house-download-pdfs", action="store_true")
    p.add_argument("--house-sleep-between-years", type=float, default=5.0)

    p.add_argument("--senate-since", default="2012-01-01")
    p.add_argument("--senate-max-pages", type=int, default=0)
    p.add_argument("--senate-download", action="store_true")
    p.add_argument("--senate-overwrite", action="store_true")

    p.add_argument("--auto-map-yfinance", action="store_true", help="Let pipeline fill sector/subsector using yfinance")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    setup_logging(project_root, verbose=args.verbose)

    LOGGER.info("=== PPF Master start ===")
    LOGGER.info("project_root=%s", project_root)

    ensure_expected_scripts(project_root)

    if not (args.do_house or args.do_senate or args.do_unify or args.do_pipeline or args.do_analysis):
        LOGGER.error("No stages selected.")
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

        if args.do_unify:
            stage_unify(project_root=project_root, verbose=args.verbose)

        if args.do_pipeline:
            stage_pipeline(
                project_root=project_root,
                verbose=args.verbose,
                auto_map_yfinance=args.auto_map_yfinance,
            )

        if args.do_analysis:
            stage_analysis(project_root=project_root, verbose=args.verbose)

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        return 130
    except Exception as e:
        LOGGER.exception("Master failed: %s", repr(e))
        return 1

    LOGGER.info("=== PPF Master end (success) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
