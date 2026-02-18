#!/usr/bin/env python3
"""
unify_ppf.py

Unifies transaction records into a single canonical CSV for downstream PPF pipeline.

Key behaviors:
- Reads outputs/efd_reports_index.csv (Senate eFD crawl index)
- Parses BOTH HTML and PDF artifacts referenced by local_path
- HTML PTR pages are parsed via pandas.read_html() (high yield for Senate PTR)
- PDF parsing is best-effort via pdfplumber table extraction; HTML is primary for PTR
- Never crashes on empty runs: emits diagnostics and writes an empty unified CSV

This file is designed to be a drop-in replacement that addresses:
- ValueError: No objects to concatenate
- Corpus dominated by HTML (is_html=True) causing empty ptr_df if HTML ignored

CLI:
  python unify_ppf.py --project-root . [--verbose]

Outputs:
  outputs/ppf_transactions_unified.csv
  outputs/unify_manifest.json
  logs/unify_ppf.log
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Optional PDF dependency. HTML parsing is the main path.
try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore


# ----------------------------
# Utilities
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("unify_ppf")
    logger.setLevel(logging.DEBUG)

    # avoid duplicate handlers if reloaded
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)sZ | %(levelname)-7s | %(name)s | %(message)s")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO if verbose else logging.WARNING)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.info("Logging initialized")
    logger.info(f"log_file={log_path}")
    return logger


def safe_read_csv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        logger.error(f"Missing required file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.exception(f"Failed reading CSV {path}: {e}")
        return pd.DataFrame()


def normalize_bool_series(s: pd.Series) -> pd.Series:
    # Handles True/False, "True"/"False", 1/0, etc.
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def clean_colname(c: Any) -> str:
    return re.sub(r"\s+", " ", str(c).strip()).lower()


def coerce_amount_range(val: Any) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Attempts to parse Senate PTR amount ranges like:
      "$1,001 - $15,000"
      "$15,001 - $50,000"
      "None"
    Returns (low, high, raw_text)
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return (None, None, None)
    raw = str(val).strip()
    if not raw or raw.lower() in {"none", "n/a", "na"}:
        return (None, None, raw)

    # extract numeric tokens
    nums = [n.replace(",", "") for n in re.findall(r"\$?\s*([0-9][0-9,]*)", raw)]
    if len(nums) >= 2:
        try:
            return (float(nums[0]), float(nums[1]), raw)
        except Exception:
            return (None, None, raw)
    if len(nums) == 1:
        try:
            x = float(nums[0])
            return (x, x, raw)
        except Exception:
            return (None, None, raw)
    return (None, None, raw)


def coerce_date(val: Any) -> Optional[str]:
    """
    Attempts to normalize a date string to YYYY-MM-DD.
    Returns None if not parseable.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    # common formats: MM/DD/YYYY, M/D/YYYY, YYYY-MM-DD
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date().isoformat()
    except Exception:
        return None


# ----------------------------
# Canonical schema
# ----------------------------

CANON_COLS = [
    # provenance
    "source",              # e.g., "senate_efd_ptr_html" / "senate_efd_pdf"
    "report_id",           # from index
    "local_path",          # from index
    "filer_first",
    "filer_last",
    "date_received_raw",
    "url_full",

    # transaction fields (best-effort)
    "owner",
    "transaction_date",
    "transaction_type",
    "asset_name",
    "asset_type",
    "ticker",
    "amount_raw",
    "amount_low",
    "amount_high",
    "comments",
]


def empty_unified_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CANON_COLS)


# ----------------------------
# HTML PTR parsing
# ----------------------------

PTR_COL_KEYWORDS = {
    "transaction_date": ["transaction date", "date", "transactiondate"],
    "owner": ["owner"],
    "transaction_type": ["transaction type", "type"],
    "asset_name": ["asset name", "asset", "name"],
    "asset_type": ["asset type"],
    "ticker": ["ticker"],
    "amount_raw": ["amount"],
    "comments": ["comment", "comments", "description", "notes"],
}


def pick_transaction_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Heuristic selection of the table that represents transactions on PTR HTML.
    """
    best = None
    best_score = -1

    for t in tables:
        if t is None or t.empty:
            continue
        cols = [clean_colname(c) for c in t.columns]
        colblob = " | ".join(cols)

        # Score presence of key concepts; require at least some density
        score = 0
        for k in ["transaction", "amount", "ticker", "asset", "owner", "date", "type"]:
            if k in colblob:
                score += 1

        # Additional heuristics: transaction tables usually have multiple rows and >=5 columns
        if len(t) >= 1:
            score += 1
        if t.shape[1] >= 5:
            score += 1

        if score > best_score:
            best_score = score
            best = t

    # Require a minimum score so we don't select irrelevant tables
    if best is None:
        return None
    if best_score < 4:
        return None
    return best


def map_ptr_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Maps input dataframe columns to canonical fields using keyword matching.
    Returns mapping: canonical -> source_column_name
    """
    cols = list(df.columns)
    cleaned = {c: clean_colname(c) for c in cols}

    mapping: Dict[str, str] = {}
    for canon, keys in PTR_COL_KEYWORDS.items():
        for c in cols:
            cc = cleaned[c]
            if any(k in cc for k in keys):
                mapping[canon] = c
                break
    return mapping


def parse_ptr_html(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Parses a Senate eFD PTR HTML file into a dataframe of transactions (raw columns).
    """
    try:
        tables = pd.read_html(str(path))
    except Exception as e:
        logger.debug(f"read_html failed: {path} err={e}")
        return pd.DataFrame()

    t = pick_transaction_table(tables)
    if t is None or t.empty:
        return pd.DataFrame()

    # Clean up: strip whitespace in column names
    t = t.copy()
    t.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in t.columns]

    # Some pages may include repeated header rows inside the table; drop obvious duplicates
    # If "Transaction Date" appears as a value row, drop those rows.
    for col in t.columns:
        if "date" in clean_colname(col):
            t = t[t[col].astype(str).str.lower().ne(str(col).lower())]
            break

    return t


def normalize_ptr_rows(
    raw: pd.DataFrame,
    idx_row: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Converts raw PTR table rows to canonical schema.
    """
    if raw is None or raw.empty:
        return empty_unified_df()

    mapping = map_ptr_columns(raw)
    # Build canonical rows
    out_rows: List[Dict[str, Any]] = []
    for _, r in raw.iterrows():
        def get(canon: str) -> Any:
            col = mapping.get(canon)
            if col is None:
                return None
            return r.get(col)

        amount_raw = get("amount_raw")
        low, high, _raw = coerce_amount_range(amount_raw)

        out_rows.append({
            "source": "senate_efd_ptr_html",
            "report_id": idx_row.get("report_id"),
            "local_path": idx_row.get("local_path"),
            "filer_first": idx_row.get("filer_first"),
            "filer_last": idx_row.get("filer_last"),
            "date_received_raw": idx_row.get("date_received_raw"),
            "url_full": idx_row.get("url_full"),

            "owner": get("owner"),
            "transaction_date": coerce_date(get("transaction_date")),
            "transaction_type": get("transaction_type"),
            "asset_name": get("asset_name"),
            "asset_type": get("asset_type"),
            "ticker": get("ticker"),
            "amount_raw": None if amount_raw is None else str(amount_raw),
            "amount_low": low,
            "amount_high": high,
            "comments": get("comments"),
        })

    df = pd.DataFrame(out_rows)
    # Ensure all columns present
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = None
    return df[CANON_COLS]


# ----------------------------
# PDF parsing (best-effort)
# ----------------------------

def parse_pdf_best_effort(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Best-effort PDF parsing.
    This is intentionally conservative: if we cannot reliably extract a transaction table,
    we return empty and rely on HTML for PTR coverage.

    If you later want higher recall, swap in camelot/tabula or a custom layout parser.
    """
    if pdfplumber is None:
        logger.debug("pdfplumber not available; skipping PDF parsing")
        return pd.DataFrame()

    try:
        with pdfplumber.open(str(path)) as pdf:
            all_tables: List[pd.DataFrame] = []
            # Limit pages to reduce runtime spikes; adjust as needed
            max_pages = min(len(pdf.pages), 8)
            for i in range(max_pages):
                page = pdf.pages[i]
                tables = page.extract_tables() or []
                for t in tables:
                    if not t or len(t) < 2:
                        continue
                    # first row is header candidate
                    header = [re.sub(r"\s+", " ", (h or "")).strip() for h in t[0]]
                    rows = t[1:]
                    # Build df
                    try:
                        df = pd.DataFrame(rows, columns=header)
                        # Heuristic: transaction-ish
                        cols = [clean_colname(c) for c in df.columns]
                        blob = " ".join(cols)
                        score = sum(k in blob for k in ["transaction", "amount", "asset", "owner", "ticker", "date", "type"])
                        if score >= 3 and len(df) >= 1 and df.shape[1] >= 4:
                            all_tables.append(df)
                    except Exception:
                        continue

            if not all_tables:
                return pd.DataFrame()

            # pick best table
            best = None
            best_score = -1
            for df in all_tables:
                cols = [clean_colname(c) for c in df.columns]
                blob = " ".join(cols)
                score = sum(k in blob for k in ["transaction", "amount", "asset", "owner", "ticker", "date", "type"])
                if score > best_score:
                    best_score = score
                    best = df
            return best if best is not None else pd.DataFrame()
    except Exception as e:
        logger.debug(f"PDF parse failed: {path} err={e}")
        return pd.DataFrame()


def normalize_pdf_rows(raw: pd.DataFrame, idx_row: Dict[str, Any]) -> pd.DataFrame:
    """
    Best-effort normalization for PDF-extracted tables.
    Column names vary; use keyword mapping similar to HTML.
    """
    if raw is None or raw.empty:
        return empty_unified_df()

    mapping = map_ptr_columns(raw)

    out_rows: List[Dict[str, Any]] = []
    for _, r in raw.iterrows():
        def get(canon: str) -> Any:
            col = mapping.get(canon)
            if col is None:
                return None
            return r.get(col)

        amount_raw = get("amount_raw")
        low, high, _raw = coerce_amount_range(amount_raw)

        out_rows.append({
            "source": "senate_efd_pdf",
            "report_id": idx_row.get("report_id"),
            "local_path": idx_row.get("local_path"),
            "filer_first": idx_row.get("filer_first"),
            "filer_last": idx_row.get("filer_last"),
            "date_received_raw": idx_row.get("date_received_raw"),
            "url_full": idx_row.get("url_full"),

            "owner": get("owner"),
            "transaction_date": coerce_date(get("transaction_date")),
            "transaction_type": get("transaction_type"),
            "asset_name": get("asset_name"),
            "asset_type": get("asset_type"),
            "ticker": get("ticker"),
            "amount_raw": None if amount_raw is None else str(amount_raw),
            "amount_low": low,
            "amount_high": high,
            "comments": get("comments"),
        })

    df = pd.DataFrame(out_rows)
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = None
    return df[CANON_COLS]


# ----------------------------
# Main unify pipeline
# ----------------------------

@dataclass
class Inputs:
    project_root: Path
    efd_index_csv: Path


@dataclass
class Outputs:
    out_dir: Path
    unified_csv: Path
    manifest_json: Path
    log_file: Path


def build_paths(project_root: Path) -> Tuple[Inputs, Outputs]:
    out_dir = project_root / "outputs"
    log_dir = project_root / "logs"
    return (
        Inputs(
            project_root=project_root,
            efd_index_csv=out_dir / "efd_reports_index.csv",
        ),
        Outputs(
            out_dir=out_dir,
            unified_csv=out_dir / "ppf_transactions_unified.csv",
            manifest_json=out_dir / "unify_manifest.json",
            log_file=log_dir / "unify_ppf.log",
        ),
    )


def unify(inputs: Inputs, outputs: Outputs, logger: logging.Logger) -> pd.DataFrame:
    idx = safe_read_csv(inputs.efd_index_csv, logger)
    if idx.empty:
        logger.error("efd_reports_index.csv is empty or missing; nothing to unify.")
        return empty_unified_df()

    # Normalize booleans
    for c in ["downloaded", "is_pdf", "is_html"]:
        if c in idx.columns:
            idx[c] = normalize_bool_series(idx[c])

    # Filter to downloaded with a local_path that exists
    if "downloaded" in idx.columns:
        idx = idx[idx["downloaded"] == True]  # noqa: E712
    if "local_path" not in idx.columns:
        logger.error("efd_reports_index.csv missing local_path column; cannot proceed.")
        return empty_unified_df()

    idx["local_path"] = idx["local_path"].fillna("").astype(str)
    idx = idx[idx["local_path"].str.len() > 0]

    # Ensure on-disk existence
    def exists(p: str) -> bool:
        try:
            return Path(p).exists()
        except Exception:
            return False

    idx = idx[idx["local_path"].map(exists)]
    logger.info(f"Index rows after downloaded+exists filtering: {len(idx)}")

    # Split html/pdf
    html_idx = idx[idx["local_path"].str.lower().str.endswith(".html")]
    pdf_idx = idx[idx["local_path"].str.lower().str.endswith(".pdf")]
    logger.info(f"Candidate HTML: {len(html_idx)}")
    logger.info(f"Candidate PDF:  {len(pdf_idx)}")

    # Parse HTML PTR
    html_frames: List[pd.DataFrame] = []
    html_ok = 0
    html_zero = 0
    for i, row in html_idx.iterrows():
        p = Path(row["local_path"])
        raw = parse_ptr_html(p, logger)
        if raw is None or raw.empty:
            html_zero += 1
            continue
        html_ok += 1
        html_frames.append(normalize_ptr_rows(raw, row.to_dict(), logger))

        # periodic progress
        if html_ok % 250 == 0:
            logger.info(f"HTML parsed ok={html_ok} zero={html_zero} scanned={html_ok+html_zero}/{len(html_idx)}")

    # Parse PDFs (best-effort)
    pdf_frames: List[pd.DataFrame] = []
    pdf_ok = 0
    pdf_zero = 0
    for i, row in pdf_idx.iterrows():
        p = Path(row["local_path"])
        raw = parse_pdf_best_effort(p, logger)
        if raw is None or raw.empty:
            pdf_zero += 1
            continue
        pdf_ok += 1
        pdf_frames.append(normalize_pdf_rows(raw, row.to_dict()))

        if pdf_ok % 100 == 0:
            logger.info(f"PDF parsed ok={pdf_ok} zero={pdf_zero} scanned={pdf_ok+pdf_zero}/{len(pdf_idx)}")

    # Concatenate safely (never crash)
    parts = [df for df in (html_frames + pdf_frames) if df is not None and not df.empty]
    if not parts:
        logger.error("unify_ppf produced 0 unified rows (no parseable HTML/PDF tables matched heuristics).")
        logger.error(f"html candidates={len(html_idx)} html_ok={html_ok} html_zero={html_zero}")
        logger.error(f"pdf  candidates={len(pdf_idx)}  pdf_ok={pdf_ok}  pdf_zero={pdf_zero}")

        # write empty unified output anyway for downstream determinism
        out = empty_unified_df()
        outputs.out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outputs.unified_csv, index=False)
        manifest = {
            "generated_utc": utc_now_iso(),
            "index_rows": int(len(idx)),
            "html_candidates": int(len(html_idx)),
            "pdf_candidates": int(len(pdf_idx)),
            "html_ok": int(html_ok),
            "html_zero": int(html_zero),
            "pdf_ok": int(pdf_ok),
            "pdf_zero": int(pdf_zero),
            "unified_rows": 0,
            "note": "No parseable tables found; adjust heuristics or add dedicated parsers.",
        }
        outputs.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    unified = pd.concat(parts, ignore_index=True)
    # basic cleanup
    unified = unified.dropna(how="all")
    for c in CANON_COLS:
        if c not in unified.columns:
            unified[c] = None
    unified = unified[CANON_COLS]

    outputs.out_dir.mkdir(parents=True, exist_ok=True)
    unified.to_csv(outputs.unified_csv, index=False)

    manifest = {
        "generated_utc": utc_now_iso(),
        "index_rows": int(len(idx)),
        "html_candidates": int(len(html_idx)),
        "pdf_candidates": int(len(pdf_idx)),
        "html_ok": int(html_ok),
        "html_zero": int(html_zero),
        "pdf_ok": int(pdf_ok),
        "pdf_zero": int(pdf_zero),
        "unified_rows": int(len(unified)),
        "output_csv": str(outputs.unified_csv),
    }
    outputs.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(f"Wrote unified: {outputs.unified_csv} rows={len(unified)}")
    logger.info(f"Wrote manifest: {outputs.manifest_json}")
    return unified


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True, help="Project root (same as used by ppf_pipeline.py)")
    ap.add_argument("--verbose", action="store_true", help="Verbose console logging")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    inputs, outputs = build_paths(project_root)
    logger = setup_logging(outputs.log_file, verbose=args.verbose)

    logger.info("=== unify_ppf start ===")
    logger.info(f"project_root={project_root}")

    df = unify(inputs, outputs, logger)

    logger.info("=== unify_ppf complete ===")
    logger.info(f"rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
