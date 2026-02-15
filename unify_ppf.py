#!/usr/bin/env python3
"""
PPF Unified Extractor (Disclosures-Only) — v11 (PTR2 pages 2–4 fix + logging + exception hardening)

Reads PDFs / optional scanned-form PNGs from:   ./data/raw/
Writes normalized artifacts to:                 ./data/processed/  and ./outputs/
Logs to:                                        ./logs/ppf_extract.log

Primary goals:
  1) Preserve EVERYTHING extractable (text, tables) with provenance.
  2) Produce unified "superset" records across all sources, record_type discriminator.
  3) Produce a canonical TRANSACTION ledger suitable for PPF scoring (filing_datetime anchored).

Key fixes in v11:
  - PTR tables may include leading empty columns; row number can appear in column 3 ("#") as seen in PTR2 pages 2–4.
    We now locate row_number by scanning the first few cells for a numeric token.
  - Amount buckets and other fields may contain newlines; we normalize all cell text by collapsing whitespace.
  - Ticker value "--" is treated as missing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pdfplumber

# Optional deps for scanned PNG checkbox semantics
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None


# -----------------------------
# Glob patterns / constants
# -----------------------------

PTR_FILE_GLOB = "eFD_ Print Periodic Transaction Report*.pdf"
ANNUAL_FILE_GLOB = "eFD_ Annual Report*.pdf"
OTHER_PDF_GLOB = "*.pdf"
SCAN_PNG_GLOB = "ppf_tmp_report5_p*.png"

AMOUNT_BUCKETS_STD = [
    "$1,001 - $15,000",
    "$15,001 - $50,000",
    "$50,001 - $100,000",
    "$100,001 - $250,000",
    "$250,001 - $500,000",
    "$500,001 - $1,000,000",
    "$1,000,001 - $5,000,000",
    "$5,000,001 - $25,000,000",
    "$25,000,001 - $50,000,000",
    "Over $50,000,000",
]

OWNER_SET = {"Self", "Spouse", "Joint", "Dependent Child"}


# -----------------------------
# Logging
# -----------------------------

def setup_logging(log_dir: Path, run_id: str, verbose: bool) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ppf_extract.log"

    logger = logging.getLogger("ppf")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("=== PPF Extractor start ===")
    logger.info("run_id=%s log_path=%s", run_id, str(log_path))
    logger.info("opencv_available=%s pytesseract_available=%s", cv2 is not None, pytesseract is not None)
    return logger


def exc_to_dict(e: BaseException) -> Dict[str, Any]:
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }


# -----------------------------
# Utilities
# -----------------------------

def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _is_missing_str(x: Any) -> bool:
    """Treat NaN, None, empty, and whitespace-only as missing."""
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def normalize_ticker(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"^(NYSE|NASDAQ|AMEX|OTC)\s*:\s*", "", s)
    s = s.replace("$", "")
    s = re.sub(r"[^A-Z0-9\.\-]", "", s)
    return s


def parse_date_any(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_amount_bucket_to_minmax(bucket: str) -> Tuple[Optional[float], Optional[float]]:
    b = (bucket or "").strip()
    if not b:
        return None, None
    b0 = b.replace(",", "")
    m = re.match(r"^\$?(\d+)\s*-\s*\$?(\d+)$", b0)
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.match(r"^Over\s*\$?(\d+)$", b0, flags=re.IGNORECASE)
    if m2:
        return float(m2.group(1)), None
    return None, None


def infer_tx_type_and_scope(raw: str) -> Tuple[str, str]:
    x = norm_ws(raw).lower()
    scope = ""
    if "purchase" in x:
        return "BUY", scope
    if "sale" in x:
        if "partial" in x:
            scope = "PARTIAL"
        elif "full" in x:
            scope = "FULL"
        return "SELL", scope
    if "exchange" in x:
        return "EXCHANGE", scope
    return "OTHER", scope


# -----------------------------
# 1) Preserve EVERYTHING extractable: PDF pages
# -----------------------------

def extract_pdf_pages(pdf_path: Path, logger: logging.Logger, errors: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    logger.info("Extracting PDF pages (text+tables): %s", pdf_path.name)

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    errors.append({"stage": "pdf_page_text", "file": pdf_path.name, "page": i, **exc_to_dict(e)})
                    logger.exception("Failed extract_text file=%s page=%s", pdf_path.name, i)
                    text = ""

                try:
                    tables = page.extract_tables() or []
                except Exception as e:
                    errors.append({"stage": "pdf_page_tables", "file": pdf_path.name, "page": i, **exc_to_dict(e)})
                    logger.exception("Failed extract_tables file=%s page=%s", pdf_path.name, i)
                    tables = []

                rows.append(
                    {
                        "record_type": "PDF_PAGE",
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "page_number": i,
                        "page_width": float(page.width),
                        "page_height": float(page.height),
                        "page_text": text,
                        "page_tables_json": json.dumps(tables, ensure_ascii=False),
                    }
                )
    except Exception as e:
        errors.append({"stage": "pdf_open", "file": pdf_path.name, **exc_to_dict(e)})
        logger.exception("Failed opening PDF: %s", pdf_path.name)

    logger.info("PDF pages extracted: %s rows=%d", pdf_path.name, len(rows))
    return pd.DataFrame(rows)


# -----------------------------
# 2) PTR parsing: TABLE-FIRST with robust header scan
# -----------------------------

def scan_header_metadata(pdf: pdfplumber.PDF, logger: logging.Logger) -> Tuple[str, str]:
    """
    Senate eFD PTR PDFs often do not expose a reliable 'Name:' header token as text.
    Instead, the official identity is present as '(Last, First)', and the filing time
    appears as: 'Filed MM/DD/YYYY @ H:MM AM/PM'.

    Deterministic, text-only (no OCR), validated across your current dataset.
    """
    official_name = ""
    filing_dt_iso = ""

    name_pat = re.compile(r"\(([A-Za-z\-\.\' ]+,\s*[A-Za-z\-\.\' ]+)\)")
    filed_pat = re.compile(
        r"\bFiled\s+(\d{2}/\d{2}/\d{4})\s*@\s*(\d{1,2}:\d{2})\s*(AM|PM)\b",
        re.I,
    )

    pages_to_scan = min(2, len(pdf.pages))
    for j in range(pages_to_scan):
        page = pdf.pages[j]
        try:
            words_info = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            words = [w["text"] for w in words_info if isinstance(w, dict) and w.get("text")]
            text = " ".join(words)
        except Exception as e:
            logger.exception("Header word extraction failed page=%s err=%s", j + 1, str(e))
            continue

        if not official_name:
            m = name_pat.search(text)
            if m:
                official_name = norm_ws(m.group(1))

        if not filing_dt_iso:
            mf = filed_pat.search(text)
            if mf:
                try:
                    d = dt.datetime.strptime(mf.group(1), "%m/%d/%Y").date()
                    tm = dt.datetime.strptime((mf.group(2) + " " + mf.group(3)).upper(), "%I:%M %p").time()
                    filing_dt_iso = dt.datetime.combine(d, tm).replace(second=0).isoformat()
                except Exception:
                    filing_dt_iso = ""

        if official_name or filing_dt_iso:
            logger.info(
                "Header scan page=%d official_name=%s filing_datetime=%s",
                j + 1,
                official_name if official_name else "<missing>",
                filing_dt_iso if filing_dt_iso else "<missing>",
            )

        if official_name and filing_dt_iso:
            break

    if not official_name:
        logger.warning("Header metadata: official_name not found after scanning %d pages", pages_to_scan)
    if not filing_dt_iso:
        logger.warning("Header metadata: filing_datetime not found after scanning %d pages", pages_to_scan)

    return official_name, filing_dt_iso


def _normalize_cell(c: Any) -> str:
    if c is None:
        return ""
    # collapse all whitespace, including newlines from extracted tables
    return norm_ws(str(c))


def _find_row_number(cells: List[str]) -> Optional[int]:
    """
    PTR tables may include leading empty columns; the '#' column may be at index 2.
    We scan the first N cells for a pure integer token.
    """
    for i in range(min(6, len(cells))):
        if re.fullmatch(r"\d+", cells[i]):
            try:
                return int(cells[i])
            except Exception:
                return None
    # fallback: any integer token near the start of the row
    joined = " ".join(cells[:8])
    m = re.match(r"^\s*(\d+)\b", joined)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def parse_ptr_table_row(row: List[Any], logger: logging.Logger) -> Optional[Dict[str, Any]]:
    cells = [_normalize_cell(c) for c in (row or [])]

    if not cells:
        return None

    row_number = _find_row_number(cells)
    if row_number is None:
        return None

    # Transaction Date
    tx_date_iso = ""
    for c in cells:
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", c):
            tx_date_iso = parse_date_any(c) or ""
            break

    # Owner
    owner = ""
    for c in cells:
        if c in OWNER_SET:
            owner = c
            break

    # Amount bucket (handle newlines already collapsed by _normalize_cell)
    amount_bucket = ""
    for c in cells[::-1]:
        if (
            c in AMOUNT_BUCKETS_STD
            or re.match(r"^\$\d[\d,]*\s*-\s*\$\d[\d,]*$", c)
            or re.match(r"^Over\s*\$\d[\d,]*$", c, re.I)
        ):
            amount_bucket = c
            break

    # Transaction type (raw)
    tx_type_raw = ""
    for c in cells:
        cl = c.lower()
        if "sale (partial)" in cl:
            tx_type_raw = "Sale (Partial)"
            break
        if "sale (full)" in cl:
            tx_type_raw = "Sale (Full)"
            break
        if "purchase" in cl:
            tx_type_raw = "Purchase"
            break
        if "exchange" in cl:
            tx_type_raw = "Exchange"
            break
        if cl == "sale":
            tx_type_raw = "Sale"
            break

    tx_type, sale_scope = infer_tx_type_and_scope(tx_type_raw)

    # Asset type
    asset_type = ""
    for c in cells:
        cl = c.lower()
        if cl in {"stock", "municipal security", "mutual fund", "etf", "bond", "crypto", "option"}:
            asset_type = c
            break

    # Ticker: ignore '--'
    ticker = ""
    for c in cells:
        if c in {"--", "-"}:
            continue
        t = normalize_ticker(c)
        if t and len(t) <= 10 and re.fullmatch(r"[A-Z0-9\.\-]+", t):
            if t not in {"SELF", "SPOUSE", "JOINT", "DEPENDENTCHILD"}:
                ticker = t
                break

    # Asset name: best-effort choose longest "semantic" cell (exclude obvious fields)
    blacklist = {
        str(row_number),
        owner,
        tx_type_raw,
        amount_bucket,
        asset_type,
        "",
        "--",
    }
    candidates = [
        c for c in cells
        if c not in blacklist
        and not re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", c)
        and not re.fullmatch(r"\d+", c)
    ]
    asset_name = norm_ws(max(candidates, key=len)) if candidates else ""

    amin, amax = parse_amount_bucket_to_minmax(amount_bucket)

    rec = {
        "record_type": "TRANSACTION",
        "source_kind": "PTR_TABLE_PDF",
        "row_number": row_number,
        "transaction_date": tx_date_iso,
        "owner": owner,
        "ticker": ticker,
        "asset_name": asset_name,
        "asset_type": asset_type,
        "transaction_type_raw": tx_type_raw,
        "transaction_type": tx_type,
        "sale_scope": sale_scope,
        "amount_bucket_raw": amount_bucket,
        "amount_min_usd": amin,
        "amount_max_usd": amax,
    }

    if not tx_date_iso:
        logger.debug("PTR row missing transaction_date row=%s cells=%s", row_number, cells)
    if not owner:
        logger.debug("PTR row missing owner row=%s cells=%s", row_number, cells)
    if not amount_bucket:
        logger.debug("PTR row missing amount_bucket row=%s cells=%s", row_number, cells)

    return rec


def parse_ptr_pdf_transactions(pdf_path: Path, logger: logging.Logger, errors: List[Dict[str, Any]]) -> pd.DataFrame:
    logger.info("Parsing PTR PDF: %s", pdf_path.name)
    records: List[Dict[str, Any]] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            official_name, filing_dt_iso = scan_header_metadata(pdf, logger)

            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables() or []
                except Exception as e:
                    errors.append({"stage": "ptr_extract_tables", "file": pdf_path.name, "page": page_idx, **exc_to_dict(e)})
                    logger.exception("PTR extract_tables failed file=%s page=%s", pdf_path.name, page_idx)
                    tables = []

                parsed_any = False
                for t_idx, table in enumerate(tables, start=1):
                    if not table or len(table) < 1:
                        continue

                    header = [_normalize_cell(x) for x in (table[0] or [])]
                    looks_like_header = any(("Transaction" in h) or ("Owner" in h) or ("Asset" in h) or (h == "#") for h in header)
                    data_rows = table[1:] if looks_like_header and len(table) > 1 else table

                    for r_idx, row in enumerate(data_rows, start=1):
                        try:
                            rec = parse_ptr_table_row(row, logger)
                            if rec:
                                rec.update({
                                    "source_file": pdf_path.name,
                                    "source_page": page_idx,
                                    "official_name": official_name,
                                    "filing_datetime": filing_dt_iso,
                                })
                                records.append(rec)
                                parsed_any = True
                        except Exception as e:
                            errors.append({
                                "stage": "ptr_parse_row",
                                "file": pdf_path.name,
                                "page": page_idx,
                                "table_index": t_idx,
                                "row_index": r_idx,
                                "row_repr": repr(row),
                                **exc_to_dict(e),
                            })
                            logger.exception("PTR row parse failed file=%s page=%s table=%s row=%s", pdf_path.name, page_idx, t_idx, r_idx)

                # Keep this warning: it is now a strong indicator of true non-transaction tables.
                if not parsed_any and tables:
                    logger.warning(
                        "No PTR rows parsed on file=%s page=%s (tables=%d).",
                        pdf_path.name, page_idx, len(tables)
                    )

            if not records:
                logger.error("PTR parsing produced 0 records for %s", pdf_path.name)

    except Exception as e:
        errors.append({"stage": "ptr_open_pdf", "file": pdf_path.name, **exc_to_dict(e)})
        logger.exception("PTR open/parse failed: %s", pdf_path.name)

    df = pd.DataFrame(records)
    logger.info("PTR parsed: %s rows=%d", pdf_path.name, len(df))
    return df


# -----------------------------
# 3) Scanned form checkbox extraction (optional)
# -----------------------------

@dataclass
class ScanTemplate:
    crop_x0: int = 300
    crop_x1: int = 2300
    crop_y0: int = 930
    crop_y1: int = 1640
    tx_lines: Optional[List[int]] = None
    amt_lines: Optional[List[int]] = None
    row_lines: Optional[List[int]] = None
    mark_threshold: float = 0.02


def _cluster_positions(idxs: np.ndarray, gap: int = 3) -> List[int]:
    if len(idxs) == 0:
        return []
    out: List[int] = []
    cur = [int(idxs[0])]
    for v in idxs[1:]:
        v = int(v)
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(int(np.mean(cur)))
            cur = [v]
    out.append(int(np.mean(cur)))
    return out


def _detect_lines(binary_crop: np.ndarray, axis: str) -> np.ndarray:
    if axis == "v":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, binary_crop.shape[0] // 30)))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, binary_crop.shape[1] // 20), 1))
    er = cv2.erode(binary_crop, kernel, iterations=1)
    di = cv2.dilate(er, kernel, iterations=2)
    return di


def auto_calibrate_template(img_path: Path, tmpl: ScanTemplate, logger: logging.Logger) -> ScanTemplate:
    if cv2 is None:
        return tmpl
    im = cv2.imread(str(img_path))
    if im is None:
        logger.warning("Scan read failed: %s", img_path.name)
        return tmpl
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bininv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10)
    crop = bininv[tmpl.crop_y0:tmpl.crop_y1, tmpl.crop_x0:tmpl.crop_x1]
    if crop.size == 0:
        logger.warning("Scan crop empty: %s", img_path.name)
        return tmpl

    v = _detect_lines(crop, "v")
    h = _detect_lines(crop, "h")

    colsum = v.sum(axis=0)
    rowsum = h.sum(axis=1)

    xs = np.where(colsum > 0.6 * colsum.max())[0] if colsum.max() > 0 else np.array([])
    ys = np.where(rowsum > 0.6 * rowsum.max())[0] if rowsum.max() > 0 else np.array([])

    x_positions = _cluster_positions(xs, gap=3)
    y_positions = _cluster_positions(ys, gap=3)

    w = crop.shape[1]
    right_candidates = sorted([p for p in x_positions if p >= int(0.48 * w)])
    mid_candidates = sorted([p for p in x_positions if int(0.20 * w) <= p <= int(0.50 * w)])

    if len(right_candidates) >= 12:
        best = None
        for i in range(0, len(right_candidates) - 11):
            win = right_candidates[i:i + 12]
            gaps = np.diff(win)
            if gaps.mean() <= 0:
                continue
            cvv = float(gaps.std() / gaps.mean())
            span = win[-1] - win[0]
            score = (cvv, span)
            if best is None or score < best[0]:
                best = (score, win)
        if best:
            tmpl.amt_lines = list(best[1])

    if len(mid_candidates) >= 4:
        best = None
        for i in range(0, len(mid_candidates) - 3):
            win = mid_candidates[i:i + 4]
            gaps = np.diff(win)
            if gaps.mean() <= 0:
                continue
            cvv = float(gaps.std() / gaps.mean())
            span = win[-1] - win[0]
            score = (cvv, span)
            if best is None or score < best[0]:
                best = (score, win)
        if best:
            tmpl.tx_lines = list(best[1])

    if len(y_positions) >= 2:
        tmpl.row_lines = list(sorted(y_positions))

    logger.info(
        "Scan calibrated: %s tx_lines=%s amt_lines=%s row_lines=%s",
        img_path.name,
        "ok" if tmpl.tx_lines and len(tmpl.tx_lines) == 4 else "missing",
        "ok" if tmpl.amt_lines and len(tmpl.amt_lines) == 12 else "missing",
        "ok" if tmpl.row_lines and len(tmpl.row_lines) >= 2 else "missing",
    )
    return tmpl


def _cell_density(bininv: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 6) -> float:
    roi = bininv[y + pad:y + h - pad, x + pad:x + w - pad]
    if roi.size == 0:
        return 0.0
    return float(roi.mean() / 255.0)


def parse_scanned_form_png(
    img_path: Path,
    logger: logging.Logger,
    errors: List[Dict[str, Any]],
    tmpl: Optional[ScanTemplate] = None,
) -> pd.DataFrame:
    if cv2 is None or pytesseract is None:
        logger.warning("Scan parsing skipped (opencv or pytesseract missing). file=%s", img_path.name)
        return pd.DataFrame()

    tmpl = tmpl or ScanTemplate()
    try:
        if tmpl.tx_lines is None or tmpl.amt_lines is None or tmpl.row_lines is None:
            tmpl = auto_calibrate_template(img_path, tmpl, logger)

        if tmpl.tx_lines is None or tmpl.amt_lines is None or tmpl.row_lines is None:
            logger.error("Scan template incomplete; cannot parse checkbox semantics. file=%s", img_path.name)
            return pd.DataFrame()

        im = cv2.imread(str(img_path))
        if im is None:
            logger.error("Scan image load failed: %s", img_path.name)
            return pd.DataFrame()

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        bininv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10)

        crop_bininv = bininv[tmpl.crop_y0:tmpl.crop_y1, tmpl.crop_x0:tmpl.crop_x1]
        crop_gray = gray[tmpl.crop_y0:tmpl.crop_y1, tmpl.crop_x0:tmpl.crop_x1]

        tx_lines = sorted(tmpl.tx_lines)[:4]
        amt_lines = sorted(tmpl.amt_lines)[:12]
        row_lines = sorted(tmpl.row_lines)

        tx_cells = [(tx_lines[i], tx_lines[i + 1]) for i in range(3)]
        amt_cells = [(amt_lines[i], amt_lines[i + 1]) for i in range(11)]
        bands = [(row_lines[i], row_lines[i + 1]) for i in range(len(row_lines) - 1)]
        bands = [(a, b) for (a, b) in bands if (b - a) >= 30]

        recs: List[Dict[str, Any]] = []
        for band_idx, (yt, yb) in enumerate(bands, start=1):
            try:
                left_x1 = tx_lines[0]
                left_img = crop_gray[yt:yb, 0:left_x1]
                ocr_text = pytesseract.image_to_string(left_img, config="--psm 6").strip()
                ocr_text = re.sub(r"\s+", " ", ocr_text)

                mrow = re.match(r"^\s*(\d+)", ocr_text)
                row_number = int(mrow.group(1)) if mrow else band_idx

                tx_date_iso = ""
                mdate = re.search(r"(\d{1,2}/\d{1,2}/\d{2})", ocr_text)
                if mdate:
                    try:
                        tx_date_iso = dt.datetime.strptime(mdate.group(1), "%m/%d/%y").date().isoformat()
                    except ValueError:
                        tx_date_iso = ""

                owner = ""
                mown = re.search(r"\((S|J|DC)\)", ocr_text)
                if mown:
                    owner = {"S": "Self", "J": "Joint", "DC": "Dependent Child"}.get(mown.group(1), "")

                tx_scores = [_cell_density(crop_bininv, xl, yt, xr - xl, yb - yt) for (xl, xr) in tx_cells]
                tx_type = "UNKNOWN"
                if any(s > tmpl.mark_threshold for s in tx_scores):
                    tx_type = ["BUY", "SELL", "EXCHANGE"][int(np.argmax(tx_scores))]

                amt_scores = [_cell_density(crop_bininv, xl, yt, xr - xl, yb - yt) for (xl, xr) in amt_cells]
                chosen = [i for i, s in enumerate(amt_scores) if s > tmpl.mark_threshold]

                bucket_raw = ""
                status = "NONE"
                ordinal = None
                if len(chosen) == 1:
                    status = "SINGLE"
                    ordinal = chosen[0] + 1
                    bucket_raw = AMOUNT_BUCKETS_STD[chosen[0]] if chosen[0] < len(AMOUNT_BUCKETS_STD) else f"BUCKET_{chosen[0] + 1}"
                elif len(chosen) > 1:
                    status = "MULTI"
                    bucket_raw = "; ".join([
                        AMOUNT_BUCKETS_STD[i] if i < len(AMOUNT_BUCKETS_STD) else f"BUCKET_{i + 1}"
                        for i in chosen
                    ])

                amin, amax = (None, None)
                if status == "SINGLE" and bucket_raw:
                    amin, amax = parse_amount_bucket_to_minmax(bucket_raw)

                recs.append({
                    "record_type": "TRANSACTION",
                    "source_kind": "PTR_SCAN_FORM",
                    "source_file": img_path.name,
                    "source_page": img_path.stem,
                    "official_name": "",
                    "filing_datetime": "",
                    "row_number": row_number,
                    "transaction_date": tx_date_iso,
                    "owner": owner,
                    "ticker": "",
                    "asset_name": ocr_text,
                    "asset_type": "",
                    "transaction_type_raw": tx_type,
                    "transaction_type": tx_type,
                    "sale_scope": "",
                    "amount_bucket_raw": bucket_raw,
                    "amount_bucket_status": status,
                    "amount_bucket_ordinal": ordinal,
                    "amount_min_usd": amin,
                    "amount_max_usd": amax,
                    "tx_scores_json": json.dumps(tx_scores),
                    "amt_scores_json": json.dumps(amt_scores),
                })
            except Exception as e:
                errors.append({"stage": "scan_parse_rowband", "file": img_path.name, "band_index": band_idx, **exc_to_dict(e)})
                logger.exception("Scan rowband parse failed file=%s band=%s", img_path.name, band_idx)

        df = pd.DataFrame(recs)
        logger.info("Scan parsed: %s rows=%d", img_path.name, len(df))
        return df

    except Exception as e:
        errors.append({"stage": "scan_parse", "file": img_path.name, **exc_to_dict(e)})
        logger.exception("Scan parse failed: %s", img_path.name)
        return pd.DataFrame()


# -----------------------------
# Annual report tables preserved (generic)
# -----------------------------

def extract_annual_report_tables(pdf_path: Path, logger: logging.Logger, errors: List[Dict[str, Any]]) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    logger.info("Extracting annual tables: %s", pdf_path.name)
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables() or []
                except Exception as e:
                    errors.append({"stage": "annual_extract_tables", "file": pdf_path.name, "page": page_idx, **exc_to_dict(e)})
                    logger.exception("Annual extract_tables failed file=%s page=%s", pdf_path.name, page_idx)
                    tables = []

                for t_idx, table in enumerate(tables, start=1):
                    try:
                        for r_idx, row in enumerate(table, start=1):
                            out.append({
                                "record_type": "ANNUAL_TABLE_ROW",
                                "source_file": pdf_path.name,
                                "source_page": page_idx,
                                "table_index": t_idx,
                                "row_index": r_idx,
                                "row_json": json.dumps(row, ensure_ascii=False),
                            })
                    except Exception as e:
                        errors.append({"stage": "annual_parse_table", "file": pdf_path.name, "page": page_idx, "table_index": t_idx, **exc_to_dict(e)})
                        logger.exception("Annual parse_table failed file=%s page=%s table=%s", pdf_path.name, page_idx, t_idx)

    except Exception as e:
        errors.append({"stage": "annual_open_pdf", "file": pdf_path.name, **exc_to_dict(e)})
        logger.exception("Annual open/parse failed: %s", pdf_path.name)

    df = pd.DataFrame(out)
    logger.info("Annual tables extracted: %s rows=%d", pdf_path.name, len(df))
    return df


# -----------------------------
# Orchestrator
# -----------------------------

def run_extract(project_root: Path, verbose: bool) -> None:
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    outputs_dir = project_root / "outputs"
    logs_dir = project_root / "logs"

    for d in (raw_dir, processed_dir, outputs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_dir, run_id, verbose)

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    logger.info("project_root=%s", str(project_root))
    logger.info("raw_dir=%s", str(raw_dir))

    pdf_files = sorted(raw_dir.glob(OTHER_PDF_GLOB))
    png_files = sorted(raw_dir.glob(SCAN_PNG_GLOB))
    ptr_pdfs = sorted(raw_dir.glob(PTR_FILE_GLOB))
    annual_pdfs = sorted(raw_dir.glob(ANNUAL_FILE_GLOB))

    logger.info(
        "inputs: pdf_files=%d ptr_pdfs=%d annual_pdfs=%d png_files=%d",
        len(pdf_files), len(ptr_pdfs), len(annual_pdfs), len(png_files)
    )

    # 1) Preserve pages for all PDFs
    page_frames = []
    for p in pdf_files:
        try:
            page_frames.append(extract_pdf_pages(p, logger, errors))
        except Exception as e:
            errors.append({"stage": "extract_pdf_pages", "file": p.name, **exc_to_dict(e)})
            logger.exception("extract_pdf_pages failed file=%s", p.name)

    pages_df = pd.concat(page_frames, ignore_index=True) if page_frames else pd.DataFrame()
    pdf_pages_path = processed_dir / "pdf_pages.csv"
    if not pages_df.empty:
        try:
            pages_df.to_csv(pdf_pages_path, index=False)
            logger.info("Wrote: %s rows=%d", pdf_pages_path, len(pages_df))
        except Exception as e:
            errors.append({"stage": "write_pdf_pages_csv", "file": str(pdf_pages_path), **exc_to_dict(e)})
            logger.exception("Write failed: %s", str(pdf_pages_path))
    else:
        logger.warning("No pdf_pages extracted.")

    # 2) PTR transactions
    ptr_frames = []
    for p in ptr_pdfs:
        try:
            ptr_frames.append(parse_ptr_pdf_transactions(p, logger, errors))
        except Exception as e:
            errors.append({"stage": "parse_ptr_pdf_transactions", "file": p.name, **exc_to_dict(e)})
            logger.exception("parse_ptr_pdf_transactions failed file=%s", p.name)

    ptr_df = pd.concat(ptr_frames, ignore_index=True) if ptr_frames else pd.DataFrame()
    if ptr_df.empty:
        logger.warning("No PTR transactions extracted from table-native PDFs.")
    else:
        miss_name = float(ptr_df["official_name"].apply(_is_missing_str).mean()) if "official_name" in ptr_df.columns else 1.0
        miss_filed = float(ptr_df["filing_datetime"].apply(_is_missing_str).mean()) if "filing_datetime" in ptr_df.columns else 1.0
        miss_amt = float(ptr_df["amount_bucket_raw"].apply(_is_missing_str).mean()) if "amount_bucket_raw" in ptr_df.columns else 1.0

        if miss_name > 0.0:
            warnings.append({"kind": "metadata_missing", "field": "official_name", "fraction": miss_name})
            logger.warning("PTR metadata coverage: official_name missing fraction=%.3f", miss_name)
        if miss_filed > 0.0:
            warnings.append({"kind": "metadata_missing", "field": "filing_datetime", "fraction": miss_filed})
            logger.warning("PTR metadata coverage: filing_datetime missing fraction=%.3f", miss_filed)
        if miss_amt > 0.0:
            warnings.append({"kind": "field_missing", "field": "amount_bucket_raw", "fraction": miss_amt})
            logger.warning("PTR field coverage: amount_bucket_raw missing fraction=%.3f", miss_amt)

    # 3) Scanned form transactions (optional)
    scan_frames = []
    if png_files:
        if cv2 is None or pytesseract is None:
            logger.warning("PNG scan inputs present, but opencv/pytesseract unavailable. Skipping scan parsing.")
        for img in png_files:
            try:
                scan_frames.append(parse_scanned_form_png(img, logger, errors))
            except Exception as e:
                errors.append({"stage": "parse_scanned_form_png", "file": img.name, **exc_to_dict(e)})
                logger.exception("parse_scanned_form_png failed file=%s", img.name)

    scan_df = pd.concat(scan_frames, ignore_index=True) if scan_frames else pd.DataFrame()

    # 4) Annual tables
    annual_frames = []
    for p in annual_pdfs:
        try:
            annual_frames.append(extract_annual_report_tables(p, logger, errors))
        except Exception as e:
            errors.append({"stage": "extract_annual_report_tables", "file": p.name, **exc_to_dict(e)})
            logger.exception("extract_annual_report_tables failed file=%s", p.name)

    annual_df = pd.concat(annual_frames, ignore_index=True) if annual_frames else pd.DataFrame()
    annual_path = processed_dir / "annual_table_rows.csv"
    if not annual_df.empty:
        try:
            annual_df.to_csv(annual_path, index=False)
            logger.info("Wrote: %s rows=%d", annual_path, len(annual_df))
        except Exception as e:
            errors.append({"stage": "write_annual_table_rows_csv", "file": str(annual_path), **exc_to_dict(e)})
            logger.exception("Write failed: %s", str(annual_path))
    else:
        logger.warning("No annual tables extracted.")

    # 5) Unified transactions
    tx_frames = [df for df in (ptr_df, scan_df) if df is not None and not df.empty]
    tx_out_path = outputs_dir / "ppf_transactions_unified.csv"
    if tx_frames:
        tx = pd.concat(tx_frames, ignore_index=True)
        try:
            tx.to_csv(tx_out_path, index=False)
            logger.info("Wrote: %s rows=%d", tx_out_path, len(tx))
        except Exception as e:
            errors.append({"stage": "write_transactions_csv", "file": str(tx_out_path), **exc_to_dict(e)})
            logger.exception("Write failed: %s", str(tx_out_path))
    else:
        logger.warning("No unified transactions produced.")

    # 6) Superset output
    superset_frames = [df for df in (pages_df, ptr_df, scan_df, annual_df) if df is not None and not df.empty]
    superset_out_path = outputs_dir / "ppf_superset_all_records.csv"
    if superset_frames:
        superset = pd.concat(superset_frames, ignore_index=True)
        try:
            superset.to_csv(superset_out_path, index=False)
            logger.info("Wrote: %s rows=%d", superset_out_path, len(superset))
        except Exception as e:
            errors.append({"stage": "write_superset_csv", "file": str(superset_out_path), **exc_to_dict(e)})
            logger.exception("Write failed: %s", str(superset_out_path))
    else:
        logger.warning("No superset produced.")

    # 7) Manifest
    manifest = {
        "run_id": run_id,
        "run_utc": now_utc_iso(),
        "project_root": str(project_root),
        "raw_dir": str(raw_dir),
        "outputs_dir": str(outputs_dir),
        "counts": {
            "pdf_pages_rows": int(len(pages_df)) if pages_df is not None else 0,
            "ptr_transactions_rows": int(len(ptr_df)) if ptr_df is not None else 0,
            "scan_transactions_rows": int(len(scan_df)) if scan_df is not None else 0,
            "annual_table_rows": int(len(annual_df)) if annual_df is not None else 0,
        },
        "inputs": {
            "pdf_files": [p.name for p in pdf_files],
            "ptr_pdfs": [p.name for p in ptr_pdfs],
            "annual_pdfs": [p.name for p in annual_pdfs],
            "scan_pngs": [p.name for p in png_files],
        },
        "outputs": {
            "pdf_pages_csv": str(pdf_pages_path),
            "annual_table_rows_csv": str(annual_path),
            "transactions_csv": str(tx_out_path),
            "superset_csv": str(superset_out_path),
            "log_file": str(logs_dir / "ppf_extract.log"),
        },
        "optional_deps": {"opencv": cv2 is not None, "pytesseract": pytesseract is not None},
        "warnings": warnings,
        "errors": errors,
    }

    manifest_path = outputs_dir / "extraction_manifest.json"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Wrote manifest: %s", str(manifest_path))
    except Exception as e:
        logger.exception("Failed writing manifest: %s err=%s", str(manifest_path), str(e))
        print(json.dumps(manifest, indent=2))

    logger.info("=== PPF Extractor end ===")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".", help="Path to PROJECT_ROOT containing data/raw, outputs, etc.")
    ap.add_argument("--verbose", action="store_true", help="Enable debug console logging.")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    run_extract(project_root, verbose=args.verbose)


if __name__ == "__main__":
    main()
