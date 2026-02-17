#!/usr/bin/env python3
"""
PPF Unifier (Disclosures-Only)

Goal:
- Parse table-native Senate eFD PTR PDFs into a canonical transaction ledger.
- Parse scanned/paper PTR pages with checkbox/X-mark semantics into the same ledger (template-based).
- Emit:
  - outputs/ppf_transactions_unified.csv
  - outputs/ppf_transactions_unified.parquet (if pyarrow available)
  - outputs/debug_overlay_*.png for scanned forms (optional)

Design constraints:
- Availability anchored on filing_datetime extracted from PDF text.
- Amount ranges preserved as bucket text + min/max floats (ordinal-friendly).
- All parsing is deterministic and auditable (row-level provenance included).
"""

from __future__ import annotations

import re
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pdfplumber

# Optional (only needed for scanned form parsing + debug overlays)
import cv2
import pytesseract


# ---------------------------
# Canonical bucket definitions (paper form column order)
# ---------------------------

AMOUNT_BUCKETS = [
    "$1,001 - $15,000",
    "$15,001 - $50,000",
    "$50,001 - $100,000",
    "$100,001 - $250,000",
    "$250,001 - $500,000",
    "$500,001 - $1,000,000",
    "Over $1,000,000",
    "$1,000,001 - $5,000,000",
    "$5,000,001 - $25,000,000",
    "$25,000,001 - $50,000,000",
    "Over $50,000,000",
]


# ---------------------------
# Helpers
# ---------------------------

def parse_amount_bucket_to_minmax(bucket: str) -> Tuple[Optional[float], Optional[float]]:
    s = (bucket or "").strip().replace(" ", "")
    m = re.match(r"^\$([\d,]+)-\$([\d,]+)$", s)
    if m:
        return float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
    m = re.match(r"^Over\$([\d,]+)$", s, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", "")), None
    return None, None


def parse_filed_datetime(text: str) -> Optional[str]:
    """
    eFD prints: 'Filed 02/04/2026 @ 1:40 PM'
    """
    m = re.search(r"Filed\s+(\d{2}/\d{2}/\d{4})\s+@\s+(\d{1,2}:\d{2}\s+[AP]M)", text)
    if not m:
        return None
    try:
        dtm = dt.datetime.strptime(m.group(1) + " " + m.group(2), "%m/%d/%Y %I:%M %p")
        return dtm.isoformat()
    except ValueError:
        return None


def collect_multiline_rows(lines: List[str]) -> List[str]:
    """
    eFD text extraction often wraps rows. We reconstruct by:
      - Start of new row: 'NN 01/20/2026 ...'
      - Continuation lines appended until next start.
    """
    start_re = re.compile(r"^\d+\s+\d{2}/\d{2}/\d{4}\s+")
    rows: List[str] = []
    cur: Optional[str] = None
    for l in lines:
        l = (l or "").strip()
        if not l:
            continue
        if start_re.match(l):
            if cur:
                rows.append(cur.strip())
            cur = l
        else:
            if cur:
                cur += " " + l
    if cur:
        rows.append(cur.strip())
    return rows


KNOWN_ASSET_TYPES = [
    "Municipal Security",
    "Municipal",
    "Security",
    "Stock",
    "ETF",
    "Mutual Fund",
    "Fund",
    "Bond",
    "Crypto",
]


def _extract_bucket_flexible(row: str) -> Optional[str]:
    """
    Table-native PTR rows sometimes include stray '--' that breaks a simple range regex.
    Robust approach:
      - If 'Over $X' exists: use that.
      - Else take first two $ amounts in the row and construct '$A - $B'.
    """
    over = re.search(r"Over\s+\$[\d,]+", row, re.IGNORECASE)
    if over:
        return over.group(0).replace("  ", " ").strip()

    amounts = re.findall(r"\$[\d,]+", row)
    if len(amounts) >= 2:
        return f"{amounts[0]} - {amounts[1]}"
    if len(amounts) == 1:
        # degraded case: keep what we saw
        return amounts[0]
    return None


def parse_ptr_row(row: str) -> Optional[Dict[str, object]]:
    """
    Parses a reconstructed row string into canonical fields.

    Strategy (robust to messy text):
    - bucket: from dollar amounts (first two) or 'Over $...'
    - tokens prior to first $ include row_number, tx_date, owner, ticker, and mid text.
    - mid text contains asset name + asset type + tx keyword (Purchase/Sale/Exchange)
    """
    bucket = _extract_bucket_flexible(row)
    if not bucket:
        return None

    m_amt = re.search(r"\$[\d,]+", row)
    if not m_amt:
        return None

    pre = row[:m_amt.start()].strip()
    toks = pre.split()
    if len(toks) < 5:
        return None

    rownum = int(toks[0])
    txdate = toks[1]
    owner = toks[2]
    ticker = toks[3]
    mid = " ".join(toks[4:])

    m_type = re.search(r"\b(Purchase|Sale|Exchange)\b", mid, re.IGNORECASE)
    tx_kw = m_type.group(1).title() if m_type else "UNKNOWN"

    # Sale scope (Partial/Full) may appear later in the row; search globally
    scope = ""
    mscope = re.search(r"\((Partial|Full)\)", row, re.IGNORECASE)
    if mscope and tx_kw == "Sale":
        scope = mscope.group(1).upper()

    tx_type_raw = f"{tx_kw} ({scope.title()})" if scope else tx_kw
    tx_type_norm = {"Purchase": "BUY", "Sale": "SELL", "Exchange": "EXCHANGE"}.get(tx_kw, "UNKNOWN")

    before = mid[:m_type.start()].strip() if m_type else mid

    asset_type = "UNKNOWN"
    asset_name = before

    # identify asset type as the last known phrase before tx type
    for at in sorted(KNOWN_ASSET_TYPES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(at)}\b$", before, re.IGNORECASE):
            asset_type = at
            asset_name = before[: -len(at)].strip()
            break

    # parse date
    try:
        tx_date_iso = dt.datetime.strptime(txdate, "%m/%d/%Y").date().isoformat()
    except ValueError:
        tx_date_iso = ""

    amin, amax = parse_amount_bucket_to_minmax(bucket.replace(" ", ""))

    return {
        "row_number": rownum,
        "transaction_date": tx_date_iso,
        "owner": owner,
        "ticker": "" if ticker == "--" else ticker,
        "asset_name": asset_name,
        "asset_type": asset_type,
        "transaction_type_raw": tx_type_raw,
        "transaction_type": tx_type_norm,
        "sale_scope": scope,
        "amount_bucket_raw": bucket,
        "amount_min_usd": amin,
        "amount_max_usd": amax,
    }


def parse_ptr_pdf(pdf_path: Path) -> pd.DataFrame:
    recs: List[Dict[str, object]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            filed_dt = parse_filed_datetime(text)
            off = None
            m = re.search(r"The Honorable\s+(.+?)\s+\(", text)
            if m:
                off = m.group(1).strip()

            rows = collect_multiline_rows(text.splitlines())
            for row in rows:
                if not re.match(r"^\d+\s+\d{2}/\d{2}/\d{4}\s+", row.strip()):
                    continue
                parsed = parse_ptr_row(row)
                if not parsed:
                    continue
                recs.append(
                    {
                        "record_type": "TRANSACTION",
                        "source_kind": "PTR_TABLE_PDF",
                        "source_pdf": pdf_path.name,
                        "source_page": page.page_number,
                        "official_name": off or "",
                        "filing_datetime": filed_dt or "",
                        **parsed,
                    }
                )
    return pd.DataFrame(recs)


# ---------------------------
# Scanned form parsing (checkbox/X-mark semantics)
# ---------------------------

@dataclass
class ScanTemplate:
    """
    Template for one scanned-form page family.

    The default values below are intentionally explicit and should be treated as:
      - calibrated constants per form family (or derived by auto-calibration).

    The provided helper `auto_detect_columns_rows` can be used to derive
    x-lines and y-lines; then you freeze them into a template for repeatable runs.
    """
    crop_x0: int = 300
    crop_x1: int = 2300
    crop_y0: int = 1100  # should be below the EXAMPLE block
    crop_y1: int = 1625

    # Vertical grid lines in crop-coordinates (x positions)
    # Expected:
    # - tx lines: 4 lines -> 3 columns (Purchase/Sale/Exchange)
    # - amt lines: 12 lines -> 11 columns (buckets)
    tx_lines: Optional[List[int]] = None
    amt_lines: Optional[List[int]] = None

    # Horizontal row separator lines in crop-coordinates
    row_lines: Optional[List[int]] = None

    mark_threshold: float = 0.12  # density threshold inside cell (tune if needed)


def _detect_grid_lines(binary: np.ndarray, axis: str, scale: int) -> np.ndarray:
    # binary expected inverted: lines/ink are white (255)
    if axis == "v":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, binary.shape[0] // scale)))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, binary.shape[1] // scale), 1))
    er = cv2.erode(binary, kernel, iterations=1)
    di = cv2.dilate(er, kernel, iterations=2)
    return di


def _cluster_positions(idxs: np.ndarray, gap: int = 2) -> List[int]:
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


def auto_detect_columns_rows(image_path: Path, tmpl: ScanTemplate) -> ScanTemplate:
    """
    Auto-detect grid lines in the crop window. Use once to calibrate a new scan family.
    """
    im = cv2.imread(str(image_path))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bininv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )

    crop = bininv[tmpl.crop_y0 : tmpl.crop_y1, tmpl.crop_x0 : tmpl.crop_x1]

    vlines = _detect_grid_lines(crop, "v", scale=40)
    hlines = _detect_grid_lines(crop, "h", scale=25)

    colsum = vlines.sum(axis=0)
    xs = np.where(colsum > 0.6 * colsum.max())[0]
    x_positions = _cluster_positions(xs)

    rowsum = hlines.sum(axis=1)
    ys = np.where(rowsum > 0.6 * rowsum.max())[0]
    y_positions = _cluster_positions(ys)

    tx_lines = sorted([p for p in x_positions if 650 <= p <= 900])
    amt_lines = sorted([p for p in x_positions if 1000 <= p <= 1700])

    tmpl.tx_lines = tx_lines if len(tx_lines) >= 4 else tmpl.tx_lines
    tmpl.amt_lines = amt_lines if len(amt_lines) >= 12 else tmpl.amt_lines
    tmpl.row_lines = y_positions if len(y_positions) >= 2 else tmpl.row_lines

    return tmpl


def _cell_density(bininv: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 6) -> float:
    roi = bininv[y + pad : y + h - pad, x + pad : x + w - pad]
    if roi.size == 0:
        return 0.0
    return float(roi.mean() / 255.0)


def parse_scanned_form_page(
    image_path: Path,
    source_pdf_name: str,
    filing_datetime: str = "",
    official_name: str = "",
    tmpl: Optional[ScanTemplate] = None,
) -> pd.DataFrame:
    """
    Extracts per-row checkbox semantics (Purchase/Sale/Exchange + 11 amount buckets)
    from a scanned form page *after* template calibration.

    Output is canonical TRANSACTION rows, but:
      - ticker may be missing
      - asset_name comes from OCR of left band (auditable, not perfect)
      - transaction_type and amount_bucket_raw come from mark detection in cells
    """
    tmpl = tmpl or ScanTemplate()

    # If lines are not provided, attempt auto-detection within the crop
    if tmpl.tx_lines is None or tmpl.amt_lines is None or tmpl.row_lines is None:
        tmpl = auto_detect_columns_rows(image_path, tmpl)

    if tmpl.tx_lines is None or tmpl.amt_lines is None or tmpl.row_lines is None:
        raise RuntimeError("Template calibration failed: missing tx_lines/amt_lines/row_lines")

    im = cv2.imread(str(image_path))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bininv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )

    # Build row bands from row separator lines
    row_lines = sorted(tmpl.row_lines)
    bands = [(row_lines[i], row_lines[i + 1]) for i in range(len(row_lines) - 1)]

    # Heuristic: ignore very small bands
    bands = [(a, b) for (a, b) in bands if (b - a) >= 30]

    # tx type columns: 3 cells from 4 grid lines
    tx_lines = sorted(tmpl.tx_lines)[:4]
    tx_cells = [(tx_lines[i], tx_lines[i + 1]) for i in range(3)]

    # amount columns: 11 cells from 12 grid lines
    amt_lines = sorted(tmpl.amt_lines)[:12]
    amt_cells = [(amt_lines[i], amt_lines[i + 1]) for i in range(11)]

    records: List[Dict[str, object]] = []

    for band_idx, (yt, yb) in enumerate(bands, start=1):
        # OCR left band (row number / asset / date / owner codes)
        left_x1 = tx_lines[0]
        left_img = gray[
            tmpl.crop_y0 + yt : tmpl.crop_y0 + yb,
            tmpl.crop_x0 : tmpl.crop_x0 + left_x1,
        ]

        ocr_text = pytesseract.image_to_string(left_img, config="--psm 6").strip()
        ocr_text = re.sub(r"\s+", " ", ocr_text)

        # row number
        mrow = re.match(r"^\s*(\d+)", ocr_text)
        row_number = int(mrow.group(1)) if mrow else band_idx

        # transaction date (MM/DD/YY)
        tx_date_iso = ""
        mdate = re.search(r"(\d{1,2}/\d{1,2}/\d{2})", ocr_text)
        if mdate:
            try:
                tx_date_iso = dt.datetime.strptime(mdate.group(1), "%m/%d/%y").date().isoformat()
            except ValueError:
                pass

        # owner code pattern (S/J/DC) in parentheses
        mown = re.search(r"\((S|J|DC)\)", ocr_text)
        owner = {"S": "Self", "J": "Joint", "DC": "Dependent Child"}.get(mown.group(1), "") if mown else ""

        # Transaction type marks
        tx_scores = []
        for (xl, xr) in tx_cells:
            d = _cell_density(
                bininv,
                tmpl.crop_x0 + xl,
                tmpl.crop_y0 + yt,
                xr - xl,
                yb - yt,
            )
            tx_scores.append(d)

        tx_checked = [s > tmpl.mark_threshold for s in tx_scores]
        tx_type = "UNKNOWN"
        if any(tx_checked):
            tx_type = ["BUY", "SELL", "EXCHANGE"][int(np.argmax(tx_scores))]

        # Amount bucket marks
        amt_scores = []
        for (xl, xr) in amt_cells:
            d = _cell_density(
                bininv,
                tmpl.crop_x0 + xl,
                tmpl.crop_y0 + yt,
                xr - xl,
                yb - yt,
            )
            amt_scores.append(d)

        amt_checked = [s > tmpl.mark_threshold for s in amt_scores]
        chosen = [i for i, b in enumerate(amt_checked) if b]

        bucket_raw = ""
        status = "NONE"
        ordinal = None
        if len(chosen) == 1:
            status = "SINGLE"
            ordinal = chosen[0] + 1
            bucket_raw = AMOUNT_BUCKETS[chosen[0]]
        elif len(chosen) > 1:
            status = "MULTI"

        amin, amax = parse_amount_bucket_to_minmax(bucket_raw.replace(" ", ""))

        records.append(
            {
                "record_type": "TRANSACTION",
                "source_kind": "PTR_SCAN_FORM",
                "source_pdf": source_pdf_name,
                "source_page": "",  # set by caller if known
                "official_name": official_name,
                "filing_datetime": filing_datetime,
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
                "amount_min_usd": amin,
                "amount_max_usd": amax,
                "amount_bucket_status": status,
                "amount_bucket_ordinal": ordinal,
                "tx_scores": tx_scores,
                "amt_scores": amt_scores,
            }
        )

    return pd.DataFrame(records)


# ---------------------------
# Unify orchestrator
# ---------------------------

def unify(inputs_dir: Path, outputs_dir: Path) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Table-native PTR PDFs
    ptr_pdfs = sorted(inputs_dir.glob("eFD_ Print Periodic Transaction Report*.pdf"))
    ptr_frames = [parse_ptr_pdf(p) for p in ptr_pdfs]
    ptr_df = pd.concat(ptr_frames, ignore_index=True) if ptr_frames else pd.DataFrame()

    # 2) Scanned form pages (optional)
    scan_frames: List[pd.DataFrame] = []
    for img in sorted(inputs_dir.glob("ppf_tmp_report5_p*.png")):
        # NOTE: caller can pass filing_datetime/official_name once you parse cover page
        scan_frames.append(
            parse_scanned_form_page(
                img,
                source_pdf_name="eFD_ Print Report5.pdf",
                tmpl=ScanTemplate(),
            )
        )
    scan_df = pd.concat(scan_frames, ignore_index=True) if scan_frames else pd.DataFrame()

    # 3) Unified
    unified = pd.concat([df for df in [ptr_df, scan_df] if not df.empty], ignore_index=True)

    # Stable column order (superset-friendly)
    preferred_cols = [
        "record_type",
        "source_kind",
        "source_pdf",
        "source_page",
        "official_name",
        "filing_datetime",
        "row_number",
        "transaction_date",
        "owner",
        "ticker",
        "asset_name",
        "asset_type",
        "transaction_type_raw",
        "transaction_type",
        "sale_scope",
        "amount_bucket_raw",
        "amount_min_usd",
        "amount_max_usd",
        "amount_bucket_status",
        "amount_bucket_ordinal",
        "tx_scores",
        "amt_scores",
    ]
    for c in preferred_cols:
        if c not in unified.columns:
            unified[c] = ""

    unified = unified[preferred_cols]

    # Write outputs
    csv_path = outputs_dir / "ppf_transactions_unified.csv"
    unified.to_csv(csv_path, index=False)

    # Parquet if available
    try:
        import pyarrow  # noqa: F401
        pq_path = outputs_dir / "ppf_transactions_unified.parquet"
        unified.to_parquet(pq_path, index=False)
    except Exception:
        pass

    return unified


if __name__ == "__main__":
    root = Path(".").resolve()
    inputs = root
    outputs = root / "outputs"

    df = unify(inputs, outputs)
    print(f"[PPF] Unified transactions: {len(df):,}")
    print(df.head(10).to_string(index=False))
