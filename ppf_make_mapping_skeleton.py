#!/usr/bin/env python3
"""
PPF Next-Step Pipeline (hardened): Enrich → Mapping → Impulses → PPF Scores

Fixes:
  - Normalizes mapping sector/subsector to prevent NaN/"nan" masquerading as mapped.
  - Adds explicit mapping quality checks so this failure mode is impossible to miss.
  - Keeps deterministic, disclosures-only posture (filing_datetime is availability timestamp).

INPUTS:
  - outputs/ppf_transactions_unified.csv

OUTPUTS:
  - data/processed/transactions_enriched.csv
  - data/reference/asset_to_sector_mapping.csv            (created if missing; user populates)
  - data/reference/sector_to_etf.csv                      (created if missing; optional later)
  - outputs/ppf_asset_universe.csv                        (unique assets with normalization)
  - outputs/ppf_assets_unmapped.csv                       (assets missing mapping or missing sector labels)
  - outputs/ppf_impulses_sector_daily.csv                 (availability-anchored impulses)
  - outputs/ppf_scores_sector_daily.csv                   (PPF timeseries)
  - outputs/ppf_scores_sector_snapshot_latest.csv         (latest ranks)
  - outputs/ppf_pipeline_manifest.json                    (run metadata + counts + errors)

Notes:
  - Mapping join key is asset_id (ticker_norm if present else issuer_norm).
  - UNMAPPED excluded from scoring by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Logging + error utilities
# -----------------------------

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def exc_to_dict(e: BaseException) -> Dict[str, Any]:
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }


def setup_logger(project_root: Path, verbose: bool) -> logging.Logger:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ppf_pipeline.log"

    logger = logging.getLogger("ppf_pipeline")
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

    logger.info("=== PPF Pipeline start ===")
    logger.info("project_root=%s", str(project_root))
    logger.info("log_path=%s", str(log_path))
    return logger


class StageTimer:
    def __init__(self, logger: logging.Logger, name: str) -> None:
        self.logger = logger
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info("Stage start: %s", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt_s = time.time() - self.t0
        self.logger.info("Stage end: %s elapsed=%.3fs", self.name, dt_s)
        return False


# -----------------------------
# Normalization
# -----------------------------

CORP_SUFFIXES = [
    r"\bINCORPORATED\b", r"\bINC\b\.?",
    r"\bCORPORATION\b", r"\bCORP\b\.?",
    r"\bCOMPANY\b", r"\bCO\b\.?",
    r"\bLIMITED\b", r"\bLTD\b\.?",
    r"\bPLC\b", r"\bLLC\b", r"\bLP\b", r"\bL\.P\.\b",
    r"\bHOLDINGS\b", r"\bHLDGS\b",
    r"\bGROUP\b",
    r"\bCOMMON STOCK\b",
]


def normalize_ticker(x: str) -> str:
    s = (x or "").strip().upper()
    if s in {"--", "-", ""}:
        return ""
    s = re.sub(r"^(NYSE|NASDAQ|AMEX|OTC)\s*:\s*", "", s)
    s = s.replace("$", "")
    s = re.sub(r"[^A-Z0-9\.\-]", "", s)
    if s in {"SELF", "SPOUSE", "JOINT", "DEPENDENTCHILD"}:
        return ""
    return s


def normalize_issuer_name(asset_name: str) -> str:
    s = (asset_name or "").upper()
    s = s.replace("’", "'")
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^A-Z0-9'\s&\-]", " ", s)
    for pat in CORP_SUFFIXES:
        s = re.sub(pat, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\bTHE\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def choose_asset_id(ticker_norm: str, issuer_norm: str) -> str:
    return ticker_norm if ticker_norm else issuer_norm


def clean_text_series(s: pd.Series) -> pd.Series:
    """
    Convert NaN → "", strip, and treat literal "nan"/"none" as blank.
    """
    x = s.fillna("").astype(str).map(lambda v: v.strip())
    x = x.replace({"nan": "", "NaN": "", "None": "", "NONE": ""})
    return x


# -----------------------------
# Data contracts
# -----------------------------

REQUIRED_TX_COLS = [
    "source_file", "source_page", "official_name", "filing_datetime",
    "transaction_date", "owner", "transaction_type", "amount_bucket_raw", "asset_name",
]


def assert_tx_contract(df: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
    missing_fracs: Dict[str, float] = {}
    ok = True

    for c in REQUIRED_TX_COLS:
        if c not in df.columns:
            missing_fracs[c] = 1.0
            ok = False
            continue
        s = df[c]
        frac = (s.isna() | (s.astype(str).str.strip() == "") | (s.astype(str).str.lower() == "nan")).mean()
        missing_fracs[c] = float(frac)
        if frac > 0:
            ok = False

    if "source_file" in df.columns and "filing_datetime" in df.columns:
        nunq = df.groupby("source_file")["filing_datetime"].nunique(dropna=True)
        if (nunq > 1).any():
            ok = False
            missing_fracs["filing_datetime_nunique_violation"] = float((nunq > 1).mean())

    return ok, missing_fracs


# -----------------------------
# Reference tables
# -----------------------------

MAPPING_COLUMNS: List[str] = [
    "asset_id",
    "ticker_norm",
    "issuer_norm",
    "sector",
    "subsector",
    "mapping_source",
    "confidence",
    "last_updated_utc",
]

SECTOR_TO_ETF_COLUMNS: List[str] = [
    "sector",
    "etf_ticker",
    "etf_name",
    "notes",
    "last_updated_utc",
]


def ensure_reference_files(project_root: Path, logger: logging.Logger) -> Dict[str, str]:
    ref_dir = project_root / "data" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = ref_dir / "asset_to_sector_mapping.csv"
    if not mapping_path.exists():
        pd.DataFrame(columns=MAPPING_COLUMNS).to_csv(mapping_path, index=False)
        logger.info("Created mapping skeleton: %s", str(mapping_path))

    sector_etf_path = ref_dir / "sector_to_etf.csv"
    if not sector_etf_path.exists():
        pd.DataFrame(columns=SECTOR_TO_ETF_COLUMNS).to_csv(sector_etf_path, index=False)
        logger.info("Created sector ETF skeleton: %s", str(sector_etf_path))

    return {
        "asset_to_sector_mapping": str(mapping_path),
        "sector_to_etf": str(sector_etf_path),
    }


# -----------------------------
# Step A: Enrich transactions
# -----------------------------

def enrich_transactions(project_root: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    in_path = project_root / "outputs" / "ppf_transactions_unified.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing required input: {in_path}")

    tx = pd.read_csv(in_path)

    ok, stats = assert_tx_contract(tx)
    logger.info("Tx contract ok=%s stats=%s", ok, {k: round(v, 4) for k, v in stats.items()})
    if not ok:
        raise ValueError("Transaction contract failed; fix ingestion before proceeding.")

    if "ticker" in tx.columns:
        tx["ticker_norm"] = tx["ticker"].astype(str).map(normalize_ticker)
    else:
        tx["ticker_norm"] = ""

    tx["issuer_norm"] = tx["asset_name"].astype(str).map(normalize_issuer_name)
    tx["asset_id"] = tx.apply(lambda r: choose_asset_id(r.get("ticker_norm", ""), r.get("issuer_norm", "")), axis=1)

    tx["filing_datetime"] = tx["filing_datetime"].astype(str)
    tx["filing_date"] = tx["filing_datetime"].str.slice(0, 10)

    def sign(tt: str) -> int:
        x = (tt or "").upper().strip()
        if x == "BUY":
            return 1
        if x == "SELL":
            return -1
        return 0

    tx["tx_sign"] = tx["transaction_type"].astype(str).map(sign)
    tx["tx_weight"] = 1.0
    tx["tx_impulse"] = tx["tx_sign"] * tx["tx_weight"]

    out_path = project_root / "data" / "processed" / "transactions_enriched.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tx.to_csv(out_path, index=False)
    logger.info("Wrote enriched tx: %s rows=%d cols=%d", str(out_path), len(tx), len(tx.columns))

    assets = (
        tx[["asset_id", "ticker_norm", "issuer_norm", "asset_name"]]
        .drop_duplicates()
        .sort_values(["ticker_norm", "issuer_norm", "asset_id"])
        .reset_index(drop=True)
    )
    assets_path = project_root / "outputs" / "ppf_asset_universe.csv"
    assets.to_csv(assets_path, index=False)
    logger.info("Wrote asset universe: %s rows=%d", str(assets_path), len(assets))

    meta = {
        "transactions_in": str(in_path),
        "transactions_enriched_out": str(out_path),
        "asset_universe_out": str(assets_path),
        "rows": int(len(tx)),
        "assets": int(len(assets)),
    }
    return tx, meta


# -----------------------------
# Step B: Load/apply mapping
# -----------------------------

def load_mapping(project_root: Path, logger: logging.Logger) -> pd.DataFrame:
    mapping_path = project_root / "data" / "reference" / "asset_to_sector_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_path}")

    m = pd.read_csv(mapping_path)

    for c in MAPPING_COLUMNS:
        if c not in m.columns:
            m[c] = ""

    # Clean fields (this is the critical fix)
    m["ticker_norm"] = clean_text_series(m["ticker_norm"]).map(normalize_ticker)
    m["issuer_norm"] = clean_text_series(m["issuer_norm"]).map(normalize_issuer_name)
    m["sector"] = clean_text_series(m["sector"])
    m["subsector"] = clean_text_series(m["subsector"])
    m["mapping_source"] = clean_text_series(m["mapping_source"])
    m["confidence"] = clean_text_series(m["confidence"])
    m["last_updated_utc"] = clean_text_series(m["last_updated_utc"])

    # Rebuild asset_id deterministically from cleaned keys
    m["asset_id"] = m.apply(lambda r: choose_asset_id(r.get("ticker_norm", ""), r.get("issuer_norm", "")), axis=1)
    m["asset_id"] = clean_text_series(m["asset_id"])

    # Mapping quality checks
    blank_sector = (m["sector"] == "").mean() if len(m) else 1.0
    logger.info("Mapping loaded: rows=%d blank_sector_fraction=%.4f", len(m), float(blank_sector))
    if len(m) and blank_sector == 1.0:
        logger.warning("All mapping rows have blank sector after cleaning. This will yield 100%% UNMAPPED.")

    return m[MAPPING_COLUMNS]


def apply_mapping(
    tx: pd.DataFrame,
    mapping: pd.DataFrame,
    project_root: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Deterministic join on asset_id.
    Treat blank/NaN sector/subsector as UNMAPPED.
    """
    m = mapping.copy()

    out = tx.merge(
        m[["asset_id", "sector", "subsector", "mapping_source", "confidence", "last_updated_utc"]],
        on="asset_id",
        how="left",
    )

    # Clean merged mapping values; then set UNMAPPED where blank
    out["sector"] = clean_text_series(out["sector"])
    out["subsector"] = clean_text_series(out["subsector"])
    out["mapping_source"] = clean_text_series(out["mapping_source"])
    out["confidence"] = clean_text_series(out["confidence"])
    out["last_updated_utc"] = clean_text_series(out["last_updated_utc"])

    out.loc[out["sector"] == "", "sector"] = "UNMAPPED"
    out.loc[out["subsector"] == "", "subsector"] = "UNMAPPED"

    unmapped_fraction = float((out["sector"] == "UNMAPPED").mean())
    logger.info("Mapping coverage: unmapped_fraction=%.4f", unmapped_fraction)

    # Assets still needing mapping (either key missing OR sector blank)
    assets_unmapped = (
        out.loc[out["sector"] == "UNMAPPED", ["asset_id", "ticker_norm", "issuer_norm", "asset_name"]]
        .drop_duplicates()
        .sort_values(["ticker_norm", "issuer_norm", "asset_id"])
        .reset_index(drop=True)
    )
    unmapped_path = project_root / "outputs" / "ppf_assets_unmapped.csv"
    assets_unmapped.to_csv(unmapped_path, index=False)
    logger.info("Wrote unmapped assets: %s rows=%d", str(unmapped_path), len(assets_unmapped))

    meta = {
        "mapping_rows": int(len(m)),
        "unmapped_fraction": unmapped_fraction,
        "unmapped_assets_out": str(unmapped_path),
        "unmapped_assets": int(len(assets_unmapped)),
    }
    return out, meta


# -----------------------------
# Step C: Build impulses
# -----------------------------

def build_impulses_sector_daily(mapped_tx: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    base = mapped_tx.copy()
    base = base.loc[(base["tx_sign"] != 0) & (base["sector"] != "UNMAPPED")].copy()

    if base.empty:
        logger.warning("No mappable BUY/SELL transactions after filters (tx_sign!=0 and sector!=UNMAPPED).")
        return pd.DataFrame(columns=["filing_date", "sector", "official_name", "tx_impulse"]), {"rows": 0}

    base["filing_date"] = base["filing_date"].astype(str)

    g = (
        base.groupby(["filing_date", "sector", "official_name"], dropna=False)["tx_impulse"]
        .sum()
        .reset_index()
    )

    logger.info(
        "Impulses built: rows=%d unique_dates=%d unique_sectors=%d unique_officials=%d",
        len(g), g["filing_date"].nunique(), g["sector"].nunique(), g["official_name"].nunique()
    )

    meta = {
        "rows": int(len(g)),
        "unique_dates": int(g["filing_date"].nunique()) if len(g) else 0,
        "unique_sectors": int(g["sector"].nunique()) if len(g) else 0,
        "unique_officials": int(g["official_name"].nunique()) if len(g) else 0,
    }
    return g, meta


# -----------------------------
# Step D: PPF score
# -----------------------------

@dataclass
class PPFConfig:
    half_life_days: int = 120
    eps: float = 1e-9
    concentration_alpha: float = 1.0


def exp_smooth(series: np.ndarray, half_life_days: int) -> np.ndarray:
    if half_life_days <= 0:
        return series.astype(float)
    lam = 1.0 - float(np.exp(np.log(0.5) / float(half_life_days)))
    out = np.zeros_like(series, dtype=float)
    prev = 0.0
    for i, v in enumerate(series.astype(float)):
        prev = lam * v + (1.0 - lam) * prev
        out[i] = prev
    return out


def compute_ppf_sector_daily(impulses: pd.DataFrame, cfg: PPFConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if impulses.empty:
        logger.warning("Impulses empty; PPF scores will be empty.")
        return pd.DataFrame(columns=["filing_date", "sector", "ppf_score"]), {"rows": 0}

    df = impulses.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.dropna(subset=["filing_date"])
    if df.empty:
        logger.warning("All impulse dates failed parsing; PPF scores will be empty.")
        return pd.DataFrame(columns=["filing_date", "sector", "ppf_score"]), {"rows": 0}

    df["filing_date"] = df["filing_date"].dt.date.astype(str)

    min_d = pd.to_datetime(df["filing_date"]).min()
    max_d = pd.to_datetime(df["filing_date"]).max()
    all_days = pd.date_range(min_d, max_d, freq="D").date.astype(str)

    officials = sorted(df["official_name"].unique())
    sectors = sorted(df["sector"].unique())

    idx = pd.MultiIndex.from_product(
        [officials, sectors, list(all_days)],
        names=["official_name", "sector", "filing_date"],
    )

    dense = (
        df.set_index(["official_name", "sector", "filing_date"])["tx_impulse"]
        .reindex(idx)
        .fillna(0.0)
        .reset_index()
        .rename(columns={"tx_impulse": "impulse"})
    )
    dense = dense.sort_values(["official_name", "sector", "filing_date"]).reset_index(drop=True)

    smoothed = []
    for (_off, _sec), g in dense.groupby(["official_name", "sector"], sort=False):
        s = exp_smooth(g["impulse"].to_numpy(), cfg.half_life_days)
        smoothed.append(pd.Series(s, index=g.index))
    dense["smoothed"] = pd.concat(smoothed).sort_index().values

    dense["abs_smoothed"] = dense["smoothed"].abs()
    per_off_day_sum = dense.groupby(["official_name", "filing_date"])["abs_smoothed"].sum().reset_index().rename(columns={"abs_smoothed": "abs_sum"})
    per_off_day_max = dense.groupby(["official_name", "filing_date"])["abs_smoothed"].max().reset_index().rename(columns={"abs_smoothed": "abs_max"})
    conc = per_off_day_sum.merge(per_off_day_max, on=["official_name", "filing_date"], how="left")
    conc["concentration"] = conc["abs_max"] / (conc["abs_sum"] + cfg.eps)

    dense = dense.merge(conc[["official_name", "filing_date", "concentration"]], on=["official_name", "filing_date"], how="left")
    dense["concentration"] = dense["concentration"].fillna(0.0)
    dense["conc_weight"] = dense["concentration"] ** float(cfg.concentration_alpha)
    dense["contrib"] = dense["smoothed"] * dense["conc_weight"]

    scores = (
        dense.groupby(["filing_date", "sector"], dropna=False)["contrib"]
        .sum()
        .reset_index()
        .rename(columns={"contrib": "ppf_score"})
    )

    logger.info(
        "PPF sector scores: rows=%d dates=%d sectors=%d",
        len(scores), scores["filing_date"].nunique(), scores["sector"].nunique()
    )

    meta = {
        "rows": int(len(scores)),
        "dates": int(scores["filing_date"].nunique()) if len(scores) else 0,
        "sectors": int(scores["sector"].nunique()) if len(scores) else 0,
        "half_life_days": int(cfg.half_life_days),
        "concentration_alpha": float(cfg.concentration_alpha),
    }
    return scores, meta


# -----------------------------
# Manifest + I/O helpers
# -----------------------------

def safe_write_csv(df: pd.DataFrame, path: Path, logger: logging.Logger, errors: list) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Wrote: %s rows=%d", str(path), int(len(df)))
    except Exception as e:
        errors.append({"stage": "write_csv", "path": str(path), **exc_to_dict(e)})
        logger.exception("Write CSV failed: %s", str(path))


def safe_write_json(obj: Dict[str, Any], path: Path, logger: logging.Logger, errors: list) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        logger.info("Wrote: %s", str(path))
    except Exception as e:
        errors.append({"stage": "write_json", "path": str(path), **exc_to_dict(e)})
        logger.exception("Write JSON failed: %s", str(path))


# -----------------------------
# Main orchestrator
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".", help="PROJECT_ROOT containing data/, outputs/, logs/")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--half-life-days", type=int, default=120, help="PPF persistence half-life in days")
    ap.add_argument("--concentration-alpha", type=float, default=1.0, help="Concentration weighting exponent")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    logger = setup_logger(project_root, args.verbose)

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    errors: list = []
    warnings: list = []

    outputs_dir = project_root / "outputs"
    processed_dir = project_root / "data" / "processed"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "run_utc": utc_now_iso(),
        "project_root": str(project_root),
        "args": {
            "half_life_days": int(args.half_life_days),
            "concentration_alpha": float(args.concentration_alpha),
            "verbose": bool(args.verbose),
        },
        "counts": {},
        "outputs": {},
        "warnings": warnings,
        "errors": errors,
    }

    try:
        with StageTimer(logger, "ensure_reference_files"):
            ref_paths = ensure_reference_files(project_root, logger)
            manifest["outputs"].update(ref_paths)

        with StageTimer(logger, "enrich_transactions"):
            tx, meta_enrich = enrich_transactions(project_root, logger)
            manifest["counts"]["transactions"] = int(len(tx))
            manifest["counts"]["assets"] = int(meta_enrich.get("assets", 0))
            manifest["outputs"]["transactions_enriched"] = str(processed_dir / "transactions_enriched.csv")
            manifest["outputs"]["asset_universe"] = str(outputs_dir / "ppf_asset_universe.csv")

        with StageTimer(logger, "load_and_apply_mapping"):
            mapping = load_mapping(project_root, logger)
            mapped, meta_map = apply_mapping(tx, mapping, project_root, logger)
            manifest["counts"]["mapping_rows"] = int(meta_map.get("mapping_rows", 0))
            manifest["counts"]["unmapped_assets"] = int(meta_map.get("unmapped_assets", 0))
            manifest["counts"]["unmapped_fraction"] = float(meta_map.get("unmapped_fraction", 1.0))
            manifest["outputs"]["assets_unmapped"] = str(outputs_dir / "ppf_assets_unmapped.csv")

        with StageTimer(logger, "build_impulses_sector_daily"):
            impulses, meta_imp = build_impulses_sector_daily(mapped, logger)
            manifest["counts"]["impulses_rows"] = int(meta_imp.get("rows", 0))
            impulses_path = outputs_dir / "ppf_impulses_sector_daily.csv"
            safe_write_csv(impulses, impulses_path, logger, errors)
            manifest["outputs"]["impulses_sector_daily"] = str(impulses_path)

        with StageTimer(logger, "compute_ppf_sector_daily"):
            cfg = PPFConfig(
                half_life_days=int(args.half_life_days),
                concentration_alpha=float(args.concentration_alpha),
            )
            scores, meta_scores = compute_ppf_sector_daily(impulses, cfg, logger)
            manifest["counts"]["scores_rows"] = int(meta_scores.get("rows", 0))
            scores_path = outputs_dir / "ppf_scores_sector_daily.csv"
            safe_write_csv(scores, scores_path, logger, errors)
            manifest["outputs"]["scores_sector_daily"] = str(scores_path)

            if not scores.empty:
                latest_day = sorted(scores["filing_date"].unique())[-1]
                snap = scores.loc[scores["filing_date"] == latest_day].copy()
                snap["rank_desc"] = snap["ppf_score"].rank(ascending=False, method="min").astype(int)
                snap = snap.sort_values(["rank_desc", "sector"]).reset_index(drop=True)
                snap_path = outputs_dir / "ppf_scores_sector_snapshot_latest.csv"
                safe_write_csv(snap, snap_path, logger, errors)
                manifest["outputs"]["scores_snapshot_latest"] = str(snap_path)
                manifest["latest_score_date"] = latest_day
            else:
                warnings.append({
                    "kind": "scores_empty",
                    "reason": "No mapped BUY/SELL impulses (mapping empty/blank sectors or all UNMAPPED).",
                })
                logger.warning("Scores empty.")

        safe_write_json(manifest, outputs_dir / "ppf_pipeline_manifest.json", logger, errors)
        exit_code = 0 if len(errors) == 0 else 5
        logger.info("=== PPF Pipeline end === exit_code=%d", exit_code)
        return exit_code

    except Exception as e:
        errors.append({"stage": "fatal", **exc_to_dict(e)})
        logger.exception("Fatal pipeline error")
        safe_write_json(manifest, outputs_dir / "ppf_pipeline_manifest.json", logger, errors)
        logger.info("=== PPF Pipeline end === exit_code=9")
        return 9


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise
