#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppf_pipeline.py

PPF Next-Step Pipeline (v4): Enrich → Mapping → Impulses → PPF Scores
(Hardened logging + stage timing + manifest + optional yfinance auto-mapping)

Design intent:
  - Disclosures-only clocking: filing_datetime is treated as *availability timestamp*.
  - Sector/subsector is a taxonomy join needed to aggregate PPF at sector level.
  - Mapping is *auditable*: asset_to_sector_mapping.csv is persisted and is the join authority.
  - Optional auto-mapping uses *public* metadata (yfinance) to populate sector/subsector
    for tickers, to avoid manual mapping busywork.

Inputs:
  - outputs/ppf_transactions_unified.csv  (from unify_ppf.py)

Outputs:
  - data/processed/transactions_enriched.csv
  - outputs/ppf_asset_universe.csv
  - data/reference/asset_to_sector_mapping.csv
  - outputs/ppf_assets_unmapped.csv
  - outputs/ppf_impulses_sector_daily.csv
  - outputs/ppf_scores_sector_daily.csv
  - outputs/ppf_scores_sector_snapshot_latest.csv
  - outputs/ppf_pipeline_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore


TX_REQUIRED_COLUMNS = [
    "source_file",
    "source_page",
    "official_name",
    "filing_datetime",
    "transaction_date",
    "owner",
    "transaction_type",
    "amount_bucket_raw",
    "asset_name",
]

MAPPING_COLUMNS = [
    "asset_id",
    "ticker_norm",
    "issuer_norm",
    "sector",
    "subsector",
    "mapping_source",
    "confidence",
    "last_updated_utc",
]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(project_root: Path, verbose: bool) -> logging.Logger:
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "ppf_pipeline.log"

    logger = logging.getLogger("ppf_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)sZ | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=== PPF Pipeline start ===")
    logger.info("project_root=%s", str(project_root))
    logger.info("log_path=%s", str(log_path))
    return logger


@contextmanager
def stage_timer(stage_name: str, logger: logging.Logger):
    t0 = time.time()
    logger.info("Stage start: %s", stage_name)
    try:
        yield
    finally:
        logger.info("Stage end: %s elapsed=%.3fs", stage_name, time.time() - t0)


def normalize_ticker(x: str) -> str:
    s = (str(x) if x is not None else "").strip().upper()
    s = s.replace("$", "").replace(".", "-")
    return "" if s in {"", "NAN", "NONE"} else s


def normalize_issuer_name(x: str) -> str:
    s = (str(x) if x is not None else "").upper()
    for ch in ["\u2013", "\u2014", "–", "—"]:
        s = s.replace(ch, "-")
    s = " ".join(s.split())
    s = s.replace(",", "").replace(".", "")
    s = s.replace(" INCORPORATED", " INC")
    s = s.replace(" CORPORATION", " CORP")
    s = s.replace(" COMPANY", " CO")
    return "" if s in {"", "NAN", "NONE"} else s.strip()


def choose_asset_id(ticker_norm: str, issuer_norm: str) -> str:
    return ticker_norm if ticker_norm else issuer_norm


def ensure_reference_files(project_root: Path, logger: logging.Logger) -> Path:
    ref_dir = project_root / "data" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = ref_dir / "asset_to_sector_mapping.csv"
    if not mapping_path.exists():
        pd.DataFrame(columns=MAPPING_COLUMNS).to_csv(mapping_path, index=False)
        logger.info("Created mapping skeleton: %s", mapping_path)
    return mapping_path


def enrich_transactions(project_root: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, str]]:
    in_path = project_root / "outputs" / "ppf_transactions_unified.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing required input: {in_path}")

    tx = pd.read_csv(in_path)
    missing = [c for c in TX_REQUIRED_COLUMNS if c not in tx.columns]
    if missing:
        raise ValueError(f"Unified transactions missing required columns: {missing}")

    stats = {c: float(tx[c].isna().mean()) for c in TX_REQUIRED_COLUMNS}
    logger.info("Tx contract ok=True stats=%s", stats)

    tx["filing_datetime"] = pd.to_datetime(tx["filing_datetime"], errors="coerce", utc=True)
    tx["filing_date"] = tx["filing_datetime"].dt.date

    if "ticker" in tx.columns:
        tx["ticker_norm"] = tx["ticker"].astype(str).map(normalize_ticker)
    else:
        tx["ticker_norm"] = ""

    tx["issuer_norm"] = tx["asset_name"].astype(str).map(normalize_issuer_name)
    tx["asset_id"] = tx.apply(lambda r: choose_asset_id(r.get("ticker_norm", ""), r.get("issuer_norm", "")), axis=1)

    tx["transaction_type"] = tx["transaction_type"].astype(str).str.upper().str.strip()
    tx["tx_sign"] = tx["transaction_type"].map({"BUY": 1, "SELL": -1}).fillna(0).astype(int)
    tx["tx_weight"] = tx["tx_sign"].abs().astype(float)  # conservative default = 1
    tx["tx_impulse"] = tx["tx_sign"].astype(float) * tx["tx_weight"].astype(float)

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transactions_enriched.csv"
    tx.to_csv(out_path, index=False)
    logger.info("Wrote enriched tx: %s rows=%d cols=%d", out_path, len(tx), len(tx.columns))

    asset_universe = (
        tx[["asset_id", "ticker_norm", "issuer_norm", "asset_name"]]
        .drop_duplicates()
        .sort_values(["ticker_norm", "issuer_norm", "asset_id"])
    )
    universe_path = project_root / "outputs" / "ppf_asset_universe.csv"
    ensure_parent_dir(universe_path)
    asset_universe.to_csv(universe_path, index=False)
    logger.info("Wrote asset universe: %s rows=%d", universe_path, len(asset_universe))

    return tx, {"transactions_enriched": str(out_path), "asset_universe": str(universe_path)}


def load_mapping(mapping_path: Path) -> pd.DataFrame:
    if not mapping_path.exists():
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    try:
        m = pd.read_csv(mapping_path).fillna("")
    except pd.errors.EmptyDataError:
        m = pd.DataFrame(columns=MAPPING_COLUMNS)
    for c in MAPPING_COLUMNS:
        if c not in m.columns:
            m[c] = ""
    # normalize labels
    for c in ["sector", "subsector", "mapping_source"]:
        m[c] = m[c].astype(str).fillna("").map(lambda s: s.strip()).replace({"nan": "", "None": ""})
    return m[MAPPING_COLUMNS]


def auto_map_yfinance(
    mapping: pd.DataFrame,
    asset_universe: pd.DataFrame,
    logger: logging.Logger,
    mapping_path: Path,
    rate_limit_secs: float,
    max_assets: int,
) -> pd.DataFrame:
    if yf is None:
        logger.warning("yfinance not installed; skipping auto-mapping.")
        return mapping

    # candidates: tickers in asset universe
    tickers = asset_universe["ticker_norm"].astype(str).fillna("").tolist()
    tickers = [t.strip().upper() for t in tickers if t and t.strip() and t.strip().upper() not in {"NAN", "NONE"}]
    tickers = sorted(set(tickers))
    if max_assets and max_assets > 0:
        tickers = tickers[:max_assets]

    # build index by ticker
    by_ticker = {}
    for i, row in mapping.iterrows():
        t = str(row.get("ticker_norm", "")).strip().upper()
        if t:
            by_ticker[t] = i

    updated = 0
    checked = 0

    for t in tickers:
        checked += 1
        idx = by_ticker.get(t)
        if idx is not None:
            sector = str(mapping.at[idx, "sector"] or "").strip()
            if sector and sector.upper() != "UNMAPPED":
                continue

        try:
            info = yf.Ticker(t).info
        except Exception as e:
            logger.debug("yfinance info failed ticker=%s err=%s", t, repr(e))
            continue

        sector = str(info.get("sector", "") or "").strip()
        industry = str(info.get("industry", "") or "").strip()
        if not sector:
            continue

        now = utc_now_iso()
        if idx is None:
            new_row = {c: "" for c in MAPPING_COLUMNS}
            new_row.update(
                {
                    "asset_id": t,
                    "ticker_norm": t,
                    "issuer_norm": "",
                    "sector": sector,
                    "subsector": industry,
                    "mapping_source": "yfinance",
                    "confidence": "medium",
                    "last_updated_utc": now,
                }
            )
            mapping = pd.concat([mapping, pd.DataFrame([new_row])], ignore_index=True)
            by_ticker[t] = len(mapping) - 1
            updated += 1
        else:
            mapping.at[idx, "sector"] = sector
            mapping.at[idx, "subsector"] = industry
            mapping.at[idx, "mapping_source"] = "yfinance"
            mapping.at[idx, "confidence"] = str(mapping.at[idx, "confidence"] or "medium") or "medium"
            mapping.at[idx, "last_updated_utc"] = now
            updated += 1

        if updated % 25 == 0:
            logger.info("Auto-mapping progress updated=%d checked=%d", updated, checked)

        if rate_limit_secs and rate_limit_secs > 0:
            time.sleep(rate_limit_secs)

    if updated:
        mapping.to_csv(mapping_path, index=False)
        logger.info("Auto-mapping updated mapping_path=%s updated_rows=%d", mapping_path, updated)

    logger.info("Auto-mapping complete updated_rows=%d checked=%d", updated, checked)
    return mapping


def apply_mapping(tx: pd.DataFrame, mapping: pd.DataFrame, project_root: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tx = tx.copy()
    tx["ticker_norm"] = tx["ticker_norm"].astype(str).map(normalize_ticker)
    tx["issuer_norm"] = tx["issuer_norm"].astype(str).map(normalize_issuer_name)
    tx["asset_id"] = tx.apply(lambda r: choose_asset_id(r.get("ticker_norm", ""), r.get("issuer_norm", "")), axis=1)

    m = mapping.copy()
    m["ticker_norm"] = m["ticker_norm"].astype(str).map(normalize_ticker)
    m["issuer_norm"] = m["issuer_norm"].astype(str).map(normalize_issuer_name)
    m["asset_id"] = m.apply(lambda r: choose_asset_id(r.get("ticker_norm", ""), r.get("issuer_norm", "")), axis=1)

    joined = tx.merge(m[["asset_id", "sector", "subsector", "mapping_source", "confidence"]], on="asset_id", how="left")
    joined["sector"] = joined["sector"].fillna("").astype(str).str.strip()
    joined["subsector"] = joined["subsector"].fillna("").astype(str).str.strip()
    joined.loc[joined["sector"] == "", "sector"] = "UNMAPPED"
    joined.loc[joined["subsector"] == "", "subsector"] = "UNMAPPED"

    unmapped_fraction = float((joined["sector"] == "UNMAPPED").mean()) if len(joined) else 1.0
    logger.info("Mapping coverage: unmapped_fraction=%.4f", unmapped_fraction)

    out_unmapped = project_root / "outputs" / "ppf_assets_unmapped.csv"
    ensure_parent_dir(out_unmapped)
    (
        joined.loc[joined["sector"] == "UNMAPPED", ["asset_id", "ticker_norm", "issuer_norm", "asset_name"]]
        .drop_duplicates()
        .sort_values(["ticker_norm", "issuer_norm", "asset_id"])
        .to_csv(out_unmapped, index=False)
    )
    logger.info("Wrote unmapped assets: %s rows=%d", out_unmapped, int((joined["sector"] == "UNMAPPED").sum()))
    return joined, {"assets_unmapped": str(out_unmapped), "unmapped_fraction": unmapped_fraction}


def build_impulses_sector_daily(mapped: pd.DataFrame, project_root: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = mapped.copy()
    df = df[(df["tx_sign"] != 0) & (df["sector"] != "UNMAPPED")].copy()

    out_path = project_root / "outputs" / "ppf_impulses_sector_daily.csv"
    ensure_parent_dir(out_path)

    if df.empty:
        logger.warning("No mappable BUY/SELL transactions after filters.")
        pd.DataFrame(columns=["filing_date", "official_name", "sector", "impulse"]).to_csv(out_path, index=False)
        return pd.DataFrame(), {"impulses_path": str(out_path)}

    impulses = (
        df.groupby(["filing_date", "official_name", "sector"], as_index=False)["tx_impulse"]
        .sum()
        .rename(columns={"tx_impulse": "impulse"})
        .sort_values(["filing_date", "official_name", "sector"])
    )
    impulses.to_csv(out_path, index=False)
    logger.info(
        "Impulses built rows=%d dates=%d sectors=%d officials=%d",
        len(impulses),
        impulses["filing_date"].nunique(),
        impulses["sector"].nunique(),
        impulses["official_name"].nunique(),
    )
    return impulses, {"impulses_path": str(out_path)}


def compute_ppf_sector_daily(
    impulses: pd.DataFrame,
    project_root: Path,
    logger: logging.Logger,
    half_life_days: int,
    concentration_alpha: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out_path = project_root / "outputs" / "ppf_scores_sector_daily.csv"
    ensure_parent_dir(out_path)

    if impulses is None or impulses.empty:
        logger.warning("Impulses empty; scores will be empty.")
        pd.DataFrame(columns=["date", "sector", "ppf_score"]).to_csv(out_path, index=False)
        return pd.DataFrame(), {"scores_path": str(out_path)}

    impulses = impulses.copy()
    impulses["date"] = pd.to_datetime(impulses["filing_date"])

    all_dates = pd.date_range(impulses["date"].min(), impulses["date"].max(), freq="D")
    officials = sorted(impulses["official_name"].unique().tolist())
    sectors = sorted(impulses["sector"].unique().tolist())

    lam = (math.log(2.0) / float(half_life_days)) if half_life_days > 0 else 0.0
    decay = math.exp(-lam) if lam > 0 else 1.0

    S = {(i, k): 0.0 for i in officials for k in sectors}
    lookup = impulses.groupby(["date", "official_name", "sector"])["impulse"].sum()

    rows = []
    for d in all_dates:
        for i in officials:
            for k in sectors:
                S[(i, k)] = S[(i, k)] * decay + float(lookup.get((d, i, k), 0.0))

            abs_total = sum(abs(S[(i, k)]) for k in sectors)
            eps = 1e-9
            conc = sum(abs(S[(i, k)]) ** 2 for k in sectors) / (abs_total ** 2 + eps) if abs_total > 0 else 0.0
            conc_weight = conc ** float(concentration_alpha)

            for k in sectors:
                rows.append(
                    {
                        "date": d.date(),
                        "sector": k,
                        "ppf_score": S[(i, k)] * conc_weight,
                    }
                )

    scores = (
        pd.DataFrame(rows)
        .groupby(["date", "sector"], as_index=False)["ppf_score"]
        .sum()
        .sort_values(["date", "sector"])
    )
    scores.to_csv(out_path, index=False)
    logger.info("PPF sector scores rows=%d dates=%d sectors=%d", len(scores), scores["date"].nunique(), scores["sector"].nunique())

    latest_date = scores["date"].max()
    snap = scores[scores["date"] == latest_date].sort_values("ppf_score", ascending=False)
    snap_path = project_root / "outputs" / "ppf_scores_sector_snapshot_latest.csv"
    snap.to_csv(snap_path, index=False)
    logger.info("Wrote snapshot: %s rows=%d latest_date=%s", snap_path, len(snap), latest_date)
    return scores, {"scores_path": str(out_path), "snapshot_path": str(snap_path), "latest_date": str(latest_date)}


def write_manifest(project_root: Path, manifest: Dict[str, Any], logger: logging.Logger) -> None:
    out = project_root / "outputs" / "ppf_pipeline_manifest.json"
    ensure_parent_dir(out)
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote manifest: %s", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--half-life-days", type=int, default=120)
    ap.add_argument("--concentration-alpha", type=float, default=1.0)
    ap.add_argument("--auto-map-yfinance", action="store_true")
    ap.add_argument("--yfinance-rate-limit-secs", type=float, default=0.6)
    ap.add_argument("--yfinance-max-assets", type=int, default=0)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    logger = setup_logger(project_root, args.verbose)

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_utc": utc_now_iso(),
        "project_root": str(project_root),
        "args": vars(args),
        "paths": {},
        "warnings": [],
        "errors": [],
    }

    try:
        with stage_timer("ensure_reference_files", logger):
            mapping_path = ensure_reference_files(project_root, logger)
            manifest["paths"]["asset_to_sector_mapping"] = str(mapping_path)

        with stage_timer("enrich_transactions", logger):
            tx, meta = enrich_transactions(project_root, logger)
            manifest["paths"].update(meta)

        with stage_timer("load_and_apply_mapping", logger):
            mapping = load_mapping(mapping_path)
            if args.auto_map_yfinance:
                try:
                    au = pd.read_csv(project_root / "outputs" / "ppf_asset_universe.csv")
                    mapping = auto_map_yfinance(
                        mapping=mapping,
                        asset_universe=au,
                        logger=logger,
                        mapping_path=mapping_path,
                        rate_limit_secs=float(args.yfinance_rate_limit_secs),
                        max_assets=int(args.yfinance_max_assets),
                    )
                except Exception as e:
                    manifest["warnings"].append({"stage": "auto_map_yfinance", "error": repr(e), "traceback": traceback.format_exc(limit=20)})
                    logger.warning("auto-map-yfinance failed; continuing with existing mapping.")

            mapped, meta = apply_mapping(tx, mapping, project_root, logger)
            manifest["paths"].update(meta)

        with stage_timer("build_impulses_sector_daily", logger):
            impulses, meta = build_impulses_sector_daily(mapped, project_root, logger)
            manifest["paths"].update(meta)

        with stage_timer("compute_ppf_sector_daily", logger):
            scores, meta = compute_ppf_sector_daily(
                impulses,
                project_root,
                logger,
                half_life_days=int(args.half_life_days),
                concentration_alpha=float(args.concentration_alpha),
            )
            manifest["paths"].update(meta)

        manifest["ended_utc"] = utc_now_iso()
        manifest["exit_code"] = 0

    except Exception as e:
        manifest["errors"].append({"stage": "pipeline", "error": repr(e), "traceback": traceback.format_exc(limit=40)})
        manifest["ended_utc"] = utc_now_iso()
        manifest["exit_code"] = 1
        logger.exception("Pipeline failed")

    finally:
        write_manifest(project_root, manifest, logger)
        logger.info("=== PPF Pipeline end === exit_code=%s", manifest.get("exit_code"))

    return int(manifest.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
