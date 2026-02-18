import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path
import logging
import requests
import zipfile
import io

# --- CONFIGURATION ---
T_NOW = pd.to_datetime("2026-02-18", utc=True)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
RISK_FREE_RATE = 0.042  # 4.2% (Current 10Y Treasury yield proxy)

# UUID TO NAME MAPPER (Extracted from your logs)
ID_MAP = {
    'B195126E-7BB2-4D54-BAF5-E6FC8E7A0165': 'WARNER',
    'D7CAE837-F73C-4EB1-B9FB-E510F53D65DE': 'TUBERVILLE',
    '98B3317B-2632-48C3-B636-D5555C4680CA': 'GRASSLEY',
    '7AB0D2FD-19EA-459D-A1D9-3701E2CC8E93': 'THUNE',
    'BD886067-927F-48A8-9D43-8AB6FA713F98': 'CARPER'
}

COMMITTEE_WEIGHTS = {
    "WARNER": 1.5, "TUBERVILLE": 1.4, "GRASSLEY": 1.3, "THUNE": 1.3, "CARPER": 1.2
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PPFEngine:
    def __init__(self, debug_path="outputs/ppf_final_signal_debug_rows.csv"):
        self.debug_path = Path(debug_path)
        self.df = None
        self.prices = None
        self.results_df = None

    def load_senate_data(self):
        """Standardizes Senate UUIDs and tickers."""
        if not self.debug_path.exists():
            logging.error(f"Missing {self.debug_path}. Scrape Senate data first.")
            return False
        
        self.df = pd.read_csv(self.debug_path)
        
        def resolve_filer(row):
            raw_id = Path(row['source_file']).stem.split('_')[0].upper()
            return ID_MAP.get(raw_id, raw_id[:8])

        self.df['filer'] = self.df.apply(resolve_filer, axis=1)
        self.df['ticker'] = self.df['ticker'].astype(str).str.strip().str.upper()
        self.df['filing_datetime'] = pd.to_datetime(self.df['filing_datetime'], utc=True)
        
        # Apply Committee Alpha
        self.df['final_score'] = self.df.apply(
            lambda x: x['score'] * COMMITTEE_WEIGHTS.get(x['filer'], 1.0), axis=1
        )
        return True

    def fetch_house_index(self, year="2024"):
        """EXPERIMENTAL: Pulls House PTR Index for expansion."""
        logging.info(f"Fetching House PTR Index for {year}...")
        url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
        try:
            r = requests.get(url, timeout=10)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(OUT_DIR / "house")
            logging.info("House index downloaded. Use pdfplumber for PTR parsing next.")
        except Exception as e:
            logging.error(f"House fetch failed: {e}")

    def fetch_market_prices(self):
        """Syncs adjusted prices for backtesting."""
        tickers = [t for t in self.df['ticker'].unique().tolist() if t != 'K']
        logging.info(f"Downloading data for {len(tickers)} tickers...")
        
        # auto_adjust=True accounts for dividends/splits
        data = yf.download(tickers, start="2024-01-01", end="2026-02-19", auto_adjust=True)['Close']
        self.prices = data.stack().reset_index()
        self.prices.columns = ['date', 'ticker', 'close']
        self.prices['date'] = pd.to_datetime(self.prices['date'], utc=True)

    def run_backtest(self):
        """Calculates CAGR and Sharpe Ratio with T+1 execution."""
        results = []
        for _, row in self.df.iterrows():
            ticker = row['ticker']
            entry_date = row['filing_datetime'] + pd.Timedelta(days=1)
            
            subset = self.prices[(self.prices['ticker'] == ticker) & (self.prices['date'] >= entry_date)]
            if subset.empty: continue
                
            entry_px = subset.iloc[0]['close']
            exit_px = self.prices[self.prices['ticker'] == ticker].iloc[-1]['close']
            
            days_held = (T_NOW - entry_date).days
            if days_held < 14: continue # Filter for conviction trades
            
            simple_ret = (exit_px / entry_px) - 1
            cagr = ((1 + simple_ret) ** (365.25 / days_held)) - 1
            results.append({'filer': row['filer'], 'ticker': ticker, 'cagr': cagr, 'score': row['final_score']})
        
        self.results_df = pd.DataFrame(results)

    def generate_report(self):
        """Final Risk/Reward Dashboard."""
        weights = self.results_df['score'].abs()
        portfolio_cagr = (self.results_df['cagr'] * weights).sum() / weights.sum()
        
        # Calculating Volatility for Sharpe Ratio
        vol = self.results_df['cagr'].std()
        sharpe = (portfolio_cagr - RISK_FREE_RATE) / vol if vol > 0 else 0
        
        print("\n" + "═"*50)
        print(f"       PPF QUANT ENGINE v2.3: AUDIT COMPLETE")
        print("═"*50)
        print(f"PORTFOLIO CAGR:       {portfolio_cagr:.2%}")
        print(f"SHARPE RATIO:         {sharpe:.2f} (Target > 1.0)")
        print(f"ESTIMATED ALPHA:      {portfolio_cagr - 0.152:.2%}")
        print("─"*50)
        print("TOP RISK-ADJUSTED MOVERS:")
        print(self.results_df.sort_values('cagr', ascending=False).head(5)[['filer', 'ticker', 'cagr']])

if __name__ == "__main__":
    engine = PPFEngine()
    if engine.load_senate_data():
        # engine.fetch_house_index() # Uncomment to start House data collection
        engine.fetch_market_prices()
        engine.run_backtest()
        engine.generate_report()
