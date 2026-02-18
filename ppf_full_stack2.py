import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
import io
import pdfplumber
import re

# --- HELPER FUNCTIONS ---
def normalize_ticker(t: str) -> str:
    """
    Cleans tickers for Yahoo Finance compatibility by removing prefixes and non-standard characters.
    """
    if not isinstance(t, str): return ""
    t = t.strip().upper()
    t = t.lstrip("$")            # Fixes ticker normalization for yfinance
    t = t.replace(".", "-")      # BRK.B -> BRK-B
    t = re.sub(r"[^A-Z0-9\-]", "", t) 
    return t

# --- CONFIG ---
T_NOW = pd.to_datetime("2026-02-18", utc=True)
OUT_DIR = Path("outputs")
HOUSE_DIR = OUT_DIR / "house"
OUT_DIR.mkdir(exist_ok=True); HOUSE_DIR.mkdir(exist_ok=True)
TIMEOUT = 10

# THE AUDIT MAP: Strictly verified 36-char GUIDs only
ID_MAP = {
    'B195126E-7BB2-4D54-BAF5-E6FC8E7A0165': 'WARNER',
    'D7CAE837-F73C-4EB1-B9FB-E510F53D65DE': 'TUBERVILLE',
    '98B3317B-2632-48C3-B636-D5555C4680CA': 'GRASSLEY',
    '7AB0D2FD-19EA-459D-A1D9-3701E2CC8E93': 'THUNE',
    'BD886067-927F-48A8-9D43-8AB6FA713F98': 'CARPER',
    '996378A6-200B-48F6-96F0-A3A1F45E445E': 'MORAN'
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PPFEngine:
    def __init__(self, senate_debug="outputs/ppf_final_signal_debug_rows.csv"):
        self.senate_path = Path(senate_debug)
        self.df = pd.DataFrame()
        self.session = self._get_session()

    def _get_session(self):
        """Creates a resilient session with retries."""
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        return session

    def _open_text_safely(self, p: Path):
        with open(p, "rb") as fb:
            head = fb.read(4)
        if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
            return open(p, "r", encoding="utf-16")
        return open(p, "r", encoding="utf-8", errors="replace")

    def sync_house(self, year="2024"):
        """Streams index and uses persistent cache for parsed PDFs."""
        cache_file = HOUSE_DIR / f"parsed_{year}.csv"
        processed_ids = set()
        
        # Resume mechanism: Load existing cache
        if cache_file.exists():
            cached = pd.read_csv(cache_file)
            cached['filing_datetime'] = pd.to_datetime(cached['filing_datetime'], utc=True)
            self.df = pd.concat([self.df, cached], ignore_index=True)
            processed_ids = set(cached['doc_id'].astype(str))

        url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
        zip_path = OUT_DIR / f"house_{year}_index.zip"
        if not zip_path.exists():
            r = self.session.get(url, timeout=TIMEOUT)
            with open(zip_path, 'wb') as f: f.write(r.content)
            
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(HOUSE_DIR)
        
        txt_file = max(HOUSE_DIR.glob("*.txt"), key=lambda p: p.stat().st_size)
        new_signals = []

        # Iterate stream rather than preloading all lines
        total_lines = sum(1 for _ in open(txt_file, 'rb'))
        with self._open_text_safely(txt_file) as f:
            for i, line in enumerate(f, start=1):
                parts = [p.strip() for p in line.split('\t')]
                if len(parts) < 5 or "P" not in parts[3:6]: continue
                
                doc_id = str(parts[-1])
                if doc_id in processed_ids: continue
                
                if i % 100 == 0: logging.info(f"House Index Stream: {i}/{total_lines} rows...")
                
                signals = self._parse_house_pdf(year, doc_id, parts[0].upper(), parts[-2])
                for s in signals: s['doc_id'] = doc_id
                new_signals.extend(signals)

        if new_signals:
            new_df = pd.DataFrame(new_signals)
            new_df.to_csv(cache_file, mode='a', header=not cache_file.exists(), index=False)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            logging.info(f"House: {len(new_signals)} new signals parsed.")

    def _parse_house_pdf(self, year, doc_id, name, date):
        pdf_path = HOUSE_DIR / f"{doc_id}.pdf"
        if not pdf_path.exists():
            try:
                r = self.session.get(f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf", timeout=TIMEOUT)
                if r.status_code == 200:
                    with open(pdf_path, 'wb') as f: f.write(r.content)
                else: return []
            except: return []
            
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "".join([pg.extract_text() or "" for pg in pdf.pages]).upper()
                
                # Transaction direction (impulse) extraction
                buy, sell = "PURCHASE" in text, "SALE" in text
                impulse = 1 if buy and not sell else (-1 if sell and not buy else 0)
                
                raw_tickers = set(re.findall(r'TICKER\s*[:\-]\s*([A-Z]{1,5})', text))
                raw_tickers |= set(re.findall(r'\(([A-Z]{1,5})\)', text))
                
                # Cleanup and Noise filtering
                noise = {"THE","AND","INC","LLC","USD","UNIT","PAGE","DATE","FORM"}
                tickers = {normalize_ticker(t) for t in raw_tickers}
                tickers = {t for t in tickers if 2 <= len(t) <= 5 and t not in noise}
                
                return [{'ticker': t, 'filer': name, 'filing_datetime': pd.to_datetime(date, utc=True), 'impulse': impulse} for t in tickers]
        except: return []

    def sync_senate(self):
        """Strict Senate ingestion with deterministic attribution."""
        if not self.senate_path.exists(): return False
        raw = pd.read_csv(self.senate_path)
        
        def resolve(row):
            # Strict GUID extraction to prevent pseudo-IDs
            m = re.search(r"\{?([A-F0-9-]{36})\}?", Path(str(row["source_file"])).name, flags=re.I)
            if not m: return "UNKNOWN"
            guid = m.group(1).upper()
            return ID_MAP.get(guid, f"UNKNOWN:{guid}")

        raw['filer'] = raw.apply(resolve, axis=1)
        raw['ticker'] = raw['ticker'].apply(normalize_ticker)
        raw['filing_datetime'] = pd.to_datetime(raw['filing_datetime'], utc=True)
        raw['impulse'] = 1 
        self.df = pd.concat([self.df, raw[['ticker', 'filer', 'filing_datetime', 'impulse']]], ignore_index=True)
        logging.info(f"Senate: {len(raw)} rows loaded.")
        return True

    def run_audit(self):
        """Vectorized event-study using merge_asof and local price caching."""
        self.df['ticker'] = self.df['ticker'].apply(normalize_ticker)
        self.df = self.df[self.df['ticker'].str.len() >= 2].drop_duplicates()
        if self.df.empty: return
        
        # Local price caching to avoid repeated yfinance calls
        cache_path = OUT_DIR / "prices_close.parquet"
        tickers = list(self.df['ticker'].unique()) + ['SPY']
        
        if cache_path.exists():
            prices = pd.read_parquet(cache_path)
            missing = [t for t in tickers if t not in prices.columns]
            if missing:
                logging.info(f"Fetching {len(missing)} missing tickers...")
                new_p = yf.download(missing, start="2024-01-01", auto_adjust=True, progress=False)['Close']
                if isinstance(new_p, pd.Series): new_p = new_p.to_frame(name=missing[0])
                prices = pd.concat([prices, new_p], axis=1)
                prices.to_parquet(cache_path)
        else:
            logging.info("Downloading initial market data...")
            prices = yf.download(tickers, start="2024-01-01", auto_adjust=True, progress=False)['Close']
            prices.to_parquet(cache_path)

        prices.index = pd.to_datetime(prices.index, utc=True)
        prices_long = prices.stack().reset_index(name='close').rename(columns={'level_0':'date','level_1':'ticker'}).sort_values(['ticker','date'])

        # Vectorized calculation: align events to nearest trading dates
        events = self.df.copy().sort_values('filing_datetime')
        events['entry_target'] = events['filing_datetime'] + pd.Timedelta(days=1)
        
        events = pd.merge_asof(events, prices_long, left_on='entry_target', right_on='date', by='ticker', direction='forward')
        events = events.rename(columns={'close':'p_entry','date':'actual_entry_dt'}).dropna(subset=['p_entry'])

        audit_results = []
        spy_long = prices_long[prices_long['ticker'] == 'SPY'].drop(columns='ticker')

        for h in [5, 20, 60]:
            events['exit_target'] = events['actual_entry_dt'] + pd.Timedelta(days=h)
            h_df = pd.merge_asof(events.sort_values('exit_target'), prices_long, left_on='exit_target', right_on='date', by='ticker', direction='forward')
            h_df = h_df.rename(columns={'close':'p_exit','date':'actual_exit_dt'}).dropna(subset=['p_exit'])
            
            # Benchmark normalization against SPY
            h_df = pd.merge_asof(h_df.sort_values('actual_entry_dt'), spy_long, left_on='actual_entry_dt', right_on='date', direction='nearest').rename(columns={'close':'spy_entry'})
            h_df = pd.merge_asof(h_df.sort_values('actual_exit_dt'), spy_long, left_on='actual_exit_dt', right_on='date', direction='nearest').rename(columns={'close':'spy_exit'})

            h_df['ret'] = (h_df['p_exit'] / h_df['p_entry'] - 1) * h_df['impulse']
            h_df['spy_ret'] = (h_df['spy_exit'] / h_df['spy_entry'] - 1) * h_df['impulse']
            h_df['alpha'] = h_df['ret'] - h_df['spy_ret']
            h_df['horizon'] = h
            audit_results.append(h_df[['filer', 'horizon', 'alpha', 'ret']])

        final_audit = pd.concat(audit_results)
        summary = final_audit.groupby('horizon').agg(
            n=('alpha','size'), 
            mean_alpha=('alpha','mean'), 
            std_alpha=('alpha','std')
        ).reset_index()
        summary['se_alpha'] = summary['std_alpha'] / np.sqrt(summary['n'])
        
        print("\n" + "═"*65 + "\n  PPF QUANT AUDIT v4.4 | VECTORIZED EVENT-STUDY\n" + "═"*65)
        print(summary[['horizon', 'n', 'mean_alpha', 'se_alpha']].to_string(index=False))
        print(f"\nCoverage: {len(events)} valid events | {prices.shape[1]-1} priced tickers.")

if __name__ == "__main__":
    engine = PPFEngine()
    engine.sync_senate()
    engine.sync_house("2024")
    engine.run_audit()
