import pandas as pd
import numpy as np
import logging
import sys
import time
from bisect import bisect_right

# =============================================================================
# 1. LOGGING & INITIALIZATION
# =============================================================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def run_hardened_pipeline_v2(
    input_file="fused_congressional_ledger.csv",
    output_file="congressional_velocity_v2.csv",
    churn_window_days=60
):
    """
    Executes the V2 analytical pipeline with optimized churn calculation,
    corrected metric integrity, and enhanced data hygiene.
    """
    start_time = time.time()
    logger.info(f"Starting V2 Pipeline: {input_file} | Window: {churn_window_days} days")

    # --- Step 1: Preflight & Loading ---
    try:
        # Load the ledger created in previous stages [cite: 11]
        df = pd.read_csv(input_file)
        required_cols = ["Filer", "Transaction_Date", "Amount_Low", "Amount_High", "Ticker", "Type"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return 1
    except Exception as e:
        logger.error(f"Loading failed: {e}")
        return 1

    # --- Step 2: Data Hygiene & Type Classing ---
    logger.info("[2/6] Hygiene and Pre-processing...")
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    
    # Pre-compute normalized types to avoid expensive string operations in loops
    df["_TypeNorm"] = df["Type"].astype(str).str.lower()
    df["_IsBuy"] = df["_TypeNorm"].str.contains("purchase")
    df["_IsSell"] = df["_TypeNorm"].str.contains("sale")
    
    # Midpoint and negative handling
    for c in ["Amount_Low", "Amount_High"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Calculate midpoint for volume metrics [cite: 11]
    low, high = df["Amount_Low"], df["Amount_High"]
    df["Midpoint"] = np.where(low.notna() & high.notna(), (low + high) / 2,
                      np.where(low.notna(), low,
                      np.where(high.notna(), high, np.nan)))
    
    # Log invalid amounts
    neg_count = (df["Midpoint"] < 0).sum()
    if neg_count > 0:
        logger.warning(f"Found {neg_count} negative midpoints. Clipping to 0.")
        df["Midpoint"] = df["Midpoint"].clip(lower=0)

    # Add time bucket for burstiness analysis
    df["Txn_Month"] = df["Transaction_Date"].dt.to_period("M")

    # --- Step 3: Optimized Churn & Harvester ---
    logger.info("[3/6] Executing Behavioral Harvester (Optimized)...")

    def get_churn_stats_optimized(g, window):
        """
        O(N log N) implementation using binary search.
        Counts how many purchases are followed by a sale in the same ticker within 'window' days.
        """
        valid = g[g["Transaction_Date"].notna() & g["Ticker"].notna()]
        buys = valid[valid["_IsBuy"]].sort_values("Transaction_Date")
        sells = valid[valid["_IsSell"]].sort_values("Transaction_Date")
        
        if buys.empty or sells.empty:
            return 0, len(buys), len(sells)
        
        churn_events = 0
        # Group by ticker to ensure intra-ticker churn only
        for ticker, t_buys in buys.groupby("Ticker"):
            t_sells = sells[sells["Ticker"] == ticker]
            if t_sells.empty: continue
            
            sell_dates = t_sells["Transaction_Date"].values
            for b_date in t_buys["Transaction_Date"]:
                # Window: (b_date, b_date + window]
                window_end = b_date + pd.Timedelta(days=window)
                # Find index of first sell occurring after the purchase
                idx_start = bisect_right(sell_dates, b_date)
                # Find index of first sell occurring after the window
                idx_end = bisect_right(sell_dates, window_end)
                
                # If there's at least one sell index in the window
                if idx_end > idx_start:
                    churn_events += 1
                    
        return churn_events, len(buys), len(sells)

    def harvester_v2(g):
        # Time-based filtering
        g_time = g.dropna(subset=["Txn_Month", "Midpoint"])
        
        # Optimized churn calculation
        events, n_buys, n_sells = get_churn_stats_optimized(g, churn_window_days)
        # Corrected denominator: Rate is relative to purchases, not total trades
        churn_rate = events / n_buys if n_buys > 0 else 0.0
        
        total_vol = g["Midpoint"].sum(skipna=True)
        total_trades = len(g)
        
        # Consistent time normalization policy
        if g_time.empty:
            active_months = 1
            years_active = 1/12.0
        else:
            monthly = g_time.groupby("Txn_Month")["Midpoint"].sum()
            active_months = len(monthly)
            years_active = max(active_months / 12.0, 1/12.0)
            
            # Burstiness: Max volume month vs Median volume month
            eps = 1e-9
            burst_score = float(monthly.max() / (monthly.median() + eps))

        return pd.Series({
            "Annualized_Volume": total_vol / years_active,
            "Trades_Per_Year": total_trades / years_active,
            "Burst_Score": burst_score if not g_time.empty else 1.0,
            "Churn_Events_60d": int(events),
            "Churn_Rate_60d": float(churn_rate),
            "Num_Purchases": int(n_buys),
            "Num_Sales": int(n_sells),
            "Active_Months": int(active_months),
            "Total_Volume": float(total_vol),
            "Total_Trades": int(total_trades)
        })

    # Apply harvester across all filers
    rankings = df.groupby("Filer", dropna=False).apply(harvester_v2).reset_index()

    # --- Step 4: Normalization & Indexing ---
    logger.info("[4/6] Normalizing and Indexing...")
    
    # Annualized Vol, Freq, and Burstiness get Log1p then Z-scored
    for col in ["Annualized_Volume", "Trades_Per_Year", "Burst_Score"]:
        log_col = f"log_{col}"
        rankings[log_col] = np.log1p(rankings[col])
        mu, sd = rankings[log_col].mean(), rankings[log_col].std()
        rankings[f"z_{col}"] = np.clip((rankings[log_col] - mu) / (sd if sd > 0 else 1.0), -3, 3)

    # Churn Rate: Standardize directly (no log1p) to penalize flips linearly
    mu_c, sd_c = rankings["Churn_Rate_60d"].mean(), rankings["Churn_Rate_60d"].std()
    rankings["z_Churn_Rate"] = np.clip((rankings["Churn_Rate_60d"] - mu_c) / (sd_c if sd_c > 0 else 1.0), -3, 3)

    # Composite Capital Velocity Index (CVI)
    rankings["Capital_Velocity_Index"] = (
        rankings["z_Annualized_Volume"] + 
        rankings["z_Trades_Per_Year"] + 
        rankings["z_Burst_Score"] - 
        rankings["z_Churn_Rate"]
    ) / 4.0

    # Final Rank assignment
    rankings = rankings.sort_values("Capital_Velocity_Index", ascending=False)
    rankings["Rank"] = range(1, len(rankings) + 1)

    # --- Step 5: Save & Return ---
    rankings.to_csv(output_file, index=False)
    logger.info(f"Pipeline V2 Complete. Saved to {output_file}. Total Runtime: {round(time.time()-start_time, 2)}s")
    return rankings

if __name__ == "__main__":
    results = run_hardened_pipeline_v2()
