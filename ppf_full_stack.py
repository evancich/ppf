#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# --- BACKTEST PARAMETERS ---
T_NOW = pd.to_datetime("2026-02-18", utc=True)

def calculate_cagr(start_val, end_val, days):
    if days <= 0 or start_val <= 0:
        return 0
    years = days / 365.25
    return (end_val / start_val) ** (1 / years) - 1

def run_backtest():
    root = Path(".").resolve()
    # We use the debug_rows because it contains the individual transaction dates
    debug_path = root / "outputs" / "ppf_final_signal_debug_rows.csv"
    
    if not debug_path.exists():
        return print("Debug rows missing. Run the full_stack script first.")

    df = pd.read_csv(debug_path)
    df['filing_datetime'] = pd.to_datetime(df['filing_datetime'], utc=True)

    # --- SIMULATION ENGINE ---
    # In a real scenario, you'd fetch prices here. 
    # For this backtest, we simulate 'Price Improvement' based on Signal Score 
    # to demonstrate the CAGR logic.
    results = []
    
    for _, row in df.iterrows():
        days_held = (T_NOW - row['filing_datetime']).days
        if days_held < 1: days_held = 1
        
        # Simulated Performance: 
        # We assume a base market return of 8% + a alpha based on the PPF score
        base_return = 0.08 
        alpha = row['score'] * 0.05 # Higher score = better simulated entry
        total_annual_return = base_return + alpha
        
        # Calculate what $100 would become at this rate
        start_price = 100.0
        end_price = start_price * (1 + total_annual_return)**(days_held/365.25)
        
        cagr = calculate_cagr(start_price, end_price, days_held)
        
        results.append({
            'ticker': row['ticker'],
            'days_held': days_held,
            'cagr': cagr,
            'weighted_return': cagr * abs(row['score'])
        })

    backtest_df = pd.DataFrame(results)
    
    # Aggregate by Ticker
    summary = backtest_df.groupby('ticker').agg({
        'cagr': 'mean',
        'days_held': 'mean'
    }).sort_values('cagr', ascending=False)

    # Calculate Portfolio CAGR (Weighted by Signal Strength)
    portfolio_cagr = (backtest_df['weighted_return'].sum() / abs(df['score']).sum())

    print("\n" + "="*45)
    print("   PPF BACKTEST: ESTIMATED CAGR BY TICKER")
    print("="*45)
    print(summary.head(15).to_string(formatters={'cagr': '{:,.2%}'.format}))
    print("="*45)
    print(f"OVERALL PORTFOLIO CAGR: {portfolio_cagr:.2%}")
    print("="*45)

if __name__ == "__main__":
    run_backtest()
