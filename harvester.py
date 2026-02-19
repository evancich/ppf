import pandas as pd
import numpy as np

# Load the dataset
# Ensure the file path matches your current environment
file_path = 'fused_congressional_ledger.csv'
df = pd.read_csv(file_path)

# 1. Pre-processing & Midpoint Calculation
# We use the midpoint of reported ranges to avoid bias from wide bands.
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
df['Midpoint'] = (df['Amount_Low'] + df['Amount_High']) / 2

def activity_intelligence_harvester(group):
    """
    Calculates behavioral metrics for each filer.
    """
    # Define Time Exposure
    dates = group['Transaction_Date'].dropna().sort_values()
    if len(dates) < 2:
        years = 1.0  # Default for single-trade filers to avoid division errors
    else:
        # Measure years of active exposure in the dataset
        years = max((dates.max() - dates.min()).days, 1) / 365.25
    
    # Volume Metrics
    total_volume = group['Midpoint'].sum()
    annualized_volume = total_volume / years
    
    # Burstiness Score: Measures episodic vs. steady trading.
    # Ratio of Max Monthly Volume to Average Monthly Volume
    monthly_series = group.groupby(group['Transaction_Date'].dt.to_period('M'))['Midpoint'].sum()
    burst_score = monthly_series.max() / monthly_series.mean() if not monthly_series.empty else 1.0
    
    # Trade Cadence
    trades_per_year = len(group) / years
    
    return pd.Series({
        'Annualized_Volume': annualized_volume,
        'Trades_Per_Year': trades_per_year,
        'Burst_Score': burst_score,
        'Total_Trades': len(group),
        'Years_Active': round(years, 2)
    })

# Run the group-by analysis
rankings = df.groupby('Filer').apply(activity_intelligence_harvester).reset_index()

# 2. Capital Velocity Index (CVI) Normalization
# Creates a composite Z-score to rank overall intensity.
metrics_to_normalize = ['Annualized_Volume', 'Trades_Per_Year', 'Burst_Score']
for col in metrics_to_normalize:
    rankings[f'z_{col}'] = (rankings[col] - rankings[col].mean()) / rankings[col].std()

# Combine normalized scores into the final Index
rankings['Capital_Velocity_Index'] = rankings[[f'z_{m}' for m in metrics_to_normalize]].mean(axis=1)

# 3. Output Generation
rankings = rankings.sort_values('Capital_Velocity_Index', ascending=False)
rankings['Rank'] = range(1, len(rankings) + 1)

# Save the updated analytics report
output_name = 'congressional_activity_intelligence.csv'
rankings.to_csv(output_name, index=False)

# Display the Top 10 High-Velocity Traders
top_10 = rankings[['Rank', 'Filer', 'Annualized_Volume', 'Burst_Score', 'Capital_Velocity_Index']].head(10)
print("--- Top 10 Congressional Traders by Capital Velocity ---")
print(top_10.to_string(index=False))
