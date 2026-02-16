import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
TOP_N = 5
INITIAL_CAPITAL = 100000
BENCHMARK = "SPY"

TX_PATH = "data/processed/transactions_enriched.csv"

# -----------------------------
# LOAD DISCLOSURES
# -----------------------------
tx = pd.read_csv(TX_PATH, parse_dates=["filing_date"])

# Use BUY only
tx = tx[tx["transaction_type"] == "BUY"].copy()

if tx.empty:
    raise ValueError("No BUY transactions found.")

# Convert filing_date to quarter period
tx["quarter"] = tx["filing_date"].dt.to_period("Q")

# Rank within each quarter by amount_max_usd
tx["amount_rank"] = tx.groupby("quarter")["amount_max_usd"].rank(
    method="first", ascending=False
)

# Select top N per quarter
selected = tx[tx["amount_rank"] <= TOP_N]

# Build quarter → tickers mapping
quarter_portfolios = (
    selected.groupby("quarter")["asset_id"]
    .apply(lambda x: list(x.unique()))
    .sort_index()
)

print("\nQuarterly Portfolios:")
print(quarter_portfolios)

# -----------------------------
# GET PRICE DATA
# -----------------------------
all_tickers = sorted(set(selected["asset_id"].unique()) | {BENCHMARK})

start_date = tx["filing_date"].min() - pd.Timedelta(days=10)
end_date = datetime.today()

prices = yf.download(
    all_tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)["Close"]

prices = prices.dropna(how="all")

# -----------------------------
# BACKTEST
# -----------------------------
portfolio_value = INITIAL_CAPITAL
portfolio_history = []
benchmark_history = []

dates = prices.index

for i, (quarter, tickers) in enumerate(quarter_portfolios.items()):
    q_start = quarter.start_time
    if i + 1 < len(quarter_portfolios):
        q_end = list(quarter_portfolios.index)[i + 1].start_time
    else:
        q_end = dates[-1]

    period_prices = prices.loc[(prices.index >= q_start) & (prices.index < q_end)]

    if period_prices.empty:
        continue

    valid_tickers = [t for t in tickers if t in period_prices.columns]
    if not valid_tickers:
        continue

    returns = period_prices[valid_tickers].pct_change().mean(axis=1).fillna(0)
    bench_returns = period_prices[BENCHMARK].pct_change().fillna(0)

    for r, br in zip(returns, bench_returns):
        portfolio_value *= (1 + r)
        benchmark_value = INITIAL_CAPITAL * (1 + br)

        portfolio_history.append(portfolio_value)
        benchmark_history.append(benchmark_value)

# -----------------------------
# RESULTS
# -----------------------------
portfolio_series = pd.Series(portfolio_history)
benchmark_series = pd.Series(benchmark_history[:len(portfolio_series)])

cagr = (portfolio_series.iloc[-1] / INITIAL_CAPITAL) ** (
    252 / len(portfolio_series)
) - 1

print("\nFinal Portfolio Value:", round(portfolio_series.iloc[-1], 2))
print("CAGR:", round(cagr * 100, 2), "%")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(portfolio_series.values, label="Congress Strategy")
plt.plot(benchmark_series.values, label="SPY")
plt.legend()
plt.title("Congress Disclosure Strategy vs SPY")
plt.show()
