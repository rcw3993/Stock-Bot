# backtest_quarterly.py

import yfinance as yf
import pandas as pd
import numpy as np

# --- scoring function ---
def score_stock(pe, fcf, sharpe=None):
    score = 0
    if pe is not None and pe > 0:
        score += 1 / pe  # lower PE = better
    if fcf is not None and fcf > 0:
        score += np.log1p(fcf) / 1e6  # scaled log
    if sharpe is not None:
        score += sharpe
    return score

def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]  # first table is the S&P 500 list
    tickers = df['Symbol'].tolist()
    return tickers

# --- backtest function ---
def backtest_quarterly(tickers, start, end, top_n=20):
    print(f"Downloading daily prices for {len(tickers)} stocks...")
    data = yf.download(tickers, start=start, end=end, interval="1d")["Close"]
    data = data.dropna(axis=1, how="any")  # drop tickers with missing data

    quarterly_prices = data.resample("Q").last()

    portfolio_returns = []
    benchmark_returns = []

    for i in range(len(quarterly_prices) - 1):
        quarter_start = quarterly_prices.index[i]
        quarter_end = quarterly_prices.index[i+1]

        print(f"\nQuarter {quarter_start.date()} → {quarter_end.date()}")

        # --- snapshot scores (using today's fundamentals for now) ---
        scores = {}
        for ticker in quarterly_prices.columns:
            try:
                stock = yf.Ticker(ticker)
                pe = stock.info.get("trailingPE")
                fcf = None
                if hasattr(stock, "cashflow") and "Total Cash From Operating Activities" in stock.cashflow:
                    fcf = stock.cashflow.loc["Total Cash From Operating Activities"].iloc[0]
                sharpe = None  # placeholder
                scores[ticker] = score_stock(pe, fcf, sharpe)
            except Exception:
                continue

        top_stocks = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # --- calculate returns for this quarter ---
        start_prices = data.loc[quarter_start, top_stocks]
        end_prices = data.loc[quarter_end, top_stocks]

        returns = (end_prices.values - start_prices.values) / start_prices.values
        portfolio_return = np.nanmean(returns)

        sp500_return = (data.loc[quarter_end].mean() - data.loc[quarter_start].mean()) / data.loc[quarter_start].mean()

        print(f"Portfolio return this quarter: {portfolio_return*100:.2f}% | Benchmark: {sp500_return*100:.2f}%")

        portfolio_returns.append(portfolio_return)
        benchmark_returns.append(sp500_return)

    total_portfolio = np.prod([1+r for r in portfolio_returns]) - 1
    total_benchmark = np.prod([1+r for r in benchmark_returns]) - 1

    return total_portfolio, total_benchmark, portfolio_returns, benchmark_returns


if __name__ == "__main__":
    # test with a small set first
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    portfolio, benchmark, port_rets, bench_rets = backtest_quarterly(
        tickers,
        start="2015-01-01",
        end="2017-01-01",
        top_n=3
    )

    print("\n=== Final Results ===")
    print(f"Portfolio total return: {portfolio*100:.2f}%")
    print(f"Benchmark total return: {benchmark*100:.2f}%")
