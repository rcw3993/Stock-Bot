import pandas as pd
import yfinance as yf

def load_scores(csv_file):
    return pd.read_csv(csv_file)


def backtest(scores_df, start="2015-01-01", end="2017-01-01", top_n=50):
    # Download SPY benchmark
    spy = yf.download("SPY", start=start, end=end, interval="1d")

    # Handle MultiIndex vs. single-level DataFrame
    if isinstance(spy.columns, pd.MultiIndex):
        spy = spy.loc[:, ("Adj Close", "SPY")]
    else:
        spy = spy["Adj Close"]

    spy = spy.rename("SPY")  # make Series with nice name

    # --- Align scores with price data ---
    scores_df = scores_df.copy()
    scores_df["Date"] = pd.to_datetime(scores_df["Date"])
    scores_df = scores_df.set_index("Date")
    scores_df = scores_df.loc[start:end]  # filter to backtest window

    portfolio_value = 1.0
    spy_value = 1.0

    # Quarterly rebalancing
    for quarter_start, quarter_scores in scores_df.groupby(pd.Grouper(freq="Q")):
        if quarter_scores.empty:
            continue

        # Pick top N tickers
        top_tickers = (
            quarter_scores.drop(columns="Date", errors="ignore")
            .iloc[-1]  # take last scores of quarter
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )

        # Download prices for portfolio tickers
        prices = yf.download(top_tickers, start=quarter_start, end=quarter_start + pd.DateOffset(months=3), interval="1d")

        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices["Adj Close"]

        # Calculate returns
        returns = prices.pct_change().dropna()
        if returns.empty:
            continue

        # Portfolio return = equal weight average
        portfolio_return = (returns.mean(axis=1) + 1).prod() - 1

        # SPY return over same window
        spy_window = spy.loc[quarter_start:quarter_start + pd.DateOffset(months=3)]
        spy_return = (spy_window.pct_change().dropna() + 1).prod() - 1 if not spy_window.empty else 0

        # Update portfolio & spy values
        portfolio_value *= 1 + portfolio_return
        spy_value *= 1 + spy_return

        print(f"Quarter {quarter_start.date()} | Portfolio: {portfolio_return*100:.2f}% | SPY: {spy_return*100:.2f}%")

    print(f"\nFinal portfolio value: {portfolio_value:.2f}")
    print(f"Final SPY value: {spy_value:.2f}")

def main():
    scores_df = load_scores("sp500_scores.csv")
    # Example: backtest from 2015 → 2017
    backtest(scores_df, start="2015-01-01", end="2017-01-01", top_n=50)

if __name__ == "__main__":
    main()
