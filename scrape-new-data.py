import pandas as pd
import yfinance as yf
import numpy as np

def compute_metrics(history: pd.DataFrame) -> dict:
    """
    Given historical price data (with a 'Close' column), compute:
      - 1-month return (momentum)
      - 3-month return (momentum)
      - 30-day rolling volatility (std of daily returns)
      - Trend ratio (50-day MA vs 200-day MA)
    Returns a dict with these metrics.
    """
    metrics = {}
    history = history.dropna(subset=['Close'])
    # Compute returns
    history['Return'] = history['Close'].pct_change()
    # 1-month (~30 trading days) momentum
    if len(history) >= 30:
        metrics['Momentum_1m'] = (history['Close'].iloc[-1] / history['Close'].iloc[-30]) - 1
    else:
        metrics['Momentum_1m'] = np.nan
    # 3-month (~90 trading days) momentum
    if len(history) >= 90:
        metrics['Momentum_3m'] = (history['Close'].iloc[-1] / history['Close'].iloc[-90]) - 1
    else:
        metrics['Momentum_3m'] = np.nan
    # Volatility: standard deviation of daily returns over last 30 days
    if len(history) >= 30:
        metrics['Volatility_30d'] = history['Return'].iloc[-30:].std()
    else:
        metrics['Volatility_30d'] = np.nan
    # Trend ratio: 50-day vs 200-day moving average
    if len(history) >= 200:
        ma50 = history['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        metrics['Trend_50_200'] = ma50 / ma200
    else:
        metrics['Trend_50_200'] = np.nan
    return metrics

def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to 0-1 (min-max)."""
    minv = series.min()
    maxv = series.max()
    if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
        return pd.Series(np.nan, index=series.index)
    return (series - minv) / (maxv - minv)

def main():
    input_file = "sp500_scores_cleaned.csv"
    output_file = "sp500_scores_dynamic.csv"

    df = pd.read_csv(input_file)
    tickers = df["Ticker"].tolist()

    # Prepare storage for new metrics
    metrics_list = []
    for ticker in tickers:
        print(f"Processing {ticker} …")
        try:
            hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
            mets = compute_metrics(hist)
        except Exception as e:
            print(f"  Error for {ticker}: {e}")
            mets = {'Momentum_1m': np.nan,
                    'Momentum_3m': np.nan,
                    'Volatility_30d': np.nan,
                    'Trend_50_200': np.nan}
        mets['Ticker'] = ticker
        metrics_list.append(mets)

    metrics_df = pd.DataFrame(metrics_list).set_index("Ticker")

    # Merge with original
    df = df.set_index("Ticker")
    df = df.join(metrics_df, how="left")

    # Normalize each of the new metric columns to 0-1
    for col in ['Momentum_1m', 'Momentum_3m', 'Volatility_30d', 'Trend_50_200']:
        norm_col = f"{col}_Score"
        df[norm_col] = normalize_series(df[col])

    # Save result
    df.reset_index().to_csv(output_file, index=False)
    print(f"✅ Saved updated file to {output_file}")

if __name__ == "__main__":
    main()
