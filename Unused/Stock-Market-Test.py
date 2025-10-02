"""
sp500_scoring.py

Fetch S&P 500 tickers from Wikipedia, pull fundamentals & prices with yfinance,
compute P/E, Free Cash Flow, 4-year Sharpe (monthly), normalize scores,
and save a ranked CSV.

Requirements:
  pip install yfinance pandas numpy tqdm requests tabulate

Run:
  python sp500_scoring.py
"""

import time
import math
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

# -----------------------
# SETTINGS
# -----------------------
YEARS = 4  # lookback for Sharpe (years)
RISK_FREE_RATE = 0.02  # annual risk-free rate (approx)
FINAL_TOP_N = 50  # example: how many top stocks to show / consider
OUTPUT_CSV = "sp500_scores.csv"
WIKI_TABLE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Weights for combining scores (tweak these)
WEIGHT_FCF = 0.4
WEIGHT_PE = 0.3
WEIGHT_SHARPE = 0.3

# Timeout / polite pause to avoid hammering APIs
PER_TICKER_PAUSE = 0.2  # seconds between .info calls (yfinance can be slow)


# -----------------------
# HELPERS
# -----------------------
def fetch_sp500_tickers():
    """Fetch S&P 500 tickers and sectors from a stable CSV source."""
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    # Columns are: Symbol, Name, Sector
    df = df.rename(columns={"Symbol": "Ticker", "Sector": "Sector"})
    df["Ticker"] = df["Ticker"].str.replace('.', '-', regex=False)
    return df

def compute_sharpe_monthly(ticker, years=YEARS, rf=RISK_FREE_RATE):
    """
    Compute approximate Sharpe ratio using monthly returns for 'years' back.
    Returns NaN on failure. Annualized Sharpe = (annual_return - rf)/annual_vol
    """
    try:
        period = f"{years}y"
        hist = yf.Ticker(ticker).history(period=period, interval="1mo", actions=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return np.nan
        # monthly returns
        returns = hist["Close"].pct_change().dropna()
        if returns.empty:
            return np.nan
        avg_monthly = returns.mean()
        std_monthly = returns.std()
        # annualize
        annual_return = avg_monthly * 12
        annual_vol = std_monthly * math.sqrt(12)
        if annual_vol == 0 or math.isnan(annual_vol):
            return np.nan
        sharpe = (annual_return - rf) / annual_vol
        return float(sharpe)
    except Exception:
        return np.nan


def safe_info_fetch(ticker):
    """
    Fetch fundamental info via yfinance.Ticker.info with error handling.
    Returns dict with keys 'trailingPE' and 'freeCashflow' (may be NaN).
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        pe = info.get("trailingPE", np.nan)
        fcf = info.get("freeCashflow", np.nan)
        # sometimes freeCashflow is string or None; coerce
        if isinstance(fcf, str):
            try:
                fcf = float(fcf)
            except Exception:
                fcf = np.nan
        return {"trailingPE": pe if pe is not None else np.nan,
                "freeCashflow": fcf if fcf is not None else np.nan}
    except Exception:
        return {"trailingPE": np.nan, "freeCashflow": np.nan}


# -----------------------
# MAIN PIPELINE
# -----------------------
def main():
    print("Fetching S&P 500 tickers from Wikipedia...")
    tickers_df = fetch_sp500_tickers()
    print(f"Found {len(tickers_df)} tickers (including duplicates if any).")

    records = []
    failed = []

    # iterate tickers
    for idx, row in tqdm(tickers_df.iterrows(), total=len(tickers_df), desc="Tickers"):
        ticker = row["Ticker"]
        sector = row.get("Sector", None)
        # fundamentals
        info = safe_info_fetch(ticker)
        pe = info["trailingPE"]
        fcf = info["freeCashflow"]

        # sharpe
        sharpe = compute_sharpe_monthly(ticker)

        records.append({
            "Ticker": ticker,
            "Sector": sector,
            "P/E": pe,
            "FreeCashFlow": fcf,
            "SharpeRatio": sharpe
        })

        # be polite — some rate limiting protection
        time.sleep(PER_TICKER_PAUSE)

    df = pd.DataFrame(records)

    # -----------------------
    # CLEANING & TRANSFORMS
    # -----------------------
    # Convert numeric columns to floats, handle zeros/negatives as needed
    for col in ["P/E", "FreeCashFlow", "SharpeRatio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional: replace extreme FCF outliers (very large) by applying log scale for scoring
    # We'll compute a transformed column for FCF to reduce effect of huge companies
    df["FreeCashFlow_Pos"] = df["FreeCashFlow"].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan)
    # Use log(1 + FCF) to reduce skew
    df["FreeCashFlow_Log"] = df["FreeCashFlow_Pos"].apply(lambda x: np.log1p(x) if pd.notna(x) else np.nan)

    # For P/E, extremely large/unrealistic P/E (or negative) can be troublesome.
    # We'll treat P/E <= 0 or extremely high as NaN for the normalization step,
    # then later we can re-score/penalize them.
    df["P/E_Pos"] = df["P/E"].apply(lambda x: x if pd.notna(x) and x > 0 and x < 500 else np.nan)

    # For Sharpe, NaNs remain. (There may be many NaNs for thinly traded stocks; S&P500 is okay)
    # -----------------------
    # NORMALIZE (min-max) to 0-1
    # -----------------------
    # We will create *score* columns:
    # - FreeCashFlow_Score (higher better)
    # - P/E_Score (lower better -> flipped)
    # - SharpeRatio_Score (higher better)
    def minmax(series):
        s = series.dropna()
        if s.empty:
            return pd.Series(np.nan, index=series.index)
        mn = s.min()
        mx = s.max()
        if mx == mn:
            # all same -> map to 0.5
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    df["FreeCashFlow_Score"] = minmax(df["FreeCashFlow_Log"])
    df["P/E_Score_raw"] = minmax(df["P/E_Pos"])
    df["P/E_Score"] = 1 - df["P/E_Score_raw"]  # lower P/E is better
    df["SharpeRatio_Score"] = minmax(df["SharpeRatio"])

    # If a metric was NaN, score remains NaN. Decide replacement policy:
    # Option A: treat NaN as worst (0.0)
    # Option B: drop rows with too many NaNs
    # We'll set NaN scores to 0 to be conservative.
    for sc in ["FreeCashFlow_Score", "P/E_Score", "SharpeRatio_Score"]:
        df[sc] = df[sc].fillna(0.0)

    # -----------------------
    # COMBINE SCORES
    # -----------------------
    df["FinalScore"] = (
        WEIGHT_FCF * df["FreeCashFlow_Score"] +
        WEIGHT_PE * df["P/E_Score"] +
        WEIGHT_SHARPE * df["SharpeRatio_Score"]
    )

    df.sort_values("FinalScore", ascending=False, inplace=True)

    # Add rank
    df["Rank"] = range(1, len(df) + 1)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")

    # Pretty print the top N
    display_cols = ["Rank", "Ticker", "Sector", "P/E", "FreeCashFlow", "SharpeRatio",
                    "P/E_Score", "FreeCashFlow_Score", "SharpeRatio_Score", "FinalScore"]
    top_df = df[display_cols].head(FINAL_TOP_N)
    print("\nTop results:")
    print(tabulate(top_df, headers="keys", tablefmt="pretty", floatfmt=".4f"))

    # Also print some simple sector exposure info for the selected top group
    top_sector_counts = top_df["Sector"].value_counts().head(10)
    print("\nSector counts in top selection:")
    print(tabulate(top_sector_counts.reset_index().values, headers=["Sector", "Count"], tablefmt="pretty"))

    print("\nDone.")


if __name__ == "__main__":
    main()
