import pandas as pd
import requests

# --- Step 1: Load your scores CSV ---
df = pd.read_csv("sp500_scores.csv")

# --- Step 2: Scrape S&P 500 data from Wikipedia using requests ---
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}  # Pretend to be a browser
response = requests.get(url, headers=headers)
response.raise_for_status()  # Throw error if request fails

# Parse the first table
sp500 = pd.read_html(response.text, header=0)[0]

# --- Step 3: Build a ticker → sector map ---
sector_map = dict(zip(sp500["Symbol"], sp500["GICS Sector"]))

# --- Step 4: Map into your dataframe, defaulting to "Unknown" ---
df["Sector"] = df["Ticker"].map(sector_map).fillna("Unknown")

# --- Step 5: Save new file ---
df.to_csv("sp500_scores_with_sectors.csv", index=False)

print("✅ Done! Saved as sp500_scores_with_sectors.csv")
missing = (df["Sector"] == "Unknown").sum()
if missing > 0:
    print(f"⚠️ {missing} tickers did not match Wikipedia and were set to 'Unknown'")
