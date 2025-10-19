import pandas as pd

# Input and output filenames
input_file = "sp500_scores_with_sectors.csv"
output_file = "sp500_scores_cleaned.csv"

# Columns to keep
keep_columns = [
    "Ticker",
    "Sector",
    "FreeCashFlow_Score",
    "P/E_Score_raw",
    "P/E_Score",
    "SharpeRatio_Score",
    "FinalScore",
    "Rank"
]

# Load, filter, and save
df = pd.read_csv(input_file)
df = df[keep_columns]
df.to_csv(output_file, index=False)

print(f"✅ Cleaned CSV saved as {output_file}")