import pandas as pd

# Load the CSV
df = pd.read_csv("sp500_scores_dynamic.csv")

# Define desired column order
cols = df.columns.tolist()

# Identify the new metric columns
new_metrics = [
    "Momentum_1m", "Momentum_3m", "Volatility_30d", "Trend_50_200",
    "Momentum_1m_Score", "Momentum_3m_Score", "Volatility_30d_Score", "Trend_50_200_Score"
]

# Reorder: place new metrics before FinalScore and Rank
if "FinalScore" in cols and "Rank" in cols:
    final_idx = cols.index("FinalScore")
    reordered = (
        cols[:final_idx] + new_metrics +
        [c for c in cols[final_idx:] if c not in new_metrics]
    )
    df = df[reordered]

# Save
df.to_csv("sp500_scores_dynamic.csv", index=False)
print("✅ Columns reordered successfully.")
