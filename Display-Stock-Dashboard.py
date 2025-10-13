import streamlit as st
import pandas as pd

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("sp500_scores_with_sectors.csv")
    return df

df = load_data()

st.set_page_config(page_title="S&P 500 Stock Scores", layout="wide")

# --- Title ---
st.title("S&P 500 Stock Scores Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filters")

# Search bar for ticker
search_ticker = st.sidebar.text_input("Search by Ticker (e.g., AAPL, NVDA):").upper().strip()

# --- Sector filter with Select All ---
sectors = df["Sector"].unique().tolist()
sectors = sorted(sectors)  # keep it tidy

# Add a "Select All" checkbox
select_all = st.sidebar.checkbox("Select All Sectors", value=True)

if select_all:
    selected_sector = st.sidebar.multiselect("Select Sector(s):", sectors, default=sectors)
else:
    selected_sector = st.sidebar.multiselect("Select Sector(s):", sectors, default=[])

sector_filter = df["Sector"].isin(selected_sector)

# --- Ranking slider ---
max_rank = int(df["Rank"].max())
selected_rank = st.sidebar.slider("Maximum Rank:", 1, max_rank, (1, 50))

# --- Apply filters ---
filtered_df = df[
    sector_filter &
    (df["Rank"].between(selected_rank[0], selected_rank[1]))
]

# Apply ticker search if entered
if search_ticker:
    filtered_df = filtered_df[filtered_df["Ticker"].str.contains(search_ticker, case=False, na=False)]

# --- Main Table ---
st.subheader("Stock Rankings")
st.dataframe(
    filtered_df.sort_values("Rank").reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

# --- Metrics / Summary ---
st.subheader("Summary Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Number of Stocks", len(filtered_df))
col2.metric("Average Final Score", f"{filtered_df['FinalScore'].mean():.3f}")
col3.metric("Best Final Score", f"{filtered_df['FinalScore'].max():.3f}")

# --- Charts ---
st.subheader("Score Distribution")

tab1, tab2, tab3, tab4 = st.tabs(["Final Score", "P/E Ratio", "Free Cash Flow", "Sharpe Ratio"])

with tab1:
    st.bar_chart(filtered_df.set_index("Ticker")["FinalScore"])

with tab2:
    st.bar_chart(filtered_df.set_index("Ticker")["P/E"])

with tab3:
    st.bar_chart(filtered_df.set_index("Ticker")["FreeCashFlow"])

with tab4:
    st.bar_chart(filtered_df.set_index("Ticker")["SharpeRatio"])

