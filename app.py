import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Setting page configuration
st.set_page_config(page_title="Marketing Campaign Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Utility function to find columns by keywords
def find_col(cols, keywords):
    """Return first column in cols that contains any of the keywords (case-insensitive)."""
    cols_lower = [c.lower() for c in cols]
    for kw in keywords:
        kw = kw.lower()
        for i, c in enumerate(cols_lower):
            if kw in c:
                return cols[i]
    return None

# Load and preprocess data
@st.cache_data
def load_and_clean(path="data/marketing_campaign.csv"):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read file at '{path}'. Ensure the file exists. Error: {e}")
        return None

    df = df.copy()

    # Finding important columns by keywords
    col_date = find_col(df.columns, ["date", "day", "campaign_date"])
    col_channel = find_col(df.columns, ["channel", "channel_used", "channel_name"])
    col_segment = find_col(df.columns, ["segment", "customer_segment", "segment_name"])
    col_roi = find_col(df.columns, ["roi"])
    col_conv_rate = find_col(df.columns, ["conversion_rate", "conv_rate", "conversion"])
    col_engagement = find_col(df.columns, ["engagement", "engagement_score", "engagementscore"])
    col_acq_cost = find_col(df.columns, ["acquisition_cost", "acq_cost", "acquisition"])
    col_clicks = find_col(df.columns, ["click", "clicks"])
    col_impr = find_col(df.columns, ["impression", "impressions", "impr"])

    # Standardizing column names
    rename_map = {}
    if col_date: rename_map[col_date] = "Date"
    if col_channel: rename_map[col_channel] = "Channel"
    if col_segment: rename_map[col_segment] = "Segment"
    if col_roi: rename_map[col_roi] = "ROI"
    if col_conv_rate: rename_map[col_conv_rate] = "Conversion_Rate"
    if col_engagement: rename_map[col_engagement] = "Engagement_Score"
    if col_acq_cost: rename_map[col_acq_cost] = "Acquisition_Cost"
    if col_clicks: rename_map[col_clicks] = "Clicks"
    if col_impr: rename_map[col_impr] = "Impressions"

    df.rename(columns=rename_map, inplace=True)

    # Converting Date to datetime
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        except Exception as e:
            st.warning(f"Date parsing failed: {e}. Some dates may be invalid.")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Cleaning and converting numeric columns
    numeric_cols = ["ROI", "Conversion_Rate", "Engagement_Score", "Acquisition_Cost", "Clicks", "Impressions"]
    for c in numeric_cols:
        if c in df.columns:
            # Removing dollar signs, commas, and % signs
            df[c] = df[c].astype(str).str.replace("[\$,%]", "", regex=True).str.strip()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Converting Conversion_Rate to percentage if between 0-1
            if c == "Conversion_Rate" and df[c].max(skipna=True) <= 1.0:
                df[c] = df[c] * 100

    # Estimating Conversions
    if "Conversions" not in df.columns and "Clicks" in df.columns and "Conversion_Rate" in df.columns:
        df["Conversions"] = (df["Clicks"] * (df["Conversion_Rate"] / 100)).round(0)

    # Estimating CTR
    if "CTR" not in df.columns and "Clicks" in df.columns and "Impressions" in df.columns:
        df["CTR"] = (df["Clicks"] / df["Impressions"] * 100).round(2)

    return df

# Loading data
df = load_and_clean()
if df is None:
    st.error("Data loading failed. Please check the file path and format.")
    st.stop()

# Sidebar: Filters
st.sidebar.title("Filters")

# Channel filter
channels = ["All"] + sorted(df["Channel"].dropna().unique().tolist()) if "Channel" in df.columns else ["All"]
selected_channels = st.sidebar.multiselect("Channel", channels, default=["All"])

# Segment filter
segments = ["All"] + sorted(df["Segment"].dropna().unique().tolist()) if "Segment" in df.columns else ["All"]
selected_segments = st.sidebar.multiselect("Customer Segment", segments, default=["All"])

# Date range filter
if df["Date"].notna().any():
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
else:
    date_range = None
    st.sidebar.warning("No valid date column found. Date filter disabled.")

# Metric selector for visualizations
metric_options = [c for c in ["ROI", "CTR", "Conversion_Rate", "Engagement_Score", "Acquisition_Cost"] if c in df.columns] or ["ROI"]
selected_metric = st.sidebar.selectbox("Main Metric for Visualizations", options=metric_options)

# Applying filters
df_filtered = df.copy()
if selected_channels and "All" not in selected_channels and "Channel" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Channel"].isin(selected_channels)]
if selected_segments and "All" not in selected_segments and "Segment" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Segment"].isin(selected_segments)]
if date_range and df_filtered["Date"].notna().any():
    start, end = date_range
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_filtered = df_filtered[(df_filtered["Date"] >= start_dt) & (df_filtered["Date"] <= end_dt)]

# Title and Introduction
st.title("📊 Marketing Campaign Analysis Dashboard")
st.write("Interactive dashboard to analyze campaign performance metrics (ROI, CTR, Conversion Rate, Engagement, Acquisition Cost). Use the sidebar to filter and explore data.")

# KPI Cards
st.write("### Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
kpi_metrics = [
    ("Total Conversions", "Conversions", lambda x: f"{int(x):,}" if not np.isnan(x) else "—"),
    ("Avg ROI", "ROI", lambda x: f"{x:.2f}" if not np.isnan(x) else "—"),
    ("Avg CTR (%)", "CTR", lambda x: f"{x:.2f}%" if not np.isnan(x) else "—"),
    ("Avg Engagement", "Engagement_Score", lambda x: f"{x:.2f}" if not np.isnan(x) else "—")
]
with col1:
    value = df_filtered["Conversions"].sum() if "Conversions" in df_filtered.columns else None
    st.metric(label=kpi_metrics[0][0], value=kpi_metrics[0][2](value))
with col2:
    value = df_filtered["ROI"].mean() if "ROI" in df_filtered.columns else None
    st.metric(label=kpi_metrics[1][0], value=kpi_metrics[1][2](value))
with col3:
    value = df_filtered["CTR"].mean() if "CTR" in df_filtered.columns else None
    st.metric(label=kpi_metrics[2][0], value=kpi_metrics[2][2](value))
with col4:
    value = df_filtered["Engagement_Score"].mean() if "Engagement_Score" in df_filtered.columns else None
    st.metric(label=kpi_metrics[3][0], value=kpi_metrics[3][2](value))

# Main Visualizations
st.write("### Key Visualizations")
col1, col2 = st.columns((3, 2))

with col1:
    # Trend over time
    st.subheader("Metric Trend Over Time")
    if df_filtered["Date"].notna().any() and selected_metric in df_filtered.columns:
        df_time = df_filtered.set_index("Date").resample("M").mean(numeric_only=True).reset_index()
        fig_trend = px.line(
            df_time,
            x="Date",
            y=selected_metric,
            title=f"{selected_metric} Over Time",
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No valid date or metric data available for trend visualization.")

    # Average by Channel
    st.subheader("Average by Channel")
    if "Channel" in df_filtered.columns and selected_metric in df_filtered.columns:
        by_ch = df_filtered.groupby("Channel")[selected_metric].mean().reset_index().sort_values(selected_metric, ascending=False)
        fig_bar = px.bar(
            by_ch,
            x="Channel",
            y=selected_metric,
            title=f"Average {selected_metric} per Channel",
            text=by_ch[selected_metric].round(2)
        )
        fig_bar.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Channel or metric data not available for bar chart.")

with col2:
    # Channel Share (Pie Chart)
    st.subheader("Channel Share by Conversions")
    if "Channel" in df_filtered.columns and "Conversions" in df_filtered.columns:
        ch_share = df_filtered.groupby("Channel")["Conversions"].sum().reset_index()
        fig_pie = px.pie(
            ch_share,
            names="Channel",
            values="Conversions",
            title="Share by Conversions",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Channel or Conversions data not available for pie chart.")

    # Box Plot for Metric Distribution
    st.subheader("Metric Distribution")
    if "Channel" in df_filtered.columns and selected_metric in df_filtered.columns:
        fig_box = px.box(
            df_filtered,
            x="Channel",
            y=selected_metric,
            title=f"Distribution of {selected_metric} by Channel"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Channel or metric data not available for box plot.")

# Scatter Plot with Log Scale Option
st.write("### ROI vs Engagement Insights")
use_log_scale = st.checkbox("Use Log Scale for Scatter Plot", value=False)
if "ROI" in df_filtered.columns and "Engagement_Score" in df_filtered.columns:
    scatter_df = df_filtered.dropna(subset=["ROI", "Engagement_Score"]).sample(n=min(200, len(df_filtered)), random_state=42)
    if scatter_df.shape[0] > 0:
        size_col = "Conversion_Rate" if "Conversion_Rate" in scatter_df.columns else ("Clicks" if "Clicks" in scatter_df.columns else None)
        fig_scatter = px.scatter(
            scatter_df,
            x="Engagement_Score",
            y="ROI",
            size=size_col,
            color="Channel" if "Channel" in scatter_df.columns else None,
            hover_data=["Date", "Segment"],
            title="ROI vs Engagement (bubble size = Conversion Rate or Clicks)",
            log_y=use_log_scale
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Not enough data for scatter plot (ROI & Engagement).")
else:
    st.info("ROI or Engagement_Score columns not found in data.")

# Data Preview and Download
st.write("### Data Preview & Download")
st.write("Preview of filtered data (top 10 rows):")
st.dataframe(df_filtered.head(10))
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Data (CSV)",
    data=csv,
    file_name="marketing_campaign_filtered.csv",
    mime="text/csv"
)

# Recommendations
st.write("### Strategic Recommendations")
st.write("""
- **Prioritize Google Ads and Email**: These channels consistently show high ROI and Conversion Rates, making them ideal for budget allocation.
- **Optimize Underperforming Channels**: Enhance content and targeting for Instagram, YouTube, and Website to improve Engagement and CTR.
- **Leverage Seasonal Trends**: Allocate larger budgets in high-ROI months (e.g., December: 3,475 ROI, July, May) and adjust strategies in low periods like February (3,057 ROI).
- **Balance Cost and Value**: Email offers high ROI with lower acquisition costs, while Google Ads justifies higher costs with strong returns.
- **A/B Test Creatives**: Experiment with ad copies and visuals on mid-performing channels (Facebook, YouTube) to boost CTR and conversions.
""")