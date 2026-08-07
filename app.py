"""
Inventory & Demand Forecasting Dashboard
-----------------------------------------
A general-purpose tool: upload ANY company's sales/stock data and get
demand forecasts, reorder points, safety stock, and stockout risk flags.

Built by Francisca [Tilburg IBA] as a portfolio project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pathlib import Path

# Always resolve paths relative to THIS script's location, regardless of
# which folder the terminal happened to be in when it was launched.
APP_DIR = Path(__file__).parent
DEMO_DATA_PATH = APP_DIR / "sample_data" / "demo_data.csv"

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Inventory & Demand Forecasting Dashboard",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Inventory & Demand Forecasting Dashboard")
st.caption(
    "Upload any company's historical sales/stock data and get demand "
    "forecasts, reorder points, and stockout risk — built for retail, "
    "e-commerce, or manufacturing inventory planning."
)

# ----------------------------------------------------------------------
# STEP 1: DATA INPUT
# ----------------------------------------------------------------------
st.sidebar.header("1. Load your data")

data_source = st.sidebar.radio(
    "Choose a data source",
    ["Use demo dataset", "Upload my own CSV"],
)

if data_source == "Upload my own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("👈 Upload a CSV in the sidebar to get started, or switch to the demo dataset.")
        st.stop()
    raw_df = pd.read_csv(uploaded_file)
else:
    if not DEMO_DATA_PATH.exists():
        st.error(
            f"Demo file not found at: {DEMO_DATA_PATH}\n\n"
            "Make sure the 'sample_data' folder (containing demo_data.csv) "
            "sits directly inside the same folder as app.py."
        )
        st.stop()
    raw_df = pd.read_csv(DEMO_DATA_PATH)
    st.sidebar.success("Using built-in demo dataset (retail inventory, 2023-2024).")

st.subheader("Raw data preview")
st.dataframe(raw_df.head(10), use_container_width=True)

# ----------------------------------------------------------------------
# STEP 2: COLUMN MAPPING (this is what makes the app work for ANY company)
# ----------------------------------------------------------------------
st.sidebar.header("2. Map your columns")
st.sidebar.caption("Tell the app which column in YOUR file means what.")

columns = list(raw_df.columns)


def guess(col_options, keywords):
    for kw in keywords:
        for c in col_options:
            if kw in c.lower():
                return c
    return col_options[0]


date_col = st.sidebar.selectbox("Date column", columns, index=columns.index(guess(columns, ["date"])))
product_col = st.sidebar.selectbox("Product / SKU column", columns, index=columns.index(guess(columns, ["product", "sku", "item"])))
demand_col = st.sidebar.selectbox("Demand / units sold column", columns, index=columns.index(guess(columns, ["demand", "sales", "sold", "qty"])))
stock_col_options = ["(none)"] + columns
stock_col = st.sidebar.selectbox("Current stock column (optional)", stock_col_options,
                                  index=(stock_col_options.index(guess(columns, ["stock", "inventory"]))
                                         if guess(columns, ["stock", "inventory"]) in stock_col_options else 0))

# Clean & standardize
df = raw_df.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, product_col, demand_col])
df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")
df = df.dropna(subset=[demand_col])
df = df.sort_values(date_col)

products = sorted(df[product_col].unique().tolist())

# ----------------------------------------------------------------------
# STEP 3: FORECAST SETTINGS
# ----------------------------------------------------------------------
st.sidebar.header("3. Forecast settings")
selected_product = st.sidebar.selectbox("Product to analyze", products)
forecast_days = st.sidebar.slider("Forecast horizon (days)", 7, 60, 14)
lead_time_days = st.sidebar.slider("Supplier lead time (days)", 1, 30, 7)
service_level = st.sidebar.select_slider(
    "Target service level (chance of NOT running out of stock)",
    options=[0.90, 0.95, 0.975, 0.99],
    value=0.95,
)

# z-score for service level (standard safety stock formula)
Z_TABLE = {0.90: 1.28, 0.95: 1.645, 0.975: 1.96, 0.99: 2.33}
z = Z_TABLE[service_level]

# ----------------------------------------------------------------------
# FORECASTING FUNCTION
# ----------------------------------------------------------------------
def forecast_demand(series: pd.Series, periods: int):
    """
    Forecast future demand using Holt's Exponential Smoothing (trend-aware).
    Falls back to a simple moving-average trend if there isn't enough data.
    """
    series = series.reset_index(drop=True)
    if len(series) >= 14:
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            forecast = fit.forecast(periods)
            fitted_values = fit.fittedvalues
            return forecast.values, fitted_values.values
        except Exception:
            pass
    # Fallback: flat average
    avg = series.mean()
    return np.repeat(avg, periods), np.repeat(avg, len(series))


product_df = df[df[product_col] == selected_product].groupby(date_col, as_index=False)[demand_col].sum()
product_df = product_df.set_index(date_col).asfreq("D").ffill().reset_index()

demand_series = product_df[demand_col]
forecast_values, fitted_values = forecast_demand(demand_series, forecast_days)

last_date = product_df[date_col].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

# Backtest accuracy: compare fitted vs actual on historical data (MAPE)
fitted_len = min(len(fitted_values), len(demand_series))
actual_tail = demand_series.values[-fitted_len:]
fitted_tail = fitted_values[-fitted_len:]
nonzero_mask = actual_tail != 0
mape = np.mean(np.abs((actual_tail[nonzero_mask] - fitted_tail[nonzero_mask]) / actual_tail[nonzero_mask])) * 100 if nonzero_mask.any() else np.nan

# ----------------------------------------------------------------------
# INVENTORY METRICS
# ----------------------------------------------------------------------
avg_daily_demand = demand_series.tail(30).mean()
std_daily_demand = demand_series.tail(30).std()

safety_stock = z * std_daily_demand * np.sqrt(lead_time_days)
reorder_point = (avg_daily_demand * lead_time_days) + safety_stock

if stock_col != "(none)":
    current_stock = df[df[product_col] == selected_product].sort_values(date_col)[stock_col].iloc[-1]
else:
    current_stock = None

# ----------------------------------------------------------------------
# LAYOUT: KPI ROW
# ----------------------------------------------------------------------
st.divider()
st.subheader(f"Forecast & Inventory Plan — Product {selected_product}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Avg daily demand (last 30d)", f"{avg_daily_demand:.1f}")
kpi2.metric("Reorder point", f"{reorder_point:.0f} units")
kpi3.metric("Safety stock", f"{safety_stock:.0f} units")
kpi4.metric("Forecast accuracy (MAPE)", f"{mape:.1f}%" if not np.isnan(mape) else "N/A")

if current_stock is not None:
    st.divider()
    if current_stock <= reorder_point:
        st.error(
            f"⚠️ **Stockout risk**: current stock ({current_stock:.0f}) is at or below "
            f"the reorder point ({reorder_point:.0f}). Reorder now."
        )
    else:
        buffer_days = (current_stock - reorder_point) / avg_daily_demand if avg_daily_demand > 0 else np.nan
        st.success(
            f"✅ Stock healthy: {current_stock:.0f} units on hand, "
            f"~{buffer_days:.0f} days until you hit the reorder point."
        )

# ----------------------------------------------------------------------
# CHART: HISTORY + FORECAST
# ----------------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=product_df[date_col], y=demand_series, name="Actual demand", line=dict(color="#4C78A8")))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_values, name="Forecast", line=dict(color="#F58518", dash="dash")))
fig.add_hline(y=reorder_point, line_dash="dot", line_color="red", annotation_text="Reorder point")
fig.update_layout(
    xaxis_title="Date", yaxis_title="Units demanded",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# ALL-PRODUCTS OVERVIEW TABLE
# ----------------------------------------------------------------------
st.divider()
st.subheader("All products at a glance")

overview_rows = []
for p in products:
    p_series = df[df[product_col] == p].groupby(date_col)[demand_col].sum()
    p_avg = p_series.tail(30).mean()
    p_std = p_series.tail(30).std()
    p_safety = z * p_std * np.sqrt(lead_time_days)
    p_rop = (p_avg * lead_time_days) + p_safety
    row = {"Product": p, "Avg daily demand": round(p_avg, 1), "Reorder point": round(p_rop, 0), "Safety stock": round(p_safety, 0)}
    if stock_col != "(none)":
        p_stock = df[df[product_col] == p].sort_values(date_col)[stock_col].iloc[-1]
        row["Current stock"] = p_stock
        row["Status"] = "🔴 Reorder now" if p_stock <= p_rop else "🟢 OK"
    overview_rows.append(row)

st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Methodology: demand forecast uses Holt's Exponential Smoothing (trend-aware). "
    "Reorder point = (avg daily demand × lead time) + safety stock. "
    "Safety stock = Z-score(service level) × std. dev. of demand × √(lead time)."
)
