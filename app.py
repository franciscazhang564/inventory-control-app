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
# PAGE CONFIG + VISUAL THEME
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Inventory & Demand Forecasting Dashboard",
    page_icon="📦",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2F4B7C 50%, #F58518 150%);
        padding: 2rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.6rem;
    }
    .app-header h1 { color: #FFFFFF; margin: 0; font-size: 2rem; }
    .app-header p { color: #E7ECF7; margin: .4rem 0 0 0; font-size: 1.02rem; }

    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1rem 0.6rem 1rem;
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; opacity: 0.85; }

    /* Section headers */
    h2, h3 { border-left: 4px solid #F58518; padding-left: .6rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>📦 Inventory &amp; Demand Forecasting Dashboard</h1>
        <p>Upload any company's historical sales/stock data (CSV or Excel) and get demand
        forecasts, reorder points, and stockout risk — built for retail, e-commerce,
        or manufacturing inventory planning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"

# ----------------------------------------------------------------------
# STEP 1: DATA INPUT
# ----------------------------------------------------------------------
st.sidebar.header("1. Load your data")

data_source = st.sidebar.radio(
    "Choose a data source",
    ["Use demo dataset", "Upload my own file (CSV or Excel)"],
)

if data_source == "Upload my own file (CSV or Excel)":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file", type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is None:
        st.info("👈 Upload a CSV or Excel file in the sidebar to get started, or switch to the demo dataset.")
        st.stop()

    file_name = uploaded_file.name.lower()
    if file_name.endswith((".xlsx", ".xls")):
        # An Excel file might have multiple sheets — let the user pick one.
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = excel_file.sheet_names[0]
        if len(excel_file.sheet_names) > 1:
            sheet_name = st.sidebar.selectbox("Excel sheet to use", excel_file.sheet_names)
        raw_df = excel_file.parse(sheet_name)
    else:
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
st.sidebar.caption("Tell the app which column in YOUR file means what — auto-detected below, change if it guessed wrong.")

columns = list(raw_df.columns)


def keyword_guess(col_options, keywords):
    """Try to match a column name against a list of likely keywords."""
    for kw in keywords:
        for c in col_options:
            if kw in str(c).lower():
                return c
    return None


def detect_date_column(df_sample, col_options, exclude=None):
    """Fall back to data-type detection: pick whichever column parses
    as dates most successfully, if no name matches."""
    exclude = exclude or set()
    best_col, best_score = None, 0.0
    for c in col_options:
        if c in exclude:
            continue
        try:
            parsed = pd.to_datetime(df_sample[c], errors="coerce")
            score = parsed.notna().mean()
        except Exception:
            score = 0.0
        if score > best_score:
            best_col, best_score = c, score
    return best_col if best_score > 0.5 else None


def detect_numeric_column(df_sample, col_options, exclude=None):
    """Fall back to data-type detection: pick the numeric column with the
    most variation, skipping obvious ID-like columns."""
    exclude = exclude or set()
    candidates = []
    for c in col_options:
        if c in exclude:
            continue
        if "id" in str(c).lower():
            continue
        numeric = pd.to_numeric(df_sample[c], errors="coerce")
        valid_ratio = numeric.notna().mean()
        if valid_ratio > 0.8 and numeric.nunique() > 3:
            candidates.append((c, numeric.std()))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def detect_category_column(df_sample, col_options, exclude=None):
    """Fall back to data-type detection: pick a text/categorical column
    with a reasonable number of repeated distinct values (not an ID,
    not free text, not all-unique)."""
    exclude = exclude or set()
    best_col, best_score = None, -1
    n = len(df_sample)
    for c in col_options:
        if c in exclude:
            continue
        if "id" in str(c).lower():
            continue
        nunique = df_sample[c].nunique(dropna=True)
        if nunique < 2 or nunique > max(50, n * 0.5):
            continue
        # Prefer columns with fewer distinct values (more product-like, less free-text-like)
        score = -nunique
        if score > best_score:
            best_col, best_score = c, score
    return best_col


def guess(col_options, keywords, sample_df=None, kind=None, exclude=None):
    """Name-based guess first; falls back to data-type detection if no
    keyword matches, so the app still works on unfamiliar column names."""
    match = keyword_guess(col_options, keywords)
    if match:
        return match
    if sample_df is not None and kind == "date":
        detected = detect_date_column(sample_df, col_options, exclude)
        if detected:
            return detected
    if sample_df is not None and kind == "numeric":
        detected = detect_numeric_column(sample_df, col_options, exclude)
        if detected:
            return detected
    if sample_df is not None and kind == "category":
        detected = detect_category_column(sample_df, col_options, exclude)
        if detected:
            return detected
    return col_options[0]


_sample = raw_df.head(200)  # detection only needs a sample, keeps it fast

date_col = st.sidebar.selectbox(
    "Date column", columns,
    index=columns.index(guess(columns, ["date", "day", "time", "period"], _sample, kind="date")),
)
product_col = st.sidebar.selectbox(
    "Product / SKU column", columns,
    index=columns.index(guess(columns, ["product", "sku", "item", "category", "name"], _sample, kind="category", exclude={date_col})),
)
demand_col = st.sidebar.selectbox(
    "Demand / units sold column", columns,
    index=columns.index(guess(columns, ["demand", "sales", "sold", "qty", "quantity", "units", "volume", "orders"], _sample, kind="numeric", exclude={date_col, product_col})),
)
stock_col_options = ["(none)"] + columns
_stock_guess = keyword_guess(columns, ["stock", "inventory", "on hand", "on_hand"])
stock_col = st.sidebar.selectbox(
    "Current stock column (optional)", stock_col_options,
    index=(stock_col_options.index(_stock_guess) if _stock_guess else 0),
)

# Clean & standardize
df = raw_df.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, product_col, demand_col])
df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")
df = df.dropna(subset=[demand_col])
df = df.sort_values(date_col)

# ------------------------------------------------------------------
# DATA QUALITY CHECK — fail gracefully instead of crashing on messy
# real-world files (title rows, merged cells, wrong file shape, etc.)
# ------------------------------------------------------------------
if len(df) < 5:
    st.error(
        "⚠️ **After mapping, fewer than 5 usable rows remained.**\n\n"
        "This usually means one of two things:\n\n"
        "1. **Most likely** — the column mapping above picked the wrong "
        "column. Double-check that Date, Product, and Demand are each "
        "pointing at the right column in *your* file (the auto-guess "
        "isn't perfect, especially with unusual column names).\n\n"
        "2. This file genuinely isn't sales/demand data (e.g. a task "
        "list or report with title rows above the real headers).\n\n"
        "**What this app expects:** one row per date per product, with "
        "a real date column and a numeric demand/units-sold/quantity "
        "column, ideally at least a few weeks of history per product.\n\n"
        "Try adjusting the column mapping in the sidebar first, or use "
        "the demo dataset to see the expected shape."
    )
    st.stop()

products = sorted(df[product_col].unique().tolist())

# ----------------------------------------------------------------------
# STEP 3: FORECAST SETTINGS
# ----------------------------------------------------------------------
st.sidebar.header("3. Forecast settings")
st.sidebar.caption(f"{len(products)} product(s) detected in your data.")
selected_product = st.sidebar.selectbox("Product to analyze", products)
forecast_days = st.sidebar.number_input(
    "Forecast horizon (days)", min_value=1, value=14, step=1,
    help="How many days into the future to forecast. No upper limit — type any number.",
)
lead_time_days = st.sidebar.number_input(
    "Supplier lead time (days)", min_value=1, value=7, step=1,
    help="How many days it takes your supplier to deliver after you order. No upper limit — type any number.",
)
service_level = st.sidebar.select_slider(
    "Target service level (chance of NOT running out of stock)",
    options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999],
    value=0.95,
)

# z-score for service level (standard safety stock formula)
Z_TABLE = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.645, 0.975: 1.96,
           0.99: 2.33, 0.995: 2.576, 0.999: 3.09}
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

try:
    # Build a complete daily date range and fill any gaps, so the
    # forecast model sees an evenly-spaced series (not just sparse dates).
    full_range = pd.date_range(product_df[date_col].min(), product_df[date_col].max(), freq="D")
    product_df = (
        product_df.set_index(date_col)
        .reindex(full_range)
        .ffill()
        .rename_axis(date_col)
        .reset_index()
    )
except Exception:
    st.warning(
        "⚠️ Couldn't fill in a continuous daily date range for this product "
        "(the dates may be irregular or too sparse) — proceeding with the "
        "raw dates found in your file instead."
    )
    product_df = product_df.sort_values(date_col)

demand_series = product_df[demand_col]

if len(demand_series) < 2 or demand_series.dropna().empty:
    st.error(
        f"⚠️ Not enough valid demand history for product **{selected_product}** "
        "to build a forecast. Try a different product, or check that the "
        "demand column contains real numbers for this product."
    )
    st.stop()

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
fig.add_trace(go.Scatter(x=product_df[date_col], y=demand_series, name="Actual demand",
                          line=dict(color="#5B8FE8", width=2)))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_values, name="Forecast",
                          line=dict(color="#F58518", width=2, dash="dash")))
fig.add_hline(y=reorder_point, line_dash="dot", line_color="#E45756", annotation_text="Reorder point")
fig.update_layout(
    template=PLOTLY_TEMPLATE,
    xaxis_title="Date", yaxis_title="Units demanded",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=450,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# ALL-PRODUCTS OVERVIEW TABLE
# ----------------------------------------------------------------------
st.divider()
st.subheader("All products at a glance")

if len(products) > 10:
    search_term = st.text_input("🔍 Search products", "")
    products_to_show = [p for p in products if search_term.lower() in str(p).lower()] if search_term else products
else:
    products_to_show = products

overview_rows = []
for p in products_to_show:
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
