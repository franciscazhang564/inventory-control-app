# 📦 Inventory & Demand Forecasting Dashboard

A general-purpose web app for demand forecasting and inventory planning.
Upload any company's historical sales/stock data (any column names — you map
them yourself) and instantly get:

- **Demand forecasts** per product, using Holt's Exponential Smoothing
- **Reorder point & safety stock** calculations based on your chosen service level
- **Stockout risk flags** comparing current stock against the reorder point
- **Forecast accuracy tracking** (MAPE) so you can trust the numbers
- A full **multi-product overview table** for quick decision-making

Built as a self-contained tool — not tied to one company's dataset — so it can
be demoed with any retail, e-commerce, or manufacturing inventory data.

## 🔗 Live demo
[Add your Streamlit Cloud link here once deployed]

## Why I built this
I'm an International Business Administration student (Tilburg University)
completing a minor in Supply Chain / Operations Management. I wanted a hands-on
project that applies inventory planning concepts (reorder points, safety stock,
service levels) in a tool that's actually usable — not just a spreadsheet exercise.

## How it works
1. Upload a CSV (or use the built-in demo dataset)
2. Map your columns (date, product, demand, stock) — works with any schema
3. Pick a product, lead time, and target service level
4. Get a forecast chart, reorder point, safety stock, and stockout status

## Tech stack
- **Python** — pandas, numpy for data processing
- **statsmodels** — Holt's Exponential Smoothing for demand forecasting
- **Streamlit** — interactive web app framework
- **Plotly** — interactive charts

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Methodology
- **Forecast**: Holt's Exponential Smoothing (trend-aware), fit on historical
  daily demand per product.
- **Safety stock** = Z(service level) × standard deviation of demand × √(lead time)
- **Reorder point** = (average daily demand × lead time) + safety stock

## Dataset
Demo data: [Inventory Demand Forecasting dataset, Kaggle] — 2 years of daily
sales, price, promotion, and stock data across 5 products / categories.
