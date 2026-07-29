import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Supply Chain & Inventory Demand Forecaster",
    page_icon="📦",
    layout="wide"
)

# -----------------------------
# Load Saved Files
# -----------------------------
with open("inventory_model.pkl", "rb") as f:
    model_final = pickle.load(f)

with open("product_encoder.pkl", "rb") as f:
    product_encoder = pickle.load(f)

with open("inventory_df.pkl", "rb") as f:
    df = pickle.load(f)

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("📦 Supply Chain & Inventory Demand Forecaster")

st.markdown("""
Predict inventory demand for premium products using Machine Learning.

### Features
- 📅 30-Day Forecast
- 📅 60-Day Forecast
- 📦 Inventory Planning
- 📈 Demand Trend
- ⚠ Demand Alerts
- 📥 Download Forecast
""")

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Forecast Settings")

product = st.sidebar.selectbox(
    "Select Product",
    ["Saffron", "Makhana", "Chia Seeds"]
)

days = st.sidebar.radio(
    "Forecast Period",
    [30, 60]
)

st.divider()
# ==============================
# Forecast Function
# ==============================

def forecast(product_name_str, days):

    try:
        encoded_product = product_encoder.transform([product_name_str])[0]

    except ValueError:
        st.error("Invalid Product Selected")
        return pd.DataFrame()

    product_df = df[df["Product"] == encoded_product].copy()

    if product_df.empty:
        st.error("No data available for this product.")
        return pd.DataFrame()

    if "Date" not in product_df.columns:
        product_df = product_df.reset_index()

    product_df["Date"] = pd.to_datetime(product_df["Date"])

    product_df = product_df.sort_values("Date")

    last = product_df.iloc[-1].copy()

    future = []

    for i in range(days):

        next_date = last["Date"] + timedelta(days=1)

        last["Date"] = next_date
        last["Month"] = next_date.month
        last["Weekend"] = 1 if next_date.weekday() >= 5 else 0

        X_new = pd.DataFrame({

            "Product":[last["Product"]],
            "Inventory":[last["Inventory"]],
            "Price":[last["Price"]],
            "Promotion":[last["Promotion"]],
            "Festival":[last["Festival"]],
            "Temperature":[last["Temperature"]],
            "Month":[last["Month"]],
            "Weekend":[last["Weekend"]],
            "Lag_1":[last["Lag_1"]],
            "Lag_7":[last["Lag_7"]],
            "MA_7":[last["MA_7"]],
            "MA_30":[last["MA_30"]],
            "STD_7":[last["STD_7"]],
            "Demand_Spike":[last["Demand_Spike"]]

        })

        prediction = model_final.predict(X_new)[0]

        prediction = round(max(5, prediction))

        future.append({

            "Date": next_date,
            "Product": product_name_str,
            "Forecast Demand": prediction

        })

        last["Lag_7"] = last["Lag_1"]
        last["Lag_1"] = prediction

        last["MA_7"] = (last["MA_7"] * 6 + prediction) / 7

        last["MA_30"] = (last["MA_30"] * 29 + prediction) / 30

        last["Inventory"] = max(
            0,
            last["Inventory"] - prediction
        )

    return pd.DataFrame(future)
# ==============================
# Generate Forecast
# ==============================

if st.button("🚀 Generate Forecast"):

    result = forecast(product, days)

    if not result.empty:

        st.success(f"{days}-Day Forecast Generated Successfully!")

        # -----------------------------
        # Forecast Table
        # -----------------------------
        st.subheader("📋 Forecast Table")

        st.dataframe(
            result,
            use_container_width=True
        )

        # -----------------------------
        # Line Chart
        # -----------------------------
        st.subheader("📈 Demand Forecast")

        chart_df = result.set_index("Date")

        st.line_chart(chart_df["Forecast Demand"])

        # -----------------------------
        # Summary Metrics
        # -----------------------------
        total_demand = int(result["Forecast Demand"].sum())

        avg_demand = round(result["Forecast Demand"].mean(), 2)

        max_demand = int(result["Forecast Demand"].max())

        safety_stock = int(avg_demand * 7)

        reorder_point = total_demand + safety_stock

        st.subheader("📊 Inventory Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Forecast Demand",
            total_demand
        )

        col2.metric(
            "Average Daily Demand",
            avg_demand
        )

        col3.metric(
            "Maximum Daily Demand",
            max_demand
        )

        col4, col5 = st.columns(2)

        col4.metric(
            "Safety Stock",
            safety_stock
        )

        col5.metric(
            "Reorder Point",
            reorder_point
        )

        # -----------------------------
        # Demand Status
        # -----------------------------
        st.subheader("📦 Demand Status")

        if avg_demand >= 120:

            st.error("🔴 High Demand Spike Expected")

        elif avg_demand >= 80:

            st.warning("🟡 Moderate Demand")

        else:

            st.success("🟢 Normal Demand")

        # -----------------------------
        # Download CSV
        # -----------------------------
        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Forecast CSV",
            data=csv,
            file_name=f"{product}_{days}_Day_Forecast.csv",
            mime="text/csv"
        )

