import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt

st.set_page_config(page_title="Demand Forecasting App", layout="wide")

# ==============================
# LOAD DATA (UPLOAD VERSION)
# ==============================
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df = df[df["LCM"] == "Local"]
    return df

uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

if uploaded_file is None:
    st.warning("Please upload your dataset to continue.")
    st.stop()

df = load_data(uploaded_file)

# ==============================
# PREPROCESSING
# ==============================
static_cols = ["Item Code", "Item Name", "Price"]

# Remove unwanted columns like Total
month_cols = [
    c for c in df.columns
    if c not in static_cols and c not in ["LCM", "Total"]
]

df_long = df.melt(
    id_vars=static_cols,
    value_vars=month_cols,
    var_name="Date",
    value_name="Demand"
)

# ✅ ROBUST DATE FIX
df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
df_long = df_long.dropna(subset=["Date"])

df_long["Demand"] = df_long["Demand"].fillna(0)
df_long = df_long.sort_values(["Item Code", "Date"])

# ==============================
# FEATURE ENGINEERING
# ==============================
df_long["Lag1"] = df_long.groupby("Item Code")["Demand"].shift(1)
df_long["Lag2"] = df_long.groupby("Item Code")["Demand"].shift(2)
df_long["Lag3"] = df_long.groupby("Item Code")["Demand"].shift(3)

df_long["RollingMean3"] = (
    df_long.groupby("Item Code")["Demand"]
    .shift(1)
    .rolling(3)
    .mean()
)

df_long = df_long.dropna()

features = ["Lag1", "Lag2", "Lag3", "RollingMean3", "Price"]

# ==============================
# UI
# ==============================
st.title("📊 Demand Forecasting Dashboard")

item_list = df_long["Item Code"].unique()
selected_item = st.selectbox("Select Item Code", item_list)

model_option = st.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost", "SVR", "ARIMA"]
)

# ==============================
# FILTER ITEM DATA
# ==============================
item_df = df_long[df_long["Item Code"] == selected_item].copy()

X = item_df[features]
y = item_df["Demand"]

# ==============================
# TRAIN MODEL
# ==============================
if model_option == "Random Forest":
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X, y)

elif model_option == "XGBoost":
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
    model.fit(X, y)

elif model_option == "SVR":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVR(C=100, gamma=0.1)
    model.fit(X_scaled, y)

elif model_option == "ARIMA":
    ts = item_df.set_index("Date")["Demand"].asfreq("MS")
    model = ARIMA(ts, order=(1,1,1)).fit()

# ==============================
# FORECAST NEXT 6 MONTHS
# ==============================
forecast_horizon = 6

future_dates = pd.date_range(
    start=item_df["Date"].max() + pd.DateOffset(months=1),
    periods=forecast_horizon,
    freq="MS"
)

forecast_values = []

temp_df = item_df.copy()

for i in range(forecast_horizon):

    last_row = temp_df.iloc[-1]

    lag1 = last_row["Demand"]
    lag2 = temp_df.iloc[-2]["Demand"]
    lag3 = temp_df.iloc[-3]["Demand"]
    rolling = temp_df["Demand"].iloc[-3:].mean()
    price = last_row["Price"]

    X_new = np.array([[lag1, lag2, lag3, rolling, price]])

    if model_option == "SVR":
        X_new = scaler.transform(X_new)

    if model_option == "ARIMA":
        forecast = model.forecast(steps=forecast_horizon)
        forecast_values = forecast.values
        break
    else:
        pred = model.predict(X_new)[0]
        forecast_values.append(pred)

        # Recursive update
        new_row = last_row.copy()
        new_row["Demand"] = pred
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])])

# ==============================
# RESULTS
# ==============================
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": forecast_values
})

st.subheader("📅 6-Month Forecast")
st.dataframe(forecast_df)

# ==============================
# PLOT
# ==============================
fig, ax = plt.subplots()

ax.plot(item_df["Date"], item_df["Demand"], label="Historical")
ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast")

ax.legend()
ax.set_title(f"Forecast for Item: {selected_item}")

st.pyplot(fig)
