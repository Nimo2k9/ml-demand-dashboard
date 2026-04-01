import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

import plotly.graph_objects as go

st.set_page_config(page_title="ML Demand Forecast Dashboard", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df = df[df["LCM"] == "Local"]
    return df

uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

if uploaded_file is None:
    st.warning("Upload file to continue")
    st.stop()

df = load_data(uploaded_file)

# ==============================
# PREPROCESS
# ==============================
static_cols = ["Item Code","Item Name","Price"]

month_cols = [
    c for c in df.columns
    if c not in static_cols and c not in ["LCM","Total"]
]

df_long = df.melt(
    id_vars=static_cols,
    value_vars=month_cols,
    var_name="Date",
    value_name="Demand"
)

df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
df_long = df_long.dropna(subset=["Date"])

df_long["Demand"] = df_long["Demand"].fillna(0)
df_long = df_long.sort_values(["Item Code","Date"])

# ==============================
# FEATURES
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

features = ["Lag1","Lag2","Lag3","RollingMean3","Price"]

# ==============================
# UI
# ==============================
st.title("📊 ML Demand Forecasting Dashboard")

item_list = df_long["Item Code"].unique()
selected_item = st.selectbox("Select Item", item_list)

item_df = df_long[df_long["Item Code"] == selected_item].copy()

# ==============================
# TRAIN / VALID SPLIT
# ==============================
split = int(len(item_df)*0.8)

train = item_df.iloc[:split]
val = item_df.iloc[split:]

X_train = train[features]
y_train = train["Demand"]

X_val = val[features]
y_val = val["Demand"]

models = {}
rmse_scores = {}

# RF
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_val)
rmse_scores["RF"] = np.sqrt(mean_squared_error(y_val, pred))
models["RF"] = rf

# XGB
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
xgb.fit(X_train, y_train)
pred = xgb.predict(X_val)
rmse_scores["XGB"] = np.sqrt(mean_squared_error(y_val, pred))
models["XGB"] = xgb

# SVR
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

svr = SVR(C=100, gamma=0.1)
svr.fit(X_train_s, y_train)
pred = svr.predict(X_val_s)
rmse_scores["SVR"] = np.sqrt(mean_squared_error(y_val, pred))
models["SVR"] = (svr, scaler)

# ARIMA
try:
    ts = train.set_index("Date")["Demand"].asfreq("MS")
    arima = ARIMA(ts, order=(1,1,1)).fit()
    forecast = arima.forecast(steps=len(val))
    rmse_scores["ARIMA"] = np.sqrt(mean_squared_error(y_val, forecast))
    models["ARIMA"] = arima
except:
    pass

# ==============================
# BEST MODEL
# ==============================
best_model_name = min(rmse_scores, key=rmse_scores.get)

st.success(f"🏆 Best Model: {best_model_name}")

# ==============================
# FORECAST 6 MONTHS
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

    last = temp_df.iloc[-1]

    lag1 = last["Demand"]
    lag2 = temp_df.iloc[-2]["Demand"]
    lag3 = temp_df.iloc[-3]["Demand"]
    rolling = temp_df["Demand"].iloc[-3:].mean()
    price = last["Price"]

    X_new = np.array([[lag1, lag2, lag3, rolling, price]])

    if best_model_name == "SVR":
        model, scaler = models["SVR"]
        X_new = scaler.transform(X_new)
        pred = model.predict(X_new)[0]

    elif best_model_name == "ARIMA":
        forecast_values = models["ARIMA"].forecast(steps=forecast_horizon)
        break

    else:
        model = models[best_model_name]
        pred = model.predict(X_new)[0]

    forecast_values.append(pred)

    new_row = last.copy()
    new_row["Demand"] = pred
    temp_df = pd.concat([temp_df, pd.DataFrame([new_row])])

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": forecast_values
})

# ==============================
# KPI CALCULATION
# ==============================
rmse = rmse_scores[best_model_name]
LT = 1.2
Z = 1.65

safety_stock = Z * rmse * np.sqrt(LT)
avg_forecast = forecast_df["Forecast"].mean()
rop = avg_forecast * LT + safety_stock

# ==============================
# DISPLAY KPIs
# ==============================
col1, col2, col3 = st.columns(3)

col1.metric("RMSE", round(rmse,2))
col2.metric("Safety Stock", round(safety_stock,2))
col3.metric("Reorder Point", round(rop,2))

# ==============================
# PLOT (INTERACTIVE)
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=item_df["Date"], y=item_df["Demand"],
    mode='lines', name='Historical'
))

fig.add_trace(go.Scatter(
    x=forecast_df["Date"], y=forecast_df["Forecast"],
    mode='lines', name='Forecast'
))

st.plotly_chart(fig, use_container_width=True)

# ==============================
# RMSE COMPARISON
# ==============================
rmse_df = pd.DataFrame(list(rmse_scores.items()), columns=["Model","RMSE"])
st.subheader("📊 Model Comparison")
st.dataframe(rmse_df)

# ==============================
# DOWNLOAD
# ==============================
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download Forecast",
    csv,
    "forecast.csv",
    "text/csv"
)
