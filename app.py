import streamlit as st
import pandas as pd
import numpy as np

# ML Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

import plotly.graph_objects as go

st.set_page_config(page_title="Advanced Demand Forecast Dashboard", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    if "LCM" in df.columns:
        df = df[df["LCM"] == "Local"]
    return df

uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df = load_data(uploaded_file)

# ==============================
# PREPROCESS
# ==============================
static_cols = ["Item Code","Item Name","Price"]

month_cols = [c for c in df.columns if c not in static_cols + ["LCM","Total"]]

df_long = df.melt(
    id_vars=[c for c in static_cols if c in df.columns],
    value_vars=month_cols,
    var_name="Date",
    value_name="Demand"
)

df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
df_long = df_long.dropna(subset=["Date"])

df_long["Demand"] = pd.to_numeric(df_long["Demand"], errors="coerce")
df_long["Price"] = pd.to_numeric(df_long["Price"], errors="coerce")

df_long = df_long.dropna()
df_long = df_long.sort_values(["Item Code","Date"])

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

features = ["Lag1","Lag2","Lag3","RollingMean3","Price"]

for col in features:
    df_long[col] = pd.to_numeric(df_long[col], errors='coerce')

df_long = df_long.dropna()

# ==============================
# UI
# ==============================
st.title("📊 Advanced ML Demand Forecast Dashboard")

item_list = df_long["Item Code"].unique()
item_map = df_long[["Item Code","Item Name"]].drop_duplicates()

selected_item = st.selectbox("Select Item Code", item_list)

item_name = item_map[item_map["Item Code"] == selected_item]["Item Name"].values[0]
st.markdown(f"**Item Description:** {item_name}")

item_df = df_long[df_long["Item Code"] == selected_item]

# ==============================
# TRAIN / VALID
# ==============================
split = int(len(item_df)*0.8)

train = item_df.iloc[:split]
val = item_df.iloc[split:]

X_train = train[features].astype(float)
y_train = train["Demand"]

X_val = val[features].astype(float)
y_val = val["Demand"]

models = {}
rmse_scores = {}

# ==============================
# MODELS
# ==============================
def train_model(name, model, Xtr, ytr, Xv, yv):
    try:
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        rmse_scores[name] = np.sqrt(mean_squared_error(yv, pred))
        models[name] = model
    except:
        pass

train_model("RF", RandomForestRegressor(n_estimators=200), X_train, y_train, X_val, y_val)
train_model("XGB", XGBRegressor(n_estimators=200), X_train, y_train, X_val, y_val)
train_model("GB", GradientBoostingRegressor(), X_train, y_train, X_val, y_val)
train_model("ET", ExtraTreesRegressor(n_estimators=200), X_train, y_train, X_val, y_val)
train_model("KNN", KNeighborsRegressor(), X_train, y_train, X_val, y_val)
train_model("LR", LinearRegression(), X_train, y_train, X_val, y_val)

# SVR
try:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    svr = SVR()
    svr.fit(X_train_s, y_train)
    pred = svr.predict(X_val_s)
    rmse_scores["SVR"] = np.sqrt(mean_squared_error(y_val, pred))
    models["SVR"] = (svr, scaler)
except:
    pass

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
best_model = min(rmse_scores, key=rmse_scores.get)
st.success(f"🏆 Best Model: {best_model}")

# ==============================
# FORECAST
# ==============================
forecast_horizon = 6
future_dates = pd.date_range(item_df["Date"].max() + pd.DateOffset(months=1), periods=forecast_horizon, freq="MS")

forecast_values = []
temp_df = item_df.copy()

for i in range(forecast_horizon):

    last = temp_df.iloc[-1]

    X_new = np.array([[
        last["Demand"],
        temp_df.iloc[-2]["Demand"],
        temp_df.iloc[-3]["Demand"],
        temp_df["Demand"].iloc[-3:].mean(),
        last["Price"]
    ]])

    if best_model == "SVR":
        model, scaler = models["SVR"]
        X_new = scaler.transform(X_new)
        pred = model.predict(X_new)[0]

    elif best_model == "ARIMA":
        forecast_values = models["ARIMA"].forecast(steps=forecast_horizon)
        break

    else:
        pred = models[best_model].predict(X_new)[0]

    forecast_values.append(pred)

    new_row = last.copy()
    new_row["Demand"] = pred
    temp_df = pd.concat([temp_df, pd.DataFrame([new_row])])

forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast_values})

# ==============================
# KPI
# ==============================
LT, Z = 1.2, 1.65

rmse = rmse_scores[best_model]
safety_stock = Z * rmse * np.sqrt(LT)
avg_fc = forecast_df["Forecast"].mean()
rop = avg_fc * LT + safety_stock

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", round(rmse,2))
col2.metric("Safety Stock", round(safety_stock,2))
col3.metric("ROP", round(rop,2))

# ==============================
# PLOT
# ==============================
fig = go.Figure()
fig.add_trace(go.Scatter(x=item_df["Date"], y=item_df["Demand"], name="Historical"))
fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], name="Forecast"))
st.plotly_chart(fig, use_container_width=True)

# ==============================
# MODEL COMPARISON
# ==============================
rows = []

for m, r in rmse_scores.items():
    safety = Z * r * np.sqrt(LT)
    rop_val = avg_fc * LT + safety

    rows.append({
        "Model": m,
        "RMSE": round(r,2),
        "Safety Stock": round(safety,2),
        "ROP": round(rop_val,2)
    })

rmse_df = pd.DataFrame(rows).sort_values("RMSE")

def highlight(row):
    return ['background-color: lightgreen' if row["Model"] == best_model else '' for _ in row]

st.subheader("📊 Model Comparison")
st.dataframe(rmse_df.style.apply(highlight, axis=1))

# ==============================
# DOWNLOAD
# ==============================
st.download_button(
    "📥 Download Forecast",
    forecast_df.to_csv(index=False),
    "forecast.csv"
)
