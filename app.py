import streamlit as st
import pandas as pd
import numpy as np

# ==============================
# IMPORTS
# ==============================
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="ML based Demand Forecast Dashboard", layout="wide")

# ==============================
# HEADER
# ==============================
st.title("🚀 AI Demand Forecast Dashboard")

# ==============================
# SIDEBAR SETTINGS
# ==============================
st.sidebar.header("⚙️ Inventory Settings")

LT = st.sidebar.number_input("Lead Time (months)", value=1.2)
Z = st.sidebar.number_input("Service Level (Z value)", value=1.65)

# ==============================
# DATA SOURCE
# ==============================
st.sidebar.header("📂 Data Source")

data_option = st.sidebar.radio("Choose Data Source", ["Sample Data", "Upload File"])
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

SAMPLE_URL = "https://raw.githubusercontent.com/Nimo2k9/ml-demand-dashboard/main/MRO%20consumption%20data.xlsx"

@st.cache_data
def load_data(source, is_url=False):
    try:
        df = pd.read_excel(source, engine="openpyxl" if is_url else None)
        if "LCM" in df.columns:
            df = df[df["LCM"] == "Local"]
        return df
    except Exception as e:
        st.error(e)
        return None

if data_option == "Sample Data":
    df = load_data(SAMPLE_URL, True)
else:
    if uploaded_file:
        df = load_data(uploaded_file)
    else:
        st.stop()

if df is None:
    st.stop()

# ==============================
# PREVIEW
# ==============================
with st.expander("🔍 Preview Data"):
    st.dataframe(df.head())

# ==============================
# PREPROCESS
# ==============================
static_cols = ["Item Code","Item Name","Price"]
month_cols = [c for c in df.columns if c not in static_cols + ["LCM","Total"]]

df = df.melt(
    id_vars=[c for c in static_cols if c in df.columns],
    value_vars=month_cols,
    var_name="Date",
    value_name="Demand"
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df = df.dropna().sort_values(["Item Code","Date"])

# ==============================
# FEATURE ENGINEERING (FIXED)
# ==============================
df["Lag1"] = df.groupby("Item Code")["Demand"].shift(1)
df["Lag2"] = df.groupby("Item Code")["Demand"].shift(2)
df["Lag3"] = df.groupby("Item Code")["Demand"].shift(3)

df["RollingMean3"] = df.groupby("Item Code")["Demand"].shift(1).rolling(3).mean()

df["Month"] = df["Date"].dt.month
df["Trend"] = df.groupby("Item Code").cumcount()

# ✅ Seasonality Fix
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

features = [
    "Lag1","Lag2","Lag3",
    "RollingMean3",
    "Price",
    "Month_sin","Month_cos",
    "Trend"
]

df = df.dropna()

# ==============================
# ITEM SELECTION
# ==============================
item_map = df[["Item Code","Item Name"]].drop_duplicates()

item = st.selectbox("Select Item", df["Item Code"].unique())

item_name = item_map[item_map["Item Code"] == item]["Item Name"].values[0]
st.markdown(f"### 🧾 Item Description: **{item_name}**")

item_df = df[df["Item Code"] == item].copy()

if len(item_df) < 12:
    st.warning("Not enough data")
    st.stop()

# ==============================
# TRAIN / VALIDATION SPLIT
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

def train_model(name, model):
    try:
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rmse_scores[name] = np.sqrt(mean_squared_error(y_val, pred))
        models[name] = model
    except:
        pass

# ✅ Improved Models
train_model("RF", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
train_model("XGB", XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5))
train_model("GB", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05))
train_model("ET", ExtraTreesRegressor(n_estimators=200, random_state=42))
train_model("LR", LinearRegression())

# SVR
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

svr = SVR()
svr.fit(X_train_s, y_train)
rmse_scores["SVR"] = np.sqrt(mean_squared_error(y_val, svr.predict(X_val_s)))
models["SVR"] = (svr, scaler)

# ✅ ARIMA FIX (NO LEAKAGE)
ts_train = train.set_index("Date")["Demand"].asfreq("MS")

try:
    arima = ARIMA(ts_train, order=(1,1,1)).fit()
    arima_pred = arima.forecast(len(val))
    rmse_scores["ARIMA"] = np.sqrt(mean_squared_error(y_val, arima_pred))
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
future_dates = pd.date_range(item_df["Date"].max() + pd.DateOffset(months=1), periods=6, freq="MS")

forecast = []
temp = item_df.copy()

for i in range(6):
    last = temp.iloc[-1]

    X_new = pd.DataFrame([{
        "Lag1": last["Demand"],
        "Lag2": temp.iloc[-2]["Demand"],
        "Lag3": temp.iloc[-3]["Demand"],
        "RollingMean3": temp["Demand"].iloc[-3:].mean(),
        "Price": last["Price"],
        "Month_sin": np.sin(2*np.pi*future_dates[i].month/12),
        "Month_cos": np.cos(2*np.pi*future_dates[i].month/12),
        "Trend": last["Trend"] + 1
    }])

    if best_model == "SVR":
        model, sc = models["SVR"]
        pred = model.predict(sc.transform(X_new))[0]

    elif best_model == "ARIMA":
        forecast = models["ARIMA"].forecast(6)
        forecast = np.maximum(0, forecast)
        break

    else:
        pred = models[best_model].predict(X_new)[0]

    # ✅ Prevent negative forecast
    pred = max(0, pred)

    forecast.append(pred)

    new_row = last.copy()
    new_row["Demand"] = pred
    new_row["Trend"] += 1
    temp = pd.concat([temp, pd.DataFrame([new_row])])

forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast})

# ==============================
# KPI
# ==============================
rmse = rmse_scores[best_model]
safety_stock = Z * rmse * np.sqrt(LT)
avg_fc = forecast_df["Forecast"].mean()
rop = avg_fc * LT + safety_stock

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", round(rmse,2))
col2.metric("Safety Stock", round(safety_stock,2))
col3.metric("ROP", round(rop,2))

# ==============================
# DEMAND INSIGHT
# ==============================
st.subheader("📈 Demand Insight")

trend = "Increasing 📈" if item_df["Demand"].iloc[-1] > item_df["Demand"].iloc[0] else "Stable/Decreasing 📉"

st.write(f"Trend: {trend}")
st.write(f"Average Demand: {round(item_df['Demand'].mean(),2)}")
st.write(f"Volatility: {round(item_df['Demand'].std(),2)}")

# ==============================
# PLOT (WITH LABELS)
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=item_df["Date"], y=item_df["Demand"],
    name="Actual"
))

fig.add_trace(go.Scatter(
    x=forecast_df["Date"],
    y=forecast_df["Forecast"],
    name="Forecast",
    mode="lines+markers+text",
    text=np.round(forecast_df["Forecast"],2),
    textposition="top center"
))

st.plotly_chart(fig, use_container_width=True)

# ==============================
# RMSE CHART (HIGHLIGHT BEST)
# ==============================
rmse_df = pd.DataFrame(list(rmse_scores.items()), columns=["Model","RMSE"]).sort_values("RMSE")

rmse_df["Color"] = np.where(rmse_df["Model"] == best_model, "Best", "Other")

fig2 = px.bar(
    rmse_df,
    x="Model",
    y="RMSE",
    text="RMSE",
    color="Color",
    title="Model Comparison"
)

fig2.update_traces(textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

# ==============================
# MODEL TABLE
# ==============================
st.subheader("📊 Model Performance Table")
st.dataframe(rmse_df)



# ==============================
# DOWNLOAD
# ==============================
st.download_button("Download Forecast", forecast_df.to_csv(index=False), "forecast.csv")
