import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_excel("MRO consumption data.xlsx")
    df = df[df["LCM"] == "Local"]
    return df

df = load_data()

# ==============================
# PREPROCESS
# ==============================
static_cols = ["Item Code","Item Name","Price"]
month_cols = [c for c in df.columns if c not in static_cols and c!="LCM"]

df_long = df.melt(
    id_vars=static_cols,
    value_vars=month_cols,
    var_name="Date",
    value_name="Demand"
)

df_long["Date"] = pd.to_datetime(df_long["Date"])
df_long = df_long.sort_values(["Item Code","Date"])
df_long["Demand"] = df_long["Demand"].fillna(0)

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

features = ["Lag1","Lag2","Lag3","RollingMean3","Price"]

# ==============================
# STREAMLIT UI
# ==============================
st.title("📊 Demand Forecasting App")

item_list = df_long["Item Code"].unique()
selected_item = st.selectbox("Select Item Code", item_list)

# ==============================
# FILTER DATA
# ==============================
item_df = df_long[df_long["Item Code"] == selected_item].copy()

# ==============================
# MODEL SELECTION (SIMPLE VERSION)
# ==============================
model_option = st.selectbox("Select Model", ["Random Forest","XGBoost","SVR","ARIMA"])

# ==============================
# TRAIN MODEL
# ==============================
X = item_df[features]
y = item_df["Demand"]

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

        # Append for recursive forecasting
        new_row = last_row.copy()
        new_row["Demand"] = pred
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])])

# ==============================
# DISPLAY RESULTS
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
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(item_df["Date"], item_df["Demand"], label="Historical")
ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast")

ax.legend()
ax.set_title(f"Forecast for Item: {selected_item}")

st.pyplot(fig)
