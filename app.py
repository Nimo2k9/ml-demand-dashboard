import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# SETTINGS
# ============================================================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

LT = 1.2
Z = 1.65
features = ["Lag1","Lag2","Lag3","RollingMean3","Price"]

st.set_page_config(layout="wide")
st.title("📦 Demand Forecasting & Inventory Optimization Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
run_btn = st.sidebar.button("🚀 Run Analysis")

# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true)!=0 else np.nan

    return rmse, mae, wmape

# ============================================================
# MAIN
# ============================================================

if uploaded_file and run_btn:

    with st.spinner("Running full ML pipeline... ⏳"):

        df = pd.read_excel(uploaded_file)
        df = df[df["LCM"]=="Local"]

        # ---------------- TRANSFORM ----------------
        static_cols = ["Item Code","Item Name","Price"]
        month_cols = [c for c in df.columns if c not in static_cols and c!="LCM"]

        df_long = df.melt(
            id_vars=static_cols,
            value_vars=month_cols,
            var_name="Date",
            value_name="Demand"
        )

        df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
        df_long = df_long.dropna()
        df_long["Demand"] = df_long["Demand"].fillna(0)

        df_long = df_long.sort_values(["Item Code","Date"])

        # ---------------- SEGMENT ----------------
        item_stats = df_long.groupby("Item Code").agg(mean_demand=("Demand","mean")).reset_index()

        def segment(row):
            if row["mean_demand"] > 150: return "High"
            elif row["mean_demand"] > 50: return "Medium"
            else: return "Low"

        item_stats["Segment"] = item_stats.apply(segment, axis=1)
        df_long = df_long.merge(item_stats[["Item Code","Segment"]], on="Item Code")

        # ---------------- FEATURES ----------------
        df_long["Lag1"] = df_long.groupby("Item Code")["Demand"].shift(1)
        df_long["Lag2"] = df_long.groupby("Item Code")["Demand"].shift(2)
        df_long["Lag3"] = df_long.groupby("Item Code")["Demand"].shift(3)

        df_long["RollingMean3"] = (
            df_long.groupby("Item Code")["Demand"]
            .shift(1).rolling(3).mean()
        )

        df_long["Year"] = df_long["Date"].dt.year
        df_long = df_long.dropna()

        train = df_long[df_long["Year"]<=2023]
        val = df_long[df_long["Year"]==2024]
        test = df_long[df_long["Year"]==2025]
        train_full = df_long[df_long["Year"]<=2024]

        # ============================================================
        # MODEL COMPARISON
        # ============================================================

        detailed_results=[]
        best_models={}

        for seg in df_long["Segment"].unique():

            train_seg=train[train["Segment"]==seg]
            val_seg=val[val["Segment"]==seg]

            if len(val_seg)==0:
                continue

            scores={}

            rf=RandomForestRegressor(n_estimators=300,max_depth=12,random_state=SEED)
            rf.fit(train_seg[features],train_seg["Demand"])
            preds=rf.predict(val_seg[features])
            rmse,mae,wmape=compute_metrics(val_seg["Demand"],preds)
            detailed_results.append([seg,"RF",rmse,mae,wmape])
            scores["RF"]=rmse

            xgb=XGBRegressor(n_estimators=400,learning_rate=0.05,max_depth=5,random_state=SEED)
            xgb.fit(train_seg[features],train_seg["Demand"])
            preds=xgb.predict(val_seg[features])
            rmse,mae,wmape=compute_metrics(val_seg["Demand"],preds)
            detailed_results.append([seg,"XGB",rmse,mae,wmape])
            scores["XGB"]=rmse

            scaler=StandardScaler()
            X_train_scaled=scaler.fit_transform(train_seg[features])
            X_val_scaled=scaler.transform(val_seg[features])

            svr=SVR(kernel="rbf",C=100,gamma=0.1)
            svr.fit(X_train_scaled,train_seg["Demand"])
            preds=svr.predict(X_val_scaled)
            rmse,mae,wmape=compute_metrics(val_seg["Demand"],preds)
            detailed_results.append([seg,"SVR",rmse,mae,wmape])
            scores["SVR"]=rmse

            best_models[seg]=min(scores,key=scores.get)

        results_df=pd.DataFrame(detailed_results,columns=["Segment","Model","RMSE","MAE","WMAPE"])

        # ============================================================
        # UI - MODEL COMPARISON
        # ============================================================

        st.subheader("📊 Model Comparison")
        st.dataframe(results_df)

        seg = st.selectbox("Select Segment", results_df["Segment"].unique())
        temp = results_df[results_df["Segment"]==seg]

        st.bar_chart(temp.set_index("Model")["RMSE"])
        st.success(f"Best Model: {best_models[seg]}")

        # ============================================================
        # 6-MONTH FORECAST
        # ============================================================

        st.subheader("🔮 Forecast Next 6 Months")

        item = st.selectbox("Select Item", df_long["Item Code"].unique())

        item_data = df_long[df_long["Item Code"]==item].sort_values("Date")
        seg_item = item_data["Segment"].iloc[0]
        model_name = best_models[seg_item]

        st.info(f"Segment: {seg_item}")
        st.success(f"Best Model: {model_name}")

        train_item = train_full[train_full["Item Code"]==item]

        forecast_values = []

        if model_name == "ARIMA":

            ts = train_item.set_index("Date")["Demand"].asfreq("MS")
            model = ARIMA(ts, order=(1,1,1))
            fit = model.fit()
            forecast_values = fit.forecast(steps=6).values.tolist()

        else:

            if model_name == "RF":
                model = RandomForestRegressor(n_estimators=300, random_state=SEED)
                model.fit(train_item[features], train_item["Demand"])

            elif model_name == "XGB":
                model = XGBRegressor(n_estimators=400, random_state=SEED)
                model.fit(train_item[features], train_item["Demand"])

            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(train_item[features])
                model = SVR()
                model.fit(X_scaled, train_item["Demand"])

            last_row = train_item.iloc[-1:].copy()

            for i in range(6):

                if model_name == "SVR":
                    pred = model.predict(scaler.transform(last_row[features]))[0]
                else:
                    pred = model.predict(last_row[features])[0]

                forecast_values.append(pred)

                last_row["Lag3"] = last_row["Lag2"]
                last_row["Lag2"] = last_row["Lag1"]
                last_row["Lag1"] = pred
                last_row["RollingMean3"] = (last_row["Lag1"]+last_row["Lag2"]+last_row["Lag3"])/3

        forecast_df = pd.DataFrame({
            "Month": [f"M{i+1}" for i in range(6)],
            "Forecast": forecast_values
        })

        st.dataframe(forecast_df)
        st.line_chart(forecast_df.set_index("Month"))

else:
    st.info("👈 Upload file and click Run Analysis")
