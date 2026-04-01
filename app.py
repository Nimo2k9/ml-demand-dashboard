import streamlit as st
import pandas as pd
import numpy as np
import hashlib

# ==============================
# LOGIN SYSTEM
# ==============================
USER_CREDENTIALS = {
    "admin": "1234",
    "niaz": "ml2025"
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

USER_CREDENTIALS = {u: hash_password(p) for u, p in USER_CREDENTIALS.items()}

def login():
    st.title("🔐 Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == hash_password(password):
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.rerun()
        else:
            st.error("Invalid username or password")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ==============================
# IMPORTS
# ==============================
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

import plotly.graph_objects as go

st.set_page_config(page_title="Forecast Dashboard", layout="wide")

# ==============================
# HEADER
# ==============================
col1, col2 = st.columns([8,2])

with col1:
    st.title("📊 Demand Forecast Dashboard")
    st.markdown(f"👤 Logged in as: **{st.session_state['user']}**")

with col2:
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

# ==============================
# DATA SOURCE (NEW FEATURE)
# ==============================
st.sidebar.header("📂 Data Source")

use_sample = st.sidebar.checkbox("Use Sample Data (Default)", value=True)

uploaded_file = st.sidebar.file_uploader("Upload Your Excel File", type=["xlsx"])

# 🔗 Replace this with your GitHub raw file link
SAMPLE_URL = "https://github.com/Nimo2k9/ml-demand-dashboard/blob/main/MRO%20consumption%20data.xlsx"

@st.cache_data
def load_data(source):
    try:
        df = pd.read_excel(source)
        if "LCM" in df.columns:
            df = df[df["LCM"] == "Local"]
        return df
    except:
        return None

# Decide data source
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("✅ Using uploaded file")

elif use_sample:
    df = load_data(SAMPLE_URL)
    st.info("📊 Using sample dataset from GitHub")

else:
    st.warning("Please upload a file or enable sample data")
    st.stop()

if df is None:
    st.error("❌ Failed to load data")
    st.stop()

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

features = ["Lag1","Lag2","Lag3","RollingMean3","Price"]

for col in features:
    df_long[col] = pd.to_numeric(df_long[col], errors='coerce')

df_long = df_long.dropna()

# ==============================
# ITEM SELECT
# ==============================
item_list = df_long["Item Code"].unique()
item_map = df_long[["Item Code","Item Name"]].drop_duplicates()

selected_item = st.selectbox("Select Item Code", item_list)

item_name = item_map[item_map["Item Code"] == selected_item]["Item Name"].values[0]
st.markdown(f"**Item Description:** {item_name}")

item_df = df_long[df_long["Item Code"] == selected_item]

# ==============================
# MODEL TRAINING + FORECAST (same as before)
# ==============================
# 👉 KEEP YOUR EXISTING MODEL CODE HERE (no change needed)
