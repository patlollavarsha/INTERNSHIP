import os

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "yield_df.csv")
CROPS = ["Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Sweet potatoes", "Wheat"]
CUTOFF_YEAR = 2009  # models are trained on records before this year


def build_preprocessor():
    return ColumnTransformer(
        [("crop", OneHotEncoder(handle_unknown="ignore"), ["Item"])],
        remainder="passthrough",
    )


@st.cache_resource
def load_and_train():
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df[(df["Area"] == "India") & (df["Item"].isin(CROPS))]
    df = df.drop_duplicates()

    annual = df.groupby(["Item", "Year"], as_index=False).agg(
        yield_hg=("hg/ha_yield", "first"),
        rainfall_mm=("average_rain_fall_mm_per_year", "first"),
        pesticides_tonnes=("pesticides_tonnes", "first"),
        temp_c=("avg_temp", "mean"),
    )
    train = annual[annual["Year"] < CUTOFF_YEAR].sort_values("Year")
    features = ["Item", "rainfall_mm", "pesticides_tonnes", "temp_c"]
    X, y = train[features], train["yield_hg"]

    linear = make_pipeline(build_preprocessor(), LinearRegression()).fit(X, y)
    boosting = make_pipeline(
        build_preprocessor(), GradientBoostingRegressor(random_state=42)
    ).fit(X, y)

    # Last-known-yield baseline: latest training-period yield per crop
    last_known = train.groupby("Item").tail(1).set_index("Item")["yield_hg"]

    return linear, boosting, last_known


st.set_page_config(page_title="CropCast", layout="centered")
st.title("CropCast: Crop Yield Demonstration")
st.caption(
    "Experimental estimates, not guaranteed forecasts. Pesticides are an aggregate "
    "historical total in tonnes, not a farm dose. Temperature is the mean of source "
    "values, not a verified national average. Rainfall was fixed at 1083 mm in the "
    "training data, so changing it is outside what the models learned."
)

if not os.path.exists(DATA_FILE):
    st.error("Dataset not found. Add yield_df.csv to the repository root.")
    st.stop()

linear, boosting, last_known = load_and_train()

crop = st.selectbox("Crop", CROPS, index=CROPS.index("Rice, paddy"))
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=1083.0, step=10.0)
pesticides = st.number_input("Pesticides (tonnes, aggregate)", min_value=0.0, value=14485.33, step=100.0)
temperature = st.number_input("Mean temperature (°C)", value=25.79, step=0.1)

if st.button("Predict yield"):
    row = pd.DataFrame([{
        "Item": crop,
        "rainfall_mm": rainfall,
        "pesticides_tonnes": pesticides,
        "temp_c": temperature,
    }])

    baseline = last_known[crop] / 10000
    lr_pred = linear.predict(row)[0] / 10000
    gb_pred = boosting.predict(row)[0] / 10000

    st.subheader("Yield estimates (tonnes per hectare)")
    st.write(f"**Last-known-yield baseline:** {baseline:.3f}")
    st.write(f"**Linear Regression:** {lr_pred:.3f}")
    st.write(f"**Gradient Boosting:** {gb_pred:.3f}")
    st.info(
        "These estimates come from historical records, so they show how the form works, "
        "not how accurate it is on unseen data."
    )
