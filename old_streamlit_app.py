import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

FEATURES = ["Item", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
TARGET = "hg/ha_yield"

@st.cache_resource
def load_models():
    df = pd.read_csv("annual_crop_table.csv")
    df = df[df["Year"] < 2009]  # same training cutoff as the notebook form

    def make_pipeline(model):
        pre = ColumnTransformer(
            [("crop", OneHotEncoder(handle_unknown="ignore"), ["Item"])],
            remainder="passthrough",
        )
        return Pipeline([("pre", pre), ("model", model)]).fit(df[FEATURES], df[TARGET])

    lr = make_pipeline(LinearRegression())
    gb = make_pipeline(GradientBoostingRegressor(random_state=42))
    last_known = df.sort_values("Year").groupby("Item")[TARGET].last()
    crops = sorted(df["Item"].unique())
    return lr, gb, last_known, crops

lr, gb, last_known, crops = load_models()

st.title("CropCast: Crop Yield Estimator")
st.caption("Educational prototype. Not for operational forecasting.")

crop = st.selectbox("Crop", crops)
rain = st.number_input("Average rainfall (mm/year)", min_value=0.0, value=1000.0)
pest = st.number_input("Pesticides (tonnes)", min_value=0.0, value=1000.0)
temp = st.number_input("Average temperature (°C)", value=20.0)

if st.button("Estimate"):
    X = pd.DataFrame([{
        "Item": crop,
        "average_rain_fall_mm_per_year": rain,
        "pesticides_tonnes": pest,
        "avg_temp": temp,
    }])
    st.write(f"Historical baseline (last known yield): **{last_known[crop] / 10000:.2f} t/ha**")
    for name, model in [("Linear Regression", lr), ("Gradient Boosting", gb)]:
        value = model.predict(X)[0]
        if value < 0:
            st.warning(f"{name}: negative output ({value / 10000:.2f} t/ha) is physically invalid.")
        else:
            st.write(f"{name}: **{value / 10000:.2f} t/ha**")
