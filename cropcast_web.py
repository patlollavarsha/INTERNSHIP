import os

import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "yield_df.csv")
CROPS = ["Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Sweet potatoes", "Wheat"]
CUTOFF_YEAR = 2009


def build_preprocessor():
    return ColumnTransformer(
        [("crop", OneHotEncoder(handle_unknown="ignore"), ["Item"])],
        remainder="passthrough",
    )


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
    last_known = train.groupby("Item").tail(1).set_index("Item")["yield_hg"]
    return linear, boosting, last_known


linear, boosting, last_known = load_and_train()
app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>CropCast</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; color: #1B2A41; }
  label { display: block; margin-top: 14px; font-weight: bold; }
  input, select { width: 100%; padding: 8px; font-size: 16px; margin-top: 4px; box-sizing: border-box; }
  button { margin-top: 20px; padding: 10px 20px; font-size: 16px; background: #2E6FBA; color: white; border: none; border-radius: 4px; cursor: pointer; }
  .note { color: #5A6472; font-size: 14px; }
  .result { margin-top: 24px; padding: 16px; background: #EEF3F9; border-radius: 6px; }
  .error { color: #B00020; margin-top: 16px; }
</style>
</head>
<body>
<h1>CropCast: Crop Yield Demonstration</h1>
<p class="note">Experimental estimates, not guaranteed forecasts. Pesticides are an aggregate historical total in tonnes, not a farm dose. Temperature is the mean of source values, not a verified national average. Rainfall was fixed at 1083 mm in the training data, so changing it is outside what the models learned.</p>
<form method="post">
  <label>Crop</label>
  <select name="crop">
    {% for c in crops %}
      <option value="{{ c }}" {% if c == form.crop %}selected{% endif %}>{{ c }}</option>
    {% endfor %}
  </select>
  <label>Rainfall (mm)</label>
  <input name="rainfall" type="number" step="any" value="{{ form.rainfall }}">
  <label>Pesticides (tonnes, aggregate)</label>
  <input name="pesticides" type="number" step="any" value="{{ form.pesticides }}">
  <label>Mean temperature (°C)</label>
  <input name="temperature" type="number" step="any" value="{{ form.temperature }}">
  <button type="submit">Predict yield</button>
</form>

{% if error %}<p class="error">{{
