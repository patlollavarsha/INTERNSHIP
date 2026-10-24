import os

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "yield_df.csv")
CROPS = ["Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Sweet potatoes", "Wheat"]
CUTOFF_YEAR = 2009


def design_matrix(items, rain, pest, temp):
    items = pd.Series(items).reset_index(drop=True)
    rain = pd.Series(rain).reset_index(drop=True).to_numpy(dtype=float)
    pest = pd.Series(pest).reset_index(drop=True).to_numpy(dtype=float)
    temp = pd.Series(temp).reset_index(drop=True).to_numpy(dtype=float)
    onehot = np.column_stack([(items == c).astype(float).to_numpy() for c in CROPS])
    ones = np.ones((len(items), 1))
    numeric = np.column_stack([rain, pest, temp])
    return np.hstack([ones, onehot, numeric])


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

    # Last-known-yield baseline: latest training-period yield per crop
    last_known = train.groupby("Item").tail(1).set_index("Item")["yield_hg"]

    # Linear Regression by least squares
    X = design_matrix(train["Item"], train["rainfall_mm"],
                      train["pesticides_tonnes"], train["temp_c"])
    y = train["yield_hg"].to_numpy(dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef, last_known


coef, last_known = load_and_train()
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

{% if error %}<p class="error">{{ error }}</p>{% endif %}

{% if result %}
<div class="result">
  <h3>Yield estimates (tonnes per hectare)</h3>
  <p><b>Last-known-yield baseline:</b> {{ result.baseline }}</p>
  <p><b>Linear Regression:</b> {{ result.lr }}</p>
  <p class="note">These estimates come from historical records, so they show how the form works, not how accurate it is on unseen data.</p>
</div>
{% endif %}
</body>
</html>
"""

DEFAULTS = {"crop": "Rice, paddy", "rainfall": "1083", "pesticides": "14485.33", "temperature": "25.79"}


@app.route("/", methods=["GET", "POST"])
def index():
    form = dict(DEFAULTS)
    result, error = None, None

    if request.method == "POST":
        form = {
            "crop": request.form.get("crop", DEFAULTS["crop"]),
            "rainfall": request.form.get("rainfall", ""),
            "pesticides": request.form.get("pesticides", ""),
            "temperature": request.form.get("temperature", ""),
        }
        try:
            rainfall = float(form["rainfall"])
            pesticides = float(form["pesticides"])
            temperature = float(form["temperature"])
        except ValueError:
            error = "Please enter numbers for rainfall, pesticides, and temperature."
        else:
            if rainfall < 0 or pesticides < 0:
                error = "Rainfall and pesticides cannot be negative."
            elif form["crop"] not in CROPS:
                error = "Please choose a crop from the list."
            else:
                x = design_matrix([form["crop"]], [rainfall], [pesticides], [temperature])
                result = {
                    "baseline": f"{last_known[form['crop']] / 10000:.3f}",
                    "lr": f"{(x @ coef)[0] / 10000:.3f}",
                }

    return render_template_string(PAGE, crops=CROPS, form=form, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
