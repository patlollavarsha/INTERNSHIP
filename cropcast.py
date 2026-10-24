import os

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "yield_df.csv")
CROPS = ["Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Sweet potatoes", "Wheat"]
CUTOFF_YEAR = 2009


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

    # Linear Regression by least squares: intercept, one-hot crop, rainfall, pesticides, temperature
    coef = np.linalg.lstsq(design_matrix(train["Item"], train["rainfall_mm"],
                                         train["pesticides_tonnes"], train["temp_c"]),
                           train["yield_hg"].to_numpy(dtype=float), rcond=None)[0]
    return coef, last_known


def design_matrix(items, rain, pest, temp):
    items = pd.Series(items).reset_index(drop=True)
    onehot = np.column_stack([(items == c).astype(float).to_numpy() for c in CROPS])
    ones = np.ones((len(items), 1))
    numeric = np.column_stack([
        pd.Series(rain).reset_index(drop=True).to_nu
