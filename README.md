# CropCast: Crop Yield Prediction Using Weather and Pesticide Data

Educational prototype comparing Linear Regression and Gradient Boosting with
historical baselines for six Indian crops (1990–2013).

## Open the notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patlollavarsha/cropcast/blob/main/CropCast.ipynb)

## Files
- `CropCast.ipynb`: full notebook (data preparation, models, validation, prediction form)
- `annual_crop_table.csv`: prepared annual dataset (138 records)
- `app.py`: Streamlit prediction form (web version)
- `requirements.txt`: Python packages

## Key result
The last-known-yield baseline had the lowest mean validation MAE (0.4081 t/ha).
The ML models did not consistently beat it.

## Limitations
Country-level data, not farm-level. Not an operational forecasting service.

**Author:** Varsha Patlolla, XTRAGRAD Pvt Ltd internship, 2026
