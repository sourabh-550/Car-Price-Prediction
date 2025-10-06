"""Simple validation script for the saved Random Forest model.

Usage:
    python validate_model.py

This script will:
 - Load model, scaler, and feature names
 - Load `Car.csv` (expects a `price` column)
 - Construct the feature matrix according to `feature_names.json`
 - Predict prices and compute MAE, RMSE, R2
 - Save a scatter plot `prediction_vs_actual.png`
"""
import json
from pathlib import Path
import joblib
import pickle
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


BASE = Path(__file__).parent


def load_artifacts():
    model_path = BASE / "Car_price_rf_model.pkl"
    scaler_path = BASE / "scaler_joblib.pkl"
    feat_path = BASE / "feature_names.json"

    with open(feat_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return model, scaler, feature_names


def infer_groups(feature_names):
    groups = {}
    numeric = []
    for name in feature_names:
        if "_" in name:
            prefix = name.split("_", 1)[0]
            groups.setdefault(prefix, []).append(name)
        else:
            numeric.append(name)

    categorical = {k: v for k, v in groups.items() if len(v) > 1}
    for k, v in groups.items():
        if len(v) == 1:
            numeric.extend(v)

    return numeric, categorical


def build_X_from_csv(df_raw, feature_names, numeric_feats, categorical_groups):
    # initialize zeros
    row_dicts = []

    for _, r in df_raw.iterrows():
        row = {fn: 0 for fn in feature_names}

        # numeric
        for feat in numeric_feats:
            # If the raw csv has the same column name, use it; else try to parse base name
            if feat in df_raw.columns:
                val = r[feat]
            else:
                # try to extract base name if the numeric feature had underscores (rare)
                base = feat.split("_", 1)[0]
                val = r.get(base, 0)

            try:
                row[feat] = float(val)
            except Exception:
                row[feat] = 0.0

        # categorical: for each group, set the matching one-hot
        for grp, columns in categorical_groups.items():
            raw_val = r.get(grp, None)
            if pd.isna(raw_val):
                continue
            # find column name in the group's columns that matches raw_val
            match = None
            for c in columns:
                suffix = c.split("_", 1)[1]
                if str(suffix).lower() == str(raw_val).lower():
                    match = c
                    break
            if match:
                row[match] = 1

        row_dicts.append(row)

    X = pd.DataFrame(row_dicts, columns=feature_names)
    return X


def scale_numeric(X, numeric_feats, scaler):
    if hasattr(scaler, "mean_") and len(getattr(scaler, "mean_", [])) == len(numeric_feats):
        try:
            vals = X[numeric_feats].astype(float).values
            X[numeric_feats] = scaler.transform(vals)
        except Exception:
            pass
    return X


def main():
    model, scaler, feature_names = load_artifacts()

    csv_path = BASE / "Car.csv"
    if not csv_path.exists():
        print("Car.csv not found in the project folder.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "price" not in df.columns:
        print("Target column 'price' not found in Car.csv")
        sys.exit(1)

    y = df["price"].astype(float).values

    numeric_feats, categorical_groups = infer_groups(feature_names)
    X = build_X_from_csv(df, feature_names, numeric_feats, categorical_groups)
    X = scale_numeric(X, numeric_feats, scaler)

    # Ensure no NaNs
    X = X.fillna(0)

    # Predict
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    # compute RMSE in a backwards-compatible way
    rmse = mean_squared_error(y, preds) ** 0.5
    r2 = r2_score(y, preds)

    print(f"Evaluated on {len(y)} rows")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    # Save scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y, preds, alpha=0.6)
    lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
    plt.plot(lims, lims, color="red", linewidth=1)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    out = BASE / "prediction_vs_actual.png"
    plt.savefig(out)
    print(f"Saved scatter plot to {out}")


if __name__ == "__main__":
    main()
