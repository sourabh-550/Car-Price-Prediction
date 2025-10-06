import json
from pathlib import Path
import joblib
import pickle

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from typing import List, Dict, Tuple


APP_DIR = Path(__file__).parent


@st.cache_data
def load_artifacts() -> Tuple[object, object, List[str]]:
    """Load model, scaler and feature names from disk.

    Returns:
        model, scaler, feature_names
    """
    # Paths (assume files sit next to this app)
    model_path = APP_DIR / "Car_price_rf_model.pkl"
    scaler_path = APP_DIR / "scaler_joblib.pkl"
    feat_path = APP_DIR / "feature_names.json"

    # Load feature names
    try:
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Could not load feature names from {feat_path}: {e}")

    # Load scaler
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        # try pickle fallback
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            raise FileNotFoundError(f"Could not load scaler from {scaler_path}: {e}")

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception:
            raise FileNotFoundError(f"Could not load model from {model_path}: {e}")

    return model, scaler, feature_names


def infer_feature_types(feature_names: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Infer numeric and categorical features from feature name list.

    Heuristic: columns that contain an underscore and share a prefix are treated as one-hot
    categorical columns. Single columns with no underscore are numeric.
    """
    groups: Dict[str, List[str]] = {}
    numeric: List[str] = []

    # group by prefix before first underscore
    for name in feature_names:
        if "_" in name:
            prefix, suffix = name.split("_", 1)
            groups.setdefault(prefix, []).append(name)
        else:
            numeric.append(name)

    # Only keep groups that actually look like categorical (more than 1 member)
    categorical = {k: v for k, v in groups.items() if len(v) > 1}

    # If a group has only one member, treat it as numeric (rare)
    for k, v in groups.items():
        if len(v) == 1:
            numeric.extend(v)

    return numeric, categorical


def build_sidebar_inputs(numeric_feats: List[str], categorical_groups: Dict[str, List[str]], scaler) -> Dict[str, object]:
    st.sidebar.header("Input features")
    st.sidebar.write("Enter values for features below. Defaults are reasonable estimates.")

    inputs = {}

    # If scaler has mean_ and scale_ and lengths match numeric_feats, use it for defaults
    scaler_means = None
    if hasattr(scaler, "mean_") and len(getattr(scaler, "mean_", [])) == len(numeric_feats):
        scaler_means = scaler.mean_

    for i, feat in enumerate(numeric_feats):
        default = float(scaler_means[i]) if scaler_means is not None else 0.0
        # Provide some reasonable min/max around default
        low = default - abs(default) * 0.8 - 10
        high = default + abs(default) * 1.5 + 10
        # Ensure bounds make sense
        if low >= high:
            low = default - 1000
            high = default + 1000

        # Use step for integer-like values
        step = 1 if float(default).is_integer() else 0.01
        inputs[feat] = st.sidebar.number_input(label=feat, min_value=float(low), max_value=float(high), value=float(default), step=step, format="%f")

    # Categorical dropdowns for each group
    for grp, columns in categorical_groups.items():
        # extract options as suffixes after first underscore
        options = [c.split("_", 1)[1] for c in columns]
        default = options[0] if options else ""
        choice = st.sidebar.selectbox(label=f"{grp}", options=options, index=0)
        inputs[grp] = choice

    return inputs


def preprocess_inputs(inputs: Dict[str, object], feature_names: List[str], numeric_feats: List[str], categorical_groups: Dict[str, List[str]], scaler) -> pd.DataFrame:
    """Turn the user inputs into a DataFrame matching the model's expected feature order.

    Steps:
        - Create zero-filled row for all features
        - Fill numeric features
        - Set one-hot encodings for categorical groups
        - Scale numeric columns using scaler
    """
    row = {fn: 0 for fn in feature_names}

    # Fill numeric
    for feat in numeric_feats:
        val = inputs.get(feat, 0)
        # ensure numeric
        try:
            val = float(val)
        except Exception:
            val = 0.0
        row[feat] = val

    # Fill categorical one-hot
    for grp, columns in categorical_groups.items():
        selected = inputs.get(grp)
        # If user selected something, map to column name
        if selected is not None:
            selected_col = f"{grp}_{selected}"
            if selected_col in row:
                row[selected_col] = 1
            else:
                # fallback: try to find suffix match ignoring case
                matches = [c for c in columns if c.split("_", 1)[1].lower() == str(selected).lower()]
                if matches:
                    row[matches[0]] = 1

    df = pd.DataFrame([row], columns=feature_names)

    # Scale numeric columns in-place using scaler if possible
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        try:
            # Determine order for numeric_feats and transform
            numeric_values = df[numeric_feats].astype(float).values.reshape(1, -1)
            scaled = scaler.transform(numeric_values)
            # replace scaled values back
            for i, feat in enumerate(numeric_feats):
                df.loc[0, feat] = scaled[0, i]
        except Exception:
            # if scaling fails, continue without scaling
            pass

    return df


def plot_feature_importance(model, feature_names: List[str], top_n: int = 10):
    if not hasattr(model, "feature_importances_"):
        st.info("Model does not provide feature importances.")
        return

    importances = np.array(model.feature_importances_)
    if importances.shape[0] != len(feature_names):
        # try to align by trimming or padding
        min_n = min(len(feature_names), importances.shape[0])
        importances = importances[:min_n]
        names = feature_names[:min_n]
    else:
        names = feature_names

    fi = pd.DataFrame({"feature": names, "importance": importances})
    fi = fi.sort_values("importance", ascending=True).tail(top_n)

    chart = alt.Chart(fi).mark_bar(color="#0f4c81").encode(
        x=alt.X("importance:Q", title="Importance"),
        y=alt.Y("feature:N", sort="-x", title="Feature")
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)


def main():
    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

    # Header / title
    st.markdown(
        "<div style='display:flex;align-items:center;gap:12px;'>"
        "<div style='font-size:36px;font-weight:700;color:#0f4c81;'>🚗 Car Price Predictor</div>"
        "</div>", unsafe_allow_html=True
    )
    st.write("Predict the price of a used car using a pre-trained Random Forest model. Enter feature values in the sidebar and click Predict.")

    # Load artifacts (with caching)
    try:
        model, scaler, feature_names = load_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    # Infer numeric and categorical features
    numeric_feats, categorical_groups = infer_feature_types(feature_names)

    # Layout: sidebar inputs + main results
    inputs = build_sidebar_inputs(numeric_feats, categorical_groups, scaler)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Prediction")
        if st.button("Predict"):
            try:
                df = preprocess_inputs(inputs, feature_names, numeric_feats, categorical_groups, scaler)
                pred = model.predict(df)[0]
                # If model predicts log-price or needs rounding that is unknown; assume direct price
                st.markdown(f"<div style='background:#f5faff;padding:18px;border-radius:8px;'>"
                            f"<h2 style='color:#0f4c81;margin:0;'>₹ {pred:,.2f}</h2>"
                            f"<div style='color:#333;margin-top:6px;'>Predicted sale price</div>"
                            f"</div>", unsafe_allow_html=True)

                # Show raw prediction and some details
                st.write("---")
                st.write("Raw prediction value:" , float(pred))
                st.write("Input features used:")
                st.dataframe(df.T.rename(columns={0: "value"}))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        st.subheader("Model insights")
        st.write("Top feature importances")
        plot_feature_importance(model, feature_names, top_n=10)

        st.write("---")
        # Optional: show model evaluation placeholder (if available)
        if hasattr(model, "oob_score_"):
            st.metric("OOB Score", f"{model.oob_score_:.3f}")

    # Footer / instructions
    st.markdown("---")
    st.markdown("#### Notes")
    st.markdown(
        "- Numeric defaults are estimated from the training scaler when available.\n"
        "- Categorical features are shown as dropdowns. If a category is not present in the training set, results may be unpredictable.\n"
        "- If the app errors on loading, make sure the three artifact files are in the same folder as this `app.py` file."
    )


if __name__ == "__main__":
    main()
