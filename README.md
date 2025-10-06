# Car Price Predictor (Streamlit)

This Streamlit app loads a pre-trained Random Forest model and a StandardScaler to predict used car prices interactively.

Files expected in the same folder:
- `Car_price_rf_model.pkl` - trained Random Forest model (joblib or pickle)
- `scaler_joblib.pkl` - StandardScaler (joblib or pickle)
- `feature_names.json` - list of feature names (in the exact order expected by the model)

Run:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

App highlights:
- Dynamic input generation from `feature_names.json` (numeric fields and categorical dropdowns)
- Scales numeric inputs using the provided scaler
- Shows predicted price and top-10 feature importances

If something breaks when loading artifacts, confirm the filenames and that they are in the same directory as `app.py`.
