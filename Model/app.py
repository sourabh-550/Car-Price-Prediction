from flask import Flask, render_template, request
import pickle, joblib, json
import pandas as pd
import numpy as np

app = Flask(__name__)

# ------------------------------
# Load artifacts
# ------------------------------
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

model = joblib.load("Car_price_rf_model.pkl")
scaler = joblib.load("scaler_joblib.pkl")

# ------------------------------
# Feature metadata for friendly UI
# ------------------------------
FEATURE_METADATA = {
    "symboling": {"label": "Risk Rating", "example": 1, "type": "number"},
    "wheelbase": {"label": "Wheelbase (inches)", "example": 98.0, "type": "number"},
    "carlength": {"label": "Car Length (inches)", "example": 176.0, "type": "number"},
    "carwidth": {"label": "Car Width (inches)", "example": 66.0, "type": "number"},
    "carheight": {"label": "Car Height (inches)", "example": 54.0, "type": "number"},
    "curbweight": {"label": "Curb Weight (kg)", "example": 2500, "type": "number"},
    "enginesize": {"label": "Engine Size (cc)", "example": 2000, "type": "number"},
    "boreratio": {"label": "Bore Ratio", "example": 3.5, "type": "number"},
    "stroke": {"label": "Stroke", "example": 3.4, "type": "number"},
    "compressionratio": {"label": "Compression Ratio", "example": 9.0, "type": "number"},
    "horsepower": {"label": "Horsepower (hp)", "example": 150, "type": "number"},
    "peakrpm": {"label": "Peak RPM", "example": 5000, "type": "number"},
    "citympg": {"label": "City MPG", "example": 20, "type": "number"},
    "highwaympg": {"label": "Highway MPG", "example": 27, "type": "number"},

    # Categorical features
    "fueltype_gas": {"label": "Fuel Type", "example": "Gasoline", "type": "select",
                     "options": {"Gasoline": "fueltype_gas", "Diesel": "fueltype_diesel"}},
    "aspiration_turbo": {"label": "Aspiration", "example": "Standard", "type": "select",
                         "options": {"Standard": "aspiration_std", "Turbo": "aspiration_turbo"}},
    "doornumber_two": {"label": "Number of Doors", "example": "Four", "type": "select",
                       "options": {"Two": "doornumber_two", "Four": "doornumber_four"}},
    "carbody_hardtop": {"label": "Car Body Type", "example": "Sedan", "type": "select",
                        "options": {"Hardtop": "carbody_hardtop", "Hatchback": "carbody_hatchback",
                                    "Sedan": "carbody_sedan", "Wagon": "carbody_wagon"}},
    "drivewheel_fwd": {"label": "Drive Wheel", "example": "Front-Wheel Drive", "type": "select",
                       "options": {"Front-Wheel Drive": "drivewheel_fwd", "Rear-Wheel Drive": "drivewheel_rwd"}},
    "enginetype_ohc": {"label": "Engine Type", "example": "OHC", "type": "select",
                       "options": {"OHC": "enginetype_ohc", "OHCV": "enginetype_ohcv", "OHCF": "enginetype_ohcf",
                                   "DOHCV": "enginetype_dohcv", "L": "enginetype_l", "Rotor": "enginetype_rotor"}},
    "cylindernumber_four": {"label": "Number of Cylinders", "example": 4, "type": "select",
                            "options": {2: "cylindernumber_two", 3: "cylindernumber_three", 4: "cylindernumber_four",
                                        5: "cylindernumber_five", 6: "cylindernumber_six", 12: "cylindernumber_twelve"}},
    "fuelsystem_mpfi": {"label": "Fuel System", "example": "MPFI", "type": "select",
                        "options": {"2BBL": "fuelsystem_2bbl", "4BBL": "fuelsystem_4bbl", "IDI": "fuelsystem_idi",
                                    "MFI": "fuelsystem_mfi", "MPFI": "fuelsystem_mpfi",
                                    "SPDI": "fuelsystem_spdi", "SPFI": "fuelsystem_spfi"}}
}


# ------------------------------
# Preprocess user input
# ------------------------------
def preprocess_input(form):
    row = {fn: 0 for fn in feature_names}

    for feat, meta in FEATURE_METADATA.items():
        # Numeric
        if meta["type"] == "number":
            try:
                val = float(form.get(feat, meta["example"]))
            except:
                val = float(meta["example"])
            row[feat] = val

        # Categorical (one-hot)
        elif meta["type"] == "select":
            selected = form.get(feat)
            if selected is not None:
                col_name = meta["options"].get(selected)
                if col_name and col_name in row:
                    row[col_name] = 1

    df = pd.DataFrame([row], columns=feature_names)

    # Scale numeric features
    numeric_feats = [f for f in FEATURE_METADATA if FEATURE_METADATA[f]["type"] == "number"]
    try:
        numeric_values = df[numeric_feats].values.reshape(1, -1)
        scaled = scaler.transform(numeric_values)
        for i, feat in enumerate(numeric_feats):
            df.loc[0, feat] = scaled[0, i]
    except:
        pass

    return df


# ------------------------------
# Flask route
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        df = preprocess_input(request.form)
        prediction = model.predict(df)[0]
    return render_template("index.html", features=FEATURE_METADATA, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
