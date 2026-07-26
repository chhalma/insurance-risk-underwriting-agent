"""Standalone sanity check for the local model artifacts. Run from the project root:

    python scripts/check_model.py
"""

import joblib
import pandas as pd
from xgboost import XGBRegressor

model = XGBRegressor()
model.load_model("ml-model/xgboost_risk_model.json")
preprocessor = joblib.load("ml-model/insurance_preprocessor.pkl")

cases = [
    {"age": 30, "sex": "male", "bmi": 28, "children": 0, "smoker": "no", "region": "northeast"},
    {"age": 30, "sex": "male", "bmi": 28, "children": 0, "smoker": "yes", "region": "northeast"},
    {"age": 18, "sex": "female", "bmi": 22, "children": 0, "smoker": "no", "region": "southwest"},
    {"age": 60, "sex": "female", "bmi": 30, "children": 2, "smoker": "no", "region": "southeast"},
]

for case in cases:
    df = pd.DataFrame([case])
    encoded = preprocessor.transform(df)
    prediction = model.predict(encoded)[0]
    print(f"{case} -> £{prediction:,.2f}")
