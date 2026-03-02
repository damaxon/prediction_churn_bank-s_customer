import joblib
import pandas as pd
from pathlib import Path
from typing import Dict


def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model

def predict(model, input_dict: Dict):
    df = pd.DataFrame([input_dict])
    proba = model.predict_proba(df)[0, 1]
    pred = model.predict(df)[0]
    return pred, proba