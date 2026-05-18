import pandas as pd
import joblib
from .config import MODEL_PATH

def load_best_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"], bundle["best_model_name"]

def predict_single(model, features, values_dict):
    x = pd.DataFrame([values_dict])[features]
    return float(model.predict(x)[0])