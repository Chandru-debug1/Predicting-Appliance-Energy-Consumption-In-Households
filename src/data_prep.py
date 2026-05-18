import pandas as pd
from .config import DATE_COL, TARGET

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["hour"] = df[DATE_COL].dt.hour
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["month"] = df[DATE_COL].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df

def make_xy(df):
    X = df.drop(columns=[DATE_COL, TARGET, "rv1", "rv2"], errors="ignore")
    y = df[TARGET]
    return X, y