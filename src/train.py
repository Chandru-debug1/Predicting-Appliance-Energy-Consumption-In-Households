import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from .config import DATA_PATH, MODEL_PATH, RESULTS_PATH, RANDOM_STATE, TEST_SIZE
from .data_prep import load_data, preprocess_data, make_xy
from .evaluate import regression_metrics

def build_models():
    return {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=250,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))
        ]),
    }

def train_and_select_best():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    X, y = make_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = build_models()
    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = regression_metrics(y_test, pred)
        results.append({"Model": name, **metrics})
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = results_df.loc[0, "Model"]
    best_model = fitted_models[best_model_name]

    joblib.dump(
        {
            "model": best_model,
            "features": X.columns.tolist(),
            "best_model_name": best_model_name,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred_best": best_model.predict(X_test)
        },
        MODEL_PATH
    )

    results_df.to_csv(RESULTS_PATH, index=False)
    return results_df, best_model_name

if __name__ == "__main__":
    train_and_select_best()