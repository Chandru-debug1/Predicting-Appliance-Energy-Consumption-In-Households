# Appliance Energy Prediction Dashboard

A machine learning project that predicts household appliance energy consumption using sensor and weather data, with an interactive Streamlit dashboard for model comparison, analysis, and prediction.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
## Features

- Predicts appliance energy consumption.
- Trains and compares 4 regression models.
- Automatically selects the best model based on RMSE.
- Handles missing values using imputation.
- Provides an interactive Streamlit dashboard.
- Shows model performance with charts and metrics.
- Supports single-row prediction input.
- Allows downloading filtered data and model results.

## Models Used

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Project Structure

```text
appliance-energy-prediction/
├── data/
│   └── energydata_complete.csv
├── models/
│   └── best_model.joblib
├── reports/
│   └── model_results.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_prep.py
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The dataset contains timestamped household energy data with the target column `Appliances`, along with indoor temperature, humidity, outdoor weather features, and two random variables `rv1` and `rv2`. The random variables are excluded from training. [file:11]

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/appliance-energy-prediction.git
cd appliance-energy-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Place the dataset

Put `energydata_complete.csv` inside the `data/` folder.

### 2. Train the models

```bash
python -m src.train
```

This will train all models, evaluate them, and save the best model and results. [web:85][web:82]

### 3. Run the dashboard

```bash
streamlit run app.py
```

The dashboard displays model comparison charts, actual vs predicted plots, residual analysis, target trends, and a prediction panel. Streamlit’s Plotly, tabs, columns, and download widgets are well-suited for this layout. [web:19][web:63][web:24]

## Dashboard Sections

- **Model Performance**: compares MAE, RMSE, and R² across all models.
- **Data Trends**: shows appliance usage over time, distributions, and feature correlation.
- **Predict**: allows manual input for a single prediction.
- **Downloads**: exports model results and filtered data.

## Output Files

After training, the project generates:

- `models/best_model.joblib`
- `reports/model_results.csv`

## Performance Logic

The best model is selected using the lowest RMSE on the test split. RMSE is a strong regression metric because it penalizes larger errors more heavily and is easy to interpret in the target’s original scale. [web:89][web:95]

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Notes

- The app uses imputation to handle missing values safely.
- `rv1` and `rv2` are excluded from feature selection.
- The project is structured to be GitHub-ready and easy to extend.

## Future Improvements

- Hyperparameter tuning.
- Cross-validation.
- Feature importance visualization.
- Model persistence versioning.
- Deployment to Streamlit Community Cloud.

## Project Goal

The main goal is to build a model that predicts appliance consumption accurately, then compare multiple algorithms and choose the best one. The result is presented in a Streamlit dashboard so the project looks production-ready on GitHub.

## ML Workflow

Your pipeline follows this sequence:

1. Load the CSV file.
2. Clean and preprocess the data.
3. Extract time features from `date`.
4. Drop `rv1` and `rv2`.
5. Split into train and test sets.
6. Train 4 regression models.
7. Evaluate them with regression metrics.
8. Select the best model.
9. Save the model.
10. Show performance in Streamlit.

## Models You Used

Your project compares these 4 models:

- Linear Regression.
- Ridge Regression.
- Random Forest Regressor.
- Gradient Boosting Regressor.

These give you a baseline linear model, a regularized linear model, and two nonlinear tree-based models.

## Evaluation Metrics

You evaluate the models using:

- MAE: average absolute error.
- RMSE: square-rooted average squared error.
- R²: how much variance the model explains.

These are standard metrics for regression problems.

## Feature Engineering

The most useful new features are time-based:

- `hour`
- `day_of_week`
- `month`
- `is_weekend`

These help the model capture daily and weekly appliance usage patterns.

## Missing Values Handling

Your code handles NaN values using `SimpleImputer` inside the sklearn pipeline. This is necessary because Linear Regression and some other models cannot train directly on missing values.

## Streamlit Dashboard

Your Streamlit app acts as the presentation layer for the project. It shows:

- model comparison table,
- RMSE bar chart,
- actual vs predicted chart,
- residual plot,
- trend plots,
- prediction form,
- download buttons for outputs.

## What the Dashboard Output Means

The dashboard output shows the user:

- which model performed best,
- how accurate each model was,
- what the predicted appliance value is for a given input,
- how predictions compare with actual values.

## Project Outputs

After training, the project creates:

- `best_model.joblib`,
- `model_results.csv`,
- dashboard plots,
- prediction values from the UI.

## License

This project is released for learning and portfolio use. Add an MIT license if you want it open-source.

## Author

Chandru M
