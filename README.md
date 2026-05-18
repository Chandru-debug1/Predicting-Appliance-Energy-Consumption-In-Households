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
Project goal
The main goal is to build a model that predicts appliance consumption accurately, then compare multiple algorithms and choose the best one. You also want to present the result in a Streamlit dashboard so the project looks production-ready on GitHub.

ML workflow
Your pipeline should follow this sequence:

Load the CSV file.

Clean and preprocess the data.

Extract time features from date.

Drop rv1 and rv2.

Split into train and test sets.

Train 4 regression models.

Evaluate them with regression metrics.

Select the best model.

Save the model.

Show performance in Streamlit.

Models you used
Your project compares these 4 models:

Linear Regression.

Ridge Regression.

Random Forest Regressor.

Gradient Boosting Regressor.

These give you a baseline linear model, a regularized linear model, and two nonlinear tree-based models.

Evaluation metrics
You should evaluate the models using:

MAE: average absolute error.

RMSE: square-rooted average squared error.

R²: how much variance the model explains.

These are standard metrics for regression problems.

Feature engineering
The most useful new features are time-based:

hour

day_of_week

month

is_weekend

These help the model capture daily and weekly appliance usage patterns.

Missing values handling
Your code should handle NaN values using SimpleImputer inside the sklearn pipeline. This is necessary because Linear Regression and some other models cannot train directly on missing values.

Streamlit dashboard
Your Streamlit app should act as the presentation layer for the project. It should show:

model comparison table,

RMSE bar chart,

actual vs predicted chart,

residual plot,

trend plots,

prediction form,

download buttons for outputs.

What the dashboard output means
The dashboard output should show the user:

which model performed best,

how accurate each model was,

what the predicted appliance value is for a given input,

how predictions compare with actual values.

Project outputs
After training, the project should create:

best_model.joblib,

model_results.csv,

dashboard plots,

prediction values from the U

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

## License

This project is released for learning and portfolio use. Add an MIT license if you want it open-source.

## Author

Chandru M
