import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.config import DATA_PATH, RESULTS_PATH
from src.data_prep import load_data, preprocess_data
from src.train import train_and_select_best
from src.predict import load_best_model, predict_single

st.set_page_config(page_title="Appliance Energy Dashboard", layout="wide")

@st.cache_data
def get_data():
    df = load_data(DATA_PATH)
    return preprocess_data(df)

@st.cache_data
def get_results():
    try:
        return pd.read_csv(RESULTS_PATH)
    except:
        return None

def main():
    st.title("Appliance Energy Prediction Dashboard")

    df = get_data()

    if st.sidebar.button("Train / Retrain Models"):
        results_df, best_model_name = train_and_select_best()
        st.success(f"Best model selected: {best_model_name}")
        st.dataframe(results_df, use_container_width=True)
        st.rerun()

    try:
        model, features, best_model_name = load_best_model()
    except:
        st.warning("No saved model found. Training now...")
        results_df, best_model_name = train_and_select_best()
        model, features, best_model_name = load_best_model()

    st.sidebar.header("Filters")
    min_hour = int(df["hour"].min())
    max_hour = int(df["hour"].max())
    hour_range = st.sidebar.slider("Hour range", min_hour, max_hour, (min_hour, max_hour))
    day_type = st.sidebar.selectbox("Day type", ["All", "Weekday", "Weekend"])

    filtered = df[(df["hour"] >= hour_range[0]) & (df["hour"] <= hour_range[1])]
    if day_type == "Weekday":
        filtered = filtered[filtered["is_weekend"] == 0]
    elif day_type == "Weekend":
        filtered = filtered[filtered["is_weekend"] == 1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(filtered):,}")
    col2.metric("Mean Appliances", f"{filtered['Appliances'].mean():.2f}")
    col3.metric("Best Model", best_model_name)
    col4.metric("Max Appliances", f"{filtered['Appliances'].max():.2f}")

    results_df = get_results()
    if results_df is not None:
        st.subheader("Model Performance")
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(results_df, use_container_width=True)
        with c2:
            fig = px.bar(
                results_df.sort_values("RMSE"),
                x="Model",
                y="RMSE",
                color="Model",
                title="RMSE Comparison Across 4 Models"
            )
            st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Data View", "Trends", "Predict"])

    with tab1:
        st.subheader("Filtered Data")
        st.dataframe(filtered.head(100), use_container_width=True)

    with tab2:
        st.subheader("Target Trend")
        fig1 = px.line(filtered, x="date", y="Appliances", title="Appliance Consumption Over Time")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Target Distribution")
        fig2 = px.histogram(filtered, x="Appliances", nbins=40, title="Appliances Distribution")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Correlation Heatmap")
        num_df = filtered.select_dtypes(include=np.number)
        fig3 = px.imshow(num_df.corr(), aspect="auto", title="Feature Correlation")
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("Single Prediction")
        base_row = filtered[features].mean(numeric_only=True)

        input_dict = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            default_val = float(base_row[feat]) if feat in base_row.index else 0.0
            input_dict[feat] = cols[i % 2].number_input(feat, value=default_val)

        if st.button("Predict Appliances"):
            prediction = predict_single(model, features, input_dict)
            st.success(f"Predicted Appliances value: {prediction:.2f}")

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Data",
        data=csv,
        file_name="filtered_energy_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()