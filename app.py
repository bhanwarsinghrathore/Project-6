import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rossmann Sales Forecast", layout="wide")

st.title(" Rossmann Sales Forecast App")

# Load model
model = joblib.load("models/rf_pipeline_15-02-2026-12-33-47.pkl")

st.write("Upload a CSV with future dates and store info to predict sales.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload FUTURE data CSV (no Sales column)",
    type=["csv"]
)

# Date features
def create_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday
    df["IsWeekend"] = df["Weekday"].isin([5,6]).astype(int)
    return df

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Safety check
    if "Sales" in df.columns:
        st.warning("Remove 'Sales' column. This app predicts sales.")
        st.stop()

    # Create date features
    if "Date" in df.columns:
        df = create_date_features(df)

    try:
        preds = model.predict(df)
    except Exception as e:
        st.error("Feature mismatch. Ensure CSV has required columns.")
        st.stop()

    # Closed stores = 0 sales
    if "Open" in df.columns:
        preds[df["Open"]==0] = 0

    df["Predicted_Sales"] = preds

    st.subheader("Predictions")
    st.dataframe(df.head())

    # Plot
    st.subheader("Sales Forecast Chart")
    fig, ax = plt.subplots()
    ax.plot(df["Predicted_Sales"])
    ax.set_ylabel("Sales")
    ax.set_xlabel("Rows")
    st.pyplot(fig)

    # Download
    st.download_button(
        "⬇ Download Predictions CSV",
        df.to_csv(index=False),
        "predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to start forecasting.")

