import streamlit as st
import pandas as pd
import joblib

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Customer Segment Predictor",
    page_icon="🛍️",
    layout="wide"
)

# ============================================================
# Title
# ============================================================

st.title("🛍️ Customer Segment Predictor")

st.markdown("""
This application predicts customer personas using the trained machine learning model.

Upload a **customer-level CSV dataset**, and the application will automatically assign each customer to one of the predefined customer personas.
""")

# ============================================================
# Load Pipeline
# ============================================================

@st.cache_resource
def load_pipeline():
    return joblib.load(r"models/kmeans_pipeline.pkl")

pipeline = load_pipeline()

# ============================================================
# Persona Mapping
# ============================================================

persona_mapping = {
    0: "Dissatisfied & Delayed Customers",
    1: "Satisfied Mass Market",
    2: "High-Value Spenders",
    3: "Whales / Top Tier VIPs"
}

# ============================================================
# Upload File
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Customer-Level CSV",
    type=["csv"]
)

# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    required_features = [
        "payment_value",
        "mean_review_score",
        "mean_delivery_days"
    ]

    missing_columns = [
        col
        for col in required_features
        if col not in df.columns
    ]

    if len(missing_columns) > 0:

        st.error(
            f"Missing required columns:\n{missing_columns}"
        )

        st.stop()

    if st.button("Predict Customer Segments"):

        X = df[required_features]

        prediction = pipeline.predict(X)

        result = df.copy()

        result["predicted_cluster"] = prediction

        result["customer_persona"] = (
            result["predicted_cluster"]
            .map(persona_mapping)
        )

        st.success("Prediction completed successfully!")

        # ====================================================
        # Metrics
        # ====================================================

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Customers",
            len(result)
        )

        col2.metric(
            "Unique Personas",
            result["customer_persona"].nunique()
        )

        col3.metric(
            "Most Common Persona",
            result["customer_persona"].mode()[0]
        )

        # ====================================================
        # Persona Distribution
        # ====================================================

        st.subheader("Customer Persona Distribution")

        summary = (
            result["customer_persona"]
            .value_counts()
            .reset_index()
        )

        summary.columns = [
            "Customer Persona",
            "Number of Customers"
        ]

        st.dataframe(summary)

        # ====================================================
        # Result
        # ====================================================

        st.subheader("Prediction Result")

        st.dataframe(result)

        # ====================================================
        # Download
        # ====================================================

        csv = result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Result",
            data=csv,
            file_name="customer_segmentation_result.csv",
            mime="text/csv"
        )