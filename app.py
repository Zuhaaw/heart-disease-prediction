"""
Heart Disease Prediction System - Streamlit Web App
----------------------------------------------------
A small web app built from the "Heart Disease Prediction" notebook
(adapted from the Titanic Passenger Survival Prediction example).

It loads the Support Vector Classifier saved by the notebook
(svc_trained_model.pkl) and predicts whether heart disease is present
from 4 input attributes: sex, cp, exang and oldpeak.

No label encoding is used: all four attributes are already numeric, so the
values chosen in the web form are passed straight to the model.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

# ---------------------------------------------------------------------------
# Display labels for the numeric codes used in the dataset.
# The model receives the NUMBERS (the dictionary keys), not the labels.
# ---------------------------------------------------------------------------

SEX_LABELS = {1: "Male", 0: "Female"}
CP_LABELS = {
    1: "1 - Typical angina",
    2: "2 - Atypical angina",
    3: "3 - Non-anginal pain",
    4: "4 - Asymptomatic",
}
EXANG_LABELS = {0: "No", 1: "Yes"}

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    This is a machine learning course project. Enter a patient's details below and the
    trained Support Vector Classifier (SVC) will predict whether **heart disease is present**.

    *Model performance on the UCI Cleveland heart disease data (303 patients):
    **84%** accuracy on the held-out test set (61 patients), **about 79%** in
    5-fold cross-validation.*
    """
)

st.warning(
    "Educational demo only - **not a medical tool**. It must not be used to diagnose "
    "or make any health decision. See a doctor for real medical advice."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", options=list(SEX_LABELS.keys()), format_func=lambda v: SEX_LABELS[v])
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=list(CP_LABELS.keys()),
        format_func=lambda v: CP_LABELS[v],
        index=3,
    )
with col2:
    exang = st.selectbox(
        "Exercise-Induced Angina (exang)",
        options=list(EXANG_LABELS.keys()),
        format_func=lambda v: EXANG_LABELS[v],
    )
    oldpeak = st.number_input(
        "ST Depression Induced by Exercise (oldpeak)",
        min_value=0.0,
        max_value=7.0,
        value=1.0,
        step=0.1,
        help="Measured on an exercise ECG. In the training data it ranges from 0.0 to 6.2.",
    )

if st.button("Predict Heart Disease", type="primary"):
    # Feature vector: same 4 columns, in the same order, as the training data
    feature_vector = pd.DataFrame({
        "sex": [sex],
        "cp": [cp],
        "exang": [exang],
        "oldpeak": [oldpeak],
    })

    prediction = model.predict(feature_vector)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE PRESENT (predicted)")
    else:
        st.success("### ✅ NO HEART DISEASE (predicted)")

    with st.expander("See the feature vector sent to the model"):
        st.dataframe(feature_vector, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes from the UCI Heart Disease (Cleveland) dataset using only "
    "4 input attributes. Predictions reflect patterns in a small dataset and are not a "
    "clinical assessment."
)
