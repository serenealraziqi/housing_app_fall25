import os
from typing import Any, Dict

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📞", layout="centered")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# API_URL is set in docker-compose environment
# Expect API_URL to be a BASE like "http://api:8000" (not including /predict)
API_BASE_URL = os.getenv("API_URL", "http://api:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("📞 Telco Customer Churn Prediction App")
st.write(f"This app sends your inputs to the FastAPI backend at **{API_BASE_URL}** for prediction.")

st.header("Input Features (Telco)")

user_input: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Inputs (match your model/API)
# -----------------------------------------------------------------------------
st.subheader("Customer Info")

user_input["tenure"] = st.number_input(
    "Tenure (months)",
    min_value=0,
    value=12,
    step=1,
    help="How many months the customer has stayed with the company.",
)

user_input["MonthlyCharges"] = st.number_input(
    "MonthlyCharges ($)",
    min_value=0.0,
    value=70.0,
    step=1.0,
)

user_input["TotalCharges"] = st.number_input(
    "TotalCharges ($)",
    min_value=0.0,
    value=840.0,
    step=10.0,
    help="Total charges to date. (If blank in dataset, your model handled it as missing.)",
)

user_input["Contract"] = st.selectbox(
    "Contract",
    options=["Month-to-month", "One year", "Two year"],
)

st.markdown("---")

# -----------------------------------------------------------------------------
# Predict Button
# -----------------------------------------------------------------------------
if st.button("🔮 Predict", type="primary"):
    payload = {"instances": [user_input]}

    with st.spinner("Calling API for prediction..."):
        try:
            resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request to API failed: {e}")
            st.write("Tried URL:", PREDICT_ENDPOINT)
            st.write("Payload:", payload)
        else:
            preds = data.get("predictions", [])
            probs = data.get("probabilities", None)

            if not preds:
                st.warning("⚠️ No predictions returned from API.")
                st.write("Response:", data)
            else:
                pred = int(preds[0])
                proba = float(probs[0]) if probs else None

                st.success("✅ Prediction successful!")
                st.subheader("Prediction Result")

                label = "Churn (Yes)" if pred == 1 else "No Churn (No)"
                st.metric("Predicted Class", label)

                if proba is not None:
                    st.metric("Probability of Churn", f"{proba:.4f}")

                with st.expander("📋 View Input Summary"):
                    st.json(user_input)

st.markdown("---")
st.caption(f"🌐 API Base: `{API_BASE_URL}`  |  🎯 Endpoint: `{PREDICT_ENDPOINT}`")
