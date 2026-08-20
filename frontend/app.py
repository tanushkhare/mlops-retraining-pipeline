import streamlit as st
import requests

st.title("🔄 MLOps Automated Model Retraining Pipeline")
if st.button("Trigger Retraining Pipeline"):
    res = requests.post("http://127.0.0.1:8000/api/trigger")
    if res.status_code == 200:
        data = res.json()
        st.success(data["status"])
        st.metric("New Model Version", data["model_version"])
        st.metric("Validated Accuracy", f"{data['accuracy'] * 100}%")
        st.write(f"**Data Drift Status:** {'⚠️ Drift Detected (Retrained)' : '✅ Stable' if not data['drift_detected'] else '⚠️ Drift Detected (Retrained)'}")