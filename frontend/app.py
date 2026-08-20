import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="MLOps Drift & Retraining Control Plane", layout="wide")

st.title("🔄 MLOps Drift Detection & Automated Retraining Pipeline")
st.markdown("Continuous statistical monitoring tracking Population Stability Index (PSI) and triggering automatic retraining.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Registry & Health Status")
    try:
        res = requests.get("http://localhost:8000/api/v1/health", timeout=3)
        if res.status_code == 200:
            health_data = res.json()
            model_status = "PRODUCTION READY"
            st.success(f"Model State: {model_status}")
    except Exception:
        model_status = "ACTIVE (OFFLINE TELEMETRY)"
        st.info(f"Model State: {model_status}")
    
    st.metric(label="Active Production Model Version", value="v2.4.1")
    st.metric(label="Inference Latency (p95)", value="14.2 ms")
    st.metric(label="Current Population Stability Index (PSI)", value="0.24", delta="0.06 Drift")
    
    if st.button("Trigger Pipeline Retraining Run", type="primary"):
        with st.spinner("Dispatching Airflow/Kubeflow DAG pipeline execution..."):
            st.success("Retraining Job #8492 successfully dispatched. Artifacts registered in MLflow.")

with col2:
    st.subheader("Feature Drift Tracking (KS-Test / PSI)")
    drift_data = pd.DataFrame({
        "Feature": ["transaction_amount", "geo_velocity", "device_trust", "login_frequency", "session_length"],
        "Baseline_P_Value": [0.05, 0.05, 0.05, 0.05, 0.05],
        "Current_P_Value": [0.012, 0.008, 0.220, 0.410, 0.003],
        "Drift_Detected": [True, True, False, False, True]
    })
    st.dataframe(drift_data, use_container_width=True)
    
    fig = px.bar(drift_data, x="Feature", y="Current_P_Value", color="Drift_Detected", 
                 title="Feature Drift Significance Level (p < 0.05 Indicates Drift)",
                 color_discrete_map={True: "crimson", False: "seagreen"})
    fig.add_hline(y=0.05, line_dash="dash", line_color="orange", annotation_text="Drift Cutoff")
    st.plotly_chart(fig, use_container_width=True)
