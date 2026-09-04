# ⚡ MLOps Retraining Pipeline

[![Live Web Demo](https://img.shields.io/badge/Live_App-Vercel-black?style=for-the-badge&logo=vercel)](https://mlops-retraining-pipeline.vercel.app)
[![Portfolio Hub](https://img.shields.io/badge/Portfolio_Hub-Live-blue?style=for-the-badge)](https://portfolio-showcase-hub-web11.vercel.app)

🔗 **Production URL:** [https://mlops-retraining-pipeline.vercel.app](https://mlops-retraining-pipeline.vercel.app)  
🌐 **Showcase Hub:** [https://portfolio-showcase-hub-web11.vercel.app](https://portfolio-showcase-hub-web11.vercel.app)

---

## 📌 Architectural Overview
Statistical covariate drift engine tracking Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests to trigger automated retraining pipelines.

---

## 🛠️ Technology Ecosystem
* **Core Architecture:** SciPy, NumPy, FastAPI, Statistical Engine
* **Testing & Quality:** PyTest, Automated GitHub Actions CI
* **Deployment:** Vercel Edge Runtime

---

## 🛡️ Production Standards
* **Configurable Sensitivity:** Parameters `batch_size` and `psi_threshold` control trigger thresholds dynamically.
* **Standardized Routes:** Unified `/health` and `/api/v1/drift/evaluate` endpoints.
* **Closed-Loop System:** Retraining triggers are queued as soon as distributions drift beyond PSI 0.25.

---

## 🚀 API Contracts
```http
POST /api/v1/drift/evaluate
Request:
{
  "batch_size": 1000,
  "psi_threshold": 0.25
}

Response (200 OK):
{
  "psi_score": 0.312,
  "ks_p_value": 0.0028,
  "drift_detected": true,
  "retrain_recommended": true,
  "job_id": "job_retrain_8192"
}

GET /health
Response: {"status": "healthy"}


💻 Local Quickstart

Bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
pytest tests/ -v