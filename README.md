# ⚡ MLOps Retraining Pipeline

[![Live Web Demo](https://img.shields.io/badge/Live_App-Vercel-black?style=for-the-badge&logo=vercel)](https://mlops-retraining-pipeline.vercel.app)
[![Portfolio Hub](https://img.shields.io/badge/Portfolio_Hub-Live-blue?style=for-the-badge)](https://portfolio-showcase-hub-web11.vercel.app)

🔗 **Production URL:** [https://mlops-retraining-pipeline.vercel.app](https://mlops-retraining-pipeline.vercel.app)  
🌐 **Showcase Hub:** [https://portfolio-showcase-hub-web11.vercel.app](https://portfolio-showcase-hub-web11.vercel.app)

---

## 📌 Architectural Overview
Covariate drift control plane tracking Population Stability Index (PSI) and KS-tests to trigger automated retraining.

---

## 🛠️ Technology Ecosystem
* **Core Architecture:** SciPy, NumPy, FastAPI, Statistical Engine
* **Testing & Quality:** PyTest, Automated GitHub Actions CI
* **Deployment:** Vercel Edge Runtime

---

## 🚀 API Contracts
```http
POST /api/v1/drift/evaluate
GET /health
```

---

## 💻 Local Quickstart
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
pytest tests/ -v
```
