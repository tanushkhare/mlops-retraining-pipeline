import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_drift_evaluation_retraining_trigger():
    payload = {
        "batch_size": 3000,
        "psi_threshold": 0.20,
        "ks_p_value_threshold": 0.05
    }
    res = client.post("/api/v1/mlops/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "DRIFT-" in data["evaluation_id"]
    assert len(data["feature_metrics"]) >= 5
    assert data["retraining_triggered"] is True
