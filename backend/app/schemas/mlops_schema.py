from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class DriftEvaluationRequest(BaseModel):
    batch_size: int = Field(default=2500, ge=100, le=50000, description="Inference batch size")
    psi_threshold: float = Field(default=0.25, ge=0.05, le=1.0, description="Population Stability Index threshold")
    ks_p_value_threshold: float = Field(default=0.05, ge=0.001, le=0.20, description="KS-test statistical significance threshold")

class FeatureDriftMetric(BaseModel):
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    is_drifted: bool

class DriftEvaluationResponse(BaseModel):
    evaluation_id: str
    batch_size: int
    model_version: str
    overall_psi: float
    drift_detected: bool
    retraining_triggered: bool
    feature_metrics: List[FeatureDriftMetric]
    pipeline_status: str
    timestamp: str
