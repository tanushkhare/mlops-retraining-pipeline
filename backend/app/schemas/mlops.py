from pydantic import BaseModel, Field
from typing import List

class DriftEvaluationRequest(BaseModel):
    batch_size: int = Field(default=2500, ge=100, le=100000)
    psi_threshold: float = Field(default=0.20, ge=0.01, le=1.0)
    ks_p_value_threshold: float = Field(default=0.05, ge=0.001, le=0.5)

class FeatureDriftMetric(BaseModel):
    feature_name: str
    psi_score: float
    ks_p_value: float
    is_drifted: bool

class DriftEvaluationResponse(BaseModel):
    pipeline_id: str
    model_version: str
    overall_psi: float
    drift_status: str
    retraining_recommended: bool
    features: List[FeatureDriftMetric]