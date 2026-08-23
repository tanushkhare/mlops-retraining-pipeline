from fastapi import APIRouter
from backend.app.schemas.mlops import DriftEvaluationRequest, DriftEvaluationResponse
from backend.app.services.mlops_service import mlops_service

router = APIRouter(prefix="/api/v1/mlops", tags=["MLOps Retraining Engine"])

@router.post("/evaluate", response_model=DriftEvaluationResponse)
async def evaluate_drift(payload: DriftEvaluationRequest):
    return mlops_service.evaluate_batch(payload)