from fastapi import APIRouter, HTTPException
from backend.app.schemas.mlops_schema import DriftEvaluationRequest, DriftEvaluationResponse
from backend.app.services.mlops_service import mlops_engine

router = APIRouter(prefix="/api/v1/mlops", tags=["MLOps Model Drift & Retraining Control Plane"])

@router.post("/evaluate", response_model=DriftEvaluationResponse)
async def evaluate_model_drift(payload: DriftEvaluationRequest):
    try:
        result = mlops_engine.evaluate_drift(
            payload.batch_size, payload.psi_threshold, payload.ks_p_value_threshold
        )
        return DriftEvaluationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
