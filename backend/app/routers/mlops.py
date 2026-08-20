from fastapi import APIRouter
from app.schemas.mlops import PipelineTriggerResponse
from app.services.mlops_service import run_retraining_pipeline

router = APIRouter(prefix="/api", tags=["MLOps Pipeline"])

@router.post("/trigger", response_model=PipelineTriggerResponse)
def trigger_pipeline():
    return run_retraining_pipeline()