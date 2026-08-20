from pydantic import BaseModel

class PipelineTriggerResponse(BaseModel):
    model_version: str
    accuracy: float
    drift_detected: bool
    status: str