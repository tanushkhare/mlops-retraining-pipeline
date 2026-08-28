from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers import mlops_router
import uvicorn

app = FastAPI(
    title="MLOps Continuous Retraining & Statistical Drift Control Plane",
    description="Population Stability Index (PSI) and Kolmogorov-Smirnov model drift monitoring engine.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mlops_router.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mlops-retraining-pipeline"}

@app.get("/api/v1/health")
async def health_check_v1():
    return {"status": "healthy", "service": "mlops-retraining-pipeline"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
