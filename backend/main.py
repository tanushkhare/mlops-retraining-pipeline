from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers.mlops import router as mlops_router
import uvicorn

app = FastAPI(
    title="MLOps Drift Detection & Retraining Engine",
    description="Statistical PSI and KS-test pipeline triggering automated retraining DAGs.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mlops_router)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mlops-retraining-pipeline"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)