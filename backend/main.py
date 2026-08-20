from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import mlops

app = FastAPI(title="MLOps Pipeline API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(mlops.router)

@app.get("/")
def read_root():
    return {"message": "MLOps Engine is online!"}