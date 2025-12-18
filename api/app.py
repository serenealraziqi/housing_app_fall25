"""
FastAPI service for Telco Customer Churn prediction.
Loads the trained model and exposes a /predict endpoint.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("/app/models/model.joblib")  # inside the API container

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="FastAPI service for predicting customer churn (Yes/No)",
    version="1.0.0",
)


# -----------------------------------------------------------------------------
# Load model at startup (module import time)
# -----------------------------------------------------------------------------
def load_model(path: Path):
    """Load the trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model from: {path}")
    m = joblib.load(path)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(m).__name__}")
    if hasattr(m, "named_steps"):
        print(f"  Pipeline steps: {list(m.named_steps.keys())}")
    return m


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"✗ ERROR: Failed to load model from {MODEL_PATH}")
    print(f"  Error: {e}")
    raise RuntimeError(f"Failed to load model: {e}")


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Prediction request with list of instances (dicts of features).
    """
    instances: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "tenure": 12,
                        "MonthlyCharges": 70.0,
                        "TotalCharges": 840.0,
                        "Contract": "Month-to-month",
                    }
                ]
            }
        }


class PredictResponse(BaseModel):
    predictions: List[int]                 # 1 = churn, 0 = no churn
    probabilities: Optional[List[float]]   # prob of churn (if available)
    count: int

    class Config:
        schema_extra = {
            "example": {
                "predictions": [1],
                "probabilities": [0.63],
                "count": 1,
            }
        }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Telco Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided. Please provide at least one instance.",
        )

    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format. Could not convert to DataFrame: {e}",
        )

    required_columns = ["tenure", "MonthlyCharges", "TotalCharges", "Contract"]
    missing = set(required_columns) - set(X.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    # Coerce numeric columns safely
    try:
        X["tenure"] = pd.to_numeric(X["tenure"], errors="raise")
        X["MonthlyCharges"] = pd.to_numeric(X["MonthlyCharges"], errors="raise")
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")  # allow blanks -> NaN
        X["Contract"] = X["Contract"].astype(str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad column types: {e}")

    try:
        preds = model.predict(X)
        preds_list = [int(p) for p in preds]

        probs_list = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]  # prob churn=1
            probs_list = [float(p) for p in probs]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {e}",
        )

    return PredictResponse(
        predictions=preds_list,
        probabilities=probs_list,
        count=len(preds_list),
    )


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("Telco Customer Churn Prediction API - Starting Up")
    print("=" * 80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print("API is ready to accept requests!")
    print("=" * 80 + "\n")

