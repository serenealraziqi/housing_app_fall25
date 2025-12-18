from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("/app/models/model.joblib")
_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_one(payload: dict):
    model = get_model()

    X = pd.DataFrame([{
        "tenure": float(payload["tenure"]),
        "MonthlyCharges": float(payload["MonthlyCharges"]),
        "TotalCharges": float(payload["TotalCharges"]),
        "Contract": str(payload["Contract"]),
    }])

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None
    return {"pred_class": pred, "pred_proba": proba}
