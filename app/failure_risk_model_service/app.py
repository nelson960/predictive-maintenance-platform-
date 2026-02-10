import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/data/models/current/model.joblib"))
FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "/data/features_offline"))
MAX_SCAN_FILES = int(os.getenv("MAX_SCAN_FILES", "200"))

app = FastAPI(title="Predictive Maintenance Model Service", version="0.1.0")

MODEL_ARTIFACT = None


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    failure_probability: float
    risk_level: str
    model_run_id: Optional[str]


class ReloadResponse(BaseModel):
    loaded: bool
    model_path: str
    loaded_at: str


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def ensure_model_loaded(force: bool = False):
    global MODEL_ARTIFACT
    if MODEL_ARTIFACT is not None and not force:
        return MODEL_ARTIFACT

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"model artifact not found at {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
        raise HTTPException(status_code=503, detail="model artifact is invalid")

    MODEL_ARTIFACT = artifact
    return MODEL_ARTIFACT


def vectorize(features: Dict[str, float], feature_columns):
    return np.array([[safe_float(features.get(column, 0.0)) for column in feature_columns]], dtype=np.float64)


def predict_probability(features: Dict[str, float]) -> Tuple[float, Dict]:
    artifact = ensure_model_loaded()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    vector = vectorize(features, feature_columns)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(vector)[0][1])
    else:
        probability = float(model.predict(vector)[0])

    probability = min(max(probability, 0.0), 1.0)
    return probability, artifact


def find_latest_machine_row(machine_id: str) -> Tuple[Dict, Optional[str]]:
    if not FEATURES_DIR.exists():
        raise HTTPException(status_code=404, detail=f"features directory not found: {FEATURES_DIR}")

    files = sorted(FEATURES_DIR.glob("**/*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if MAX_SCAN_FILES > 0:
        files = files[:MAX_SCAN_FILES]

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("machine_id") == machine_id:
                return row, str(path)

    raise HTTPException(status_code=404, detail=f"no feature row found for machine_id={machine_id}")


@app.on_event("startup")
def startup_event():
    try:
        ensure_model_loaded(force=True)
    except HTTPException:
        # service should still start so a model can be trained and loaded later
        pass


@app.get("/health")
def health():
    model_exists = MODEL_PATH.exists()
    return {
        "status": "ok",
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "model_exists": model_exists,
        "model_loaded": MODEL_ARTIFACT is not None,
    }


@app.get("/model/info")
def model_info():
    artifact = ensure_model_loaded()
    return {
        "run_id": artifact.get("run_id"),
        "created_at": artifact.get("created_at"),
        "training_rows": artifact.get("training_rows"),
        "feature_columns": artifact.get("feature_columns", []),
        "metrics": artifact.get("metrics", {}),
        "model_path": str(MODEL_PATH),
    }


@app.post("/reload", response_model=ReloadResponse)
def reload_model():
    ensure_model_loaded(force=True)
    return ReloadResponse(
        loaded=True,
        model_path=str(MODEL_PATH),
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    probability, artifact = predict_probability(request.features)
    return PredictResponse(
        failure_probability=probability,
        risk_level=risk_level(probability),
        model_run_id=artifact.get("run_id"),
    )


@app.get("/predict/latest/{machine_id}")
def predict_latest(machine_id: str):
    row, source_file = find_latest_machine_row(machine_id)
    probability, artifact = predict_probability(row)
    return {
        "machine_id": machine_id,
        "failure_probability": probability,
        "risk_level": risk_level(probability),
        "model_run_id": artifact.get("run_id"),
        "feature_event_ts": row.get("event_ts"),
        "source_file": source_file,
    }
