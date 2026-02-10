import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "/data/features_offline"))
MODEL_REGISTRY_DIR = Path(os.getenv("MODEL_REGISTRY_DIR", "/data/models/registry"))
MODEL_CURRENT_DIR = Path(os.getenv("MODEL_CURRENT_DIR", "/data/models/current"))

MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "100"))
MAX_FILES = int(os.getenv("MAX_FILES", "0"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

DEFAULT_FEATURE_COLUMNS = (
    "temperature,vibration,pressure,rpm,load,"
    "temp_mean_5s,temp_std_5s,temp_slope_5s,"
    "vib_mean_5s,vib_std_5s,vib_slope_5s,"
    "pressure_mean_5s,pressure_std_5s,pressure_slope_5s,"
    "event_count_5s,event_count_30s"
)
FEATURE_COLUMNS = [
    column.strip()
    for column in os.getenv("FEATURE_COLUMNS", DEFAULT_FEATURE_COLUMNS).split(",")
    if column.strip()
]


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def list_feature_files(features_dir: Path, max_files: int) -> List[Path]:
    files = sorted(features_dir.glob("**/*.jsonl"))
    if max_files > 0:
        return files[-max_files:]
    return files


def load_rows(files: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def compute_risk_score(row: Dict) -> float:
    temp = safe_float(row.get("temp_mean_30s", row.get("temp_mean_5s", row.get("temperature", 0))))
    vib = safe_float(row.get("vib_mean_30s", row.get("vib_mean_5s", row.get("vibration", 0))))
    pressure = safe_float(row.get("pressure_mean_30s", row.get("pressure_mean_5s", row.get("pressure", 0))))
    rpm = safe_float(row.get("rpm_mean_30s", row.get("rpm_mean_5s", row.get("rpm", 0))))
    load = safe_float(row.get("load_mean_30s", row.get("load_mean_5s", row.get("load", 0))))

    vib_slope = abs(safe_float(row.get("vib_slope_5s", 0)))
    temp_slope = abs(safe_float(row.get("temp_slope_5s", 0)))

    score = 0.0
    score += 1.2 * max(0.0, temp - 73.5)
    score += 3.0 * max(0.0, vib - 0.28)
    score += 0.8 * max(0.0, pressure - 36.5)
    score += 0.0010 * max(0.0, rpm - 1880.0)
    score += 1.5 * max(0.0, load - 0.75)
    score += 12.0 * vib_slope
    score += 1.5 * temp_slope
    return float(score)


def build_labels(rows: List[Dict]) -> np.ndarray:
    risk_values = np.array([compute_risk_score(row) for row in rows], dtype=np.float64)
    if risk_values.size == 0:
        return np.array([], dtype=np.int64)

    threshold = float(np.percentile(risk_values, 80))
    labels = (risk_values >= threshold).astype(np.int64)

    if np.unique(labels).size < 2 and risk_values.size > 1:
        labels = np.zeros_like(risk_values, dtype=np.int64)
        labels[np.argsort(risk_values)[risk_values.size // 2 :]] = 1

    return labels


def build_matrix(rows: List[Dict], feature_columns: List[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append([safe_float(row.get(column, 0.0)) for column in feature_columns])
    return np.array(matrix, dtype=np.float64)


def evaluate(model: Pipeline, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        y_prob = y_pred.astype(np.float64)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if np.unique(y_test).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def ensure_dirs() -> None:
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CURRENT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ensure_dirs()

    feature_files = list_feature_files(FEATURES_DIR, MAX_FILES)
    if not feature_files:
        print(f"no feature files found under {FEATURES_DIR}")
        return 2

    rows = load_rows(feature_files)
    if len(rows) < MIN_SAMPLES:
        print(
            f"not enough rows to train: found={len(rows)} min_required={MIN_SAMPLES} "
            f"files={len(feature_files)}"
        )
        return 2

    x = build_matrix(rows, FEATURE_COLUMNS)
    y = build_labels(rows)

    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        print("failed to build dataset")
        return 2

    if np.unique(y).size < 2:
        print("training labels have a single class; collect more varied data")
        return 2

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=None,
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )
    model.fit(x_train, y_train)

    metrics = evaluate(model, x_test, y_test)

    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")

    model_dir = MODEL_REGISTRY_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "created_at": created_at.isoformat(),
        "run_id": run_id,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "training_rows": int(x.shape[0]),
        "model": model,
    }

    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.json"
    metadata_path = model_dir / "metadata.json"

    joblib.dump(artifact, model_path)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    metadata = {
        "created_at": artifact["created_at"],
        "run_id": run_id,
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": artifact["training_rows"],
        "feature_files_used": len(feature_files),
        "model_path": str(model_path),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    current_model_path = MODEL_CURRENT_DIR / "model.joblib"
    current_metadata_path = MODEL_CURRENT_DIR / "metadata.json"
    shutil.copy2(model_path, current_model_path)
    shutil.copy2(metadata_path, current_metadata_path)

    print(f"trained model saved: {model_path}")
    print(f"metrics: {json.dumps(metrics, separators=(',', ':'))}")
    print(f"current model updated: {current_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
