# API Interaction Guide

This document covers every available HTTP request for the two running APIs:

- Failure-Risk Prediction API (`http://localhost:8000`)
- Anomaly Detection API (`http://localhost:8001`)

## 1) Start Services First

From project root:

```bash
cd <repo-root>

# make sure feature data exists
./scripts/pipeline.sh data_generation start --duration 120 --clear-data

# train and start failure-risk API
./scripts/pipeline.sh failure-risk-prediction train --build
./scripts/pipeline.sh failure-risk-prediction start

# train and start anomaly API
./scripts/pipeline.sh anomaly train --build
./scripts/pipeline.sh anomaly start
```

## 2) Interactive API Docs (Swagger)

FastAPI exposes interactive docs automatically:

- Failure-risk docs: <http://localhost:8000/docs>
- Failure-risk OpenAPI JSON: <http://localhost:8000/openapi.json>
- Anomaly docs: <http://localhost:8001/docs>
- Anomaly OpenAPI JSON: <http://localhost:8001/openapi.json>

## 3) Failure-Risk Prediction API (`:8000`)

Base URL:

```bash
export FAILURE_API="http://localhost:8000"
```

### 3.1 GET `/health`

Purpose: service liveness and model load state.

```bash
curl -s "$FAILURE_API/health"
```

Example response:

```json
{
  "status": "ok",
  "utc_time": "2026-02-10T11:05:01.870081+00:00",
  "model_path": "/data/models/current/model.joblib",
  "model_exists": true,
  "model_loaded": true
}
```

### 3.2 GET `/model/info`

Purpose: active model metadata and evaluation metrics.

```bash
curl -s "$FAILURE_API/model/info"
```

Example response:

```json
{
  "run_id": "20260210T105550Z",
  "created_at": "2026-02-10T10:55:50.547198+00:00",
  "training_rows": 1190,
  "feature_columns": ["temperature", "vibration", "pressure"],
  "metrics": {
    "accuracy": 0.79,
    "precision": 0.50,
    "recall": 0.12,
    "f1": 0.20,
    "roc_auc": 0.79
  },
  "model_path": "/data/models/current/model.joblib"
}
```

Common error:

- `503` if model artifact is missing or invalid.

### 3.3 POST `/reload`

Purpose: force reload model from disk.

```bash
curl -s -X POST "$FAILURE_API/reload"
```

Example response:

```json
{
  "loaded": true,
  "model_path": "/data/models/current/model.joblib",
  "loaded_at": "2026-02-10T11:10:00.000000+00:00"
}
```

Common error:

- `503` if model file is missing/invalid.

### 3.4 POST `/predict`

Purpose: score a single feature payload.

Request body shape:

```json
{
  "features": {
    "temperature": 74.2,
    "vibration": 0.31,
    "pressure": 36.7,
    "rpm": 1870,
    "load": 0.82,
    "temp_mean_5s": 73.9,
    "temp_std_5s": 0.8,
    "temp_slope_5s": 0.22,
    "vib_mean_5s": 0.29,
    "vib_std_5s": 0.04,
    "vib_slope_5s": 0.01,
    "pressure_mean_5s": 36.0,
    "pressure_std_5s": 0.7,
    "pressure_slope_5s": 0.12,
    "event_count_5s": 5,
    "event_count_30s": 30
  }
}
```

Request:

```bash
curl -s -X POST "$FAILURE_API/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "temperature": 74.2,
      "vibration": 0.31,
      "pressure": 36.7,
      "rpm": 1870,
      "load": 0.82,
      "temp_mean_5s": 73.9,
      "temp_std_5s": 0.8,
      "temp_slope_5s": 0.22,
      "vib_mean_5s": 0.29,
      "vib_std_5s": 0.04,
      "vib_slope_5s": 0.01,
      "pressure_mean_5s": 36.0,
      "pressure_std_5s": 0.7,
      "pressure_slope_5s": 0.12,
      "event_count_5s": 5,
      "event_count_30s": 30
    }
  }'
```

Example response:

```json
{
  "failure_probability": 0.62,
  "risk_level": "medium",
  "model_run_id": "20260210T105550Z"
}
```

Common errors:

- `422` invalid JSON/body schema.
- `503` model missing/invalid.

### 3.5 GET `/predict/latest/{machine_id}`

Purpose: auto-load latest feature row on disk for one machine and score it.

```bash
curl -s "$FAILURE_API/predict/latest/M-001"
```

Example response:

```json
{
  "machine_id": "M-001",
  "failure_probability": 0.0863,
  "risk_level": "low",
  "model_run_id": "20260210T105550Z",
  "feature_event_ts": "2026-02-10T10:54:02.754515+00:00",
  "source_file": "/data/features_offline/dt=2026-02-10/hour=10/part-1770720843768.jsonl"
}
```

Common errors:

- `404` no feature file or no row for that machine.
- `503` model missing/invalid.

## 4) Anomaly Detection API (`:8001`)

Base URL:

```bash
export ANOMALY_API="http://localhost:8001"
```

### 4.1 GET `/health`

Purpose: service liveness and baseline load state.

```bash
curl -s "$ANOMALY_API/health"
```

Example response:

```json
{
  "status": "ok",
  "utc_time": "2026-02-10T11:59:12.880433+00:00",
  "baseline_path": "/data/models/anomaly/current/baseline.json",
  "baseline_exists": true,
  "baseline_loaded": true
}
```

### 4.2 GET `/baseline/info`

Purpose: current baseline metadata and runtime thresholds.

```bash
curl -s "$ANOMALY_API/baseline/info"
```

Example response:

```json
{
  "run_id": "20260210T115615Z",
  "created_at": "2026-02-10T11:56:15.636960+00:00",
  "monitored_features": ["temperature", "vibration", "pressure"],
  "machines_with_baseline": 10,
  "zscore_threshold": 3.0,
  "persistence_windows": 3,
  "baseline_path": "/data/models/anomaly/current/baseline.json"
}
```

Common error:

- `503` baseline missing/invalid.

### 4.3 POST `/reload`

Purpose: force reload anomaly baseline from disk.

```bash
curl -s -X POST "$ANOMALY_API/reload"
```

Example response:

```json
{
  "loaded": true,
  "baseline_path": "/data/models/anomaly/current/baseline.json",
  "loaded_at": "2026-02-10T12:01:00.000000+00:00"
}
```

Common error:

- `503` baseline missing/invalid.

### 4.4 POST `/anomaly`

Purpose: detect anomaly from provided feature payload.

Request body shape:

```json
{
  "machine_id": "M-001",
  "features": {
    "temperature": 91.0,
    "vibration": 0.52,
    "pressure": 42.0,
    "rpm": 2050,
    "load": 0.95,
    "temp_mean_5s": 88.2,
    "vib_mean_5s": 0.49,
    "pressure_mean_5s": 40.5,
    "rpm_mean_5s": 2010,
    "load_mean_5s": 0.93
  }
}
```

Request:

```bash
curl -s -X POST "$ANOMALY_API/anomaly" \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "M-001",
    "features": {
      "temperature": 91.0,
      "vibration": 0.52,
      "pressure": 42.0,
      "rpm": 2050,
      "load": 0.95,
      "temp_mean_5s": 88.2,
      "vib_mean_5s": 0.49,
      "pressure_mean_5s": 40.5,
      "rpm_mean_5s": 2010,
      "load_mean_5s": 0.93
    }
  }'
```

Example response:

```json
{
  "machine_id": "M-001",
  "anomaly_flag": true,
  "severity": "critical",
  "reason": "temperature_above_hard_limit",
  "anomaly_score": 2.0111,
  "raw_flag": true,
  "consecutive_count": 3,
  "persistence_windows": 3,
  "trigger_count": 2,
  "triggers": [
    {
      "code": "temperature_above_hard_limit",
      "feature": "temperature",
      "value": 91.0,
      "threshold": 90.0,
      "kind": "hard_limit",
      "score": 2.0111
    }
  ],
  "feature_event_ts": null,
  "source_file": null,
  "baseline_run_id": "20260210T115615Z"
}
```

Common errors:

- `400` if `machine_id` is missing in both top-level field and `features.machine_id`.
- `422` invalid JSON/body schema.
- `503` baseline missing/invalid.

### 4.5 GET `/anomaly/latest/{machine_id}`

Purpose: auto-load latest feature row on disk for one machine and evaluate anomaly.

```bash
curl -s "$ANOMALY_API/anomaly/latest/M-001"
```

Example response:

```json
{
  "machine_id": "M-001",
  "anomaly_flag": false,
  "severity": "normal",
  "reason": "normal",
  "anomaly_score": 0.0,
  "raw_flag": false,
  "consecutive_count": 0,
  "persistence_windows": 3,
  "trigger_count": 0,
  "triggers": [],
  "feature_event_ts": "2026-02-10T10:54:02.754515+00:00",
  "source_file": "/data/features_offline/dt=2026-02-10/hour=10/part-1770720843768.jsonl",
  "baseline_run_id": "20260210T115615Z"
}
```

Common errors:

- `404` no feature file or no row for that machine.
- `503` baseline missing/invalid.

## 5) Quick End-to-End Smoke Test

```bash
curl -s "$FAILURE_API/health"
curl -s "$FAILURE_API/model/info"
curl -s "$FAILURE_API/predict/latest/M-001"

curl -s "$ANOMALY_API/health"
curl -s "$ANOMALY_API/baseline/info"
curl -s "$ANOMALY_API/anomaly/latest/M-001"
```

## 6) Notes

- Failure-risk model currently trains from synthetic labels built from engineered risk heuristics.
- Anomaly model uses hard limits + z-score deviations + persistence windows.
- C-MAPSS trainer currently generates offline artifacts and does not expose additional HTTP endpoints.
