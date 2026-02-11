import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

ANOMALY_BASELINE_PATH = Path(os.getenv("ANOMALY_BASELINE_PATH", "/data/models/anomaly/current/baseline.json"))
FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "/data/features_offline"))
MAX_SCAN_FILES = int(os.getenv("MAX_SCAN_FILES", "200"))

# Optional runtime overrides; fall back to trained baseline config if these are not set.
ZSCORE_THRESHOLD_OVERRIDE = os.getenv("ZSCORE_THRESHOLD", "").strip()
PERSISTENCE_WINDOWS_OVERRIDE = os.getenv("PERSISTENCE_WINDOWS", "").strip()

app = FastAPI(title="Predictive Maintenance Anomaly Service", version="0.1.0")

ANOMALY_BASELINE = None
CONSECUTIVE_COUNTS: Dict[str, int] = {}

HTTP_REQUESTS = Counter(
    "anomaly_api_requests_total",
    "Total HTTP requests for the anomaly API",
    ["method", "path", "status"],
)
HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "anomaly_api_request_latency_seconds",
    "HTTP request latency for the anomaly API",
    ["method", "path"],
)
ANOMALY_REQUESTS = Counter(
    "anomaly_api_detect_requests_total",
    "Number of /anomaly calls",
)
ANOMALY_LATEST_REQUESTS = Counter(
    "anomaly_api_detect_latest_requests_total",
    "Number of /anomaly/latest/{machine_id} calls",
)
BASELINE_RELOADS = Counter(
    "anomaly_api_baseline_reloads_total",
    "Number of successful anomaly baseline reloads",
)
BASELINE_LOADED = Gauge(
    "anomaly_api_baseline_loaded",
    "1 if anomaly baseline is loaded, otherwise 0",
)
BASELINE_LAST_RELOAD_UNIX = Gauge(
    "anomaly_api_baseline_last_reload_unix",
    "Unix timestamp of the last successful baseline load/reload",
)


class AnomalyRequest(BaseModel):
    machine_id: Optional[str] = None
    features: Dict[str, float] = Field(default_factory=dict)


class ReloadResponse(BaseModel):
    loaded: bool
    baseline_path: str
    loaded_at: str


def route_path_label(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path


@app.middleware("http")
async def capture_http_metrics(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        path = request.url.path
        HTTP_REQUESTS.labels(request.method, path, "500").inc()
        HTTP_REQUEST_LATENCY_SECONDS.labels(request.method, path).observe(time.perf_counter() - start)
        raise

    path = route_path_label(request)
    HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    HTTP_REQUEST_LATENCY_SECONDS.labels(request.method, path).observe(time.perf_counter() - start)
    return response


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def ensure_baseline_loaded(force: bool = False):
    global ANOMALY_BASELINE

    if ANOMALY_BASELINE is not None and not force:
        return ANOMALY_BASELINE

    if not ANOMALY_BASELINE_PATH.exists():
        BASELINE_LOADED.set(0)
        raise HTTPException(status_code=503, detail=f"anomaly baseline not found at {ANOMALY_BASELINE_PATH}")

    with ANOMALY_BASELINE_PATH.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    if not isinstance(artifact, dict):
        BASELINE_LOADED.set(0)
        raise HTTPException(status_code=503, detail="anomaly baseline is invalid")

    required = ["monitored_features", "global_baseline", "hard_limits"]
    if any(key not in artifact for key in required):
        BASELINE_LOADED.set(0)
        raise HTTPException(status_code=503, detail="anomaly baseline missing required fields")

    ANOMALY_BASELINE = artifact
    BASELINE_LOADED.set(1)
    BASELINE_LAST_RELOAD_UNIX.set(datetime.now(timezone.utc).timestamp())
    return ANOMALY_BASELINE


def get_runtime_config(artifact: Dict) -> Tuple[float, int]:
    trained_config = artifact.get("config", {})

    zscore_threshold = safe_float(trained_config.get("zscore_threshold", 3.0), 3.0)
    persistence_windows = int(safe_float(trained_config.get("persistence_windows", 3), 3))

    if ZSCORE_THRESHOLD_OVERRIDE:
        zscore_threshold = safe_float(ZSCORE_THRESHOLD_OVERRIDE, zscore_threshold)
    if PERSISTENCE_WINDOWS_OVERRIDE:
        persistence_windows = int(safe_float(PERSISTENCE_WINDOWS_OVERRIDE, persistence_windows))

    persistence_windows = max(1, persistence_windows)
    zscore_threshold = max(0.5, zscore_threshold)

    return zscore_threshold, persistence_windows


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


def baseline_stats_for_feature(artifact: Dict, machine_id: str, feature: str) -> Optional[Dict]:
    machine_baselines = artifact.get("machine_baselines", {})
    global_baseline = artifact.get("global_baseline", {})

    machine_entry = machine_baselines.get(machine_id, {})
    if feature in machine_entry:
        return machine_entry[feature]

    return global_baseline.get(feature)


def evaluate_row(machine_id: str, features: Dict, event_ts: Optional[str] = None, source_file: Optional[str] = None) -> Dict:
    artifact = ensure_baseline_loaded()
    monitored_features = artifact.get("monitored_features", [])
    hard_limits = artifact.get("hard_limits", {})
    zscore_threshold, persistence_windows = get_runtime_config(artifact)

    triggers: List[Dict] = []
    max_score = 0.0
    hard_limit_triggered = False

    for feature in monitored_features:
        if feature not in features:
            continue

        value = safe_float(features.get(feature))

        limits = hard_limits.get(feature)
        if isinstance(limits, dict):
            low = safe_float(limits.get("low"), None)
            high = safe_float(limits.get("high"), None)

            if low is not None and value < low:
                hard_limit_triggered = True
                distance = abs((low - value) / max(abs(low), 1e-6))
                score = 2.0 + distance
                max_score = max(max_score, score)
                triggers.append(
                    {
                        "code": f"{feature}_below_hard_limit",
                        "feature": feature,
                        "value": value,
                        "threshold": low,
                        "kind": "hard_limit",
                        "score": round(score, 4),
                    }
                )

            if high is not None and value > high:
                hard_limit_triggered = True
                distance = abs((value - high) / max(abs(high), 1e-6))
                score = 2.0 + distance
                max_score = max(max_score, score)
                triggers.append(
                    {
                        "code": f"{feature}_above_hard_limit",
                        "feature": feature,
                        "value": value,
                        "threshold": high,
                        "kind": "hard_limit",
                        "score": round(score, 4),
                    }
                )

        stats = baseline_stats_for_feature(artifact, machine_id, feature)
        if not stats:
            continue

        mean = safe_float(stats.get("mean"))
        std = max(safe_float(stats.get("std"), 1e-6), 1e-6)
        z = abs((value - mean) / std)

        if z >= zscore_threshold:
            score = z / zscore_threshold
            max_score = max(max_score, score)
            triggers.append(
                {
                    "code": f"{feature}_zscore_high",
                    "feature": feature,
                    "value": value,
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "zscore": round(z, 6),
                    "threshold": zscore_threshold,
                    "kind": "zscore",
                    "score": round(score, 4),
                }
            )

    raw_flag = hard_limit_triggered or max_score >= 1.0

    if raw_flag:
        CONSECUTIVE_COUNTS[machine_id] = CONSECUTIVE_COUNTS.get(machine_id, 0) + 1
    else:
        CONSECUTIVE_COUNTS[machine_id] = 0

    consecutive_count = CONSECUTIVE_COUNTS[machine_id]
    anomaly_flag = hard_limit_triggered or consecutive_count >= persistence_windows

    if hard_limit_triggered:
        severity = "critical"
    elif anomaly_flag and max_score >= 1.5:
        severity = "high"
    elif anomaly_flag:
        severity = "medium"
    elif raw_flag:
        severity = "watch"
    else:
        severity = "normal"

    reason = triggers[0]["code"] if triggers else "normal"

    return {
        "machine_id": machine_id,
        "anomaly_flag": anomaly_flag,
        "severity": severity,
        "reason": reason,
        "anomaly_score": round(max_score, 6),
        "raw_flag": raw_flag,
        "consecutive_count": consecutive_count,
        "persistence_windows": persistence_windows,
        "trigger_count": len(triggers),
        "triggers": triggers,
        "feature_event_ts": event_ts,
        "source_file": source_file,
        "baseline_run_id": artifact.get("run_id"),
    }


@app.on_event("startup")
def startup_event():
    try:
        ensure_baseline_loaded(force=True)
    except HTTPException:
        # service can run before baseline is trained
        pass


@app.get("/health")
def health():
    baseline_exists = ANOMALY_BASELINE_PATH.exists()
    return {
        "status": "ok",
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "baseline_path": str(ANOMALY_BASELINE_PATH),
        "baseline_exists": baseline_exists,
        "baseline_loaded": ANOMALY_BASELINE is not None,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/baseline/info")
def baseline_info():
    artifact = ensure_baseline_loaded()
    zscore_threshold, persistence_windows = get_runtime_config(artifact)
    return {
        "run_id": artifact.get("run_id"),
        "created_at": artifact.get("created_at"),
        "monitored_features": artifact.get("monitored_features", []),
        "machines_with_baseline": len(artifact.get("machine_baselines", {})),
        "zscore_threshold": zscore_threshold,
        "persistence_windows": persistence_windows,
        "baseline_path": str(ANOMALY_BASELINE_PATH),
    }


@app.post("/reload", response_model=ReloadResponse)
def reload_baseline():
    ensure_baseline_loaded(force=True)
    BASELINE_RELOADS.inc()
    return ReloadResponse(
        loaded=True,
        baseline_path=str(ANOMALY_BASELINE_PATH),
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/anomaly")
def detect_anomaly(request: AnomalyRequest):
    ANOMALY_REQUESTS.inc()
    machine_id = request.machine_id or request.features.get("machine_id")
    if not machine_id:
        raise HTTPException(status_code=400, detail="machine_id is required")

    return evaluate_row(machine_id=machine_id, features=request.features)


@app.get("/anomaly/latest/{machine_id}")
def detect_anomaly_latest(machine_id: str):
    ANOMALY_LATEST_REQUESTS.inc()
    row, source_file = find_latest_machine_row(machine_id)
    return evaluate_row(
        machine_id=machine_id,
        features=row,
        event_ts=row.get("event_ts") or row.get("ts"),
        source_file=source_file,
    )
