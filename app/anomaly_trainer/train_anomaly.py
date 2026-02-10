import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "/data/features_offline"))
ANOMALY_REGISTRY_DIR = Path(os.getenv("ANOMALY_REGISTRY_DIR", "/data/models/anomaly/registry"))
ANOMALY_CURRENT_DIR = Path(os.getenv("ANOMALY_CURRENT_DIR", "/data/models/anomaly/current"))

MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "200"))
MIN_MACHINE_SAMPLES = int(os.getenv("MIN_MACHINE_SAMPLES", "30"))
MAX_FILES = int(os.getenv("MAX_FILES", "0"))
STD_EPSILON = float(os.getenv("STD_EPSILON", "1e-6"))

DEFAULT_MONITORED_FEATURES = (
    "temperature,vibration,pressure,rpm,load,"
    "temp_mean_5s,vib_mean_5s,pressure_mean_5s,rpm_mean_5s,load_mean_5s"
)
MONITORED_FEATURES = [
    column.strip()
    for column in os.getenv("MONITORED_FEATURES", DEFAULT_MONITORED_FEATURES).split(",")
    if column.strip()
]

DEFAULT_HARD_LIMITS = {
    "temperature": {"low": 55.0, "high": 90.0},
    "vibration": {"low": 0.05, "high": 0.70},
    "pressure": {"low": 20.0, "high": 50.0},
    "rpm": {"low": 1200.0, "high": 2300.0},
    "load": {"low": 0.0, "high": 1.0},
    "temp_mean_5s": {"low": 55.0, "high": 90.0},
    "vib_mean_5s": {"low": 0.05, "high": 0.70},
    "pressure_mean_5s": {"low": 20.0, "high": 50.0},
    "rpm_mean_5s": {"low": 1200.0, "high": 2300.0},
    "load_mean_5s": {"low": 0.0, "high": 1.0},
}

HARD_LIMITS_JSON = os.getenv("HARD_LIMITS_JSON", "")
ZSCORE_THRESHOLD = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
PERSISTENCE_WINDOWS = int(os.getenv("PERSISTENCE_WINDOWS", "3"))


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(max(arr.std(), STD_EPSILON)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def ensure_dirs() -> None:
    ANOMALY_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    ANOMALY_CURRENT_DIR.mkdir(parents=True, exist_ok=True)


def parse_hard_limits() -> Dict[str, Dict[str, float]]:
    if not HARD_LIMITS_JSON:
        return DEFAULT_HARD_LIMITS
    try:
        loaded = json.loads(HARD_LIMITS_JSON)
    except json.JSONDecodeError:
        return DEFAULT_HARD_LIMITS

    if not isinstance(loaded, dict):
        return DEFAULT_HARD_LIMITS

    merged = dict(DEFAULT_HARD_LIMITS)
    for feature, limits in loaded.items():
        if not isinstance(limits, dict):
            continue
        low = safe_float(limits.get("low"))
        high = safe_float(limits.get("high"))
        if low is None or high is None:
            continue
        merged[feature] = {"low": float(low), "high": float(high)}
    return merged


def main() -> int:
    ensure_dirs()

    feature_files = list_feature_files(FEATURES_DIR, MAX_FILES)
    if not feature_files:
        print(f"no feature files found under {FEATURES_DIR}")
        return 2

    rows = load_rows(feature_files)
    if len(rows) < MIN_SAMPLES:
        print(f"not enough rows for anomaly baseline: found={len(rows)} min_required={MIN_SAMPLES}")
        return 2

    machine_feature_values: Dict[str, Dict[str, List[float]]] = {}
    global_feature_values: Dict[str, List[float]] = {feature: [] for feature in MONITORED_FEATURES}

    for row in rows:
        machine_id = row.get("machine_id")
        if not machine_id:
            continue

        machine_store = machine_feature_values.setdefault(
            machine_id,
            {feature: [] for feature in MONITORED_FEATURES},
        )

        for feature in MONITORED_FEATURES:
            value = safe_float(row.get(feature))
            if value is None:
                continue
            machine_store[feature].append(value)
            global_feature_values[feature].append(value)

    machine_baselines = {}
    for machine_id, feature_map in machine_feature_values.items():
        baseline = {}
        for feature, values in feature_map.items():
            if len(values) < MIN_MACHINE_SAMPLES:
                continue
            baseline[feature] = compute_stats(values)
        if baseline:
            machine_baselines[machine_id] = baseline

    global_baseline = {}
    for feature, values in global_feature_values.items():
        if len(values) < MIN_MACHINE_SAMPLES:
            continue
        global_baseline[feature] = compute_stats(values)

    if not global_baseline:
        print("failed to build global baseline; insufficient valid feature values")
        return 2

    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")

    run_dir = ANOMALY_REGISTRY_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    hard_limits = parse_hard_limits()

    baseline_artifact = {
        "created_at": created_at.isoformat(),
        "run_id": run_id,
        "monitored_features": MONITORED_FEATURES,
        "machine_baselines": machine_baselines,
        "global_baseline": global_baseline,
        "hard_limits": hard_limits,
        "config": {
            "zscore_threshold": ZSCORE_THRESHOLD,
            "persistence_windows": PERSISTENCE_WINDOWS,
            "min_samples": MIN_SAMPLES,
            "min_machine_samples": MIN_MACHINE_SAMPLES,
            "max_files": MAX_FILES,
        },
    }

    baseline_path = run_dir / "baseline.json"
    metadata_path = run_dir / "metadata.json"

    with baseline_path.open("w", encoding="utf-8") as handle:
        json.dump(baseline_artifact, handle, indent=2)

    metadata = {
        "created_at": created_at.isoformat(),
        "run_id": run_id,
        "rows_used": len(rows),
        "feature_files_used": len(feature_files),
        "machines_with_baseline": len(machine_baselines),
        "monitored_features": MONITORED_FEATURES,
        "baseline_path": str(baseline_path),
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    current_baseline_path = ANOMALY_CURRENT_DIR / "baseline.json"
    current_metadata_path = ANOMALY_CURRENT_DIR / "metadata.json"

    shutil.copy2(baseline_path, current_baseline_path)
    shutil.copy2(metadata_path, current_metadata_path)

    print(f"anomaly baseline saved: {baseline_path}")
    print(f"machines_with_baseline: {len(machine_baselines)}")
    print(f"current baseline updated: {current_baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
