import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

DATASET_DIR = Path(os.getenv("CMAPSS_DATASET_DIR", "/data/datasets/CMAPSSData"))
CMAPSS_SUBSET = os.getenv("CMAPSS_SUBSET", "FD001").strip().upper()

MODEL_REGISTRY_DIR = Path(os.getenv("CMAPSS_MODEL_REGISTRY_DIR", "/data/models/cmapss/registry"))
MODEL_CURRENT_DIR = Path(os.getenv("CMAPSS_MODEL_CURRENT_DIR", "/data/models/cmapss/current"))

RUL_CLIP_MAX = int(os.getenv("RUL_CLIP_MAX", "130"))
FAILURE_HORIZON_CYCLES = int(os.getenv("FAILURE_HORIZON_CYCLES", "30"))
ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", "5"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "300"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
N_JOBS = int(os.getenv("N_JOBS", "-1"))
MAX_DEPTH_ENV = os.getenv("MAX_DEPTH", "")
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "1000"))

BASE_COLUMN_NAMES = (
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
) + tuple(f"sensor_{idx}" for idx in range(1, 22))


def optional_int(value: str):
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


MAX_DEPTH = optional_int(MAX_DEPTH_ENV)


def load_txt_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_split(dataset_dir: Path, subset: str) -> Dict[str, np.ndarray]:
    train_path = dataset_dir / f"train_{subset}.txt"
    test_path = dataset_dir / f"test_{subset}.txt"
    rul_path = dataset_dir / f"RUL_{subset}.txt"

    missing = [str(path) for path in (train_path, test_path, rul_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing C-MAPSS files: {missing}")

    train_raw = load_txt_matrix(train_path)
    test_raw = load_txt_matrix(test_path)
    rul_values = np.loadtxt(rul_path, dtype=np.float64).reshape(-1)

    if train_raw.shape[1] < 26 or test_raw.shape[1] < 26:
        raise ValueError("unexpected C-MAPSS column count; expected at least 26")

    return {
        "train_units": train_raw[:, 0].astype(np.int32),
        "train_cycles": train_raw[:, 1].astype(np.int32),
        "train_signals": train_raw[:, 2:26].astype(np.float64),
        "test_units": test_raw[:, 0].astype(np.int32),
        "test_cycles": test_raw[:, 1].astype(np.int32),
        "test_signals": test_raw[:, 2:26].astype(np.float64),
        "test_rul": rul_values,
    }


def build_feature_names(rolling_window: int) -> List[str]:
    names: List[str] = [
        "cycle",
        "cycle_ratio",
        "remaining_ratio_to_unit_end",
    ]

    for base in BASE_COLUMN_NAMES:
        names.extend(
            [
                base,
                f"{base}_delta1",
                f"{base}_mean_{rolling_window}",
                f"{base}_std_{rolling_window}",
                f"{base}_slope_{rolling_window}",
            ]
        )

    return names


def engineer_unit_features(cycles: np.ndarray, signals: np.ndarray, rolling_window: int) -> np.ndarray:
    row_count = signals.shape[0]
    signal_count = signals.shape[1]

    features = np.zeros((row_count, 3 + (signal_count * 5)), dtype=np.float64)

    max_cycle = float(cycles[-1]) if row_count > 0 else 1.0
    if max_cycle <= 0.0:
        max_cycle = 1.0

    for row_idx in range(row_count):
        cycle = float(cycles[row_idx])
        features[row_idx, 0] = cycle
        features[row_idx, 1] = cycle / max_cycle
        features[row_idx, 2] = (max_cycle - cycle) / max_cycle

        window_start = max(0, row_idx - rolling_window + 1)
        window_cycles = cycles[window_start : row_idx + 1].astype(np.float64)

        offset = 3
        for signal_idx in range(signal_count):
            current_value = float(signals[row_idx, signal_idx])
            prev_value = float(signals[row_idx - 1, signal_idx]) if row_idx > 0 else current_value

            window_values = signals[window_start : row_idx + 1, signal_idx].astype(np.float64)
            window_mean = float(window_values.mean())
            window_std = float(window_values.std())

            if window_values.size > 1:
                cycle_delta = float(window_cycles[-1] - window_cycles[0])
                if cycle_delta > 0.0:
                    window_slope = float((window_values[-1] - window_values[0]) / cycle_delta)
                else:
                    window_slope = 0.0
            else:
                window_slope = 0.0

            features[row_idx, offset] = current_value
            features[row_idx, offset + 1] = current_value - prev_value
            features[row_idx, offset + 2] = window_mean
            features[row_idx, offset + 3] = window_std
            features[row_idx, offset + 4] = window_slope
            offset += 5

    return features


def build_train_dataset(
    units: np.ndarray,
    cycles: np.ndarray,
    signals: np.ndarray,
    rul_clip_max: int,
    rolling_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []

    for unit_id in np.unique(units):
        idx = np.where(units == unit_id)[0]
        sorted_idx = idx[np.argsort(cycles[idx])]

        unit_cycles = cycles[sorted_idx]
        unit_signals = signals[sorted_idx]

        if unit_cycles.size == 0:
            continue

        unit_x = engineer_unit_features(unit_cycles, unit_signals, rolling_window)

        max_cycle = float(unit_cycles[-1])
        unit_rul = max_cycle - unit_cycles.astype(np.float64)
        if rul_clip_max > 0:
            unit_rul = np.minimum(unit_rul, float(rul_clip_max))

        x_blocks.append(unit_x)
        y_blocks.append(unit_rul)

    if not x_blocks:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)

    return np.vstack(x_blocks), np.concatenate(y_blocks)


def build_test_last_cycle_dataset(
    units: np.ndarray,
    cycles: np.ndarray,
    signals: np.ndarray,
    final_rul: Sequence[float],
    rul_clip_max: int,
    rolling_window: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    test_units = sorted(int(unit) for unit in np.unique(units))
    final_rul = np.asarray(final_rul, dtype=np.float64)

    if len(test_units) != final_rul.size:
        raise ValueError(
            "test RUL length mismatch: "
            f"units={len(test_units)} rul_values={final_rul.size}"
        )

    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    unit_ids: List[int] = []
    last_cycles: List[int] = []

    for position, unit_id in enumerate(test_units):
        idx = np.where(units == unit_id)[0]
        sorted_idx = idx[np.argsort(cycles[idx])]

        unit_cycles = cycles[sorted_idx]
        unit_signals = signals[sorted_idx]
        if unit_cycles.size == 0:
            continue

        unit_features = engineer_unit_features(unit_cycles, unit_signals, rolling_window)
        last_row = unit_features[-1]

        target_rul = float(final_rul[position])
        if rul_clip_max > 0:
            target_rul = min(target_rul, float(rul_clip_max))

        x_rows.append(last_row)
        y_rows.append(target_rul)
        unit_ids.append(unit_id)
        last_cycles.append(int(unit_cycles[-1]))

    if not x_rows:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            [],
            [],
        )

    return np.vstack(x_rows), np.asarray(y_rows, dtype=np.float64), unit_ids, last_cycles


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    penalties = np.where(
        diff >= 0,
        np.exp(diff / 10.0) - 1.0,
        np.exp((-diff) / 13.0) - 1.0,
    )
    return float(np.sum(penalties))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if y_true.size > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = 0.0

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": r2,
        "nasa_score": nasa_score(y_true, y_pred),
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }

    if np.unique(y_true).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def ensure_dirs() -> None:
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CURRENT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ensure_dirs()

    print(f"loading C-MAPSS subset={CMAPSS_SUBSET} from {DATASET_DIR}")
    try:
        dataset = load_split(DATASET_DIR, CMAPSS_SUBSET)
    except Exception as exc:
        print(f"failed to load C-MAPSS data: {exc}")
        return 2

    feature_names = build_feature_names(ROLLING_WINDOW)

    x_train, y_train_rul = build_train_dataset(
        units=dataset["train_units"],
        cycles=dataset["train_cycles"],
        signals=dataset["train_signals"],
        rul_clip_max=RUL_CLIP_MAX,
        rolling_window=ROLLING_WINDOW,
    )

    if x_train.shape[0] < MIN_TRAIN_ROWS:
        print(
            f"not enough training rows: found={x_train.shape[0]} min_required={MIN_TRAIN_ROWS}"
        )
        return 2

    x_test, y_test_rul, test_unit_ids, last_cycles = build_test_last_cycle_dataset(
        units=dataset["test_units"],
        cycles=dataset["test_cycles"],
        signals=dataset["test_signals"],
        final_rul=dataset["test_rul"],
        rul_clip_max=RUL_CLIP_MAX,
        rolling_window=ROLLING_WINDOW,
    )

    if x_test.shape[0] == 0:
        print("no test rows were generated")
        return 2

    y_train_failure = (y_train_rul <= float(FAILURE_HORIZON_CYCLES)).astype(np.int64)
    y_test_failure = (y_test_rul <= float(FAILURE_HORIZON_CYCLES)).astype(np.int64)

    if np.unique(y_train_failure).size < 2:
        print("failure-risk labels are single-class in training; change horizon or subset")
        return 2

    regressor = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=N_JOBS,
    )

    regressor.fit(x_train, y_train_rul)
    classifier.fit(x_train, y_train_failure)

    y_pred_rul = np.maximum(regressor.predict(x_test), 0.0)
    y_pred_failure = classifier.predict(x_test)

    if hasattr(classifier, "predict_proba"):
        y_prob_failure = classifier.predict_proba(x_test)[:, 1]
    else:
        y_prob_failure = y_pred_failure.astype(np.float64)

    regression_metrics = evaluate_regression(y_test_rul, y_pred_rul)
    classification_metrics = evaluate_classification(y_test_failure, y_pred_failure, y_prob_failure)

    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")

    run_dir = MODEL_REGISTRY_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rul_artifact = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "dataset": "C-MAPSS",
        "subset": CMAPSS_SUBSET,
        "target": "remaining_useful_life",
        "feature_columns": feature_names,
        "model": regressor,
        "config": {
            "rul_clip_max": RUL_CLIP_MAX,
            "rolling_window": ROLLING_WINDOW,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "random_state": RANDOM_STATE,
        },
        "metrics": regression_metrics,
    }

    failure_artifact = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "dataset": "C-MAPSS",
        "subset": CMAPSS_SUBSET,
        "target": "failure_within_horizon",
        "feature_columns": feature_names,
        "failure_horizon_cycles": FAILURE_HORIZON_CYCLES,
        "model": classifier,
        "config": {
            "rul_clip_max": RUL_CLIP_MAX,
            "rolling_window": ROLLING_WINDOW,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "random_state": RANDOM_STATE,
        },
        "metrics": classification_metrics,
    }

    rul_model_path = run_dir / "rul_model.joblib"
    failure_model_path = run_dir / "failure_risk_model.joblib"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"

    joblib.dump(rul_artifact, rul_model_path)
    joblib.dump(failure_artifact, failure_model_path)

    metrics_payload = {
        "regression": regression_metrics,
        "failure_risk": classification_metrics,
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    metadata = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "dataset": "C-MAPSS",
        "subset": CMAPSS_SUBSET,
        "train_rows": int(x_train.shape[0]),
        "train_units": int(np.unique(dataset["train_units"]).size),
        "test_rows": int(x_test.shape[0]),
        "test_units": int(len(test_unit_ids)),
        "feature_count": int(len(feature_names)),
        "rul_clip_max": RUL_CLIP_MAX,
        "failure_horizon_cycles": FAILURE_HORIZON_CYCLES,
        "rolling_window": ROLLING_WINDOW,
        "model_paths": {
            "rul_model": str(rul_model_path),
            "failure_risk_model": str(failure_model_path),
        },
        "sample_test_units": test_unit_ids[:5],
        "sample_test_last_cycles": last_cycles[:5],
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    shutil.copy2(rul_model_path, MODEL_CURRENT_DIR / "rul_model.joblib")
    shutil.copy2(failure_model_path, MODEL_CURRENT_DIR / "failure_risk_model.joblib")
    shutil.copy2(metadata_path, MODEL_CURRENT_DIR / "metadata.json")
    shutil.copy2(metrics_path, MODEL_CURRENT_DIR / "metrics.json")

    print(f"trained C-MAPSS models saved to: {run_dir}")
    print(f"current C-MAPSS models updated under: {MODEL_CURRENT_DIR}")
    print(f"regression_metrics={json.dumps(regression_metrics, separators=(',', ':'))}")
    print(f"failure_risk_metrics={json.dumps(classification_metrics, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
