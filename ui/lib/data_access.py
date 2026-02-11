from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def list_feature_files(features_dir: Path) -> List[Path]:
    if not features_dir.exists():
        return []
    return sorted(features_dir.glob("**/*.jsonl"))


def count_feature_files(features_dir: Path) -> int:
    return len(list_feature_files(features_dir))


def latest_feature_file(features_dir: Path) -> Optional[Path]:
    files = list_feature_files(features_dir)
    return files[-1] if files else None


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def read_jsonl_tail(path: Path, n: int = 50) -> List[Dict]:
    if not path.exists() or n <= 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    rows: List[Dict] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError:
            continue
    return rows


def load_latest_rows_by_machine(features_dir: Path, max_files: int = 200) -> Dict[str, Dict]:
    files = list_feature_files(features_dir)
    if max_files > 0:
        files = files[-max_files:]

    latest: Dict[str, Dict] = {}
    for path in reversed(files):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            machine_id = row.get("machine_id")
            if not machine_id or machine_id in latest:
                continue
            row["_source_file"] = str(path)
            latest[machine_id] = row
    return latest


def load_machine_timeseries(features_dir: Path, machine_id: str, max_rows: int = 400) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in list_feature_files(features_dir):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("machine_id") != machine_id:
                continue
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    rows = rows[-max_rows:]
    frame = pd.DataFrame(rows)
    if "event_ts" in frame.columns:
        frame["event_ts"] = pd.to_datetime(frame["event_ts"], utc=True, errors="coerce")
        frame = frame.sort_values("event_ts")
    return frame


def feature_file_growth(features_dir: Path) -> pd.DataFrame:
    files = list_feature_files(features_dir)
    if not files:
        return pd.DataFrame()

    records = []
    for path in files:
        stat = path.stat()
        records.append(
            {
                "file": str(path),
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            }
        )

    frame = pd.DataFrame(records)
    frame = frame.sort_values("mtime")
    frame["count"] = range(1, len(frame) + 1)
    frame["minute"] = frame["mtime"].dt.floor("min")
    return frame


def load_cmapss_registry(registry_dir: Path) -> pd.DataFrame:
    if not registry_dir.exists():
        return pd.DataFrame()

    rows: List[Dict] = []
    for run_dir in sorted(registry_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metadata = read_json(run_dir / "metadata.json") or {}
        metrics = read_json(run_dir / "metrics.json") or {}
        regression = metrics.get("regression", {})
        failure = metrics.get("failure_risk", {})

        rows.append(
            {
                "run_id": metadata.get("run_id", run_dir.name),
                "subset": metadata.get("subset"),
                "created_at": metadata.get("created_at"),
                "train_rows": metadata.get("train_rows"),
                "test_units": metadata.get("test_units"),
                "mae": regression.get("mae"),
                "rmse": regression.get("rmse"),
                "r2": regression.get("r2"),
                "nasa_score": regression.get("nasa_score"),
                "accuracy": failure.get("accuracy"),
                "precision": failure.get("precision"),
                "recall": failure.get("recall"),
                "f1": failure.get("f1"),
                "roc_auc": failure.get("roc_auc"),
            }
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    if "created_at" in frame.columns:
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
        frame = frame.sort_values("created_at", ascending=False)
    return frame
