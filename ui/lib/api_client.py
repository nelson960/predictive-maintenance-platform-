from __future__ import annotations

from typing import Dict, Optional, Tuple

import requests


class ApiClient:
    def __init__(self, failure_risk_base: str, anomaly_base: str, timeout_seconds: float = 5.0) -> None:
        self.failure_risk_base = failure_risk_base.rstrip("/")
        self.anomaly_base = anomaly_base.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post(self, url: str, payload: Dict) -> Tuple[bool, str, Optional[Dict]]:
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            return False, str(exc), None

        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}: {response.text}", None

        try:
            return True, "ok", response.json()
        except ValueError:
            return False, "invalid JSON response", None

    def _get(self, url: str) -> Tuple[bool, str, Optional[Dict]]:
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            return False, str(exc), None

        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}: {response.text}", None

        try:
            return True, "ok", response.json()
        except ValueError:
            return False, "invalid JSON response", None

    def failure_risk_health(self) -> Tuple[bool, str, Optional[Dict]]:
        return self._get(f"{self.failure_risk_base}/health")

    def failure_risk_model_info(self) -> Tuple[bool, str, Optional[Dict]]:
        return self._get(f"{self.failure_risk_base}/model/info")

    def failure_risk_predict_latest(self, machine_id: str) -> Tuple[bool, str, Optional[Dict]]:
        return self._get(f"{self.failure_risk_base}/predict/latest/{machine_id}")

    def failure_risk_predict(self, features: Dict) -> Tuple[bool, str, Optional[Dict]]:
        return self._post(f"{self.failure_risk_base}/predict", {"features": features})

    def anomaly_health(self) -> Tuple[bool, str, Optional[Dict]]:
        return self._get(f"{self.anomaly_base}/health")

    def anomaly_baseline_info(self) -> Tuple[bool, str, Optional[Dict]]:
        return self._get(f"{self.anomaly_base}/baseline/info")

    def anomaly_latest(self, machine_id: str) -> Tuple[bool, str, Optional[Dict]]:
        return self._get(f"{self.anomaly_base}/anomaly/latest/{machine_id}")

    def anomaly_detect(self, machine_id: str, features: Dict) -> Tuple[bool, str, Optional[Dict]]:
        payload = {"machine_id": machine_id, "features": features}
        return self._post(f"{self.anomaly_base}/anomaly", payload)

    def prometheus_targets(self, base_url: str = "http://localhost:9090") -> Tuple[bool, str, Optional[Dict]]:
        return self._get(f"{base_url.rstrip('/')}/api/v1/targets")

    def prometheus_query(self, query: str, base_url: str = "http://localhost:9090") -> Tuple[bool, str, Optional[Dict]]:
        try:
            response = requests.get(
                f"{base_url.rstrip('/')}/api/v1/query",
                params={"query": query},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            return False, str(exc), None

        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}: {response.text}", None

        try:
            payload = response.json()
        except ValueError:
            return False, "invalid JSON response", None

        if payload.get("status") != "success":
            return False, f"prometheus query failed: {payload}", payload
        return True, "ok", payload
