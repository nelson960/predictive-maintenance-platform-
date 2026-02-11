from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Orchestrator:
    def __init__(
        self,
        repo_root: Path,
        compose_file: Path,
        project_name: str,
        data_dir: Path,
        log_dir: Path,
    ) -> None:
        self.repo_root = repo_root
        self.compose_file = compose_file
        self.project_name = project_name
        self.data_dir = data_dir
        self.log_dir = log_dir

    def _compose_base(self, include_ml_profile: bool = False) -> List[str]:
        cmd = [
            "docker",
            "compose",
            "-p",
            self.project_name,
            "-f",
            str(self.compose_file),
        ]
        if include_ml_profile:
            cmd.extend(["--profile", "ml"])
        return cmd

    def _run(
        self,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        include_ml_profile: bool = False,
    ) -> CommandResult:
        cmd = self._compose_base(include_ml_profile=include_ml_profile) + args
        start = time.perf_counter()
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        proc = subprocess.run(
            cmd,
            cwd=self.repo_root,
            env=process_env,
            text=True,
            capture_output=True,
        )
        duration = time.perf_counter() - start
        return CommandResult(
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_seconds=duration,
        )

    def clear_feature_data(self) -> Tuple[bool, str]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for item in self.data_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)
        return True, f"cleared {self.data_dir}"

    def data_generation_start(
        self,
        duration_seconds: int = 0,
        build: bool = False,
        clear_data: bool = False,
    ) -> List[CommandResult]:
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be >= 0")

        if clear_data:
            self.clear_feature_data()

        results: List[CommandResult] = []
        results.append(self._run(["up", "-d", "redpanda", "init-topic"]))

        up_args = ["up", "-d"]
        if build:
            up_args.append("--build")
        up_args.extend(["producer", "aggregator"])

        results.append(
            self._run(
                up_args,
                env={"MAX_RUNTIME_SECONDS": str(duration_seconds)},
            )
        )
        return results

    def data_generation_stop(self, stop_all: bool = False) -> List[CommandResult]:
        results: List[CommandResult] = []
        if stop_all:
            results.append(self._run(["stop"], include_ml_profile=True))
            results.append(self._run(["stop"]))
            return results

        results.append(self._run(["stop", "producer", "aggregator"]))
        return results

    def data_generation_status(self) -> CommandResult:
        return self._run(["ps"], include_ml_profile=True)

    def failure_risk_train(self, build: bool = False) -> List[CommandResult]:
        results: List[CommandResult] = []
        if build:
            results.append(self._run(["build", "failure-risk-trainer"], include_ml_profile=True))
        results.append(self._run(["run", "--rm", "failure-risk-trainer"], include_ml_profile=True))
        return results

    def failure_risk_start(self, build: bool = False) -> CommandResult:
        args = ["up", "-d"]
        if build:
            args.append("--build")
        args.append("failure-risk-service")
        return self._run(args, include_ml_profile=True)

    def failure_risk_stop(self) -> CommandResult:
        return self._run(["stop", "failure-risk-service"], include_ml_profile=True)

    def failure_risk_status(self) -> CommandResult:
        return self._run(["ps", "failure-risk-service"], include_ml_profile=True)

    def anomaly_train(self, build: bool = False) -> List[CommandResult]:
        results: List[CommandResult] = []
        if build:
            results.append(self._run(["build", "anomaly-trainer"], include_ml_profile=True))
        results.append(self._run(["run", "--rm", "anomaly-trainer"], include_ml_profile=True))
        return results

    def anomaly_start(self, build: bool = False) -> CommandResult:
        args = ["up", "-d"]
        if build:
            args.append("--build")
        args.append("anomaly-service")
        return self._run(args, include_ml_profile=True)

    def anomaly_stop(self) -> CommandResult:
        return self._run(["stop", "anomaly-service"], include_ml_profile=True)

    def anomaly_status(self) -> CommandResult:
        return self._run(["ps", "anomaly-service"], include_ml_profile=True)

    def cmapss_train(self, subset: str = "FD001", build: bool = False) -> List[CommandResult]:
        subset = subset.upper()
        if subset not in {"FD001", "FD002", "FD003", "FD004"}:
            raise ValueError("subset must be one of FD001, FD002, FD003, FD004")

        results: List[CommandResult] = []
        if build:
            results.append(self._run(["build", "cmapss-trainer"], include_ml_profile=True))
        results.append(
            self._run(
                ["run", "--rm", "-e", f"CMAPSS_SUBSET={subset}", "cmapss-trainer"],
                include_ml_profile=True,
            )
        )
        return results

    def prometheus_start(self) -> CommandResult:
        return self._run(["up", "-d", "prometheus"], include_ml_profile=True)

    def prometheus_stop(self) -> CommandResult:
        return self._run(["stop", "prometheus"], include_ml_profile=True)

    def prometheus_status(self) -> CommandResult:
        return self._run(["ps", "prometheus"], include_ml_profile=True)

    def compose_ps_json(self) -> List[Dict[str, str]]:
        result = self._run(["ps", "--format", "json"], include_ml_profile=True)
        if not result.ok:
            return []

        text = result.stdout.strip()
        if not text:
            return []

        # docker compose may return json array or json lines depending on version
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [row for row in parsed if isinstance(row, dict)]
        except json.JSONDecodeError:
            pass

        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
            except json.JSONDecodeError:
                continue
        return rows
