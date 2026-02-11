from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.lib.api_client import ApiClient
from ui.lib.data_access import (
    count_feature_files,
    feature_file_growth,
    latest_feature_file,
    load_latest_rows_by_machine,
    load_machine_timeseries,
    read_jsonl_tail,
)
from ui.lib.orchestrator import CommandResult, Orchestrator


def render_command_result(result: CommandResult) -> None:
    status = "success" if result.ok else "error"
    if status == "success":
        st.success(
            f"`{result.command[-1]}` finished in {result.duration_seconds:.2f}s "
            f"(exit={result.returncode})"
        )
    else:
        st.error(
            f"`{result.command[-1]}` failed in {result.duration_seconds:.2f}s "
            f"(exit={result.returncode})"
        )

    with st.expander("Command details", expanded=False):
        st.code(" ".join(result.command), language="bash")
        if result.stdout.strip():
            st.code(result.stdout, language="text")
        if result.stderr.strip():
            st.code(result.stderr, language="text")


def render_command_results(results: Sequence[CommandResult]) -> None:
    for result in results:
        render_command_result(result)


def metrics_row(label: str, value: str) -> None:
    left, right = st.columns([1, 3])
    with left:
        st.caption(label)
    with right:
        st.write(value)


def machine_id_number(machine_id: str) -> int:
    match = re.search(r"(\d+)$", str(machine_id))
    if not match:
        return 10**9
    return int(match.group(1))


def sort_machine_ids(machine_ids: Sequence[str]) -> List[str]:
    return sorted(machine_ids, key=lambda machine_id: (machine_id_number(machine_id), str(machine_id)))


def sort_frame_by_machine_id(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "machine_id" not in frame.columns:
        return frame
    ordered = frame.copy()
    ordered["_machine_order"] = ordered["machine_id"].astype(str).map(machine_id_number)
    ordered = ordered.sort_values(["_machine_order", "machine_id"]).drop(columns=["_machine_order"])
    return ordered.reset_index(drop=True)


def to_float_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


EXCLUDED_FEATURE_KEYS = {
    "machine_id",
    "event_ts",
    "ingest_ts",
    "source_file",
    "_source_file",
}


def numeric_feature_payload(row: Dict, allowed_features: Sequence[str] | None = None) -> Dict[str, float]:
    allowed = set(allowed_features) if allowed_features else None
    payload: Dict[str, float] = {}
    for key, value in row.items():
        if key in EXCLUDED_FEATURE_KEYS or str(key).startswith("_"):
            continue
        if allowed is not None and key not in allowed:
            continue
        number = to_float_or_none(value)
        if number is None:
            continue
        payload[str(key)] = number
    return payload


SCENARIO_LIBRARY: Dict[str, Dict[str, str]] = {
    "normal_drift": {
        "label": "Normal Drift",
        "description": "Small natural variation with no strong fault signature.",
        "expected": "Failure risk should remain low and anomaly should mostly stay normal.",
    },
    "bearing_wear": {
        "label": "Bearing Wear",
        "description": "Vibration rises steadily with heat and RPM instability.",
        "expected": "Anomaly should trigger first from vibration, then failure risk should rise.",
    },
    "overheating": {
        "label": "Overheating",
        "description": "Temperature climbs quickly and stresses neighboring signals.",
        "expected": "Hard-limit and z-score temperature triggers should appear early.",
    },
    "pressure_leak": {
        "label": "Pressure Leak",
        "description": "Pressure drops while system compensates with load/rpm changes.",
        "expected": "Pressure-based anomaly triggers should lead the incident timeline.",
    },
    "overload": {
        "label": "Overload",
        "description": "Load ramps to aggressive levels and introduces broad stress.",
        "expected": "Risk climbs and anomaly severity increases across multiple features.",
    },
}

SCENARIO_ORDER = list(SCENARIO_LIBRARY.keys())
SEVERITY_ORDER = ["normal", "watch", "medium", "high", "critical"]


def bump_feature(features: Dict[str, float], key: str, delta: float) -> None:
    current = to_float_or_none(features.get(key))
    if current is None:
        return
    features[key] = current + delta


def clip_feature_ranges(features: Dict[str, float]) -> Dict[str, float]:
    clipped = dict(features)
    if "load" in clipped:
        clipped["load"] = max(0.0, min(1.5, clipped["load"]))
    if "load_mean_5s" in clipped:
        clipped["load_mean_5s"] = max(0.0, min(1.5, clipped["load_mean_5s"]))
    if "rpm" in clipped:
        clipped["rpm"] = max(500.0, clipped["rpm"])
    if "rpm_mean_5s" in clipped:
        clipped["rpm_mean_5s"] = max(500.0, clipped["rpm_mean_5s"])
    return clipped


def apply_scenario_mutation(
    base_features: Dict[str, float],
    scenario_key: str,
    step: int,
    total_steps: int,
    intensity: float,
) -> Dict[str, float]:
    progress = step / max(total_steps - 1, 1)
    oscillation = math.sin((step + 1) * 0.8)
    features = dict(base_features)

    if scenario_key == "normal_drift":
        bump_feature(features, "temperature", 0.6 * oscillation * intensity)
        bump_feature(features, "vibration", 0.01 * oscillation * intensity)
        bump_feature(features, "pressure", 0.4 * oscillation * intensity)
        bump_feature(features, "rpm", 15.0 * oscillation * intensity)
        bump_feature(features, "load", 0.015 * oscillation * intensity)
        bump_feature(features, "temp_mean_5s", 0.35 * oscillation * intensity)
        bump_feature(features, "vib_mean_5s", 0.008 * oscillation * intensity)

    elif scenario_key == "bearing_wear":
        bump_feature(features, "vibration", (0.03 + 0.28 * progress) * intensity)
        bump_feature(features, "vib_mean_5s", (0.025 + 0.24 * progress) * intensity)
        bump_feature(features, "temperature", (1.0 + 7.0 * progress) * intensity)
        bump_feature(features, "temp_mean_5s", (0.8 + 5.0 * progress) * intensity)
        bump_feature(features, "rpm", (-25.0 - 200.0 * progress) * intensity)
        bump_feature(features, "rpm_mean_5s", (-20.0 - 150.0 * progress) * intensity)
        bump_feature(features, "load", (0.01 + 0.09 * progress) * intensity)
        bump_feature(features, "load_mean_5s", (0.008 + 0.07 * progress) * intensity)

    elif scenario_key == "overheating":
        bump_feature(features, "temperature", (2.0 + 18.0 * progress) * intensity)
        bump_feature(features, "temp_mean_5s", (1.5 + 16.0 * progress) * intensity)
        bump_feature(features, "vibration", (0.01 + 0.09 * progress) * intensity)
        bump_feature(features, "vib_mean_5s", (0.008 + 0.07 * progress) * intensity)
        bump_feature(features, "pressure", (0.4 + 4.0 * progress) * intensity)
        bump_feature(features, "pressure_mean_5s", (0.3 + 3.0 * progress) * intensity)
        bump_feature(features, "load", (0.01 + 0.08 * progress) * intensity)

    elif scenario_key == "pressure_leak":
        bump_feature(features, "pressure", (-1.0 - 13.0 * progress) * intensity)
        bump_feature(features, "pressure_mean_5s", (-0.8 - 11.0 * progress) * intensity)
        bump_feature(features, "rpm", (20.0 + 110.0 * progress) * intensity)
        bump_feature(features, "rpm_mean_5s", (16.0 + 90.0 * progress) * intensity)
        bump_feature(features, "load", (0.015 + 0.12 * progress) * intensity)
        bump_feature(features, "load_mean_5s", (0.01 + 0.10 * progress) * intensity)
        bump_feature(features, "temperature", (0.3 + 4.5 * progress) * intensity)

    elif scenario_key == "overload":
        bump_feature(features, "load", (0.03 + 0.42 * progress) * intensity)
        bump_feature(features, "load_mean_5s", (0.02 + 0.36 * progress) * intensity)
        bump_feature(features, "temperature", (1.0 + 11.0 * progress) * intensity)
        bump_feature(features, "temp_mean_5s", (0.8 + 8.5 * progress) * intensity)
        bump_feature(features, "vibration", (0.02 + 0.14 * progress) * intensity)
        bump_feature(features, "vib_mean_5s", (0.015 + 0.11 * progress) * intensity)
        bump_feature(features, "rpm", (-10.0 - 140.0 * progress) * intensity)
        bump_feature(features, "rpm_mean_5s", (-8.0 - 115.0 * progress) * intensity)

    return clip_feature_ranges(features)


def first_step_where(frame: pd.DataFrame, column: str, predicate) -> int | None:
    if frame.empty or column not in frame.columns:
        return None
    subset = frame[predicate(frame[column])]
    if subset.empty:
        return None
    return int(subset["step"].min())


def make_orchestrator_and_clients() -> tuple[Orchestrator, ApiClient, str]:
    st.sidebar.header("Runtime Config")
    compose_file = Path(
        st.sidebar.text_input(
            "Compose file",
            value=str(REPO_ROOT / "docker-compose.yml"),
            key="cfg_compose_file",
        )
    )
    project_name = st.sidebar.text_input(
        "Project name",
        value="predictive_maintaince_platform",
        key="cfg_project_name",
    ).strip()
    data_dir = Path(
        st.sidebar.text_input(
            "Feature data dir",
            value=str(REPO_ROOT / "data/features_offline"),
            key="cfg_data_dir",
        )
    )
    log_dir = Path(
        st.sidebar.text_input(
            "Log dir",
            value=str(REPO_ROOT / "data/run_logs"),
            key="cfg_log_dir",
        )
    )
    failure_risk_base = st.sidebar.text_input(
        "Failure-risk API",
        value="http://localhost:8000",
        key="cfg_failure_risk_api",
    ).strip()
    anomaly_base = st.sidebar.text_input(
        "Anomaly API",
        value="http://localhost:8001",
        key="cfg_anomaly_api",
    ).strip()
    prometheus_base = st.sidebar.text_input(
        "Prometheus",
        value="http://localhost:9090",
        key="cfg_prometheus_api",
    ).strip()

    orchestrator = Orchestrator(
        repo_root=REPO_ROOT,
        compose_file=compose_file,
        project_name=project_name or "predictive_maintaince_platform",
        data_dir=data_dir,
        log_dir=log_dir,
    )
    client = ApiClient(failure_risk_base=failure_risk_base, anomaly_base=anomaly_base)
    return orchestrator, client, prometheus_base


def render_overview(orchestrator: Orchestrator, data_dir: Path) -> None:
    st.subheader("System Status")
    rows = orchestrator.compose_ps_json()
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True)
    else:
        st.info("No compose containers found for the current project/profile.")

    file_count = count_feature_files(data_dir)
    latest_file = latest_feature_file(data_dir)
    stat_1, stat_2 = st.columns(2)
    with stat_1:
        st.metric("Feature files", file_count)
    with stat_2:
        st.metric("Latest feature file", latest_file.name if latest_file else "none")

    growth = feature_file_growth(data_dir)
    if not growth.empty:
        fig = px.line(
            growth,
            x="mtime",
            y="count",
            title="Offline Feature File Growth",
            labels={"mtime": "UTC time", "count": "file count"},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_data_generation(orchestrator: Orchestrator, data_dir: Path) -> None:
    st.subheader("Data Generation (Producer + Aggregator)")
    left, right = st.columns(2)
    with left:
        duration = st.number_input(
            "Max runtime seconds (0 = unlimited)",
            min_value=0,
            value=300,
            step=10,
            key="dg_duration",
        )
        build = st.checkbox("Build producer/aggregator images", value=False, key="dg_build")
        clear_data = st.checkbox("Clear feature data before start", value=False, key="dg_clear_before")

        if st.button("Start data generation", key="dg_start_btn", use_container_width=True):
            results = orchestrator.data_generation_start(
                duration_seconds=int(duration),
                build=build,
                clear_data=clear_data,
            )
            render_command_results(results)

        if st.button("Stop producer + aggregator", key="dg_stop_btn", use_container_width=True):
            render_command_results(orchestrator.data_generation_stop(stop_all=False))

        if st.button("Stop all services", key="dg_stop_all_btn", use_container_width=True):
            render_command_results(orchestrator.data_generation_stop(stop_all=True))

    with right:
        if st.button("Refresh compose status", key="dg_refresh_status", use_container_width=True):
            render_command_result(orchestrator.data_generation_status())
        if st.button("Clear offline feature data now", key="dg_clear_btn", use_container_width=True):
            ok, msg = orchestrator.clear_feature_data()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("---")
    feature_count = count_feature_files(data_dir)
    latest_file = latest_feature_file(data_dir)
    col_1, col_2 = st.columns(2)
    with col_1:
        st.metric("Feature files", feature_count)
    with col_2:
        st.metric("Latest file", latest_file.name if latest_file else "none")

    if latest_file is not None:
        tail_n = st.slider("Latest file preview rows", min_value=5, max_value=100, value=20, key="dg_tail_n")
        rows = read_jsonl_tail(latest_file, n=tail_n)
        if rows:
            frame = pd.DataFrame(rows)
            st.dataframe(frame, use_container_width=True)
        else:
            st.info("Latest file is empty or unreadable.")
    else:
        st.info("No feature files yet. Start data generation first.")

    latest_rows = load_latest_rows_by_machine(data_dir, max_files=300)
    if latest_rows:
        latest_frame = sort_frame_by_machine_id(pd.DataFrame(list(latest_rows.values())))
        st.markdown("#### Latest row by machine")
        st.dataframe(latest_frame, use_container_width=True)

        machine_options = sort_machine_ids(list(latest_rows.keys()))
        selected_machine = st.selectbox("Machine timeseries", options=machine_options, key="dg_machine_picker")
        ts = load_machine_timeseries(data_dir, selected_machine, max_rows=500)
        if not ts.empty and "event_ts" in ts.columns:
            candidate_metrics = ["temperature", "vibration", "pressure", "rpm", "load"]
            available_metrics = [col for col in candidate_metrics if col in ts.columns]
            metrics = st.multiselect(
                "Signals to plot",
                options=available_metrics,
                default=available_metrics[:2] if available_metrics else [],
                key="dg_plot_metrics",
            )
            if metrics:
                fig = px.line(
                    ts,
                    x="event_ts",
                    y=metrics,
                    title=f"Machine {selected_machine} Signal Trend",
                    labels={"event_ts": "UTC event time", "value": "signal"},
                )
                st.plotly_chart(fig, use_container_width=True)


def render_failure_risk(orchestrator: Orchestrator, client: ApiClient, data_dir: Path) -> None:
    st.subheader("Failure-Risk Pipeline")
    build = st.checkbox("Build failure-risk images", value=False, key="fr_build")
    control_1, control_2, control_3, control_4 = st.columns(4)
    with control_1:
        if st.button("Train", key="fr_train_btn", use_container_width=True):
            render_command_results(orchestrator.failure_risk_train(build=build))
    with control_2:
        if st.button("Service Start", key="fr_start_btn", use_container_width=True):
            render_command_result(orchestrator.failure_risk_start(build=build))
    with control_3:
        if st.button("Service Stop", key="fr_stop_btn", use_container_width=True):
            render_command_result(orchestrator.failure_risk_stop())
    with control_4:
        if st.button("Service Status", key="fr_status_btn", use_container_width=True):
            render_command_result(orchestrator.failure_risk_status())

    st.markdown("#### API Health & Model")
    ok_h, msg_h, health = client.failure_risk_health()
    if ok_h and health:
        st.success("Failure-risk service reachable")
        metrics_row("Model loaded", str(health.get("model_loaded")))
        metrics_row("Model path", str(health.get("model_path")))
        metrics_row("UTC time", str(health.get("utc_time")))
    else:
        st.error(f"Failure-risk service not reachable: {msg_h}")
        return

    ok_m, msg_m, info = client.failure_risk_model_info()
    model_feature_columns: List[str] = []
    if ok_m and info:
        model_feature_columns = list(info.get("feature_columns", []))
        st.json(info)
    else:
        st.warning(f"Model info unavailable: {msg_m}")

    latest_rows = load_latest_rows_by_machine(data_dir, max_files=300)
    machine_ids = sort_machine_ids(list(latest_rows.keys())) if latest_rows else []
    default_machine = machine_ids[0] if machine_ids else "M-001"
    machine = st.text_input("Machine ID for latest prediction", value=default_machine, key="fr_machine_id")

    if st.button("Predict latest machine", key="fr_predict_one_btn"):
        ok_p, msg_p, pred = client.failure_risk_predict_latest(machine)
        if ok_p and pred:
            st.success("Prediction complete")
            st.json(pred)
        else:
            st.error(f"Prediction failed: {msg_p}")

    if st.button("Fleet risk scan (latest by machine)", key="fr_predict_fleet_btn"):
        if not machine_ids:
            st.warning("No machines found in offline features.")
        else:
            rows = []
            for machine_id in machine_ids:
                ok_p, _, pred = client.failure_risk_predict_latest(machine_id)
                if ok_p and pred:
                    rows.append(pred)
            if rows:
                frame = sort_frame_by_machine_id(pd.DataFrame(rows))
                st.dataframe(frame, use_container_width=True)
                fig = px.bar(
                    frame,
                    x="machine_id",
                    y="failure_probability",
                    color="risk_level",
                    title="Failure-Risk Probability by Machine",
                    category_orders={"machine_id": frame["machine_id"].tolist()},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No fleet predictions returned from service.")

    if st.button("Explain risk drivers (latest row)", key="fr_explain_btn"):
        machine_row = latest_rows.get(machine)
        if machine_row is None:
            st.warning(f"No latest feature row found for machine {machine}.")
        else:
            base_features = numeric_feature_payload(
                machine_row,
                allowed_features=model_feature_columns if model_feature_columns else None,
            )
            if not base_features:
                st.error("No numeric features available for failure-risk prediction.")
                return

            ok_base, msg_base, base_pred = client.failure_risk_predict(base_features)
            if not ok_base or not base_pred:
                st.error(f"Could not compute base prediction: {msg_base}")
            else:
                base_probability = float(base_pred.get("failure_probability", 0.0))
                fleet_frame = pd.DataFrame(list(latest_rows.values()))
                fallback_features = [
                    "temperature",
                    "vibration",
                    "pressure",
                    "rpm",
                    "load",
                    "temp_mean_5s",
                    "vib_mean_5s",
                    "pressure_mean_5s",
                    "rpm_mean_5s",
                    "load_mean_5s",
                    "event_count_5s",
                    "event_count_30s",
                ]
                feature_candidates = model_feature_columns if model_feature_columns else fallback_features
                feature_candidates = [feature for feature in feature_candidates if feature in base_features]

                impact_rows = []
                for feature in feature_candidates:
                    if feature in fleet_frame.columns:
                        series = pd.to_numeric(fleet_frame[feature], errors="coerce").dropna()
                        reference_value = float(series.median()) if not series.empty else 0.0
                    else:
                        reference_value = 0.0

                    counterfactual = dict(base_features)
                    counterfactual[feature] = reference_value

                    ok_cf, _, cf_pred = client.failure_risk_predict(counterfactual)
                    if not ok_cf or not cf_pred:
                        continue

                    cf_probability = float(cf_pred.get("failure_probability", 0.0))
                    delta = base_probability - cf_probability
                    impact_rows.append(
                        {
                            "machine_id": machine,
                            "feature": feature,
                            "current_value": to_float_or_none(machine_row.get(feature)),
                            "reference_median": reference_value,
                            "counterfactual_probability": cf_probability,
                            "delta_risk": delta,
                            "direction": "increases risk" if delta > 0 else "reduces risk",
                        }
                    )

                if not impact_rows:
                    st.warning("No feature impact values computed.")
                else:
                    impact = pd.DataFrame(impact_rows)
                    impact["abs_delta"] = impact["delta_risk"].abs()
                    impact = impact.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"])

                    st.info(
                        f"Base failure probability for {machine}: "
                        f"{base_probability:.4f}. Positive delta means current value is pushing risk up."
                    )
                    st.dataframe(impact, use_container_width=True)

                    chart_frame = impact.sort_values("delta_risk", ascending=False)
                    fig = px.bar(
                        chart_frame,
                        x="feature",
                        y="delta_risk",
                        color="direction",
                        title=f"Local Risk Drivers for {machine}",
                    )
                    st.plotly_chart(fig, use_container_width=True)


def render_anomaly(orchestrator: Orchestrator, client: ApiClient, data_dir: Path) -> None:
    st.subheader("Anomaly Pipeline")
    build = st.checkbox("Build anomaly images", value=False, key="an_build")
    control_1, control_2, control_3, control_4 = st.columns(4)
    with control_1:
        if st.button("Train Baseline", key="an_train_btn", use_container_width=True):
            render_command_results(orchestrator.anomaly_train(build=build))
    with control_2:
        if st.button("Service Start", key="an_start_btn", use_container_width=True):
            render_command_result(orchestrator.anomaly_start(build=build))
    with control_3:
        if st.button("Service Stop", key="an_stop_btn", use_container_width=True):
            render_command_result(orchestrator.anomaly_stop())
    with control_4:
        if st.button("Service Status", key="an_status_btn", use_container_width=True):
            render_command_result(orchestrator.anomaly_status())

    st.markdown("#### API Health & Baseline")
    ok_h, msg_h, health = client.anomaly_health()
    if ok_h and health:
        st.success("Anomaly service reachable")
        metrics_row("Baseline loaded", str(health.get("baseline_loaded")))
        metrics_row("Baseline path", str(health.get("baseline_path")))
        metrics_row("UTC time", str(health.get("utc_time")))
    else:
        st.error(f"Anomaly service not reachable: {msg_h}")
        return

    ok_b, msg_b, baseline = client.anomaly_baseline_info()
    if ok_b and baseline:
        st.json(baseline)
    else:
        st.warning(f"Baseline info unavailable: {msg_b}")

    latest_rows = load_latest_rows_by_machine(data_dir, max_files=300)
    machine_ids = sort_machine_ids(list(latest_rows.keys())) if latest_rows else []
    default_machine = machine_ids[0] if machine_ids else "M-001"
    machine = st.text_input("Machine ID for latest anomaly check", value=default_machine, key="an_machine_id")

    if st.button("Detect anomaly (predict from latest features)", key="an_detect_one_btn"):
        machine_row = latest_rows.get(machine)
        if machine_row is None:
            st.warning(f"No latest feature row found for machine {machine}.")
        else:
            features = numeric_feature_payload(machine_row)
            if not features:
                st.error("No numeric features available for anomaly prediction.")
            else:
                ok_a, msg_a, payload = client.anomaly_detect(machine_id=machine, features=features)
                if ok_a and payload:
                    st.success("Anomaly prediction complete")
                    st.json(payload)
                else:
                    st.error(f"Anomaly prediction failed: {msg_a}")

    if st.button("Fleet anomaly scan", key="an_detect_fleet_btn"):
        if not machine_ids:
            st.warning("No machines found in offline features.")
        else:
            rows = []
            for machine_id in machine_ids:
                payload_features = numeric_feature_payload(latest_rows.get(machine_id, {}))
                if not payload_features:
                    continue
                ok_a, _, payload = client.anomaly_detect(machine_id=machine_id, features=payload_features)
                if ok_a and payload:
                    rows.append(payload)
            if rows:
                frame = sort_frame_by_machine_id(pd.DataFrame(rows))
                st.dataframe(frame, use_container_width=True)
                if {"machine_id", "anomaly_score", "severity"}.issubset(set(frame.columns)):
                    fig_score = px.bar(
                        frame,
                        x="machine_id",
                        y="anomaly_score",
                        color="severity",
                        title="Anomaly Score by Machine",
                        category_orders={"machine_id": frame["machine_id"].tolist()},
                    )
                    st.plotly_chart(fig_score, use_container_width=True)
                if "severity" in frame.columns:
                    severity_counts = frame["severity"].value_counts().reset_index()
                    severity_counts.columns = ["severity", "count"]
                    fig = px.bar(severity_counts, x="severity", y="count", color="severity", title="Severity Counts")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No anomaly responses returned from service.")


def render_demo_scenarios(client: ApiClient, data_dir: Path) -> None:
    st.subheader("Demo Scenarios")
    st.caption(
        "Run controlled fault simulations on top of latest machine features. "
        "This is a showcase layer to explain what is happening over time."
    )

    latest_rows = load_latest_rows_by_machine(data_dir, max_files=300)
    machine_ids = sort_machine_ids(list(latest_rows.keys())) if latest_rows else []
    if not machine_ids:
        st.info("No machine features found. Start data generation first.")
        return

    scenario_key = st.selectbox(
        "Scenario",
        options=SCENARIO_ORDER,
        format_func=lambda key: SCENARIO_LIBRARY[key]["label"],
        key="demo_scenario_key",
    )
    scenario_meta = SCENARIO_LIBRARY[scenario_key]
    st.info(
        f"{scenario_meta['description']}  \n"
        f"Expected behavior: {scenario_meta['expected']}"
    )

    selected_machines = st.multiselect(
        "Machines",
        options=machine_ids,
        default=machine_ids[: min(4, len(machine_ids))],
        key="demo_selected_machines",
    )
    steps = st.slider("Simulation steps", min_value=6, max_value=80, value=24, step=2, key="demo_steps")
    intensity = st.slider("Scenario intensity", min_value=0.5, max_value=2.5, value=1.2, step=0.1, key="demo_intensity")

    if st.button("Run scenario simulation", key="demo_run_btn", use_container_width=True):
        ok_risk, msg_risk, _ = client.failure_risk_health()
        ok_anomaly, msg_anomaly, _ = client.anomaly_health()
        if not ok_risk:
            st.error(f"Failure-risk service unavailable: {msg_risk}")
            return
        if not ok_anomaly:
            st.error(f"Anomaly service unavailable: {msg_anomaly}")
            return
        if not selected_machines:
            st.warning("Select at least one machine.")
            return

        records: List[Dict] = []
        skipped_machines = []
        for machine_id in selected_machines:
            base_features = numeric_feature_payload(latest_rows.get(machine_id, {}))
            if not base_features:
                skipped_machines.append(machine_id)
                continue

            for step in range(steps):
                simulated = apply_scenario_mutation(
                    base_features=base_features,
                    scenario_key=scenario_key,
                    step=step,
                    total_steps=steps,
                    intensity=float(intensity),
                )

                ok_pred, _, pred = client.failure_risk_predict(simulated)
                ok_an, _, anomaly = client.anomaly_detect(machine_id=machine_id, features=simulated)
                if not ok_pred or not pred or not ok_an or not anomaly:
                    continue

                records.append(
                    {
                        "scenario": scenario_key,
                        "machine_id": machine_id,
                        "step": step + 1,
                        "progress": round((step + 1) / steps, 4),
                        "failure_probability": float(pred.get("failure_probability", 0.0)),
                        "risk_level": pred.get("risk_level", "unknown"),
                        "anomaly_score": float(anomaly.get("anomaly_score", 0.0)),
                        "anomaly_flag": bool(anomaly.get("anomaly_flag", False)),
                        "severity": anomaly.get("severity", "normal"),
                        "reason": anomaly.get("reason", "normal"),
                        "trigger_count": int(anomaly.get("trigger_count", 0)),
                    }
                )

        if skipped_machines:
            st.warning(f"Skipped machines with no numeric baseline features: {', '.join(skipped_machines)}")
        if not records:
            st.error("No simulation results were produced.")
            return

        results = pd.DataFrame(records)
        st.session_state["demo_results"] = results
        st.session_state["demo_run_scenario_key"] = scenario_key
        st.session_state["demo_steps_run"] = steps
        st.session_state["demo_intensity_run"] = float(intensity)

    results: pd.DataFrame = st.session_state.get("demo_results", pd.DataFrame())
    if results.empty:
        st.info("Run a scenario to generate timeline charts and incident narrative.")
        return

    scenario_key_saved = st.session_state.get("demo_run_scenario_key", scenario_key)
    scenario_meta_saved = SCENARIO_LIBRARY.get(scenario_key_saved, scenario_meta)
    st.markdown("#### What Is Happening")
    st.write(
        f"Scenario: **{scenario_meta_saved['label']}**. "
        f"This simulation creates controlled signal shifts and sends each step "
        f"to both failure-risk and anomaly models."
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    with metric_1:
        st.metric("Simulated points", len(results))
    with metric_2:
        st.metric("Max failure probability", f"{results['failure_probability'].max():.3f}")
    with metric_3:
        anomaly_rate = float(results["anomaly_flag"].mean()) if not results.empty else 0.0
        st.metric("Anomaly rate", f"{anomaly_rate:.1%}")
    with metric_4:
        st.metric("Max anomaly score", f"{results['anomaly_score'].max():.3f}")

    risk_fig = px.line(
        results,
        x="step",
        y="failure_probability",
        color="machine_id",
        markers=True,
        title="Failure-Risk Timeline",
    )
    st.plotly_chart(risk_fig, use_container_width=True)

    anomaly_fig = px.line(
        results,
        x="step",
        y="anomaly_score",
        color="machine_id",
        markers=True,
        title="Anomaly Score Timeline",
    )
    st.plotly_chart(anomaly_fig, use_container_width=True)

    severity_counts = (
        results.groupby(["step", "severity"], as_index=False).size().rename(columns={"size": "count"})
    )
    sev_fig = px.bar(
        severity_counts,
        x="step",
        y="count",
        color="severity",
        title="Severity Distribution Over Time",
        category_orders={"severity": SEVERITY_ORDER},
    )
    st.plotly_chart(sev_fig, use_container_width=True)

    final_snapshot = (
        results.sort_values("step")
        .groupby("machine_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    final_snapshot = sort_frame_by_machine_id(final_snapshot)
    st.markdown("#### Final Step Snapshot")
    st.dataframe(final_snapshot, use_container_width=True)

    final_risk_fig = px.bar(
        final_snapshot,
        x="machine_id",
        y="failure_probability",
        color="risk_level",
        title="Final Failure Probability by Machine",
        category_orders={"machine_id": final_snapshot["machine_id"].tolist()},
    )
    st.plotly_chart(final_risk_fig, use_container_width=True)

    final_anomaly_fig = px.bar(
        final_snapshot,
        x="machine_id",
        y="anomaly_score",
        color="severity",
        title="Final Anomaly Score by Machine",
        category_orders={"machine_id": final_snapshot["machine_id"].tolist()},
    )
    st.plotly_chart(final_anomaly_fig, use_container_width=True)

    st.markdown("#### Incident Narrative")
    for machine_id in sort_machine_ids(final_snapshot["machine_id"].tolist()):
        machine_frame = results[results["machine_id"] == machine_id].sort_values("step")
        medium_step = first_step_where(machine_frame, "failure_probability", lambda s: s >= 0.4)
        high_step = first_step_where(machine_frame, "failure_probability", lambda s: s >= 0.7)
        anomaly_step = first_step_where(machine_frame, "anomaly_flag", lambda s: s.astype(bool))

        active_reasons = machine_frame[machine_frame["reason"] != "normal"]["reason"]
        top_reason = active_reasons.mode().iloc[0] if not active_reasons.empty else "normal_pattern"

        summary_bits = [f"{machine_id}: dominant anomaly reason `{top_reason}`."]
        if medium_step is not None:
            summary_bits.append(f"Risk crossed medium at step {medium_step}.")
        else:
            summary_bits.append("Risk stayed below medium threshold.")
        if high_step is not None:
            summary_bits.append(f"Risk crossed high at step {high_step}.")
        if anomaly_step is not None:
            summary_bits.append(f"Anomaly flag first fired at step {anomaly_step}.")
        else:
            summary_bits.append("No anomaly flag during this run.")

        st.write(" ".join(summary_bits))


def render_monitoring(orchestrator: Orchestrator, client: ApiClient, prometheus_base: str) -> None:
    st.subheader("Monitoring")
    control_1, control_2, control_3 = st.columns(3)
    with control_1:
        if st.button("Start Prometheus", key="mon_start_btn", use_container_width=True):
            render_command_result(orchestrator.prometheus_start())
    with control_2:
        if st.button("Stop Prometheus", key="mon_stop_btn", use_container_width=True):
            render_command_result(orchestrator.prometheus_stop())
    with control_3:
        if st.button("Prometheus Status", key="mon_status_btn", use_container_width=True):
            render_command_result(orchestrator.prometheus_status())

    st.markdown("#### Useful URLs")
    st.markdown(f"- Prometheus UI: [{prometheus_base}]({prometheus_base})")
    st.markdown("- Failure-risk metrics: [http://localhost:8000/metrics](http://localhost:8000/metrics)")
    st.markdown("- Anomaly metrics: [http://localhost:8001/metrics](http://localhost:8001/metrics)")

    ok_t, msg_t, payload = client.prometheus_targets(base_url=prometheus_base)
    if ok_t and payload:
        targets = payload.get("data", {}).get("activeTargets", [])
        if targets:
            target_rows = []
            for item in targets:
                labels = item.get("labels", {})
                target_rows.append(
                    {
                        "job": labels.get("job"),
                        "instance": labels.get("instance"),
                        "health": item.get("health"),
                        "lastScrape": item.get("lastScrape"),
                        "scrapeUrl": item.get("scrapeUrl"),
                    }
                )
            st.dataframe(pd.DataFrame(target_rows), use_container_width=True)
        else:
            st.info("No active scrape targets found.")
    else:
        st.warning(f"Prometheus targets unavailable: {msg_t}")

    st.markdown("#### Ad-hoc PromQL")
    query = st.text_input("Query", value="up", key="mon_query")
    if st.button("Run query", key="mon_query_btn"):
        ok_q, msg_q, payload_q = client.prometheus_query(query=query, base_url=prometheus_base)
        if ok_q and payload_q:
            result = payload_q.get("data", {}).get("result", [])
            if result:
                rows = []
                for item in result:
                    metric = item.get("metric", {})
                    value = item.get("value", [None, None])
                    rows.append(
                        {
                            "job": metric.get("job"),
                            "instance": metric.get("instance"),
                            "metric": json.dumps(metric, sort_keys=True),
                            "timestamp": value[0],
                            "value": value[1],
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("Query returned no rows.")
        else:
            st.error(f"Prometheus query failed: {msg_q}")


def main() -> None:
    st.set_page_config(page_title="Predictive Maintenance Control Plane", layout="wide")
    st.title("Predictive Maintenance Control Plane")
    st.caption("Operate data generation, training, serving, inference, and monitoring from one UI.")

    orchestrator, client, prometheus_base = make_orchestrator_and_clients()

    tabs = st.tabs(
        [
            "Overview",
            "Data Generation",
            "Failure-Risk",
            "Anomaly",
            "Demo Scenarios",
            "Monitoring",
        ]
    )

    with tabs[0]:
        render_overview(orchestrator=orchestrator, data_dir=orchestrator.data_dir)
    with tabs[1]:
        render_data_generation(orchestrator=orchestrator, data_dir=orchestrator.data_dir)
    with tabs[2]:
        render_failure_risk(orchestrator=orchestrator, client=client, data_dir=orchestrator.data_dir)
    with tabs[3]:
        render_anomaly(orchestrator=orchestrator, client=client, data_dir=orchestrator.data_dir)
    with tabs[4]:
        render_demo_scenarios(client=client, data_dir=orchestrator.data_dir)
    with tabs[5]:
        render_monitoring(orchestrator=orchestrator, client=client, prometheus_base=prometheus_base)


if __name__ == "__main__":
    main()
