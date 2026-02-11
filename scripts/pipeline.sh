#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-predictive_maintaince_platform}"
LOG_DIR="${ROOT_DIR}/data/run_logs"
DATA_DIR="${ROOT_DIR}/data/features_offline"

DEFAULT_TIMED_SECONDS=60

print_usage() {
  cat <<'USAGE'
Usage:
  scripts/pipeline.sh data_generation <start|stop|status> [--duration SECONDS] [--build] [--clear-data] [--stop-redpanda] [--all] [--compose-file PATH] [--project-name NAME] [--log-dir PATH] [--data-dir PATH]
  scripts/pipeline.sh failure-risk-prediction <train|start|stop|status> [--build] [--compose-file PATH] [--project-name NAME]
  scripts/pipeline.sh anomaly <train|start|stop|status> [--build] [--compose-file PATH] [--project-name NAME]
  scripts/pipeline.sh cmapss <train|status> [--build] [--subset FD001|FD002|FD003|FD004] [--compose-file PATH] [--project-name NAME]

Commands:
  data_generation  Data stream lifecycle: start, stop, status.
  failure-risk-prediction  Failure-risk lifecycle: train, start, stop, status.
  anomaly   Anomaly lifecycle subcommands: train, start, stop, status.
  cmapss  NASA C-MAPSS model lifecycle: train and status.

Options:
  --duration SECONDS   Timed run length for 'start'.
  --build              Build images before starting or training.
  --clear-data         Clear offline feature data before start or after stop.
  --stop-redpanda      With timed start, stop redpanda when run finishes.
  --all                With 'stop', stop all compose services (core + ml profile).
  --compose-file PATH  Override compose file.
  --project-name NAME  Override compose project name.
  --log-dir PATH       Timed-run log output directory (default: data/run_logs).
  --data-dir PATH      Offline feature directory (default: data/features_offline).
  --subset NAME        C-MAPSS subset for training (FD001, FD002, FD003, FD004).
  -h, --help           Show help.

Examples:
  scripts/pipeline.sh data_generation start
  scripts/pipeline.sh data_generation start --duration 120 --clear-data
  scripts/pipeline.sh data_generation stop --all --clear-data
  scripts/pipeline.sh failure-risk-prediction train --build
  scripts/pipeline.sh failure-risk-prediction start
  scripts/pipeline.sh anomaly train
  scripts/pipeline.sh anomaly start
  scripts/pipeline.sh cmapss train --subset FD001

Compatibility Aliases (still supported):
  start, stop, status, run, train (legacy),
  failure_risk_prediction,
  anomaly-train, anomaly-start, anomaly-stop, anomaly-status,
  cmapss-train
USAGE
}

log() {
  printf '[pipeline] %s\n' "$*"
}

compose() {
  docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" "$@"
}

compose_ml() {
  docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" --profile ml "$@"
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2
    exit 1
  fi
}

require_positive_int() {
  local value="$1"
  local name="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -le 0 ]]; then
    echo "${name} must be a positive integer" >&2
    exit 1
  fi
}

clear_feature_data() {
  if [[ -z "${DATA_DIR}" || "${DATA_DIR}" == "/" ]]; then
    echo "unsafe DATA_DIR value: ${DATA_DIR}" >&2
    exit 1
  fi

  mkdir -p "${DATA_DIR}"
  log "clearing offline data in ${DATA_DIR}"
  find "${DATA_DIR}" -mindepth 1 -delete
}

start_stream() {
  local build_flag="$1"
  local duration_seconds="$2"
  local clear_data="$3"
  local stop_redpanda="$4"

  if [[ "${clear_data}" == "true" ]]; then
    clear_feature_data
  fi

  log "starting redpanda and init-topic"
  compose up -d redpanda init-topic >/dev/null

  local -a compose_cmd=(docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}")
  compose_cmd=(env MAX_RUNTIME_SECONDS="${duration_seconds}" "${compose_cmd[@]}")

  if [[ "${build_flag}" == "true" ]]; then
    log "starting producer and aggregator (build=true)"
    "${compose_cmd[@]}" up -d --build producer aggregator >/dev/null
  else
    log "starting producer and aggregator"
    "${compose_cmd[@]}" up -d producer aggregator >/dev/null
  fi

  if [[ "${duration_seconds}" -le 0 ]]; then
    return
  fi

  mkdir -p "${LOG_DIR}"

  local run_ts
  run_ts="$(date -u +%Y%m%dT%H%M%SZ)"
  local run_log_file="${LOG_DIR}/pipeline_run_${run_ts}.log"
  local summary_file="${LOG_DIR}/pipeline_run_${run_ts}.summary"

  log "timed run started duration=${duration_seconds}s"
  log "capturing logs to ${run_log_file}"

  compose logs -f --no-color producer aggregator >"${run_log_file}" 2>&1 &
  local logs_pid=$!

  sleep "${duration_seconds}"

  kill "${logs_pid}" >/dev/null 2>&1 || true
  wait "${logs_pid}" >/dev/null 2>&1 || true

  stop_stream "false"
  if [[ "${stop_redpanda}" == "true" ]]; then
    compose stop redpanda >/dev/null 2>&1 || true
  fi

  local feature_count
  feature_count="$(find "${DATA_DIR}" -type f -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')"

  {
    echo "run_timestamp_utc=${run_ts}"
    echo "duration_seconds=${duration_seconds}"
    echo "compose_file=${COMPOSE_FILE}"
    echo "project_name=${PROJECT_NAME}"
    echo "log_file=${run_log_file}"
    echo "data_dir=${DATA_DIR}"
    echo "feature_jsonl_count=${feature_count}"
  } >"${summary_file}"

  log "timed run finished"
  log "summary: ${summary_file}"
}

stop_stream() {
  local stop_all="$1"

  if [[ "${stop_all}" == "true" ]]; then
    log "stopping all compose services (core + ml profile) in project=${PROJECT_NAME}"
    compose_ml stop >/dev/null 2>&1 || true
    compose stop >/dev/null 2>&1 || true
    return
  fi

  log "stopping producer and aggregator"
  compose stop producer aggregator >/dev/null 2>&1 || true
}

status_stream() {
  compose_ml ps
}

failure_risk_train() {
  local build_flag="$1"
  if [[ "${build_flag}" == "true" ]]; then
    log "building failure-risk-trainer image"
    compose_ml build failure-risk-trainer >/dev/null
  fi
  log "running failure-risk-trainer"
  compose_ml run --rm failure-risk-trainer
}

failure_risk_start() {
  local build_flag="$1"
  if [[ "${build_flag}" == "true" ]]; then
    log "starting failure-risk-service (build=true)"
    compose_ml up -d --build failure-risk-service >/dev/null
  else
    log "starting failure-risk-service"
    compose_ml up -d failure-risk-service >/dev/null
  fi
}

failure_risk_stop() {
  log "stopping failure-risk-service"
  compose_ml stop failure-risk-service >/dev/null 2>&1 || true
}

failure_risk_status() {
  compose_ml ps failure-risk-service
}

anomaly_train() {
  local build_flag="$1"
  if [[ "${build_flag}" == "true" ]]; then
    log "building anomaly-trainer image"
    compose_ml build anomaly-trainer >/dev/null
  fi
  log "running anomaly-trainer"
  compose_ml run --rm anomaly-trainer
}

anomaly_start() {
  local build_flag="$1"
  if [[ "${build_flag}" == "true" ]]; then
    log "starting anomaly-service (build=true)"
    compose_ml up -d --build anomaly-service >/dev/null
  else
    log "starting anomaly-service"
    compose_ml up -d anomaly-service >/dev/null
  fi
}

anomaly_stop() {
  log "stopping anomaly-service"
  compose_ml stop anomaly-service >/dev/null 2>&1 || true
}

anomaly_status() {
  compose_ml ps anomaly-service
}

cmapss_train() {
  local build_flag="$1"
  local subset="$2"

  case "${subset}" in
    FD001|FD002|FD003|FD004)
      ;;
    *)
      echo "invalid --subset: ${subset} (expected FD001|FD002|FD003|FD004)" >&2
      exit 1
      ;;
  esac

  if [[ "${build_flag}" == "true" ]]; then
    log "building cmapss-trainer image"
    compose_ml build cmapss-trainer >/dev/null
  fi
  log "running cmapss-trainer subset=${subset}"
  compose_ml run --rm -e CMAPSS_SUBSET="${subset}" cmapss-trainer
}

cmapss_status() {
  local metadata_path="${ROOT_DIR}/models/cmapss/current/metadata.json"
  local metrics_path="${ROOT_DIR}/models/cmapss/current/metrics.json"

  if [[ ! -f "${metadata_path}" ]]; then
    log "no C-MAPSS model found at ${metadata_path}"
    return
  fi

  log "current C-MAPSS metadata: ${metadata_path}"
  cat "${metadata_path}"
  if [[ -f "${metrics_path}" ]]; then
    printf '\n'
    log "current C-MAPSS metrics: ${metrics_path}"
    cat "${metrics_path}"
  fi
}

main() {
  require_docker

  if [[ "$#" -eq 0 ]]; then
    print_usage
    exit 1
  fi

  local command="$1"
  shift

  local action=""
  local run_alias="false"

  # Domain-first commands.
  case "${command}" in
    data_generation|data-generation)
      action="${1:-}"
      if [[ -z "${action}" ]]; then
        echo "missing subcommand for data_generation (use start|stop|status)" >&2
        print_usage
        exit 1
      fi
      shift
      case "${action}" in
        start|stop|status)
          command="${action}"
          ;;
        *)
          echo "unknown data_generation action: ${action}" >&2
          print_usage
          exit 1
          ;;
      esac
      ;;
    failure-risk-prediction|failure_risk_prediction|failure-risk|failure_risk)
      command="failure-risk"
      action="${1:-}"
      if [[ -z "${action}" ]]; then
        echo "missing subcommand for failure-risk-prediction (use train|start|stop|status)" >&2
        print_usage
        exit 1
      fi
      shift
      ;;
    cmapss)
      action="${1:-}"
      if [[ -z "${action}" ]]; then
        echo "missing subcommand for cmapss (use train|status)" >&2
        print_usage
        exit 1
      fi
      shift
      ;;
  esac

  # Backward-compatible aliases.
  case "${command}" in
    run)
      command="start"
      run_alias="true"
      ;;
    train)
      command="failure-risk"
      action="train"
      ;;
    model)
      # keep as-is for backward compatibility with previous interface
      command="failure-risk"
      ;;
    model-start)
      command="failure-risk"
      action="start"
      ;;
    model-stop)
      command="failure-risk"
      action="stop"
      ;;
    model-status)
      command="failure-risk"
      action="status"
      ;;
    anomaly-train)
      command="anomaly"
      action="train"
      ;;
    anomaly-start)
      command="anomaly"
      action="start"
      ;;
    anomaly-stop)
      command="anomaly"
      action="stop"
      ;;
    anomaly-status)
      command="anomaly"
      action="status"
      ;;
    cmapss-train)
      command="cmapss"
      action="train"
      ;;
  esac

  if [[ "${command}" == "failure-risk" || "${command}" == "anomaly" || "${command}" == "cmapss" ]]; then
    if [[ -z "${action}" ]]; then
      action="${1:-}"
      if [[ -z "${action}" ]]; then
        echo "missing subcommand for ${command}" >&2
        print_usage
        exit 1
      fi
      shift
    fi
  fi

  local duration_seconds="0"
  local duration_set="false"
  local build_flag="false"
  local clear_data="false"
  local stop_redpanda="false"
  local stop_all="false"
  local cmapss_subset="FD001"

  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --duration)
        duration_seconds="$2"
        duration_set="true"
        shift 2
        ;;
      --build)
        build_flag="true"
        shift
        ;;
      --clear-data)
        clear_data="true"
        shift
        ;;
      --stop-redpanda)
        stop_redpanda="true"
        shift
        ;;
      --all)
        stop_all="true"
        shift
        ;;
      --compose-file)
        COMPOSE_FILE="$2"
        shift 2
        ;;
      --project-name)
        PROJECT_NAME="$2"
        shift 2
        ;;
      --log-dir)
        LOG_DIR="$2"
        shift 2
        ;;
      --data-dir)
        DATA_DIR="$2"
        shift 2
        ;;
      --subset)
        cmapss_subset="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
        shift 2
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "unknown option: $1" >&2
        print_usage
        exit 1
        ;;
    esac
  done

  case "${command}" in
    start)
      if [[ "${run_alias}" == "true" && "${duration_set}" == "false" ]]; then
        duration_seconds="${DEFAULT_TIMED_SECONDS}"
        duration_set="true"
      fi

      if [[ "${duration_set}" == "true" ]]; then
        require_positive_int "${duration_seconds}" "--duration"
      fi

      start_stream "${build_flag}" "${duration_seconds}" "${clear_data}" "${stop_redpanda}"
      status_stream
      ;;
    stop)
      stop_stream "${stop_all}"
      if [[ "${clear_data}" == "true" ]]; then
        clear_feature_data
      fi
      status_stream
      ;;
    status)
      status_stream
      ;;
    failure-risk)
      case "${action}" in
        train)
          failure_risk_train "${build_flag}"
          ;;
        start)
          failure_risk_start "${build_flag}"
          failure_risk_status
          ;;
        stop)
          failure_risk_stop
          failure_risk_status
          ;;
        status)
          failure_risk_status
          ;;
        *)
          echo "unknown failure-risk action: ${action}" >&2
          print_usage
          exit 1
          ;;
      esac
      ;;
    anomaly)
      case "${action}" in
        train)
          anomaly_train "${build_flag}"
          ;;
        start)
          anomaly_start "${build_flag}"
          anomaly_status
          ;;
        stop)
          anomaly_stop
          anomaly_status
          ;;
        status)
          anomaly_status
          ;;
        *)
          echo "unknown anomaly action: ${action}" >&2
          print_usage
          exit 1
          ;;
      esac
      ;;
    cmapss)
      case "${action}" in
        train)
          cmapss_train "${build_flag}" "${cmapss_subset}"
          ;;
        status)
          cmapss_status
          ;;
        *)
          echo "unknown cmapss action: ${action}" >&2
          print_usage
          exit 1
          ;;
      esac
      ;;
    help|-h|--help)
      print_usage
      ;;
    *)
      echo "unknown command: ${command}" >&2
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
