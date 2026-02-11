#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=8501
HOST=0.0.0.0
INSTALL=0

usage() {
  cat <<'EOF'
Usage:
  scripts/ui.sh [--install] [--port PORT] [--host HOST]

Options:
  --install     Install ui dependencies from ui/requirements.txt before launch.
  --port PORT   Streamlit port (default: 8501).
  --host HOST   Streamlit bind host (default: 0.0.0.0).
  -h, --help    Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      INSTALL=1
      shift
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ui] unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${INSTALL}" -eq 1 ]]; then
  python3 -m pip install -r "${ROOT_DIR}/ui/requirements.txt"
fi

if ! python3 -c 'import streamlit' >/dev/null 2>&1; then
  echo "[ui] streamlit is not installed. Run: scripts/ui.sh --install" >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

exec python3 -m streamlit run "${ROOT_DIR}/ui/streamlit_app.py" \
  --server.port "${PORT}" \
  --server.address "${HOST}" \
  --server.headless true
