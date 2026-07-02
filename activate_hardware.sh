#!/usr/bin/env bash
set -Eeuo pipefail

CAN_IFACE="${CAN_IFACE:-can0}"
CAN_BITRATE="${CAN_BITRATE:-1000000}"
CONDA_ENV="${CONDA_ENV:-piper}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MAIN=0
CHECK_ONLY=0
SKIP_CONDA=0
OPEN_VIEWER=0

usage() {
  cat <<'EOF'
Usage:
  bash activate_hardware.sh [options]

Options:
  --can IFACE          CAN interface name. Default: can0
  --bitrate BITRATE    CAN bitrate. Default: 1000000
  --conda-env NAME     Conda env name. Default: piper
  --no-conda           Skip conda activation
  --check-only         Only run hardware and Python dependency checks
  --viewer             Open realsense-viewer after checks
  --run                Run python main.py after activation
  -h, --help           Show this help

Examples:
  bash activate_hardware.sh
  bash activate_hardware.sh --run
  CAN_IFACE=can1 bash activate_hardware.sh --run
EOF
}

log() {
  printf '\n[block_piper] %s\n' "$*"
}

warn() {
  printf '\n[warning] %s\n' "$*" >&2
}

die() {
  printf '\n[error] %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --can)
      CAN_IFACE="${2:?missing CAN interface}"
      shift 2
      ;;
    --bitrate)
      CAN_BITRATE="${2:?missing bitrate}"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="${2:?missing conda env name}"
      shift 2
      ;;
    --no-conda)
      SKIP_CONDA=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --viewer)
      OPEN_VIEWER=1
      shift
      ;;
    --run)
      RUN_MAIN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

maybe_activate_conda() {
  if [[ "${SKIP_CONDA}" -eq 1 ]]; then
    log "Skipping conda activation"
    return
  fi

  if ! command -v conda >/dev/null 2>&1; then
    warn "conda command not found; continuing with current Python environment"
    return
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  PYTHON_BIN="python"
  log "Activated conda env: ${CONDA_ENV}"
}

check_usb_devices() {
  require_cmd lsusb

  log "Checking USB-CAN adapter"
  if lsusb | grep -Eiq 'CAN|OpenMoko|Geschwister|1d50:606f'; then
    lsusb | grep -Ei 'CAN|OpenMoko|Geschwister|1d50:606f' || true
  else
    warn "USB-CAN adapter was not found in lsusb output"
  fi

  log "Checking Intel RealSense camera"
  if lsusb | grep -Eiq 'RealSense|Intel.*Depth|8086'; then
    lsusb | grep -Ei 'RealSense|Intel.*Depth|8086' || true
  else
    warn "RealSense camera was not found in lsusb output"
  fi
}

activate_can() {
  require_cmd ip

  log "Activating CAN interface: ${CAN_IFACE}, bitrate: ${CAN_BITRATE}"

  if ! ip link show "${CAN_IFACE}" >/dev/null 2>&1; then
    warn "${CAN_IFACE} does not exist yet; trying to load gs_usb"
    ${SUDO} modprobe gs_usb || true
    sleep 0.5
  fi

  ip link show "${CAN_IFACE}" >/dev/null 2>&1 \
    || die "CAN interface ${CAN_IFACE} not found. Check USB-CAN connection or use --can can1."

  ${SUDO} ip link set "${CAN_IFACE}" down || true
  ${SUDO} ip link set "${CAN_IFACE}" type can bitrate "${CAN_BITRATE}"
  ${SUDO} ip link set "${CAN_IFACE}" up

  ip -details link show "${CAN_IFACE}"
}

check_python_deps() {
  log "Checking Python dependencies with: ${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY'
imports = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("pyrealsense2", "pyrealsense2"),
    ("yaml", "pyyaml"),
    ("scipy", "scipy"),
    ("ultralytics", "ultralytics"),
]

missing = []
for module, package in imports:
    try:
        __import__(module)
    except Exception as exc:
        missing.append((package, str(exc)))

try:
    from piper_sdk import C_PiperInterface_V2  # noqa: F401
except Exception as exc:
    missing.append(("piper_sdk", str(exc)))

if missing:
    print("Missing Python dependencies:")
    for package, error in missing:
        print(f"  - {package}: {error}")
    raise SystemExit(1)

print("Python dependencies OK")
PY
}

check_model_file() {
  log "Checking YOLO model path from config/camera.yaml"
  local model_path
  model_path="$("${PYTHON_BIN}" - <<'PY'
import yaml
with open("config/camera.yaml", "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["detection"]["model_path"])
PY
)"

  if [[ -f "${model_path}" ]]; then
    printf 'Model OK: %s\n' "${model_path}"
  else
    warn "YOLO model file not found: ${model_path}"
    warn "Put the .pt file there or update config/camera.yaml"
  fi
}

open_realsense_viewer() {
  if [[ "${OPEN_VIEWER}" -ne 1 ]]; then
    return
  fi

  if command -v realsense-viewer >/dev/null 2>&1; then
    log "Opening realsense-viewer"
    realsense-viewer
  else
    warn "realsense-viewer not found"
  fi
}

run_main() {
  if [[ "${RUN_MAIN}" -ne 1 ]]; then
    log "Hardware activation complete. Use --run to start main.py automatically."
    return
  fi

  log "Starting main.py"
  exec "${PYTHON_BIN}" main.py
}

maybe_activate_conda
check_usb_devices

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
  activate_can
else
  log "Check-only mode: skipping CAN activation"
fi

check_python_deps
check_model_file
open_realsense_viewer
run_main
