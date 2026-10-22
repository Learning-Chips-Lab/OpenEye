#!/bin/bash
# Manage the dedicated iic-osic-tools worker container for OpenEye ASIC work.
#
# The container mounts the projects directory at /foss/designs, so the
# OpenEye repository is available inside as /foss/designs/OpenEye.
#
# Usage:
#   container.sh status
#   container.sh start
#   container.sh stop
#   container.sh exec "<command run inside the container>"
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-iic-osic-tools_asic}"
IMAGE="${IMAGE:-hpretl/iic-osic-tools:latest}"
DESIGNS="${DESIGNS:-/home/utz/projekte}"
UID_="$(id -u)"
GID_="$(id -g)"

cmd="${1:-status}"

case "$cmd" in
  status)
    docker ps -a --filter "name=${CONTAINER_NAME}" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    ;;
  start)
    if docker ps --filter "name=${CONTAINER_NAME}" -q | grep -q .; then
      echo "[INFO] ${CONTAINER_NAME} is already running."
      exit 0
    fi
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    # '-s' = skip UI startup (headless), must be first entrypoint argument.
    docker run -d \
      --name "${CONTAINER_NAME}" \
      --user "${UID_}:${GID_}" \
      --security-opt seccomp=unconfined \
      -v "${DESIGNS}:/foss/designs:rw" \
      "${IMAGE}" \
      -s /bin/bash -c "sleep infinity"
    echo "[INFO] Container ${CONTAINER_NAME} started (headless)."
    echo "[INFO] Projects mounted at /foss/designs (repo: /foss/designs/OpenEye)."
    ;;
  stop)
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    echo "[INFO] Container ${CONTAINER_NAME} stopped and removed."
    ;;
  exec)
    shift
    docker exec -it "${CONTAINER_NAME}" bash -lc "$*"
    ;;
  *)
    echo "Usage: $0 {status|start|stop|exec <command>}" >&2
    exit 1
    ;;
esac
