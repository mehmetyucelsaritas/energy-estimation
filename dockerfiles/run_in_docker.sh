#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.gpu.yml"
SERVICE="energy-estimation"
IMAGE="energy-estimation:gpu-dev"
CONTAINER_NAME="energy-estimation-gpu0"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
REBUILD=false
DETACHED=false

for arg in "$@"; do
  case "${arg}" in
    --rebuild)
      REBUILD=true
      ;;
    --detached)
      DETACHED=true
      ;;
    *)
      echo "Usage: $0 [--rebuild] [--detached]" >&2
      exit 1
      ;;
  esac
done

# Prevent name conflicts from stale containers created by prior runs.
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

if [[ "${REBUILD}" == "true" ]]; then
  if [[ "${DETACHED}" == "true" ]]; then
    HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" docker compose -f "${COMPOSE_FILE}" up --build -d "${SERVICE}"
  else
    HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" docker compose -f "${COMPOSE_FILE}" up --build --abort-on-container-exit "${SERVICE}"
  fi
  exit 0
fi

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  if [[ "${DETACHED}" == "true" ]]; then
    HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" docker compose -f "${COMPOSE_FILE}" up --no-build -d "${SERVICE}"
  else
    HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" docker compose -f "${COMPOSE_FILE}" up --no-build --abort-on-container-exit "${SERVICE}"
  fi
else
  if [[ "${DETACHED}" == "true" ]]; then
    HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" docker compose -f "${COMPOSE_FILE}" up --build -d "${SERVICE}"
  else
    HOST_UID="${HOST_UID}" HOST_GID="${HOST_GID}" docker compose -f "${COMPOSE_FILE}" up --build --abort-on-container-exit "${SERVICE}"
  fi
fi
