# Docker Run Guide (GPU 0)

This project can run inside Docker with source code bind-mounted from the host.

## Prerequisites

- Docker and Docker Compose installed.
- NVIDIA Container Toolkit installed and working.
- At least one NVIDIA GPU available on the host.

## Files

- `dockerfiles/Dockerfile.gpu`: image definition for dependencies.
- `dockerfiles/docker-compose.gpu.yml`: runtime config (mounts project and limits to GPU 0).
- `dockerfiles/run_in_docker.sh`: helper script to run with/without rebuild.
- `dockerfiles/docker_entrypoint.sh`: registers backend, then runs predictor build.

## Run

From `energy-estimation`:

```bash
./dockerfiles/run_in_docker.sh
```

Behavior:
- If image `energy-estimation:gpu-dev` exists, it runs with `--no-build`.
- If image does not exist, it builds first, then runs.

## Rebuild (only when needed)

Use rebuild when major container changes happen:
- Base image change in `dockerfiles/Dockerfile.gpu`
- Dependency change in `dockerfiles/requirements.docker.txt`

```bash
./dockerfiles/run_in_docker.sh --rebuild
```

## GPU Selection

`dockerfiles/docker-compose.gpu.yml` sets:
- `NVIDIA_VISIBLE_DEVICES=0`
- `CUDA_VISIBLE_DEVICES=0`

This makes the container use GPU 0 only.

## Notes

- Project files are mounted from host (`../:/workspace/energy-estimation` from compose file location), so normal code edits do not require image rebuild.
- Output artifacts are written into the mounted project directories on the host.

## Troubleshooting

- If GPU is not detected, verify host setup:
  - `nvidia-smi`
  - `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
- If dependencies changed but behavior looks stale, run:
  - `./dockerfiles/run_in_docker.sh --rebuild`
