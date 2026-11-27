#!/usr/bin/env bash
# Stop the URSim container started by start_ursim.sh.
set -euo pipefail
docker stop "${CONTAINER_NAME:-ursim}"
