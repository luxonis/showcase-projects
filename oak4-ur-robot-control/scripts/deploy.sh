#!/usr/bin/env bash
# Build and run the OAK4 app with a config profile.
#
#   ./scripts/deploy.sh sim              # foreground, logs streamed
#   ./scripts/deploy.sh sim --detach     # extra args go to `oakctl app run`
#   ./scripts/deploy.sh real
#   ./scripts/deploy.sh sim ROBOT_IP=10.12.222.50 RUN_MOTION=false   # KEY=VALUE overrides the profile
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-sim}"
shift || true
CONFIG="$ROOT/config/$PROFILE.env"
[[ -f "$CONFIG" ]] || { echo "No such profile: $CONFIG"; exit 1; }

declare -A vars=()
order=()
set_var() { [[ -v "vars[$1]" ]] || order+=("$1"); vars[$1]="$2"; }

while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"; line="$(echo "$line" | xargs)"
  [[ -z "$line" ]] && continue
  set_var "${line%%=*}" "${line#*=}"
done < "$CONFIG"

oakctl_args=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[A-Z_][A-Z0-9_]*= ]]; then set_var "${arg%%=*}" "${arg#*=}"; else oakctl_args+=("$arg"); fi
done

CAMERA_IP="${vars[CAMERA_IP]:-}"
[[ -n "$CAMERA_IP" ]] || { echo "CAMERA_IP missing in $CONFIG"; exit 1; }

env_args=()
for key in "${order[@]}"; do
  [[ "$key" == "CAMERA_IP" ]] && continue
  value="${vars[$key]}"
  if [[ "$value" == "CHANGE_ME" ]]; then echo "Set $key in $CONFIG (or pass $key=...) first."; exit 1; fi
  env_args+=(--env "$key=$value")
done

echo "Deploying to OAK4 at $CAMERA_IP with profile '$PROFILE': ${env_args[*]}"
exec oakctl app run "$ROOT/oakapp" -d "$CAMERA_IP" "${env_args[@]}" "${oakctl_args[@]}"
