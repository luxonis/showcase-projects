#!/usr/bin/env bash
# OAK4 app entrypoint: prepare the robot, start the UR ROS 2 driver, run the motion node.
set -eo pipefail

# shellcheck disable=SC1091
source /opt/ros/${ROS_DISTRO:-kilted}/setup.bash
export PYTHONPATH="/app:${PYTHONPATH:-}"
# FastRPC from the device OS (mounted, see oakapp.toml) for ONNX Runtime QNN on the HTP;
# appended after the ROS setup so the ROS libraries keep priority.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/opt/luxonis/npu-runtime/lib"
# Compiled QNN contexts of the held-object models: keep them in the app's persistent storage so
# a redeploy does not recompile for minutes (cache files are keyed by model + ORT version).
if [[ -d /app-storage/app && -w /app-storage/app ]]; then
  export OAK4ORT_CACHE_DIR=/app-storage/app/oak4ort_cache
fi
export PYTHONUNBUFFERED=1

log() { echo "[oak4-ur] $*"; }

if [[ -z "${ROBOT_IP:-}" ]]; then
  log "ROBOT_IP is not set. Deploy with scripts/deploy.sh <profile> or --env ROBOT_IP=<ip>."
  exit 1
fi

UR_TYPE="${UR_TYPE:-ur5e}"
HEADLESS_MODE="${HEADLESS_MODE:-true}"
RUN_MOTION="${RUN_MOTION:-true}"

log "robot=${ROBOT_IP} type=${UR_TYPE} headless=${HEADLESS_MODE} auto_power_on=${AUTO_POWER_ON:-false}"

MOCK_HARDWARE="${MOCK_HARDWARE:-false}"

# 1) Wait for the robot and (optionally) power it on and release the brakes.
if [[ "$MOCK_HARDWARE" == "true" ]]; then
  log "MOCK_HARDWARE=true - no robot needed, using ros2_control mock hardware"
else
  python3 -m ur_oak_control.robot_prep prepare
fi

# 2) Start the driver. The robot connects back to this camera on TCP 50001-50004.
driver_args=(
  "ur_type:=${UR_TYPE}"
  "robot_ip:=${ROBOT_IP}"
  "launch_rviz:=false"
  "headless_mode:=${HEADLESS_MODE}"
  "initial_joint_controller:=scaled_joint_trajectory_controller"
)
[[ "$MOCK_HARDWARE" == "true" ]] && driver_args+=("use_mock_hardware:=true")
# PolyScope X has no Dashboard Server (robot_prep leaves this marker) - skip the driver's client.
[[ -f /tmp/ur_no_dashboard ]] && driver_args+=("launch_dashboard_client:=false")
[[ -n "${REVERSE_IP:-}" ]] && driver_args+=("reverse_ip:=${REVERSE_IP}")
[[ -n "${KINEMATICS_PARAMS_FILE:-}" ]] && driver_args+=("kinematics_params_file:=${KINEMATICS_PARAMS_FILE}")

# Hand mode streams a new short trajectory every cycle. By default the trajectory controller
# starts each one from the MEASURED joint state, which always trails the command a little;
# re-anchoring 50x/s on it makes a fast feedback loop -> visible jitter. Start each new
# trajectory from the last command instead (read-only parameter: must be set at launch).
if [[ "${CONTROL_MODE:-}" == "hand" && "${JTC_FROM_DESIRED:-true}" == "true" ]]; then
  if python3 - <<'PY'
# Text insert (no YAML round trip: rcl's parser rejects some re-serialised values).
import os
from ament_index_python.packages import get_package_share_directory
src = os.path.join(get_package_share_directory("ur_robot_driver"), "config", "ur_controllers.yaml")
names = ("scaled_joint_trajectory_controller", "joint_trajectory_controller")
out, patched, target = [], [], None
for line in open(src).read().splitlines(keepends=True):
    out.append(line)
    top = line.rstrip()
    if top.endswith(":") and not line[:1].isspace() and top[:-1] in names:
        target = top[:-1]                       # top-level section of a trajectory controller
    elif target and line.strip() == "ros__parameters:":
        indent = len(line) - len(line.lstrip()) + 2
        out.append(" " * indent + "interpolate_from_desired_state: true\n")
        patched.append(target)
        target = None
    elif target and line.strip() and not line[:1].isspace():
        target = None
if "scaled_joint_trajectory_controller" not in patched:
    raise SystemExit(f"scaled_joint_trajectory_controller section not found in {src}")
with open("/tmp/ur_controllers.yaml", "w") as f:
    f.writelines(out)
print(f"[oak4-ur] interpolate_from_desired_state=true for {', '.join(patched)}")
PY
  then
    driver_args+=("controllers_file:=/tmp/ur_controllers.yaml")
  else
    log "could not patch the controllers file - using the stock one (expect more jitter)"
  fi
fi

log "launching ur_robot_driver: ${driver_args[*]}"
ros2 launch ur_robot_driver ur_control.launch.py "${driver_args[@]}" &
DRIVER_PID=$!

children=("$DRIVER_PID")
shutdown() {
  log "stopping..."
  kill -INT "${children[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}
trap shutdown INT TERM

# 3) URCap mode: load and start the External Control program once the driver listens.
if [[ "$HEADLESS_MODE" != "true" && -n "${AUTO_PLAY_PROGRAM:-}" ]]; then
  python3 -m ur_oak_control.robot_prep play &
  children+=("$!")
fi

# 4) Make sure the trajectory controller is active (also needed for MoveIt on the host).
python3 -m ur_oak_control.activate_controller &
children+=("$!")

# 5) What drives the arm: CONTROL_MODE = waypoints | hand | none
#    (legacy RUN_MOTION=false is treated as none)
if [[ -z "${CONTROL_MODE:-}" ]]; then
  [[ "$RUN_MOTION" == "true" ]] && CONTROL_MODE=waypoints || CONTROL_MODE=none
fi
log "control mode: ${CONTROL_MODE}"
case "$CONTROL_MODE" in
  waypoints) python3 -m ur_oak_control.motion_demo & children+=("$!") ;;
  hand)      python3 -m ur_oak_control.hand_follow & children+=("$!") ;;
  none)      log "no motion node - drive the arm from MoveIt/RViz or your own node" ;;
  *)         log "unknown CONTROL_MODE '${CONTROL_MODE}'"; shutdown ;;
esac

# Exit (and let oak-agent apply restart_policy) if the driver dies.
wait "$DRIVER_PID"
log "driver exited with code $?"
shutdown
