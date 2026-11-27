#!/usr/bin/env bash
# Start a simulated UR robot (URSim) on a LAN-connected x86 machine with Docker.
#
# The OAK4 camera runs the UR ROS 2 driver and connects to this machine's LAN IP,
# exactly as it would connect to a real UR controller. Therefore the robot ports
# are published on all interfaces, while the PolyScope GUI stays on localhost.
#
# Usage: ./host/start_ursim.sh            (UR5e, latest URSim e-series)
#        UR_MODEL=ur7e ./host/start_ursim.sh   (UR7e - our real arm)
#        UR_MODEL=ur10e ./host/start_ursim.sh
#        URSIM_VERSION=5.21.0 ./host/start_ursim.sh
#
# Based on https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_client_library/doc/setup/ursim_docker.html
set -euo pipefail

UR_MODEL="${UR_MODEL:-ur5e}"                # ur3e | ur5e | ur10e | ur16e (e-series image)
URSIM_VERSION="${URSIM_VERSION:-latest}"
URCAP_VERSION="${URCAP_VERSION:-1.0.5}"     # External Control URCap (only needed for URCap mode)
CONTAINER_NAME="${CONTAINER_NAME:-ursim}"
URSIM_HOME="${URSIM_HOME:-$HOME/.ursim/oak4-ur}"

case "$UR_MODEL" in
  # ur5e -> UR5; ur7e / ur12e -> UR7 / UR12 (PolyScope 5 name of the PolyScope X arms, URSim >= 5.22)
  ur3e|ur5e|ur7e|ur10e|ur12e|ur16e) URSIM_MODEL="$(echo "${UR_MODEL%e}" | tr '[:lower:]' '[:upper:]')" ;;
  *) echo "Unsupported UR_MODEL '$UR_MODEL' (use ur3e, ur5e, ur7e, ur10e, ur12e or ur16e)"; exit 1 ;;
esac

command -v docker >/dev/null || { echo "docker is not installed"; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "A container named '$CONTAINER_NAME' already exists. Stop it with ./host/stop_ursim.sh first."
  exit 1
fi

mkdir -p "$URSIM_HOME/urcaps" "$URSIM_HOME/programs"
URCAP_JAR="$URSIM_HOME/urcaps/externalcontrol-${URCAP_VERSION}.jar"
if [[ ! -f "$URCAP_JAR" ]]; then
  echo "Downloading External Control URCap ${URCAP_VERSION}..."
  curl -fL -o "$URCAP_JAR" \
    "https://github.com/UniversalRobots/Universal_Robots_ExternalControl_URCap/releases/download/v${URCAP_VERSION}/externalcontrol-${URCAP_VERSION}.jar"
fi

echo "Starting URSim ($UR_MODEL, universalrobots/ursim_e-series:${URSIM_VERSION})..."
docker run -d --rm --name "$CONTAINER_NAME" \
  -e ROBOT_MODEL="$URSIM_MODEL" \
  -p 29999:29999 \
  -p 30001-30004:30001-30004 \
  -p 127.0.0.1:6080:6080 \
  -v "$URSIM_HOME/urcaps:/urcaps" \
  -v "$URSIM_HOME/programs:/ursim/programs" \
  "universalrobots/ursim_e-series:${URSIM_VERSION}" >/dev/null

echo -n "Waiting for the Dashboard Server (port 29999)"
for _ in $(seq 1 120); do
  if (exec 3<>/dev/tcp/127.0.0.1/29999) 2>/dev/null; then echo " up."; break; fi
  echo -n "."; sleep 1
done

echo
echo "URSim is running."
echo "  PolyScope GUI:  http://localhost:6080/vnc.html"
echo "  Robot ports published on this machine's LAN IP: 29999, 30001-30004"
echo
echo "Use one of these LAN IPs as ROBOT_IP in config/sim.env:"
ip -4 -o addr show scope global 2>/dev/null | awk '{print "    " $2 ": " $4}' | grep -v -E 'docker|br-|virbr|veth' || true
echo
echo "If a firewall is active, allow TCP 29999 and 30001-30004 from the camera."
echo "Stop with: ./host/stop_ursim.sh"
