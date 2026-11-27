"""Robot preparation steps that run before / alongside the ROS 2 driver.

  python3 -m ur_oak_control.robot_prep prepare   # wait for robot, optional power on + brake release
  python3 -m ur_oak_control.robot_prep play      # URCap mode: load + play the External Control program

Environment:
  ROBOT_IP            robot / URSim IP (required)
  AUTO_POWER_ON       "true" to power on and release brakes remotely (sim default; think twice on a real arm)
  AUTO_PLAY_PROGRAM   program to load and play in URCap mode, e.g. "external_control.urp"
  PREP_TIMEOUT        seconds to wait for the robot to become reachable (default 600)

PolyScope 5 / CB3 controllers have a Dashboard Server on TCP 29999; PolyScope X (e.g. a
UR7e/UR12e/UR15 shipped with PolyScope X) does not. When 29999 stays closed, the dashboard
steps (status log, AUTO_POWER_ON, AUTO_PLAY_PROGRAM) are skipped and the operator powers
the arm on and presses Play on the pendant.
"""

from __future__ import annotations

import os
import socket
import sys
import time

from ur_oak_control.dashboard import Dashboard, DashboardError

# Needed by the ROS 2 driver on every controller generation.
ROBOT_PORTS = {
    30001: "primary",
    30002: "secondary",
    30004: "RTDE",
}
DASHBOARD_PORT = 29999  # PolyScope 5 / CB3 only
DASHBOARD_GRACE = 15.0  # seconds to wait for 29999 after the driver ports are up
# Written by prepare() when the controller has no Dashboard Server; entrypoint.sh then
# starts the driver without its dashboard_client node (it would retry forever).
NO_DASHBOARD_MARKER = "/tmp/ur_no_dashboard"
DRIVER_PORTS = {50001: "reverse", 50002: "script sender", 50003: "trajectory", 50004: "script command"}


def log(msg: str) -> None:
    print(f"[robot_prep] {msg}", flush=True)


def env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_robot(host: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_report = 0.0
    while True:
        status = {p: port_open(host, p) for p in ROBOT_PORTS}
        if all(status.values()):
            log(f"robot {host} reachable on ports {', '.join(map(str, ROBOT_PORTS))}")
            return
        now = time.monotonic()
        if now > deadline:
            missing = [f"{p} ({ROBOT_PORTS[p]})" for p, ok in status.items() if not ok]
            raise SystemExit(f"[robot_prep] robot {host} not reachable on: {', '.join(missing)}")
        if now - last_report > 10:
            missing = [str(p) for p, ok in status.items() if not ok]
            log(f"waiting for robot {host} (closed ports: {', '.join(missing)}) - is URSim / the robot running?")
            last_report = now
        time.sleep(1)


def has_dashboard(host: str) -> bool:
    """True if the controller runs a Dashboard Server (PolyScope 5 / CB3), False on PolyScope X."""
    deadline = time.monotonic() + DASHBOARD_GRACE
    while True:
        if port_open(host, DASHBOARD_PORT):
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(1)


def log_no_dashboard(host: str) -> None:
    log(f"no Dashboard Server on {host}:{DASHBOARD_PORT} - assuming PolyScope X, skipping dashboard steps")
    log("PolyScope X: enable Primary Client Interface + RTDE under Settings > Security > Services, "
        "install the External Control URCapX and run its program from the pendant")


def prepare() -> None:
    host = os.environ["ROBOT_IP"]
    wait_for_robot(host, float(os.environ.get("PREP_TIMEOUT", "600")))

    if os.path.exists(NO_DASHBOARD_MARKER):
        os.remove(NO_DASHBOARD_MARKER)
    if not has_dashboard(host):
        log_no_dashboard(host)
        open(NO_DASHBOARD_MARKER, "w").close()
        if env_true("AUTO_POWER_ON"):
            log("AUTO_POWER_ON ignored (needs the Dashboard Server) - power on and release brakes on the pendant.")
        return

    with Dashboard(host) as db:
        log(f"polyscope version: {db.cmd('PolyscopeVersion')}")
        log(f"remote control: {db.cmd('is in remote control')}")
        log(f"robot mode: {db.robot_mode()}, safety: {db.safety_status()}")

        if not env_true("AUTO_POWER_ON"):
            log("AUTO_POWER_ON is off - power on and release brakes on the teach pendant.")
            return

        safety = db.safety_status()
        if safety != "NORMAL":
            raise SystemExit(f"[robot_prep] safety status is {safety}; clear it on the pendant first.")

        db.cmd("close popup")
        if db.robot_mode() != "RUNNING":
            log(f"power on -> {db.cmd('power on')}")
            db.wait_for_mode("IDLE", timeout=60)
            log(f"brake release -> {db.cmd('brake release')}")
            db.wait_for_mode("RUNNING", timeout=60)
        log("robot is powered on and brakes are released")


def play() -> None:
    host = os.environ["ROBOT_IP"]
    program = os.environ["AUTO_PLAY_PROGRAM"]

    # The External Control program fetches its URScript from the driver on 50002.
    deadline = time.monotonic() + 120
    while not port_open("127.0.0.1", 50002, timeout=1):
        if time.monotonic() > deadline:
            raise SystemExit("[robot_prep] driver did not open port 50002")
        time.sleep(1)

    if not has_dashboard(host):
        log_no_dashboard(host)
        log(f"AUTO_PLAY_PROGRAM={program} ignored - start the External Control program on the pendant.")
        return

    with Dashboard(host) as db:
        log(f"load {program} -> {db.cmd(f'load {program}')}")
        time.sleep(1)
        log(f"play -> {db.cmd('play')}")


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    try:
        {"prepare": prepare, "play": play}[action]()
    except DashboardError as exc:
        raise SystemExit(f"[robot_prep] dashboard error: {exc}")


if __name__ == "__main__":
    main()
