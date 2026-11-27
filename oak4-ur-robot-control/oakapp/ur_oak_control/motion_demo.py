"""Scripted joint-space motion executed from the OAK4.

Waypoints are read from waypoints.yaml (degrees). Environment:
  WAYPOINTS_FILE    path to the YAML (default /app/waypoints.yaml)
  MAX_JOINT_SPEED   rad/s cap used to time each segment (default 0.5)
  MOTION_LOOPS      number of passes through the waypoints, 0 = forever (default 0)
"""

from __future__ import annotations

import os
import time

import rclpy
import yaml
from rclpy.node import Node

from ur_oak_control.arm import URArm, deg


def main() -> None:
    rclpy.init()
    node = Node("oak4_ur_motion_demo")
    log = node.get_logger()

    path = os.environ.get("WAYPOINTS_FILE", "/app/waypoints.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    names = [w["name"] for w in cfg["waypoints"]]
    poses = [deg(w["joints_deg"]) for w in cfg["waypoints"]]
    speed = float(os.environ.get("MAX_JOINT_SPEED", cfg.get("max_joint_speed", 0.5)))
    loops = int(os.environ.get("MOTION_LOOPS", "0"))
    pause = float(cfg.get("pause_between_loops", 2.0))

    arm = URArm(node, tf_prefix=os.environ.get("TF_PREFIX", ""))
    check_program = os.environ.get("MOCK_HARDWARE", "false").lower() != "true"
    try:
        arm.wait_until_ready(check_program=check_program)
        n = 0
        while rclpy.ok() and (loops == 0 or n < loops):
            n += 1
            log.info(f"loop {n}: {' -> '.join(names)} (max {speed:.2f} rad/s)")
            if not arm.move_joints(poses, max_joint_speed=speed):
                log.warning("motion failed; waiting before retry")
                time.sleep(5)
                arm.wait_until_ready(check_program=check_program)
                continue
            time.sleep(pause)
        log.info("motion demo finished")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
