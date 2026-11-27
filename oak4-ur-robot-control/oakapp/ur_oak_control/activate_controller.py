"""Make sure the trajectory controller ends up active once the driver is running.

Some driver versions leave scaled_joint_trajectory_controller inactive on real /
simulated hardware (it only comes up active with use_mock_hardware). This runs
the equivalent of `ros2 control set_controller_state <controller> active`
inside the app, so external clients such as MoveIt on the host can use it.
"""

from __future__ import annotations

import os

import rclpy
from rclpy.node import Node

from ur_oak_control.arm import CONTROLLER, URArm


def main() -> None:
    rclpy.init()
    node = Node("oak4_ur_controller_activator")
    arm = URArm(node, tf_prefix=os.environ.get("TF_PREFIX", ""),
                controller=os.environ.get("JOINT_CONTROLLER", CONTROLLER))
    try:
        arm.ensure_controller_active(timeout=float(os.environ.get("CONTROLLER_TIMEOUT", "300")))
        node.get_logger().info(f"{arm.controller} is active")
    except (RuntimeError, KeyboardInterrupt) as exc:
        node.get_logger().error(str(exc))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
