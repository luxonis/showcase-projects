"""Thin rclpy wrapper around the UR ROS 2 driver.

This is the extension point: perception code running on the OAK4 (DepthAI) can
import URArm and call move_joints() / wait_until_ready() without dealing with the
driver's topics, services and actions directly.
"""

from __future__ import annotations

import math
import os
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

UR_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

CONTROLLER = "scaled_joint_trajectory_controller"
PROGRAM_RUNNING_TOPIC = "/io_and_status_controller/robot_program_running"
RESEND_PROGRAM_SRV = "/io_and_status_controller/resend_robot_program"


def _to_duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))


class URArm:
    def __init__(self, node: Node, tf_prefix: str = "", controller: str = CONTROLLER):
        self.node = node
        self.joint_names = [tf_prefix + j for j in UR_JOINTS]
        self.controller = controller
        # Only in headless mode can the driver (re)send the program itself.
        self.headless = os.environ.get("HEADLESS_MODE", "true").lower() == "true"
        self.positions: dict[str, float] = {}
        self.program_running = False

        node.create_subscription(JointState, "/joint_states", self._on_joint_states, 1)  # newest only
        self._program_sub = None
        self._action = ActionClient(node, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
        self._resend = node.create_client(Trigger, RESEND_PROGRAM_SRV)
        self._list_ctrl = node.create_client(ListControllers, "/controller_manager/list_controllers")
        self._switch_ctrl = node.create_client(SwitchController, "/controller_manager/switch_controller")

    # Callbacks -------------------------------------------------------------
    def _on_joint_states(self, msg: JointState) -> None:
        self.positions.update(zip(msg.name, msg.position))

    def _on_program_running(self, msg: Bool) -> None:
        if msg.data != self.program_running:
            self.node.get_logger().info(f"robot program running: {msg.data}")
        self.program_running = msg.data

    # Helpers ---------------------------------------------------------------
    def _spin_until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return predicate()

    def _subscribe_program_running(self, timeout: float) -> None:
        """Match the publisher's durability so a latched value is received."""
        if self._program_sub is not None:
            return
        self._spin_until(lambda: bool(self.node.get_publishers_info_by_topic(PROGRAM_RUNNING_TOPIC)), timeout)
        infos = self.node.get_publishers_info_by_topic(PROGRAM_RUNNING_TOPIC)
        durability = DurabilityPolicy.VOLATILE
        if infos and infos[0].qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL:
            durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=durability)
        self._program_sub = self.node.create_subscription(Bool, PROGRAM_RUNNING_TOPIC, self._on_program_running, qos)

    def _call(self, client, request, timeout: float = 10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        return future.result() if future.done() else None

    def controller_state(self) -> str | None:
        res = self._call(self._list_ctrl, ListControllers.Request())
        if res is None:
            return None
        for c in res.controller:
            if c.name == self.controller:
                return c.state
        return "not loaded"

    def ensure_controller_active(self, timeout: float = 60.0) -> None:
        """Activate the trajectory controller if the driver left it inactive.

        Equivalent of `ros2 control set_controller_state <controller> active`, done
        from code so nothing has to be typed on the camera.
        """
        log = self.node.get_logger()
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            state = self.controller_state()
            if state == "active":
                return
            if state == "inactive":
                log.info(f"{self.controller} is inactive - activating")
                req = SwitchController.Request(
                    activate_controllers=[self.controller],
                    strictness=SwitchController.Request.BEST_EFFORT,
                )
                res = self._call(self._switch_ctrl, req)
                log.info(f"switch_controller ok={getattr(res, 'ok', None)}")
            else:
                log.info(f"{self.controller} state: {state} - waiting")
            time.sleep(1.0)
        raise RuntimeError(f"{self.controller} did not become active")

    def current_joints(self) -> list[float] | None:
        if not all(j in self.positions for j in self.joint_names):
            return None
        return [self.positions[j] for j in self.joint_names]

    # Public API ------------------------------------------------------------
    def wait_until_ready(self, timeout: float = 300.0, resend_after: float = 15.0,
                         check_program: bool = True) -> None:
        """Block until joint states arrive, the URScript program runs and the controller is up.

        check_program=False skips the robot-program check (mock hardware has no program).
        """
        log = self.node.get_logger()

        log.info("waiting for /joint_states ...")
        if not self._spin_until(lambda: self.current_joints() is not None, timeout):
            raise RuntimeError("no joint states from the driver")

        if check_program:
            self._wait_for_program(timeout, resend_after)

        self.ensure_controller_active()
        log.info("waiting for the trajectory controller action server ...")
        if not self._action.wait_for_server(timeout_sec=timeout):
            raise RuntimeError("trajectory controller action server not available")
        log.info("arm ready")

    def _wait_for_program(self, timeout: float, resend_after: float) -> None:
        log = self.node.get_logger()
        self._subscribe_program_running(timeout=30)
        log.info("waiting for the robot program to run ..." if self.headless else
                 "waiting for the robot program - press Play on the External Control program in PolyScope")
        start = time.monotonic()
        last_reminder = start
        resent = False
        while not self.program_running:
            now = time.monotonic()
            if self.headless and now - start > timeout:
                raise RuntimeError("robot program is not running")
            if not self.headless and now - last_reminder > 30:
                # URCap mode: the operator starts the program whenever ready - keep waiting.
                log.info(f"still waiting for Play on the External Control program ({now - start:.0f} s)")
                last_reminder = now
            if (self.headless and not resent and time.monotonic() - start > resend_after
                    and self._resend.service_is_ready()):
                log.info("program still not running - asking the driver to resend it (headless mode)")
                self._resend.call_async(Trigger.Request())
                resent = True
            rclpy.spin_once(self.node, timeout_sec=0.2)

    def move_joints(self, waypoints: list[list[float]], max_joint_speed: float = 0.5,
                    min_segment_time: float = 1.0) -> bool:
        """Move through joint-space waypoints, stopping at each one.

        Segment duration is derived from the largest joint displacement and
        max_joint_speed [rad/s], so speed stays bounded regardless of the poses.
        """
        current = self.current_joints()
        if current is None:
            raise RuntimeError("joint states unknown")

        traj = JointTrajectory(joint_names=self.joint_names)
        t = 0.0
        prev = current
        for wp in waypoints:
            if len(wp) != len(self.joint_names):
                raise ValueError(f"waypoint needs {len(self.joint_names)} values, got {wp}")
            delta = max(abs(a - b) for a, b in zip(wp, prev))
            t += max(min_segment_time, delta / max_joint_speed)
            traj.points.append(JointTrajectoryPoint(
                positions=[float(v) for v in wp],
                velocities=[0.0] * len(wp),
                time_from_start=_to_duration(t),
            ))
            prev = wp

        goal = FollowJointTrajectory.Goal(trajectory=traj, goal_time_tolerance=_to_duration(1.0))
        send = self._action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send)
        handle = send.result()
        if handle is None or not handle.accepted:
            self.node.get_logger().error("trajectory goal rejected")
            return False

        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, res, timeout_sec=t + 30)
        if not res.done():
            self.node.get_logger().error("trajectory timed out, cancelling")
            handle.cancel_goal_async()
            return False
        result = res.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.node.get_logger().error(f"trajectory failed: {result.error_code} {result.error_string}")
            return False
        return True


def deg(values: list[float]) -> list[float]:
    return [math.radians(v) for v in values]
