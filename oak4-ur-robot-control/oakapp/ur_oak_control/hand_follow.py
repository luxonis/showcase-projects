"""Make the UR tool follow a human hand seen by the OAK4.

The camera is treated as mounted on the tool (eye-in-hand): image right = tool x,
image down = tool y, optical axis = tool z (rotate with camera_rotation_deg).
In the simulation the camera stands still, so use mode "mirror" (see
follow_controller.py); with the camera really on the flange use "servo".

Environment:
  HAND_FOLLOW_CONFIG   YAML file (default /app/hand_follow.yaml)
  FOLLOW_MODE          overrides `mode` in the YAML (mirror | servo)
  EYE_IN_HAND_SIM      "true": servo mode with the camera NOT on the arm (URSim) - the hand
                       is re-projected into a virtual camera that moves with the tool
  UR_TYPE              kinematics model (ur5e, ur5, ...)
  WEB_VIEW_PORT        MJPEG debug view port (default 8090, 0 = off)
  DEPTH_SOURCE         overrides `depth_source` in the YAML (stereo | tof | fused); the
                       web view buttons switch it live - the arm follows the selected one
  FUSION_ENGINE        overrides `fusion_engine` in the YAML (heuristic | neural | vpp)
  DRY_RUN              "true": track the hand and show the would-be tool motion, but never
                       send a command to the arm (use it to check camera_rotation_deg)
  GRIPPER              log | onrobot | tool_io - close on a fist, open on an open hand, or
                       on a held object (see gripper.py; DRY_RUN forces log unless GRIPPER_IN_DRY_RUN).
                       ONROBOT_IP overrides the YAML.
  GRIPPER_IN_DRY_RUN   "true": real gripper even with DRY_RUN=true (grasp calibration: the arm
                       stays still, POST /gripper/close|open switch the gripper)
  GRASP                "false" turns grasping the held object off (`object:` / `grasp:` in the YAML)
  OBJECT_TRACKER       overrides object.tracker (xmem | cue)
  GRASP_AUTO_GRIP      overrides grasp.auto_grip ("false": approach, never close)
  DROP                 "false": after a grasp just hold (no drop sequence, `drop:` in the YAML)

`home_joints_deg: current` in the YAML uses the arm's pose at start-up as the follow
home, so the arm does not move until a hand is seen (recommended on a real arm).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ur_oak_control.arm import CONTROLLER, URArm, _to_duration
from ur_oak_control.follow_controller import FollowConfig, FollowController, eye_in_hand_view
from ur_oak_control.gripper import make_gripper
from ur_oak_control.hand_tracker import HandTracker
from ur_oak_control.kinematics import URKinematics
from ur_oak_control.grasp import GraspConfig
from ur_oak_control.held_object import HeldObject, HeldObjectConfig
from ur_oak_control.web_view import start_web_view


JOINT_INDEX = {"base": 0, "shoulder": 1, "elbow": 2, "wrist_1": 3, "wrist_2": 4, "wrist_3": 5,
               **{str(i): i for i in range(6)}}
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main() -> None:
    with open(os.environ.get("HAND_FOLLOW_CONFIG", "/app/hand_follow.yaml")) as f:
        raw = yaml.safe_load(f)
    if os.environ.get("FOLLOW_MODE"):
        raw["mode"] = os.environ["FOLLOW_MODE"]
    cfg = FollowConfig.from_dict(raw)
    rate_hz = float(raw.get("rate_hz", 20))
    home_cfg = raw.get("home_joints_deg", "current")
    home_from_current = isinstance(home_cfg, str) and home_cfg.strip().lower() == "current"
    home = None if home_from_current else np.radians(home_cfg)
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
    cam_rot = rot_z(float(raw.get("camera_rotation_deg", 0)))
    max_joint_step = float(raw.get("max_joint_step_rad", 0.15))
    lookahead = float(raw.get("command_lookahead", 0.0))   # s, 0 = legacy max(2*dt, 0.1)
    max_joint_speed = float(os.environ.get("MAX_JOINT_SPEED", "0.5"))  # rad/s, caps catch-up
    max_lead = float(raw.get("max_command_lead", 0.0))      # m, 0 = off (anti wind-up)
    # Joints held at their start angles while following; the others move the tool to the
    # target position and keep its orientation as well as they can (see ik_prioritized).
    locked = raw.get("locked_joints")
    if locked is None and raw.get("lock_wrist"):          # older config key
        locked = ["wrist_1", "wrist_2", "wrist_3"]
    locked_idx = sorted({JOINT_INDEX[str(j)] for j in (locked or [])})
    free_idx = [i for i in range(6) if i not in locked_idx]
    eih_sim = os.environ.get("EYE_IN_HAND_SIM", "false").lower() == "true"

    rclpy.init()
    node = Node("oak4_ur_hand_follow")
    log = node.get_logger()
    arm = URArm(node, tf_prefix=os.environ.get("TF_PREFIX", ""))
    kin = URKinematics(os.environ.get("UR_TYPE", "ur5e"))
    cmd_pub = node.create_publisher(JointTrajectory, f"/{CONTROLLER}/joint_trajectory", 1)
    hand_pub = node.create_publisher(PointStamped, "/hand_follow/hand", 10)
    off_pub = node.create_publisher(PointStamped, "/hand_follow/offset", 10)
    speed_scaling = [float("nan")]
    node.create_subscription(Float64, "/speed_scaling_state_broadcaster/speed_scaling",
                             lambda m: speed_scaling.__setitem__(0, m.data), 10)

    gripper_live = not dry_run or os.environ.get("GRIPPER_IN_DRY_RUN", "false").lower() == "true"
    gripper_kind = os.environ.get("GRIPPER", "log") if gripper_live else "log"
    gripper_cfg = dict(raw.get("gripper") or {})
    if os.environ.get("ONROBOT_IP"):
        gripper_cfg["onrobot_ip"] = os.environ["ONROBOT_IP"]
    gripper = make_gripper(gripper_kind, gripper_cfg, log, node)
    log.info(f"gripper backend: {gripper.name}")

    obj_cfg = HeldObjectConfig.from_dict(raw.get("object"))
    grasp_cfg = GraspConfig.from_dict(raw.get("grasp"))
    if os.environ.get("GRASP"):
        obj_cfg.enabled = os.environ["GRASP"].lower() == "true"
    if os.environ.get("OBJECT_TRACKER"):
        obj_cfg.tracker = os.environ["OBJECT_TRACKER"]
    if os.environ.get("GRASP_AUTO_GRIP"):
        grasp_cfg.auto_grip = os.environ["GRASP_AUTO_GRIP"].lower() == "true"
    held = HeldObject(obj_cfg, grasp_cfg, log=lambda m: log.info(m)) if obj_cfg.enabled else None
    # Drop sequence after a grasp: via poses -> drop pose -> open -> back to the follow home.
    # Poses are recorded by hand-guiding the arm (POST /drop/record, see web_view.py).
    dr = raw.get("drop") or {}
    drop = {"enabled": bool(dr.get("enabled", True)) and os.environ.get("DROP", "true").lower() != "false",
            "joints_deg": list(dr["joints_deg"]) if dr.get("joints_deg") else None,
            "via_deg": [list(v) for v in (dr.get("via_deg") or [])],
            "settle_s": float(dr.get("settle_s", 0.6)), "open_wait_s": float(dr.get("open_wait_s", 1.0)),
            "joint_speed": float(dr.get("joint_speed", 0.4))}
    if held is not None:
        log.info(f"held-object grasping on: tracker {obj_cfg.tracker}, grasp target "
                 f"{grasp_cfg.grasp_target or 'NOT CALIBRATED (track only)'}, auto_grip={grasp_cfg.auto_grip}")

    tracker = HandTracker(fps=int(raw.get("camera_fps", 15)),
                          min_confidence=float(raw.get("min_confidence", 0.6)),
                          grip_mode=str(raw.get("grip_mode", "off")),
                          grip_debounce=int(raw.get("grip_debounce_frames", 3)),
                          twist_open_deg=float(raw.get("twist_open_deg", 30)),
                          twist_close_deg=float(raw.get("twist_close_deg", 60)),
                          twist_reference_deg=(None if raw.get("twist_reference_deg") in (None, "auto")
                                               else float(raw["twist_reference_deg"])),
                          view_size=raw.get("view_size", 960),
                          depth_source=os.environ.get("DEPTH_SOURCE") or str(raw.get("depth_source", "stereo")),
                          stereo_engine=str(raw.get("stereo_engine", "classic")),
                          stereo_preset=str(raw.get("stereo_preset", "ROBOTICS")),
                          stereo_size=raw.get("stereo_size", [640, 400]),
                          neural_depth_model=str(raw.get("neural_depth_model", "NEURAL_DEPTH_LARGE")),
                          tof=raw.get("tof", "auto"),
                          tof_profile=str(raw.get("tof_profile", "MID_RANGE")),
                          tof_fps=raw.get("tof_fps"),
                          tof_confidence=float(raw.get("tof_confidence", 0.0)),
                          tof_min_m=float(raw.get("tof_min_m", 0.10)),
                          fuse_near_m=float(raw.get("fuse_near_m", 0.35)),
                          fuse_far_m=float(raw.get("fuse_far_m", 0.8)),
                          fusion_engine=os.environ.get("FUSION_ENGINE") or str(raw.get("fusion_engine", "heuristic")),
                          neural_fusion_confidence=float(raw.get("neural_fusion_confidence", 0.3)),
                          fusion_tof_near=bool(raw.get("fusion_tof_near", True)),
                          vpp_size=raw.get("vpp_size", [640, 400]),
                          held=held)
    tracker.drop = drop                 # shared with the web view (record / run the drop)
    tracker.arm_joints = None
    if raw.get("view_labels") is not None:
        tracker.labels = list(raw["view_labels"])
    tracker.start()
    port = int(os.environ.get("WEB_VIEW_PORT", "8090"))
    if port:
        start_web_view(tracker, port, rotation=int(raw.get("view_rotation_deg", 0)),
                       fps=float(raw.get("view_fps", 12)),
                       depth_size=int(raw.get("depth_view_size", 960)),
                       depth_range_m=raw.get("depth_view_range_m", [0.15, 1.5]))
        log.info(f"web view: http://<camera-ip>:{port}/  (depth source buttons, /rgb, /depth)")

    if held is not None:   # held-object models go onto the NPU only after the camera pipeline is up
        import threading
        threading.Thread(target=lambda: (tracker.pipeline_up.wait(), held.models_go.set()),
                         daemon=True, name="models-go").start()
    tracker.ready.wait(timeout=120)
    if tracker.error:
        log.error(f"hand tracker failed: {tracker.error}")
        return

    try:
        if dry_run:
            log.warning("DRY_RUN=true - no commands will be sent to the arm")
            log.info("waiting for /joint_states ...")
            if not arm._spin_until(lambda: arm.current_joints() is not None, 300):
                raise RuntimeError("no joint states from the driver")
        else:
            check_program = os.environ.get("MOCK_HARDWARE", "false").lower() != "true"
            arm.wait_until_ready(check_program=check_program)
        if home_from_current:
            home = np.array(arm.current_joints())
            log.info(f"follow home = current pose {np.degrees(home).round(1).tolist()} (no initial move)")
        elif dry_run:
            log.info(f"follow home {np.degrees(home).round(1).tolist()} (dry run: not moving there)")
        else:
            log.info(f"moving to follow home {np.degrees(home).round(1).tolist()}")
            arm.move_joints([home.tolist()], max_joint_speed=float(os.environ.get("MAX_JOINT_SPEED", "0.5")))

        t_home = kin.fk(home)
        r_cam = t_home[:3, :3] @ cam_rot          # camera axes in base frame
        ctrl = FollowController(cfg)
        q_cmd = np.array(arm.current_joints() or home)
        dt = 1.0 / rate_hz
        t_cmd = lookahead if lookahead > 0 else max(2 * dt, 0.1)
        last_diag = 0.0
        loop_max = 0.0       # longest loop body since the last diag line (s)
        late = 0             # cycles that overran dt
        if eih_sim and cfg.mode != "servo":
            log.warning("EYE_IN_HAND_SIM only applies to FOLLOW_MODE=servo - ignored")
            eih_sim = False
        if eih_sim:
            log.info(f"eye-in-hand simulation: virtual tool camera, focal {tracker.focal_norm}")
        log.info(f"camera axes in base frame (columns x=right, y=down, z=optical):\n{r_cam.round(2)}")
        if not locked_idx:
            log.info("tool keeps its start orientation (base frame); all joints move")
        else:
            log.info("locked at start angle: " + ", ".join(
                f"{JOINT_NAMES[i]} {np.degrees(home[i]):.1f} deg" for i in locked_idx)
                + f" | moving: {', '.join(JOINT_NAMES[i] for i in free_idx)}")
        log.info(f"hand follow running: mode={cfg.mode}{' DRY RUN' if dry_run else ''} rate={rate_hz:.0f} Hz "
                 f"max {cfg.max_tcp_speed:.2f} m/s, workspace +-{cfg.workspace} m")

        grasp = held.grasp if held is not None else None
        lc = grasp_cfg
        # approach: speed_factor x hand tracking (the tracked object lags; full gains orbit it)
        f = lc.speed_factor

        def scaled(own, hand):
            return own if own is not None else f * hand

        grasp_overrides = {"max_tcp_speed": scaled(lc.max_tcp_speed, cfg.max_tcp_speed),
                           "max_tcp_accel": scaled(lc.max_tcp_accel, cfg.max_tcp_accel),
                           "kp_xy": scaled(lc.kp_xy, cfg.kp_xy), "kp_z": scaled(lc.kp_z, cfg.kp_z),
                           "workspace": lc.workspace, "deadband": lc.deadband,
                           "deadband_z": lc.deadband_z, "lost_timeout": lc.lost_timeout}
        if held is not None:
            log.info("object approach: " + ", ".join(
                f"{k} {v:.3g}" for k, v in grasp_overrides.items() if isinstance(v, float)))
        tracker.arm_home = [round(float(v), 2) for v in np.degrees(home)]
        drop_at = None                   # time to start the drop sequence (after the grip settles)
        if held is not None:
            log.info(f"drop after grasp: {'on' if drop['enabled'] else 'off'}, pose "
                     f"{drop['joints_deg'] or 'NOT RECORDED (hold the object instead)'}, "
                     f"{len(drop['via_deg'])} via pose(s)")

        def run_drop() -> bool:
            """Carry the gripped object to the drop pose, open, go back home (blocking)."""
            target = np.radians(drop["joints_deg"])
            vias = [np.radians(v) for v in drop["via_deg"]]
            speed = min(drop["joint_speed"], max_joint_speed)
            tracker.hud = {**tracker.hud, "state": "dropping"}
            if dry_run:
                log.info(f"dry run: would carry the object via {len(vias)} pose(s) to "
                         f"{drop['joints_deg']} deg, open, and return home")
                return True
            log.info(f"drop: carrying the object to {drop['joints_deg']} deg at {speed:.2f} rad/s")
            if not arm.move_joints([*(v.tolist() for v in vias), target.tolist()], max_joint_speed=speed):
                log.error("drop: move to the drop pose failed - holding the object (POST /object/release)")
                return False
            gripper.set(False)
            arm._spin_until(lambda: False, drop["open_wait_s"])     # let it fall out, keep spinning
            log.info("drop: object released, returning to the follow home")
            if not arm.move_joints([*(v.tolist() for v in reversed(vias)), home.tolist()],
                                   max_joint_speed=speed):
                log.error("drop: return to home failed")
            return True

        next_t = time.monotonic()
        last_state = None
        last_source = None
        while rclpy.ok():
            if tracker.error:
                log.error(f"hand tracker stopped: {tracker.error} - holding position")
                break
            now = time.monotonic()
            t_body = now
            obs = tracker.latest
            if eih_sim and obs is not None:
                obs = eye_in_hand_view(obs, ctrl.offset, tracker.focal_norm, cfg.reference_depth)
                tracker.virtual_point = (obs.x, obs.y)
            elif eih_sim:
                tracker.virtual_point = None
            # web view commands: manual gripper (calibration), release after a grasp
            while tracker.commands:
                kind, arg = tracker.commands.popleft()
                if kind == "gripper":
                    log.info(f"manual gripper command: {'CLOSE' if arg else 'OPEN'}")
                    gripper.set(bool(arg), force=True)
                elif kind == "release" and grasp is not None:
                    grasp.request_release()
                elif kind == "drop" and grasp is not None and grasp.state == "gripped":
                    drop_at = now
            # the held object takes over from the hand while it is tracked / gripped
            obj_obs = None
            if grasp is not None and tracker.held is not None:
                was_active = grasp.active
                obj_obs = held.latest
                action = grasp.update(obj_obs, now)
                if action is not None:
                    log.info(f"object: {action.upper()} gripper ({grasp.detail})")
                    if action == "close":
                        # Stop right here while the jaws close: command the MEASURED joints (the
                        # stream's last point lies ahead, and the accel limit would coast on).
                        q_act = arm.current_joints()
                        if q_act and not dry_run:
                            q_cmd = np.array(q_act)
                            hold = JointTrajectory(joint_names=arm.joint_names)
                            hold.points = [JointTrajectoryPoint(positions=q_cmd.tolist(),
                                                                time_from_start=_to_duration(0.05))]
                            cmd_pub.publish(hold)
                            ctrl.offset = r_cam.T @ (kin.fk(q_cmd)[:3, 3] - t_home[:3, 3])
                        ctrl._step = np.zeros(3)
                        ctrl.reset_filter()
                        log.info(f"object position at grasp: {held.mark('auto grasp')}")
                        if drop["enabled"] and drop["joints_deg"]:
                            drop_at = now + drop["settle_s"]
                    else:   # released: forget this object, look for a new one after the cooldown
                        held.clear(suppress_s=grasp_cfg.regrip_cooldown_s, reason="released")
                    gripper.set(action == "close")
                if grasp.active != was_active:
                    ctrl.reset_filter()
            if drop_at is not None and now >= drop_at:
                drop_at = None
                if grasp is not None and grasp.state == "gripped" and drop["joints_deg"]:
                    if run_drop():
                        grasp.request_release()       # -> "open" next cycle: forget it, cooldown
                    if not dry_run:                   # the arm moved by itself: re-anchor the stream
                        q_cmd = np.array(arm.current_joints() or q_cmd)
                        ctrl.offset = np.zeros(3)
                        ctrl._step = np.zeros(3)
                        ctrl.reset_filter()
                    now = time.monotonic()
                    next_t = now
            if grasp is not None and grasp.state == "approach":
                k = grasp.slowdown(obj_obs)       # slower close to the object: no overshoot
                ov = dict(grasp_overrides)
                for key in ("max_tcp_speed", "kp_xy", "kp_z"):
                    ov[key] = grasp_overrides[key] * k
                offset = ctrl.update(obj_obs, now, dt, target_point=grasp.target(), overrides=ov)
                ctrl.state = "object approach"
            elif grasp is not None and grasp.state == "gripped":
                offset = ctrl.offset                          # frozen: no motion while gripped
                ctrl.state = "object gripped"
            else:
                offset = ctrl.update(obs, now, dt)
                # gripper follows the (debounced) hand state, only while the hand is tracked
                if ctrl.state == "following" and tracker.hand_closed is not None and tracker.grip_mode != "off":
                    gripper.set(tracker.hand_closed)
            if max_lead > 0 and not dry_run:
                # Anti wind-up: keep the command within max_lead of where the tool really is,
                # so a slow / blocked arm never piles up an offset it later jumps to.
                q_act = arm.current_joints()
                if q_act:
                    act = r_cam.T @ (kin.fk(np.array(q_act))[:3, 3] - t_home[:3, 3])
                    lead = np.clip(offset, act - max_lead, act + max_lead)
                    if not np.array_equal(lead, offset):
                        ctrl.offset = lead
                        offset = lead

            frozen = grasp is not None and grasp.state == "gripped"
            target = t_home.copy()
            target[:3, 3] = t_home[:3, 3] + r_cam @ offset
            # position first, then the start orientation as far as the free joints allow
            # (no locked joints -> orientation held exactly)
            seed = q_cmd.copy()
            seed[locked_idx] = home[locked_idx]       # locked joints pinned to the start pose
            q, ok = (q_cmd, True) if frozen else kin.ik_prioritized(target, seed, joints=free_idx)
            if frozen:
                pass       # gripping / waiting for the drop: the arm holds the measured pose
            elif ok and np.max(np.abs(q - q_cmd)) <= max_joint_step:
                q_cmd = q
                if not dry_run:
                    # One position-only point t1 ahead: the controller glides linearly towards
                    # it and every new point re-targets it, so motion is a smooth first-order
                    # follow (smoothness itself comes from the accel limit in FollowController).
                    # t1 grows if the arm lags, so it never catches up faster than
                    # max_joint_speed.
                    q_act = arm.current_joints()
                    lag = float(np.max(np.abs(q_cmd - np.array(q_act)))) if q_act else 0.0
                    t1 = max(t_cmd, lag / max_joint_speed)
                    msg = JointTrajectory(joint_names=arm.joint_names)
                    msg.points = [JointTrajectoryPoint(positions=q_cmd.tolist(),
                                                       time_from_start=_to_duration(t1))]
                    cmd_pub.publish(msg)
            elif not ok:
                log.warning("IK did not converge - holding", throttle_duration_sec=2.0)
            else:
                log.warning(f"IK jump {np.max(np.abs(q - q_cmd)):.2f} rad > {max_joint_step} - holding",
                         throttle_duration_sec=2.0)

            stamp = node.get_clock().now().to_msg()
            if obs is not None:
                hand_pub.publish(PointStamped(header=_hdr(stamp, "camera"),
                                              point=_pt(obs.x, obs.y, obs.depth_m)))
            off_pub.publish(PointStamped(header=_hdr(stamp, "camera"), point=_pt(*offset)))

            if now - last_diag > 1.0 and ctrl.state != "waiting":
                last_diag = now
                q_act = arm.current_joints()
                if q_act:
                    act = r_cam.T @ (kin.fk(np.array(q_act))[:3, 3] - t_home[:3, 3])
                    lag = float(np.max(np.abs(q_cmd - np.array(q_act))))
                    log.info(f"diag {ctrl.state}: cmd cam offset {offset.round(3).tolist()} "
                             f"actual {act.round(3).tolist()} joint lag {lag:.3f} rad "
                             f"program_running={arm.program_running} speed_scaling={speed_scaling[0]:.2f} "
                             f"loop {rate_hz:.0f} Hz max {loop_max * 1000:.1f} ms late {late}")
                loop_max, late = 0.0, 0
            if ctrl.state != last_state:
                log.info(f"state: {ctrl.state}")
                last_state = ctrl.state
            base_off = r_cam @ offset
            tracker.hud = {
                "live": not dry_run,
                "mode": cfg.mode + (" sim" if eih_sim else ""),
                "state": ctrl.state,
                "offset": offset.copy(),
                "gripper": gripper.state,
                "grip_detail": (f"object: {grasp.detail}" if grasp is not None and grasp.state != "idle"
                                else f"{tracker.grip_status}{tracker.grip_detail}")
                               + (" | ERROR" if gripper.error else ""),
                "reference_depth": cfg.reference_depth if cfg.follow_depth else None,
            }
            if tracker.depth_source != last_source:
                log.info(f"depth source: {tracker.depth_source}")
                last_source = tracker.depth_source
            if dry_run and ctrl.state in ("following", "object approach"):
                log.info(f"dry run: tool would be at base offset {base_off.round(3).tolist()} m",
                         throttle_duration_sec=1.0)

            tracker.arm_joints = arm.current_joints()
            for _ in range(4):   # joint states, speed scaling, program state, service replies
                rclpy.spin_once(node, timeout_sec=0)
            body = time.monotonic() - t_body
            loop_max = max(loop_max, body)
            if body > dt:
                late += 1
            cp = tracker.ctrl_perf
            cp["body_ms_ema"] = round(1000 * body if not cp else 0.95 * cp["body_ms_ema"] + 50 * body, 2)
            cp["late_total"] = cp.get("late_total", 0) + (body > dt)
            cp["cycles"] = cp.get("cycles", 0) + 1
            cp["rate_hz"] = rate_hz
            next_t += dt
            time.sleep(max(0.0, next_t - time.monotonic()))
            if time.monotonic() - next_t > 1.0:
                next_t = time.monotonic()   # fell behind; don't try to catch up
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        if held is not None:
            held.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _hdr(stamp, frame):
    from std_msgs.msg import Header
    return Header(stamp=stamp, frame_id=frame)


def _pt(x, y, z):
    from geometry_msgs.msg import Point
    return Point(x=float(x), y=float(y), z=float(z) if z == z else float("nan"))


if __name__ == "__main__":
    main()
