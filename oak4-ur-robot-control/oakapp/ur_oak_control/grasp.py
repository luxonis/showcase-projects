"""Grasping a held object: the grasp state machine (no ROS / DepthAI here).

Grasp target (calibrated)
  The camera sits on the flange next to the gripper, so "object between the jaws" is a fixed
  point in the camera image + a fixed depth: grasp_target {x, y, depth_m}. Measure it once:
  hold the object in the open jaws, POST /gripper/close, and if the grip holds, the object
  position recorded just before closing (GET /object -> last_mark) is the grasp target.

State machine (Grasp.update, once per control cycle)
  idle       no object -> the arm follows the hand as usual
  approach   object tracked for confirm_frames: servo it to the grasp target; while it is
             still off to the side, aim pregrasp_m farther back so the jaws come straight on
  gripped    object inside the tolerance for close_frames frames -> close, hold
  released   POST /object/release (or release_after_s) -> open; objects are ignored for
             regrip_cooldown_s so the arm does not grab it again straight away
  Without a calibrated grasp target the arm only tracks the object (centre, approach_depth_m)
  and never closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ObjectObservation:
    x: float            # 0..1 mask centroid in the detection image
    y: float
    depth_m: float      # median depth over the mask, nan if unknown
    w: float            # normalised box size
    h: float
    area: float         # mask area / image area
    stamp: float        # time.monotonic()
    confidence: float
    seq: int            # sequence number of the camera frame the mask comes from


@dataclass
class GraspConfig:
    confirm_frames: int = 3             # consecutive tracked frames before the arm reacts
    lost_timeout: float = 0.6           # s without the object -> back to the hand
    grasp_target: dict | None = None    # {x, y, depth_m} measured with the gripper (see above)
    tol_xy: float = 0.04                # normalised image error to count as "in the jaws"
    tol_z: float = 0.02                 # m
    close_frames: int = 4               # frames inside the tolerance before closing
    pregrasp_m: float = 0.08            # stay this much farther back until x / y are aligned
    approach_depth_m: float = 0.30      # no grasp target: keep the object centred this far away
    auto_grip: bool = True              # false: approach only (calibration sessions)
    release_after_s: float = 0.0        # 0 = hold until POST /object/release
    regrip_cooldown_s: float = 5.0
    speed_factor: float = 0.5           # approach = this x hand tracking (gains, speed, accel)
    slow_zone_m: float = 0.12           # within this depth of the grasp target: slow down ...
    slow_min: float = 0.3               # ... linearly to this fraction of the approach speed / gains
    max_tcp_speed: float | None = None  # m/s while approaching; None = speed_factor x hand's
    max_tcp_accel: float | None = None  # m/s^2; None = speed_factor x hand's
    workspace: tuple | None = None      # +- m (camera x, y, z) while approaching; None = hand's
    deadband: float = 0.008             # normalised image error ignored (finer than the hand)
    deadband_z: float = 0.008           # m
    kp_xy: float | None = None          # None = speed_factor x hand gains
    kp_z: float | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "GraspConfig":
        d = dict(d or {})
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if cfg.workspace is not None:
            cfg.workspace = tuple(float(v) for v in cfg.workspace)
        cfg.grasp_target = target_dict(cfg.grasp_target)
        return cfg


def target_dict(t) -> dict | None:
    if not t:
        return None
    return {"x": float(t["x"]), "y": float(t["y"]), "depth_m": float(t["depth_m"])}


@dataclass
class Grasp:
    cfg: GraspConfig
    state: str = "idle"
    detail: str = ""
    _seen: int = 0                 # consecutive frames with the object tracked
    _inside: int = 0               # consecutive frames inside the grasp tolerance
    _last_stamp: float = -1.0
    _last_seen: float = -math.inf
    _t_grip: float = 0.0
    _cooldown_until: float = 0.0
    _release_req: bool = False
    _prealigned: bool = False
    error: tuple | None = None     # (ex, ey, ez) to the grasp target, for the HUD

    @property
    def active(self) -> bool:
        """True while the object (not the hand) drives the arm."""
        return self.state in ("approach", "gripped")

    @property
    def cooling_down(self) -> bool:
        return self.state == "released"

    def request_release(self) -> None:
        self._release_req = True

    def target(self) -> tuple[float, float, float]:
        """(x, y, depth) the controller should bring the object to."""
        g = self.cfg.grasp_target
        if g is None:
            return 0.5, 0.5, self.cfg.approach_depth_m
        z = g["depth_m"] if self._prealigned else g["depth_m"] + self.cfg.pregrasp_m
        return g["x"], g["y"], z

    def slowdown(self, obj: ObjectObservation | None) -> float:
        """Speed / gain factor for the approach: 1 far away, slow_min at the grasp depth."""
        g, c = self.cfg.grasp_target, self.cfg
        if g is None or obj is None or c.slow_zone_m <= 0:
            return 1.0
        z = obj.depth_m
        if z != z:                       # depth lost - we are probably very close: go slow
            return c.slow_min
        d = max(0.0, z - g["depth_m"])
        return c.slow_min + (1.0 - c.slow_min) * min(1.0, d / c.slow_zone_m)

    def update(self, obj: ObjectObservation | None, now: float) -> str | None:
        """Advance the state machine. Returns "close" / "open" when the gripper must act."""
        c = self.cfg
        new_frame = obj is not None and obj.stamp != self._last_stamp
        fresh = obj is not None and now - obj.stamp < c.lost_timeout
        if new_frame:
            self._last_stamp = obj.stamp
            self._last_seen = now
            self._seen += 1
        elif not fresh:
            self._seen = 0

        if self.state == "gripped":
            self.detail = "holding the object"
            if self._release_req or (c.release_after_s > 0 and now - self._t_grip > c.release_after_s):
                self._release_req = False
                self.state = "released"
                self._cooldown_until = now + c.regrip_cooldown_s
                self.detail = "released"
                return "open"
            return None
        self._release_req = False

        if self.state == "released":
            if now < self._cooldown_until:
                self.detail = f"cooldown {self._cooldown_until - now:.0f} s"
                return None
            self.state = "idle"

        if self.state == "idle":
            self.detail = ""
            if fresh and self._seen >= c.confirm_frames:
                self.state, self._inside, self._prealigned = "approach", 0, False
            else:
                return None

        # approach
        if now - self._last_seen > c.lost_timeout:
            self.state, self.detail, self.error = "idle", "object lost", None
            return None
        g = c.grasp_target
        if g is None or obj is None:
            self.detail = "tracking (no grasp target calibrated)" if g is None else ""
            self.error = None
            return None
        z = obj.depth_m
        ex, ey = obj.x - g["x"], obj.y - g["y"]
        ez = (z - g["depth_m"]) if z == z else float("nan")
        self.error = (ex, ey, ez)
        xy_ok = abs(ex) < c.tol_xy and abs(ey) < c.tol_xy
        # pre-grasp: come in only once x / y are lined up (hysteresis 2.5x tolerance)
        if not self._prealigned and xy_ok:
            self._prealigned = True
        elif self._prealigned and (abs(ex) > 2.5 * c.tol_xy or abs(ey) > 2.5 * c.tol_xy):
            self._prealigned = False
        inside = xy_ok and ez == ez and abs(ez) < c.tol_z
        if new_frame:
            self._inside = self._inside + 1 if inside else 0
        self.detail = (f"err x {ex:+.3f} y {ey:+.3f} z "
                       + (f"{ez * 100:+.1f} cm" if ez == ez else "?")
                       + ("" if self._prealigned else " (pre-grasp)"))
        if self._inside >= c.close_frames:
            if not c.auto_grip:
                self.detail = "in the jaws - auto_grip off"
                return None
            self.state, self._t_grip = "gripped", now
            self.detail = "closing"
            return "close"
        return None
