"""Hand-follow control law, independent of ROS and DepthAI (unit-testable).

Offsets are expressed in the CAMERA frame of a camera mounted on the tool:
  x = image right, y = image down, z = optical axis (forward).

Modes
  mirror  The tool offset from home is proportional to the hand's offset from the
          image centre / reference distance. Use this when the camera is NOT on the
          arm (e.g. the URSim demo): hand up -> tool up, hand left -> tool left, ...
  servo   Eye-in-hand visual servoing: the tool moves so that the hand returns to
          the image centre at the reference distance. Use this when the camera is
          really mounted on the tool; with a fixed camera it would drift to the
          workspace limit - unless eye_in_hand_view() emulates the moving camera
          (EYE_IN_HAND_SIM, for trying servo mode in URSim with a camera on the desk).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class FollowConfig:
    mode: str = "mirror"
    workspace: tuple = (0.20, 0.15, 0.15)      # +- metres around home, camera frame
    max_tcp_speed: float = 0.15                # m/s
    max_tcp_accel: float = 0.0                 # m/s^2, 0 = no limit (smooths servo motion)
    deadband: float = 0.03                     # normalised image units
    deadband_z: float = 0.03                   # m of depth error ignored
    smoothing: float = 0.35                    # EMA factor for new observations (0..1, 1 = no smoothing)
    follow_depth: bool = True
    reference_depth: float = 0.6               # m, hand distance treated as "centre"
    # mirror mode
    gain_xy: float = 0.5                       # m of tool motion per full image width
    gain_z: float = 0.5                        # m of tool motion per m of hand depth change
    # servo mode
    kp_xy: float = 0.4                         # m/s per normalised image error
    kp_z: float = 0.5                          # m/s per m depth error
    lost_timeout: float = 0.5                  # s without a hand -> hold
    return_home_after: float = 5.0             # s without a hand -> drift back home (0 = never)

    @classmethod
    def from_dict(cls, d: dict) -> "FollowConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.workspace = tuple(float(v) for v in cfg.workspace)
        if cfg.mode not in ("mirror", "servo"):
            raise ValueError(f"mode must be mirror or servo, got {cfg.mode!r}")
        return cfg


@dataclass
class FollowController:
    cfg: FollowConfig
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))   # commanded, camera frame
    _filt: np.ndarray | None = None
    _last_seen: float = -math.inf
    _step: np.ndarray = field(default_factory=lambda: np.zeros(3))  # last per-cycle step
    state: str = "waiting"

    @staticmethod
    def _deadband(v: float, db: float) -> float:
        return 0.0 if abs(v) < db else v - math.copysign(db, v)

    def reset_filter(self) -> None:
        """Forget the smoothed error (call when the target point changes, e.g. hand -> held object)."""
        self._filt = None

    def update(self, obs, now: float, dt: float, target_point=None, overrides: dict | None = None) -> np.ndarray:
        """obs: HandObservation-like (x, y, depth_m, stamp) or None. Returns the new offset.

        target_point: (x, y, depth) to bring the observation to (servo mode; default image
        centre at reference_depth). overrides: per-call FollowConfig values (max_tcp_speed,
        workspace, deadband, deadband_z, kp_xy, kp_z) - used while approaching a held object."""
        c = replace(self.cfg, **{k: v for k, v in (overrides or {}).items() if v is not None}) \
            if overrides else self.cfg
        tx, ty, tz = target_point if target_point is not None else (0.5, 0.5, c.reference_depth)
        fresh = obs is not None and now - obs.stamp < c.lost_timeout
        if fresh:
            z = obs.depth_m if (c.follow_depth and obs.depth_m == obs.depth_m) else tz
            meas = np.array([obs.x - tx, obs.y - ty, z - tz])
            self._filt = meas if self._filt is None else self._filt + c.smoothing * (meas - self._filt)
            self._last_seen = now
            ex, ey = self._deadband(self._filt[0], c.deadband), self._deadband(self._filt[1], c.deadband)
            ez = self._deadband(self._filt[2], c.deadband_z)
            if c.mode == "mirror":
                target = np.array([ex * c.gain_xy, ey * c.gain_xy, ez * c.gain_z])
            else:
                target = self.offset + np.array([c.kp_xy * ex, c.kp_xy * ey, c.kp_z * ez]) * dt
            self.state = "following"
        else:
            self._filt = None
            lost_for = now - self._last_seen
            if c.return_home_after > 0 and lost_for > c.return_home_after:
                target = np.zeros(3)
                self.state = "returning home" if np.linalg.norm(self.offset) > 1e-3 else "waiting"
            else:
                target = self.offset
                self.state = "holding" if math.isfinite(lost_for) else "waiting"

        ws = np.array(c.workspace)
        target = np.clip(target, -ws, ws)
        step = target - self.offset
        n = np.linalg.norm(step)
        max_step = c.max_tcp_speed * dt
        if n > max_step:
            step *= max_step / n
        if c.max_tcp_accel > 0:
            # limit the change of velocity per cycle -> no jerks from detection noise
            dstep = step - self._step
            dn = np.linalg.norm(dstep)
            max_dstep = c.max_tcp_accel * dt * dt
            if dn > max_dstep:
                step = self._step + dstep * (max_dstep / dn)
        # a wider per-call workspace (held object) may have left the offset outside the normal one:
        # never jump back - the speed-limited step brings it in gradually
        lim = np.maximum(ws, np.abs(self.offset))
        self.offset = np.clip(self.offset + step, -lim, lim)
        self._step = step
        return self.offset

    @property
    def velocity_step(self) -> np.ndarray:
        """Offset change of the last cycle (camera frame, m)."""
        return self._step


def eye_in_hand_view(obs, offset: np.ndarray, focal: tuple, default_depth: float):
    """What a camera on the tool would see, given what a FIXED camera sees.

    The fixed camera is assumed to sit where the tool camera is at the home pose, with the
    same orientation. The tool camera has been moved by `offset` (camera frame, metres), so
    the hand appears at hand_position - offset. `focal` is (fx, fy) in image widths/heights.
    Used to try servo mode in the simulation; on a real eye-in-hand setup it is not needed.
    """
    fx, fy = focal
    has_depth = obs.depth_m == obs.depth_m
    z = obs.depth_m if has_depth else default_depth
    hand = np.array([(obs.x - 0.5) * z / fx, (obs.y - 0.5) * z / fy, z])
    v = hand - np.asarray(offset, dtype=float)
    vz = max(v[2], 0.05)
    return replace(obs, x=0.5 + fx * v[0] / vz, y=0.5 + fy * v[1] / vz,
                   depth_m=vz if has_depth else float("nan"))
