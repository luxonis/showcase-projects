"""UR forward / inverse kinematics from the nominal DH parameters.

Frames follow PolyScope: the "base" frame (z up) and the "tool0" flange frame.
Joint values are the ones published on /joint_states by the UR driver.
For millimetre-accurate Cartesian control on a real arm, use the robot's
calibrated kinematics instead (ur_calibration); for following a hand the
nominal values are more than enough.
"""

from __future__ import annotations

import numpy as np

# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
_HALF_PI = np.pi / 2
_ALPHA = [_HALF_PI, 0.0, 0.0, _HALF_PI, -_HALF_PI, 0.0]
DH = {
    "ur3e": dict(d=[0.15185, 0, 0, 0.13105, 0.08535, 0.0921], a=[0, -0.24355, -0.2132, 0, 0, 0]),
    "ur5e": dict(d=[0.1625, 0, 0, 0.1333, 0.0997, 0.0996], a=[0, -0.425, -0.3922, 0, 0, 0]),
    "ur10e": dict(d=[0.1807, 0, 0, 0.17415, 0.11985, 0.11655], a=[0, -0.6127, -0.57155, 0, 0, 0]),
    # UR7e / UR12e are the UR5e / UR10e arms renamed for their real payload (same geometry).
    # The newer long-reach variants (UR7e-920, UR12e-1300) have other dimensions and are not covered.
    "ur7e": dict(d=[0.1625, 0, 0, 0.1333, 0.0997, 0.0996], a=[0, -0.425, -0.3922, 0, 0, 0]),
    "ur12e": dict(d=[0.1807, 0, 0, 0.17415, 0.11985, 0.11655], a=[0, -0.6127, -0.57155, 0, 0, 0]),
    "ur16e": dict(d=[0.1807, 0, 0, 0.17415, 0.11985, 0.11655], a=[0, -0.4784, -0.36, 0, 0, 0]),
    "ur3": dict(d=[0.1519, 0, 0, 0.11235, 0.08535, 0.0819], a=[0, -0.24365, -0.21325, 0, 0, 0]),
    "ur5": dict(d=[0.089159, 0, 0, 0.10915, 0.09465, 0.0823], a=[0, -0.425, -0.39225, 0, 0, 0]),
    "ur10": dict(d=[0.1273, 0, 0, 0.163941, 0.1157, 0.0922], a=[0, -0.612, -0.5723, 0, 0, 0]),
}


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _rot_error(r: np.ndarray, r_goal: np.ndarray) -> np.ndarray:
    """Small-angle orientation error (axis * angle) taking r to r_goal, in base frame."""
    return 0.5 * sum(np.cross(r[:, i], r_goal[:, i]) for i in range(3))


class URKinematics:
    def __init__(self, ur_type: str = "ur5e"):
        if ur_type not in DH:
            raise ValueError(f"unknown ur_type {ur_type!r}, expected one of {sorted(DH)}")
        self.d = DH[ur_type]["d"]
        self.a = DH[ur_type]["a"]

    def fk(self, q) -> np.ndarray:
        """4x4 pose of tool0 in base."""
        t = np.eye(4)
        for i in range(6):
            t = t @ _dh(q[i], self.d[i], self.a[i], _ALPHA[i])
        return t

    def jacobian(self, q, eps: float = 1e-6) -> np.ndarray:
        """6x6 geometric Jacobian (numerical): rows = [v; w] in base frame."""
        q = np.asarray(q, dtype=float)
        t0 = self.fk(q)
        j = np.zeros((6, 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += eps
            t1 = self.fk(dq)
            j[:3, i] = (t1[:3, 3] - t0[:3, 3]) / eps
            j[3:, i] = _rot_error(t0[:3, :3], t1[:3, :3]) / eps
        return j

    def ik(self, target: np.ndarray, q_seed, iterations: int = 100, tol: float = 1e-5,
           damping: float = 0.01, max_step: float = 0.2):
        """Damped least-squares IK for a full 4x4 tool0 pose, seeded near the current joints.

        Returns (q, ok). Staying close to the seed keeps the arm in the same
        configuration (no elbow/wrist flips), which is what a following
        behaviour needs.
        """
        q = np.asarray(q_seed, dtype=float).copy()
        err = np.inf
        for _ in range(iterations):
            t = self.fk(q)
            e = np.concatenate([target[:3, 3] - t[:3, 3], _rot_error(t[:3, :3], target[:3, :3])])
            err = float(np.linalg.norm(e))
            if err < tol:
                return q, True
            j = self.jacobian(q)
            dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(6), e)
            n = np.max(np.abs(dq))
            if n > max_step:
                dq *= max_step / n
            q += dq
        return q, err < 1e-3

    def ik_position(self, target_pos, q_seed, joints=(0, 1, 2), iterations: int = 100,
                    tol: float = 1e-5, damping: float = 0.01, max_step: float = 0.2):
        """Damped least-squares IK for the tool0 POSITION only, moving just `joints`.

        The other joints stay exactly at their seed values - e.g. joints=(0, 1, 2) moves
        base / shoulder / elbow and keeps the three wrist joints locked, so the tool
        orientation follows the arm instead of being held. Returns (q, ok).
        """
        idx = list(joints)
        q = np.asarray(q_seed, dtype=float).copy()
        target_pos = np.asarray(target_pos, dtype=float)
        err = np.inf
        for _ in range(iterations):
            e = target_pos - self.fk(q)[:3, 3]
            err = float(np.linalg.norm(e))
            if err < tol:
                return q, True
            j = self.jacobian(q)[:3, idx]
            dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(3), e)
            n = np.max(np.abs(dq))
            if n > max_step:
                dq *= max_step / n
            q[idx] += dq
        return q, err < 1e-3

    def jacobian_analytic(self, q) -> tuple[np.ndarray, np.ndarray]:
        """(6x6 geometric Jacobian [v; w] in base, 4x4 tool0 pose) from one FK pass.

        Standard DH: joint i turns about z of frame i-1 located at p_(i-1).
        """
        t = np.eye(4)
        z, p = [], []
        for i in range(6):
            z.append(t[:3, 2].copy())
            p.append(t[:3, 3].copy())
            t = t @ _dh(q[i], self.d[i], self.a[i], _ALPHA[i])
        pe = t[:3, 3]
        j = np.zeros((6, 6))
        for i in range(6):
            j[:3, i] = np.cross(z[i], pe - p[i])
            j[3:, i] = z[i]
        return j, t

    def ik_prioritized(self, target: np.ndarray, q_seed, joints=(0, 1, 2, 3, 4, 5),
                       iterations: int = 30, tol: float = 1e-5, damping: float = 0.01,
                       max_step: float = 0.2):
        """IK moving only `joints`: tool0 POSITION exactly (primary task), ORIENTATION as
        close to target as the remaining freedom allows (secondary, projected into the exact
        null space of the position task, so it can never disturb the position).
        Joints not listed stay at their seed values.

        6 joints -> same as full-pose IK; 3 joints -> position only; e.g. 5 joints with
        wrist 2 locked -> wrist 1 / wrist 3 keep the orientation as well as they can.
        Called every cycle from the previous solution, so the orientation keeps improving
        across cycles; `iterations` only bounds the time per call.
        Returns (q, ok) where ok means the position was reached (< 1 mm).
        """
        idx = list(joints)
        n = len(idx)
        q = np.asarray(q_seed, dtype=float).copy()
        lam2 = damping**2
        for _ in range(iterations):
            j6, t = self.jacobian_analytic(q)
            e_p = target[:3, 3] - t[:3, 3]
            e_o = _rot_error(t[:3, :3], target[:3, :3])
            j = j6[:, idx]
            jp, jo = j[:3], j[3:]
            dq = jp.T @ np.linalg.solve(jp @ jp.T + lam2 * np.eye(3), e_p)
            null = np.eye(n) - np.linalg.pinv(jp) @ jp      # exact projector
            jo_n = jo @ null
            dq = dq + null @ (jo_n.T @ np.linalg.solve(jo_n @ jo_n.T + lam2 * np.eye(3), e_o - jo @ dq))
            m = float(np.max(np.abs(dq)))
            if m > max_step:
                dq *= max_step / m
            q[idx] += dq
            if np.linalg.norm(e_p) < tol and m < 1e-5:
                break
        pos_err = float(np.linalg.norm(target[:3, 3] - self.fk(q)[:3, 3]))
        return q, pos_err < 1e-3
