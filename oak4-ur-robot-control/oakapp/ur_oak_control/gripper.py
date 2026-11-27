"""Gripper backends for hand follow: close the gripper on a fist, open it on an open hand.

GRIPPER (environment) selects the backend:
  log       only log the command (default; also forced by DRY_RUN)
  onrobot   OnRobot RG2/RG6 through an OnRobot Compute Box: Modbus TCP, port 502, unit 65
            (target force reg 0 [1/10 N], target width reg 1 [1/10 mm], control reg 2 = 1 grip)
  tool_io   gripper switched by UR digital outputs through the driver's
            /io_and_status_controller/set_io service (tool outputs are pins 16 and 17)

Settings come from the `gripper:` section of the hand-follow YAML.
Commands run in a worker thread so a slow / unreachable gripper never stalls the control loop.
"""

from __future__ import annotations

import queue
import socket
import struct
import threading


class Gripper:
    """Log-only backend; base class for the real ones."""

    name = "log"

    def __init__(self, log, cfg: dict):
        self.log = log
        self.cfg = cfg
        self.state: bool | None = None   # True closed, False open, None unknown
        self.error: str | None = None
        self._q: queue.Queue = queue.Queue(maxsize=1)
        threading.Thread(target=self._worker, daemon=True, name="gripper").start()

    def set(self, closed: bool, force: bool = False) -> None:
        if closed == self.state and not force:
            return
        self.state = closed
        try:
            self._q.get_nowait()          # drop a command that has not run yet
        except queue.Empty:
            pass
        self._q.put_nowait(closed)

    def _worker(self) -> None:
        while True:
            closed = self._q.get()
            try:
                self._apply(closed)
                self.error = None
                self.log.info(f"gripper ({self.name}): {'CLOSE' if closed else 'OPEN'}")
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.log.error(f"gripper ({self.name}) {'close' if closed else 'open'} failed: {self.error}")

    def _apply(self, closed: bool) -> None:
        pass

    @property
    def status(self) -> str:
        s = "?" if self.state is None else ("closed" if self.state else "open")
        return f"{s} ({self.name}){' ERROR' if self.error else ''}"


class OnRobotModbus(Gripper):
    """OnRobot RG2 / RG6 behind an OnRobot Compute Box (Modbus TCP)."""

    name = "onrobot"

    def __init__(self, log, cfg: dict):
        self.host = str(cfg.get("onrobot_ip", "")).strip()
        if not self.host:
            raise ValueError("GRIPPER=onrobot needs gripper.onrobot_ip (or ONROBOT_IP)")
        self.port = int(cfg.get("onrobot_port", 502))
        self.unit = int(cfg.get("onrobot_unit", 65))
        self._tid = 0
        super().__init__(log, cfg)

    def _write_registers(self, start: int, values: list[int]) -> None:
        self._tid = (self._tid + 1) & 0xFFFF
        pdu = struct.pack(">BHHB", 16, start, len(values), 2 * len(values))
        pdu += b"".join(struct.pack(">H", v & 0xFFFF) for v in values)
        mbap = struct.pack(">HHHB", self._tid, 0, len(pdu) + 1, self.unit)
        with socket.create_connection((self.host, self.port), timeout=1.5) as s:
            s.settimeout(1.5)
            s.sendall(mbap + pdu)
            resp = s.recv(260)
        if len(resp) < 9:
            raise IOError(f"short Modbus reply {resp!r}")
        if resp[7] & 0x80:
            raise IOError(f"Modbus exception code {resp[8]}")

    def _apply(self, closed: bool) -> None:
        width_mm = float(self.cfg.get("close_mm" if closed else "open_mm", 0 if closed else 100))
        force_n = float(self.cfg.get("force_n", 20))
        self._write_registers(0, [int(force_n * 10), int(width_mm * 10), 1])


class ToolIO(Gripper):
    """Gripper switched by digital outputs through the UR driver."""

    name = "tool_io"

    def __init__(self, log, cfg: dict, node):
        from ur_msgs.srv import SetIO
        self.SetIO = SetIO
        io = cfg.get("tool_io", {}) or {}
        self.pins = {True: {int(k): float(v) for k, v in (io.get("close") or {16: 1, 17: 0}).items()},
                     False: {int(k): float(v) for k, v in (io.get("open") or {16: 0, 17: 1}).items()}}
        self.client = node.create_client(SetIO, "/io_and_status_controller/set_io")
        super().__init__(log, cfg)

    def _apply(self, closed: bool) -> None:
        if not self.client.wait_for_service(timeout_sec=2.0):
            raise IOError("/io_and_status_controller/set_io not available")
        for pin, state in self.pins[closed].items():
            req = self.SetIO.Request(fun=self.SetIO.Request.FUN_SET_DIGITAL_OUT, pin=pin, state=state)
            self.client.call_async(req)   # the main loop spins the node


def make_gripper(kind: str, cfg: dict, log, node) -> Gripper:
    kind = (kind or "log").strip().lower()
    if kind == "onrobot":
        return OnRobotModbus(log, cfg)
    if kind == "tool_io":
        return ToolIO(log, cfg, node)
    if kind not in ("log", "none", ""):
        log.warning(f"unknown GRIPPER={kind!r} - using log only")
    return Gripper(log, cfg)
