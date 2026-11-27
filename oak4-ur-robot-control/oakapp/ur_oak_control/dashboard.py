"""Minimal client for the UR Dashboard Server (TCP 29999).

Works the same against URSim and a real UR controller. Command reference:
https://www.universal-robots.com/articles/ur/dashboard-server-e-series-port-29999/
"""

from __future__ import annotations

import socket
import time


class DashboardError(RuntimeError):
    pass


class Dashboard:
    def __init__(self, host: str, port: int = 29999, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = b""

    def connect(self) -> str:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        return self._readline()  # welcome banner

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.sendall(b"quit\n")
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "Dashboard":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _readline(self) -> str:
        assert self._sock is not None
        while b"\n" not in self._buf:
            chunk = self._sock.recv(1024)
            if not chunk:
                raise DashboardError("dashboard connection closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def cmd(self, command: str) -> str:
        if self._sock is None:
            raise DashboardError("not connected")
        self._sock.sendall(command.encode() + b"\n")
        return self._readline()

    # Convenience helpers -------------------------------------------------
    def robot_mode(self) -> str:
        # "Robotmode: POWER_OFF" -> "POWER_OFF"
        return self.cmd("robotmode").split(":", 1)[-1].strip()

    def safety_status(self) -> str:
        return self.cmd("safetystatus").split(":", 1)[-1].strip()

    def wait_for_mode(self, wanted: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        mode = ""
        while time.monotonic() < deadline:
            mode = self.robot_mode()
            if mode == wanted:
                return
            time.sleep(0.5)
        raise DashboardError(f"robot did not reach {wanted} within {timeout:.0f}s (last mode: {mode})")
