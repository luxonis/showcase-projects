"""Durable append-only recording files shared by the recorder and host tools."""

import fcntl
import json
import math
import os
import re
import shutil
import threading
import zlib
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path


def sync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def make_directory(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        sync_directory(path.parent)
    sync_directory(path)
    return path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def write_all(descriptor, data):
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("Recording write made no progress")
        view = view[written:]


class DurableFile:
    def __init__(self, path):
        self.path = Path(path)
        self.fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
        sync_directory(self.path.parent)

    def append(self, data):
        offset = os.lseek(self.fd, 0, os.SEEK_END)
        write_all(self.fd, data)
        os.fdatasync(self.fd)
        return offset

    def json(self, value):
        return self.append(json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n")

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def read_jsonl(path):
    """Read complete journal records; tolerate only an unfinished final line."""
    with Path(path).open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                break
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"Corrupt complete journal record {path}:{number}") from error


def relative_height(pressure_pa, reference_pa):
    """ISA barometric estimate in metres, zero at reference pressure."""
    if not (math.isfinite(pressure_pa) and math.isfinite(reference_pa)):
        raise ValueError("Pressure must be finite")
    if pressure_pa <= 0 or reference_pa <= 0:
        raise ValueError("Pressure must be positive")
    return 44330.0 * (1.0 - (pressure_pa / reference_pa) ** 0.19029495718363465)


def annexb_units(data):
    starts = list(re.finditer(b"\x00\x00\x00\x01|\x00\x00\x01", data))
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(data)
        if match.end() < end:
            yield data[match.end()] & 0x1F, data[match.start():end]


class CameraJournal:
    """Commit encoded bytes before their timestamp/length/checksum record."""
    def __init__(self, directory, segment_seconds=10, codec="h264"):
        self.directory = make_directory(directory)
        self.index = DurableFile(self.directory / "frames.jsonl")
        self.segment_us = int(segment_seconds * 1_000_000)
        self.video = None
        self.segment = -1
        self.segment_start = 0
        self.headers = {}
        self.previous_sequence = None
        self.codec = codec

    def append(self, data, time_us, sequence, width, height):
        units = list(annexb_units(data)) if self.codec == "h264" else []
        if self.codec == "h264" and not units:
            raise ValueError("Encoder output is not an Annex-B H.264 stream")
        if self.codec == "mjpeg" and not (data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")):
            raise ValueError("Encoder output is not a complete JPEG frame")
        keyframe = self.codec == "mjpeg" or any(kind == 5 for kind, _ in units)
        for kind, unit in units:
            if kind in (7, 8):
                self.headers[kind] = unit
        if self.video is None and not keyframe:
            return False
        if keyframe and (self.video is None or time_us - self.segment_start >= self.segment_us):
            if self.video is not None:
                self.video.close()
            self.segment += 1
            self.segment_start = time_us
            self.video = DurableFile(self.directory / f"{self.segment:06d}.{self.codec}")
            present = {kind for kind, _ in units}
            data = b"".join(self.headers[kind] for kind in (7, 8) if kind not in present and kind in self.headers) + data
        offset = self.video.append(data)
        gap = 0 if self.previous_sequence is None else max(0, sequence - self.previous_sequence - 1)
        self.index.json({
            "segment": f"{self.segment:06d}.{self.codec}", "offset": offset,
            "length": len(data), "crc32": zlib.crc32(data), "time_us": time_us,
            "sequence": sequence, "keyframe": keyframe, "width": width,
            "height": height, "missing_frames": gap,
        })
        self.previous_sequence = sequence
        return True

    def close(self):
        if self.video is not None:
            self.video.close()
        self.index.close()


def verified_frames(directory):
    """Only emit frames whose complete, committed payload passes its checksum."""
    directory = Path(directory)
    for record in read_jsonl(directory / "frames.jsonl"):
        segment = record["segment"]
        if not re.fullmatch(r"\d{6}\.(h264|mjpeg)", segment):
            raise ValueError("Invalid camera segment name")
        with (directory / segment).open("rb") as stream:
            stream.seek(record["offset"])
            payload = stream.read(record["length"])
        if len(payload) != record["length"] or zlib.crc32(payload) != record["crc32"]:
            raise ValueError(f"Committed frame is corrupt: {directory / segment} at {record['offset']}")
        yield record, payload


@contextmanager
def session_lock(root):
    root = make_directory(root)
    with (root / ".sessions.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        sync_directory(root)
        yield root


def _session_catalog(root):
    """Read/migrate manifests under session_lock; wall time never orders them."""
    counter = root / ".session-counter.json"
    persisted_number = json.loads(counter.read_text())["last_number"] if counter.exists() else None
    last_number = persisted_number if persisted_number is not None else 0
    manifests = [(path, json.loads(path.read_text())) for path in sorted(root.glob("*/manifest.json")) if not path.parent.is_symlink()]
    for directory in root.iterdir():
        match = re.fullmatch(r"session-(\d+)", directory.name)
        if directory.is_dir() and match:
            last_number = max(last_number, int(match[1]))
    last_number = max([last_number] + [manifest.get("session_number", 0) for _, manifest in manifests])
    for path, manifest in manifests:
        if "session_number" not in manifest:
            match = re.fullmatch(r"session-(\d+)", path.parent.name)
            if match:
                number = int(match[1])
            else:
                # One-time migration preserves the existing legacy list order.
                last_number += 1
                number = last_number
            if persisted_number != last_number:
                atomic_json(counter, {"last_number": last_number})
                persisted_number = last_number
            manifest["session_number"] = number
            atomic_json(path, manifest)
    if persisted_number != last_number:
        atomic_json(counter, {"last_number": last_number})
    return last_number, [manifest for _, manifest in manifests]


def get_sessions(root):
    with session_lock(root) as root:
        _, manifests = _session_catalog(root)
    return sorted(manifests, key=lambda manifest: manifest["session_number"], reverse=True)


def delete_sessions(root):
    """Remove recording folders after capture stops, retaining permanent numbering."""
    with session_lock(root) as root:
        # Persist the high-water mark before removing even unfinished sessions.
        _session_catalog(root)
        deleted = 0
        for directory in sorted(root.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                continue
            if (directory / "manifest.json").is_file() or re.fullmatch(r"session-\d+|\d{8}T\d{6}Z-[0-9a-f]{8}", directory.name):
                shutil.rmtree(directory)
                sync_directory(root)
                deleted += 1
    return deleted


def reserve_session(root):
    with session_lock(root) as root:
        last_number, _ = _session_catalog(root)
        number = last_number + 1
        # Commit the counter first. A power cut can skip a number, never reuse it.
        atomic_json(root / ".session-counter.json", {"last_number": number})
        path = root / f"session-{number:06d}"
        path.mkdir()
        sync_directory(root)
    return number, path


class Session:
    def __init__(self, root, origin_us, settings):
        now = datetime.now(timezone.utc)
        self.number, self.path = reserve_session(root)
        self.id = self.path.name
        make_directory(self.path / "cameras")
        self.lock = threading.RLock()
        self.manifest = {
            "schema_version": 1, "id": self.id, "session_number": self.number,
            "started_utc": now.isoformat(),
            "origin_us": origin_us, "status": "recording", "settings": settings,
            "cameras": [], "height_model": "ISA relative to start pressure",
            "reference_pressure_pa": None,
        }
        self.update()
        self.telemetry = DurableFile(self.path / "telemetry.jsonl")
        self.events = DurableFile(self.path / "events.jsonl")

    def update(self, **values):
        with self.lock:
            self.manifest.update(values)
            atomic_json(self.path / "manifest.json", self.manifest)

    def event(self, value):
        with self.lock:
            self.events.json(value)

    def finish(self, status, **values):
        self.update(status=status, ended_utc=datetime.now(timezone.utc).isoformat(), **values)
        self.telemetry.close()
        self.events.close()


def recover_sessions(root):
    for path in Path(root).glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("status") in ("recording", "sensor_only"):
            manifest["status"] = "interrupted"
            manifest["recovered_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(path, manifest)
