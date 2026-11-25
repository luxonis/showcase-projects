#!/usr/bin/env python3
"""Manage, resume-download, convert, and play grouped OAK4 flight recordings."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from datetime import datetime, timezone
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from oakapp.recorder_storage import atomic_json, read_jsonl, sync_directory, verified_frames

# Recorder traffic stays on the local link even when the host has an HTTP proxy.
LOCAL_HTTP = build_opener(ProxyHandler({}))


def api_base(device):
    address = device.removeprefix("http://").rstrip("/")
    if address.count(":") > 1 and not address.startswith("["):
        address = f"[{address}]"
    parsed = urlsplit("http://" + address)
    if not parsed.hostname or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("Use --device IP[:API_PORT], for example --device 169.254.239.12")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{parsed.port or 8088}"


def resolve_device(device=None, *, for_start=False):
    """Find the recorder using local oakctl discovery, including link-local IPs."""
    if device and device != "auto":
        return api_base(device)
    oakctl = str(Path(__file__).with_name("oakctl"))
    discovery_error = None
    try:
        result = subprocess.run(
            [oakctl, "device", "list", "--format", "json"],
            check=True, capture_output=True, text=True, timeout=20,
        )
        devices = json.loads(result.stdout)["items"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        devices = []
        discovery_error = f" Local discovery failed ({error}); try --device IP."

    candidates = []
    matches = {}
    for item in devices:
        addresses = dict.fromkeys((item.get("ip_addresses") or []) + (item.get("ncm_ips") or []))
        bases = [api_base(address) for address in addresses]
        if not bases:
            continue
        candidates.append(bases)
        for base in bases:
            try:
                status = request_json(base, "/api/status", timeout=3)
                if "state" not in status:
                    continue
            except (OSError, ValueError):
                continue
            identity = status.get("device_id") or item.get("serial_number") or base
            matches.setdefault(identity, base)
            break
    if len(matches) > 1:
        choices = ", ".join(matches.values())
        raise RuntimeError(f"Multiple recorders found: {choices}. Select one with --device IP.")
    if matches:
        base = next(iter(matches.values()))
    elif for_start and len(candidates) == 1:
        # A stopped app has no API yet; oakctl can start it on the discovered OAK.
        base = candidates[0][0]
    else:
        # Retain access on the original LAN if multicast discovery is unavailable.
        base = api_base("10.12.222.29")
        try:
            request_json(base, "/api/status", timeout=2)
        except (OSError, ValueError) as error:
            detail = discovery_error or " Run './oakctl device list --format json' to find its address."
            raise RuntimeError(
                "No reachable recorder found on the local network."
                " For a direct offline connection, set the host Ethernet IPv4 method to Link-Local Only"
                " (169.254.x.x/16), or use --device IP if you know the camera address."
                + detail
            ) from error
    print(f"Using recorder at {base}", file=sys.stderr, flush=True)
    return base


def request_json(base, path, method="GET", timeout=600):
    request = Request(base + path, method=method)
    with LOCAL_HTTP.open(request, timeout=timeout) as response:
        return json.load(response)


def stop_recording(base):
    request_json(base, "/api/stop", method="POST")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        flights = request_json(base, "/api/sessions")
        if all(flight["status"] not in ("recording", "sensor_only") for flight in flights):
            return
        time.sleep(0.25)
    raise TimeoutError("The recording did not finish in 45 seconds")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(base, flight_id, info, directory):
    relative = Path(info["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe file path in transfer manifest")
    destination = (directory / relative).resolve()
    if not destination.is_relative_to(directory.resolve()):
        raise ValueError("Transfer path escapes flight folder")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == info["size"] and digest_file(destination) == info["sha256"]:
        print("Already verified:", relative)
        return
    partial = destination.with_name(destination.name + ".download")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > info["size"]:
        partial.unlink()
        offset = 0
    if offset < info["size"]:
        headers = {}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        url = base + f"/api/flights/{quote(flight_id)}/file/{quote(info['path'], safe='/')}"
        with LOCAL_HTTP.open(Request(url, headers=headers), timeout=120) as response:
            if offset and response.status != 206:
                raise RuntimeError("Device did not accept a resumable transfer")
            with partial.open("ab" if offset else "wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
    elif not partial.exists():
        partial.touch()
    if partial.stat().st_size != info["size"] or digest_file(partial) != info["sha256"]:
        partial.unlink()
        raise RuntimeError(f"Checksum failed for {relative}; rerun download to retry")
    os.replace(partial, destination)
    sync_directory(destination.parent)
    print(f"Verified: {relative} ({info['size']:,} bytes)", flush=True)


def collect_camera(directory, destination):
    records = []
    warning = None
    with destination.open("wb") as stream:
        try:
            for record, payload in verified_frames(directory):
                stream.write(payload)
                records.append(record)
        except (ValueError, FileNotFoundError) as error:
            # Preserve all original files. Only the verified prefix is played.
            warning = str(error)
    return records, warning


def prepare_h264(raw, records, target, fps):
    import av

    with av.open(str(raw), format="h264") as source, av.open(str(target), "w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream_from_template(source.streams.video[0])
        count = 0
        first_time = records[0]["time_us"]
        for packet in source.demux(video=0):
            if not packet.size:
                continue
            if count >= len(records):
                raise ValueError("More encoded access units than committed camera records")
            record = records[count]
            packet.pts = packet.dts = record["time_us"] - first_time
            if count + 1 < len(records):
                packet.duration = records[count + 1]["time_us"] - record["time_us"]
            else:
                packet.duration = round(1_000_000 / fps)
            packet.time_base = Fraction(1, 1_000_000)
            packet.stream = stream
            output.mux(packet)
            count += 1
        if count != len(records):
            raise ValueError(f"Decoded packet count {count} differs from journal count {len(records)}")


def prepare_mjpeg(raw, records, target, fps, width, height):
    import av

    with av.open(str(raw), format="mjpeg") as source, av.open(str(target), "w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=Fraction(str(fps)))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1_000_000)
        stream.codec_context.time_base = stream.time_base
        stream.options = {"crf": "18", "preset": "fast", "bf": "0"}
        count = 0
        first_time = records[0]["time_us"]
        for frame in source.decode(video=0):
            if count >= len(records):
                raise ValueError("More MJPEG frames than committed camera records")
            if (frame.width, frame.height) != (width, height):
                raise ValueError("MJPEG native dimensions differ from the recording manifest")
            frame.pts = records[count]["time_us"] - first_time
            frame.time_base = stream.time_base
            for packet in stream.encode(frame):
                output.mux(packet)
            count += 1
        for packet in stream.encode(None):
            output.mux(packet)
        if count != len(records):
            raise ValueError("MJPEG frame count differs from the committed journal")


def prepare_flight(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    videos = directory / "videos"
    videos.mkdir(exist_ok=True)
    result = {"manifest": manifest, "cameras": [], "duration_s": 0, "warnings": []}
    for info in manifest["cameras"]:
        name = info["name"]
        if not re.fullmatch(r"CAM_[ABCDE]", name):
            raise ValueError("Invalid camera name")
        extension = "mjpeg" if info["codec"] == "mjpeg" else "h264"
        with tempfile.TemporaryDirectory(prefix="oak-playback-") as folder:
            raw = Path(folder) / (name + "." + extension)
            records, warning = collect_camera(directory / "cameras" / name, raw)
            if warning:
                result["warnings"].append(f"{name}: playing verified prefix only: {warning}")
            if not records:
                result["warnings"].append(f"{name}: no committed frames to play")
                continue
            temporary = videos / (name + ".partial.mp4")
            if info["codec"] == "mjpeg":
                prepare_mjpeg(raw, records, temporary, info["fps"], info["width"], info["height"])
            else:
                prepare_h264(raw, records, temporary, info["fps"])
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, videos / (name + ".mp4"))
            sync_directory(videos)
            offset = records[0]["time_us"] / 1_000_000
            end = (records[-1]["time_us"] + 1_000_000 / info["fps"]) / 1_000_000
            result["cameras"].append({
                **info, "url": "/videos/" + name + ".mp4", "offset_s": offset,
                "end_s": end, "frames": len(records),
                "missing_frames": sum(record.get("missing_frames", 0) for record in records),
            })
            result["duration_s"] = max(result["duration_s"], end)
            print(f"Prepared {name}: {len(records):,} frames, native {info['width']}x{info['height']}", flush=True)
    rows = list(read_jsonl(directory / "telemetry.jsonl"))
    columns = ["time_s", "utc", "temperature_c", "pressure_pa", "height_m", "reference_pressure_pa"]
    with (directory / "telemetry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns} | {"time_s": row["time_us"] / 1_000_000})
    if rows:
        result["duration_s"] = max(result["duration_s"], rows[-1]["time_us"] / 1_000_000)
    atomic_json(directory / "playback.json", result)
    for warning in result["warnings"]:
        print("Recovery warning:", warning, file=sys.stderr)
    print("Prepared flight:", directory, flush=True)
    return result


def playback_handler(directory):
    static = Path(__file__).with_name("playback")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/playback":
                self.send_file(directory / "playback.json", "application/json")
            elif path == "/api/telemetry":
                rows = list(read_jsonl(directory / "telemetry.jsonl"))
                data = json.dumps(rows).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif re.fullmatch(r"/videos/CAM_[ABCDE]\.mp4", path):
                self.send_file(directory / path.lstrip("/"), "video/mp4")
            elif path in ("/", "/index.html", "/app.js", "/style.css"):
                filename = "index.html" if path == "/" else path.lstrip("/")
                mime = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[filename]
                self.send_file(static / filename, mime)
            else:
                self.send_error(404)

        def send_file(self, path, mime):
            if not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            ranged = self.headers.get("Range")
            if ranged:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", ranged)
                if not match:
                    self.send_error(416)
                    return
                start = int(match[1])
                end = min(int(match[2]) if match[2] else end, end)
                if start > end:
                    self.send_error(416)
                    return
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("Accept-Ranges", "bytes")
            if ranged:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def ordered_sessions(sessions):
    return sorted(sessions, key=lambda session: session["session_number"], reverse=True)


def list_sessions(sessions):
    if not sessions:
        print("No recording sessions on the device.")
        return
    print(f"{'Session':8}  {'Device time (UTC)':20}  {'Status':12}  Cameras")
    statuses = {"complete": "Finished", "error": "Failed", "sensor_only": "Sensor only"}
    for session in ordered_sessions(sessions):
        try:
            started = datetime.fromisoformat(session["started_utc"])
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            started = started.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
        except (KeyError, ValueError):
            started = "Unavailable"
        status = statuses.get(session["status"], session["status"].replace("_", " ").capitalize())
        cameras = ", ".join(f"{camera['name']} ({camera['width']}x{camera['height']})" for camera in session["cameras"]) or "No cameras"
        print(f"{session['session_number']:06d}    {started:20}  {status:12}  {cameras}")
    print("Dates use the device clock and may be incorrect offline.")


def select_sessions(sessions, selection):
    if not sessions:
        raise ValueError("No recording sessions on the device")
    sessions = ordered_sessions(sessions)
    if selection == "all":
        return sessions
    if selection == "latest":
        return [sessions[0]]
    if selection.isdecimal() or re.fullmatch(r"session-\d+", selection):
        number = int(selection.removeprefix("session-"))
        for session in sessions:
            if session["session_number"] == number:
                return [session]
        raise ValueError(f"Session number {number} not found; run 'sessions' to see the available recordings")
    for session in sessions:
        if session["id"] == selection:
            return [session]
    raise ValueError("Session not found; run 'sessions' to see the available recordings")


def start_recording(device):
    """Find the installed recorder internally; the user only supplies an IP."""
    oakctl = str(Path(__file__).with_name("oakctl"))
    base = api_base(device)
    device = urlsplit(base).hostname
    result = subprocess.run(
        [oakctl, "app", "list", "-d", device, "--format", "json"],
        check=True, capture_output=True, text=True, timeout=45,
    )
    apps = [app for app in json.loads(result.stdout) if app["identifier"] == "com.ffcdrone.flight-recorder" and app["app_id"] != "00000000-0000-0000-0000-000000000000"]
    if not apps:
        raise RuntimeError("The flight recorder is not installed on this device")
    app = max(apps, key=lambda item: (item["status"] == "running", item["enabled"], item["id"]))
    action = "restart" if app["status"] == "running" else "start"
    subprocess.run([oakctl, "app", action, app["app_id"], "-d", device], check=True, capture_output=True, text=True, timeout=60)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            status = request_json(base, "/api/status", timeout=5)
            if status["state"] in ("recording", "sensor_only"):
                print("Recording session:", status["flight_id"])
                return
            if status["state"] == "error":
                raise RuntimeError(status["error"])
        except (OSError, HTTPError):
            pass
        time.sleep(0.5)
    raise TimeoutError("The app did not begin recording in 45 seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "sessions", "stop", "download", "wipe"):
        item = sub.add_parser(name, aliases=["flights"] if name == "sessions" else [],
                              help="Stop capture and permanently delete all device sessions" if name == "wipe" else None)
        item.add_argument("--device", default="auto", help="Device IP[:API port]; default: discover on the local network")
        if name == "wipe":
            item.add_argument("--yes", action="store_true", help="Confirm deletion without prompting")
        if name == "download":
            item.add_argument("session", nargs="?", help="Session number or ID; omit to choose from a list")
            item.add_argument("--flight", help=argparse.SUPPRESS)
            item.add_argument("--output", type=Path, default=Path("recordings"))
            item.add_argument("--stop", action="store_true", help="Finish the current recording before transfer")
            item.add_argument("--raw-only", action="store_true", help="Skip MP4/CSV playback preparation")
    item = sub.add_parser("prepare")
    item.add_argument("directory", type=Path)
    item = sub.add_parser("play")
    item.add_argument("directory", type=Path)
    item.add_argument("--port", type=int, default=8090)
    item.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.command in ("prepare", "play"):
        directory = args.directory.resolve()
        if args.command == "prepare" or not (directory / "playback.json").exists():
            prepare_flight(directory)
        if args.command == "play":
            server = ThreadingHTTPServer(("127.0.0.1", args.port), playback_handler(directory))
            server.daemon_threads = True
            url = f"http://127.0.0.1:{args.port}"
            print(f"Playback: {url} (Ctrl+C to stop)", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            try:
                server.serve_forever()
            finally:
                server.server_close()
        return
    if args.command == "wipe" and not args.yes:
        target = "the locally discovered recorder" if args.device == "auto" else args.device
        try:
            answer = input(f"Stop recording and permanently delete ALL sessions on {target}? [y/N]: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Wipe cancelled.")
            return
    base = resolve_device(args.device, for_start=args.command == "run")
    address = urlsplit(base).netloc
    if args.command == "run":
        start_recording(base)
        return
    if args.command == "stop":
        stop_recording(base)
        print("Recording stopped; transfer remains available. Use 'run' to begin a new session.")
    elif args.command == "status":
        print(json.dumps(request_json(base, "/api/status"), indent=2))
    elif args.command in ("sessions", "flights"):
        list_sessions(request_json(base, "/api/sessions"))
    elif args.command == "wipe":
        result = request_json(base, "/api/sessions", method="DELETE")
        print(f"Deleted {result['deleted_sessions']} session(s) from {address}. Recording stopped; use 'run' to begin a new session.")
    elif args.command == "download":
        flights = request_json(base, "/api/sessions")
        selection = args.session or args.flight
        if selection is None:
            list_sessions(flights)
            if not flights:
                return
            selection = input("Session number or ID to download: ").strip()
        selected = select_sessions(flights, selection)
        if args.stop:
            if any(flight["status"] in ("recording", "sensor_only") for flight in selected):
                stop_recording(base)
                finished = {flight["id"]: flight for flight in request_json(base, "/api/sessions")}
                selected = [finished[flight["id"]] for flight in selected]
        for flight in selected:
            if flight["status"] in ("recording", "sensor_only"):
                raise RuntimeError(f"Session {flight['id']} is still recording; add --stop to finish it before downloading")
            snapshot = request_json(base, f"/api/flights/{quote(flight['id'])}/files")
            if not re.fullmatch(r"session-\d{6,}|\d{8}T\d{6}Z-[0-9a-f]{8}", snapshot["flight_id"]):
                raise ValueError("Invalid flight ID")
            directory = args.output.resolve() / snapshot["flight_id"]
            directory.mkdir(parents=True, exist_ok=True)
            for info in snapshot["files"]:
                download_file(base, snapshot["flight_id"], info, directory)
            atomic_json(directory / "transfer.json", snapshot)
            print("All files downloaded and SHA-256 verified:", directory)
            if not args.raw_only:
                prepare_flight(directory)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
