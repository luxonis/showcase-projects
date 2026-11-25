"""Standalone OAK4 flight recorder, persistent journals, and LAN transfer API."""

import errno
import hashlib
import json
import os
import shutil
import signal
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import depthai as dai
import qwiic_bmp581
import qwiic_i2c

if __package__:
    from .recorder_storage import CameraJournal, Session, delete_sessions, get_sessions, make_directory, recover_sessions, relative_height
else:
    from recorder_storage import CameraJournal, Session, delete_sessions, get_sessions, make_directory, recover_sessions, relative_height


class Recorder:
    def __init__(self, root):
        self.root = make_directory(root)
        recover_sessions(self.root)
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.status = {"state": "starting", "flight_id": None, "error": None}
        self.session = None
        self.fps = float(os.environ.get("RECORDER_FPS", "30"))
        self.sensor_hz = float(os.environ.get("RECORDER_SENSOR_HZ", "10"))
        self.reserve = int(os.environ.get("RECORDER_RESERVE_MB", "1024")) * 1024**2
        self.segment_seconds = float(os.environ.get("RECORDER_SEGMENT_SECONDS", "10"))
        if self.fps <= 0 or self.sensor_hz <= 0 or self.segment_seconds <= 0:
            raise ValueError("FPS, sensor rate, and segment duration must be positive")

    def time_us(self):
        return int(dai.Clock.now().total_seconds() * 1_000_000)

    def fail(self, error):
        self.status.update(state="error", error=str(error))
        self.stop.set()
        print("RECORDER ERROR:", error, flush=True)

    def guard_worker(self, function, *arguments):
        try:
            function(*arguments)
        except Exception as error:
            self.fail(error)

    def record_sensor(self):
        reference_samples = []
        reference = None
        sensor = None
        bus = int(os.environ.get("RECORDER_I2C_BUS", "4"))
        address = int(os.environ.get("RECORDER_I2C_ADDRESS", "0x47"), 0)
        deadline = time.monotonic()
        while not self.stop.is_set():
            try:
                if sensor is None:
                    transport = qwiic_i2c.get_i2c_driver(iBus=bus)
                    if transport is None:
                        raise RuntimeError(f"Unable to open I2C bus {bus}")
                    sensor = qwiic_bmp581.QwiicBMP581(address=address, i2c_driver=transport)
                    # Verify chip ID before the library's soft reset.
                    if transport.read_byte(address, sensor.kRegChipId) != sensor.kChipId or not sensor.begin():
                        raise RuntimeError("BMP581 is missing or failed initialization")
                    print(f"BMP581 initialized on I2C {bus}, 0x{address:02x}", flush=True)
                    if self.stop.wait(0.1):
                        return
                data = sensor.get_sensor_data()
                temperature = data.temperature - 256 if data.temperature >= 128 else data.temperature
                if not (-40 <= temperature <= 85 and 30000 <= data.pressure <= 125000):
                    raise ValueError("BMP581 returned an out-of-range measurement")
                if reference is None:
                    reference_samples.append(data.pressure)
                    if len(reference_samples) >= max(1, round(self.sensor_hz)):
                        reference = sum(reference_samples) / len(reference_samples)
                        self.session.update(reference_pressure_pa=reference)
                row = {
                    "time_us": self.time_us() - self.session.manifest["origin_us"],
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "temperature_c": temperature, "pressure_pa": data.pressure,
                    "height_m": relative_height(data.pressure, reference) if reference else None,
                    "reference_pressure_pa": reference,
                }
                self.session.telemetry.json(row)
                self.status["sensor"] = row
                self.status.pop("sensor_error", None)
            except OSError as error:
                if error.errno in (errno.ENOSPC, errno.EIO, errno.EROFS):
                    self.fail(error)
                    return
                sensor = None
                self.status["sensor_error"] = str(error)
                self.session.event({"type": "sensor_error", "error": str(error), "time_us": self.time_us()})
                self.stop.wait(1)
            except Exception as error:
                sensor = None
                self.status["sensor_error"] = str(error)
                self.session.event({"type": "sensor_error", "error": str(error), "time_us": self.time_us()})
                self.stop.wait(1)
            deadline = max(deadline + 1 / self.sensor_hz, time.monotonic())
            self.stop.wait(max(0, deadline - time.monotonic()))

    def record_camera(self, queue, info):
        journal = CameraJournal(self.session.path / "cameras" / info["name"], self.segment_seconds, codec=info["codec"])
        frames = 0
        last_frame = time.monotonic()
        try:
            while not self.stop.is_set():
                message = queue.tryGet()
                if message is None:
                    if time.monotonic() - last_frame > 15:
                        raise RuntimeError(f"{info['name']} stopped producing frames")
                    self.stop.wait(0.002)
                    continue
                last_frame = time.monotonic()
                width, height = message.getWidth(), message.getHeight()
                if (width, height) != (info["width"], info["height"]):
                    raise RuntimeError(f"{info['name']} returned {width}x{height}, expected native {info['width']}x{info['height']}")
                stamp = int(message.getTimestamp().total_seconds() * 1_000_000)
                if journal.append(
                    bytes(message.getData()), stamp - self.session.manifest["origin_us"],
                    message.getSequenceNum(), width, height,
                ):
                    frames += 1
                    self.status.setdefault("frames", {})[info["name"]] = frames
        except Exception as error:
            self.fail(error)
        finally:
            journal.close()

    def run(self):
        threads = []
        try:
            if shutil.disk_usage(self.root).free < self.reserve:
                raise RuntimeError("Storage reserve reached; transfer existing flights before recording")
            with dai.Device(dai.DeviceInfo("127.0.0.1")) as device:
                settings = {"fps": self.fps, "sensor_hz": self.sensor_hz, "segment_seconds": self.segment_seconds, "codec": "h264", "rotation_degrees": 180}
                self.session = Session(self.root, self.time_us(), settings)
                self.status.update(flight_id=self.session.id, session_number=self.session.number, device_id=device.getDeviceId())
                allowed = {getattr(dai.CameraBoardSocket, f"CAM_{letter}") for letter in "ABCDE"}
                features = [feature for feature in device.getConnectedCameraFeatures() if feature.socket in allowed]
                with dai.Pipeline(device) as pipeline:
                    cameras = []
                    for feature in sorted(features, key=lambda item: int(item.socket)):
                        resolutions = [(config.width, config.height) for config in feature.configs]
                        resolutions.append((feature.width, feature.height))
                        width, height = max(resolutions, key=lambda size: size[0] * size[1])
                        matching = [config for config in feature.configs if (config.width, config.height) == (width, height)]
                        if matching and max(config.maxFps for config in matching) < self.fps:
                            raise RuntimeError(f"{feature.socket.name} native {width}x{height} cannot provide {self.fps} FPS")
                        codec = "h264" if width <= 4096 and width % 32 == 0 else "mjpeg"
                        camera = pipeline.create(dai.node.Camera).build(feature.socket, sensorResolution=(width, height), sensorFps=self.fps)
                        # RVC4's encoder needs NV12 even for mono sensors.
                        output = camera.requestOutput((width, height), type=dai.ImgFrame.Type.NV12, fps=self.fps, enableUndistortion=False)
                        rotation = pipeline.create(dai.node.ImageManip)
                        rotation.setBackend(dai.node.ImageManip.Backend.GPU)
                        rotation.initialConfig.addRotateDeg(180)
                        rotation.initialConfig.setOutputSize(width, height)
                        rotation.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
                        rotation.setMaxOutputFrameSize(width * height * 3)
                        output.link(rotation.inputImage)
                        encoder = pipeline.create(dai.node.VideoEncoder)
                        profile = dai.VideoEncoderProperties.Profile.H264_MAIN if codec == "h264" else dai.VideoEncoderProperties.Profile.MJPEG
                        encoder.setDefaultProfilePreset(self.fps, profile)
                        if codec == "h264":
                            encoder.setKeyframeFrequency(max(1, round(self.fps)))
                            encoder.setNumBFrames(0)
                            encoder.setBitrate(int(os.environ.get("RECORDER_BITRATE", "0")))
                        else:
                            encoder.setQuality(90)
                        rotation.out.link(encoder.input)
                        queue = encoder.out.createOutputQueue(maxSize=32, blocking=True)
                        info = {"name": feature.socket.name, "sensor": feature.sensorName, "width": width, "height": height, "fps": self.fps, "codec": codec, "rotation_degrees": 180}
                        cameras.append((queue, info))
                    self.session.update(cameras=[info for _, info in cameras], status="recording" if cameras else "sensor_only")
                    self.status.update(state="recording" if cameras else "sensor_only", cameras=[info for _, info in cameras])
                    if cameras:
                        pipeline.start()
                    else:
                        self.status["camera_error"] = "No cameras detected on CAM_A through CAM_E"
                        print(self.status["camera_error"], flush=True)
                    threads.append(threading.Thread(target=self.guard_worker, args=(self.record_sensor,), name="sensor"))
                    threads.extend(threading.Thread(target=self.guard_worker, args=(self.record_camera, queue, info), name=info["name"]) for queue, info in cameras)
                    for thread in threads:
                        thread.start()
                    print(f"Flight {self.session.id}: {len(cameras)} camera(s); API port {os.environ.get('RECORDER_API_PORT', '8088')}", flush=True)
                    while not self.stop.wait(1):
                        self.status["free_bytes"] = shutil.disk_usage(self.root).free
                        if self.status["free_bytes"] < self.reserve:
                            self.status["state"] = "storage_full"
                            self.stop.set()
                        elif cameras and not pipeline.isRunning():
                            raise RuntimeError("Camera pipeline stopped")
                    for thread in threads:
                        thread.join(timeout=15)
                    if any(thread.is_alive() for thread in threads):
                        raise RuntimeError("A recording worker did not stop; preserving journals for recovery")
            final_state = self.status["state"]
            self.session.finish("complete" if final_state in ("recording", "sensor_only") else final_state)
            if final_state in ("recording", "sensor_only"):
                self.status["state"] = "stopped"
        except Exception as error:
            self.fail(error)
            for thread in threads:
                thread.join(timeout=15)
            if self.session and not any(thread.is_alive() for thread in threads):
                self.session.finish("error", error=str(error))
        finally:
            # A terminal status alone does not prove that all writers are closed.
            if not any(thread.is_alive() for thread in threads):
                self.finished.set()

    def wipe_sessions(self, timeout=45):
        self.stop.set()
        if not self.finished.wait(timeout):
            raise TimeoutError("Recording has not fully stopped; no sessions were deleted")
        deleted = delete_sessions(self.root)
        self.session = None
        self.status = {
            "state": "stopped", "flight_id": None, "session_number": None,
            "error": None, "free_bytes": shutil.disk_usage(self.root).free,
        }
        return {"deleted_sessions": deleted, "state": "stopped"}


def make_handler(recorder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if urlsplit(self.path).path == "/api/stop":
                recorder.stop.set()
                self.reply({"state": "stopping", "flight_id": recorder.status["flight_id"]})
            else:
                self.reply({"error": "Not found"}, 404)

        def do_DELETE(self):
            if urlsplit(self.path).path != "/api/sessions":
                self.reply({"error": "Not found"}, 404)
                return
            try:
                self.reply(recorder.wipe_sessions())
            except TimeoutError as error:
                self.reply({"error": str(error)}, 409)
            except (OSError, ValueError) as error:
                self.reply({"error": f"Could not wipe sessions: {error}"}, 500)

        def do_GET(self):
            parts = unquote(urlsplit(self.path).path).strip("/").split("/")
            try:
                if parts == ["api", "status"]:
                    self.reply(recorder.status)
                    return
                if parts in (["api", "sessions"], ["api", "flights"]):
                    self.reply(get_sessions(recorder.root))
                    return
                if len(parts) >= 4 and parts[:2] == ["api", "flights"]:
                    flight = (recorder.root / parts[2]).resolve()
                    if flight.parent != recorder.root.resolve() or not flight.is_dir():
                        raise FileNotFoundError()
                    manifest = json.loads((flight / "manifest.json").read_text())
                    if manifest["status"] in ("recording", "sensor_only"):
                        self.reply({"error": "Stop this recording before exporting a consistent snapshot"}, 409)
                        return
                    if parts[3:] == ["files"]:
                        files = []
                        for path in sorted(flight.rglob("*")):
                            if path.is_file() and path.suffix != ".tmp":
                                digest = hashlib.sha256()
                                with path.open("rb") as stream:
                                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                        digest.update(chunk)
                                files.append({"path": path.relative_to(flight).as_posix(), "size": path.stat().st_size, "sha256": digest.hexdigest()})
                        self.reply({"flight_id": flight.name, "files": files})
                        return
                    if parts[3] == "file" and len(parts) >= 5:
                        path = flight.joinpath(*parts[4:]).resolve()
                        if not path.is_relative_to(flight) or not path.is_file():
                            raise FileNotFoundError()
                        size = path.stat().st_size
                        start = 0
                        range_header = self.headers.get("Range", "")
                        if range_header:
                            import re
                            match = re.fullmatch(r"bytes=(\d+)-", range_header)
                            if not match or int(match[1]) >= size:
                                self.reply({"error": "Invalid range"}, 416)
                                return
                            start = int(match[1])
                        self.send_response(206 if range_header else 200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(size - start))
                        self.send_header("Accept-Ranges", "bytes")
                        if range_header:
                            self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
                        self.end_headers()
                        with path.open("rb") as stream:
                            stream.seek(start)
                            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
                        return
                self.reply({"error": "Not found"}, 404)
            except (FileNotFoundError, ValueError):
                self.reply({"error": "Not found"}, 404)

    return Handler


def main():
    storage = os.environ.get("RECORDER_ROOT") or os.environ.get("OAKAGENT_STORAGE_SHARED")
    if not storage:
        raise RuntimeError("Persistent OAKAGENT_STORAGE_SHARED or RECORDER_ROOT is required")
    recorder = Recorder(Path(storage) / "flights")
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("RECORDER_API_PORT", "8088"))), make_handler(recorder))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    worker = threading.Thread(target=recorder.run, name="recorder")
    worker.start()
    exit_requested = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: (recorder.stop.set(), exit_requested.set()))
    while not exit_requested.wait(1):
        # Keep transfer available after recording stops or storage fills.
        pass
    worker.join(timeout=25)
    server.shutdown()
    server.server_close()
    if worker.is_alive():
        os._exit(1)


if __name__ == "__main__":
    main()
