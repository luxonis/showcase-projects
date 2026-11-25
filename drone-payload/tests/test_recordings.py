import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from fractions import Fraction
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import av
import numpy as np
import pytest

from flight_host import download_file, list_sessions, playback_handler, prepare_flight, request_json, resolve_device, select_sessions
from oakapp.recorder_storage import CameraJournal, Session, atomic_json, delete_sessions, get_sessions, read_jsonl, recover_sessions, relative_height, verified_frames


def encoded_frames(width, height, count=12, codec="h264"):
    context = av.CodecContext.create("libx264" if codec == "h264" else "mjpeg", "w")
    context.width, context.height = width, height
    context.pix_fmt = "yuv420p" if codec == "h264" else "yuvj420p"
    context.time_base = Fraction(1, 30)
    context.framerate = Fraction(30, 1)
    if codec == "h264":
        context.options = {"preset": "ultrafast", "tune": "zerolatency", "bf": "0", "g": "3", "x264-params": "scenecut=0:repeat-headers=1"}
    packets = []
    image = np.empty((height, width, 3), dtype=np.uint8)
    for number in range(count):
        image[:, :, 0] = (number * 17) % 255
        image[:, :, 1] = np.arange(width, dtype=np.uint8)
        image[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = number
        packets.extend(bytes(packet) for packet in context.encode(frame))
    packets.extend(bytes(packet) for packet in context.encode(None))
    assert len(packets) == count
    return packets


def fixture_flight(root, cameras=5):
    session = Session(root, 0, {"fps": 30, "sensor_hz": 10})
    infos = []
    for number, size in enumerate([(320, 240), (640, 400), (640, 480), (1280, 720), (1280, 800)][:cameras]):
        name = "CAM_" + "ABCDE"[number]
        codec = "mjpeg" if number == 4 else "h264"
        info = {"name": name, "sensor": "SYNTHETIC TEST", "width": size[0], "height": size[1], "fps": 30, "codec": codec}
        journal = CameraJournal(session.path / "cameras" / name, segment_seconds=0.15, codec=codec)
        for index, packet in enumerate(encoded_frames(*size, codec=codec)):
            # A timestamp gap verifies VFR playback rather than invented FPS timing.
            stamp = 200_000 + number * 20_000 + round(index * 1_000_000 / 30) + (50_000 if index >= 6 else 0)
            journal.append(packet, stamp, index + (1 if index >= 6 else 0), *size)
        journal.close()
        infos.append(info)
    session.update(cameras=infos, reference_pressure_pa=100000)
    for index in range(20):
        pressure = 100000 - index * 3
        session.telemetry.json({"time_us": index * 100000, "utc": "2026-10-14T00:00:00Z", "temperature_c": 25 + index * 0.01, "pressure_pa": pressure, "height_m": relative_height(pressure, 100000), "reference_pressure_pa": 100000})
    session.finish("complete")
    return session.path


def test_height_direction_and_invalid_pressure():
    assert relative_height(100000, 100000) == 0
    assert relative_height(99000, 100000) > 80
    assert relative_height(101000, 100000) < -80
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            relative_height(value, 100000)


def test_session_identity_and_selection_survive_clock_reset(tmp_path, monkeypatch, capsys):
    import oakapp.recorder_storage as storage

    class DeviceClock:
        value = datetime(2050, 1, 1, tzinfo=timezone.utc)

        @classmethod
        def now(cls, _):
            return cls.value

    monkeypatch.setattr(storage, "datetime", DeviceClock)
    first = Session(tmp_path, 0, {})
    first.finish("complete")
    DeviceClock.value = datetime(1970, 1, 1, tzinfo=timezone.utc)
    second = Session(tmp_path, 0, {})
    second.finish("complete")
    assert (first.id, second.id) == ("session-000001", "session-000002")
    sessions = get_sessions(tmp_path)
    assert [session["id"] for session in sessions] == [second.id, first.id]
    assert select_sessions(sessions, "latest")[0]["id"] == second.id
    assert select_sessions(sessions, "1")[0]["id"] == first.id
    assert select_sessions(sessions, "session-000002")[0]["id"] == second.id
    list_sessions(list(reversed(sessions)))
    output = capsys.readouterr().out
    assert output.index("000002") < output.index("000001")
    assert "01 Jan 1970 00:00:00" in output
    # Removing recordings must not reset the persistent high-water mark.
    shutil.rmtree(first.path)
    shutil.rmtree(second.path)
    third = Session(tmp_path, 0, {})
    third.finish("complete")
    assert third.id == "session-000003"


def test_session_number_not_reused_after_unfinished_creation(tmp_path, monkeypatch):
    import oakapp.recorder_storage as storage

    def failed_manifest(path, value):
        if Path(path).name == "manifest.json":
            raise OSError("Simulated loss of power before the session manifest")
        atomic_json(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "atomic_json", failed_manifest)
        with pytest.raises(OSError, match="loss of power"):
            Session(tmp_path, 0, {})
    assert not (tmp_path / "session-000001" / "manifest.json").exists()
    resumed = Session(tmp_path, 0, {})
    resumed.finish("complete")
    assert resumed.id == "session-000002"


def test_legacy_recordings_receive_permanent_numbers_without_renaming(tmp_path):
    old_ids = ["20261014T113122Z-41fe61f1", "20261014T114449Z-6b7184a0"]
    for old_id in old_ids:
        directory = tmp_path / old_id
        directory.mkdir()
        (directory / "original.h264").write_bytes(b"original recording")
        atomic_json(directory / "manifest.json", {"id": old_id, "status": "complete", "cameras": []})
    sessions = get_sessions(tmp_path)
    assert [session["session_number"] for session in sessions] == [2, 1]
    assert select_sessions(sessions, "2")[0]["id"] == old_ids[1]
    assert select_sessions(sessions, old_ids[0])[0]["session_number"] == 1
    for old_id in old_ids:
        assert (tmp_path / old_id / "original.h264").read_bytes() == b"original recording"
    assert get_sessions(tmp_path) == sessions
    new = Session(tmp_path, 0, {})
    new.finish("complete")
    assert new.id == "session-000003"


def test_delete_all_sessions_preserves_counter_and_unrelated_data(tmp_path):
    root = tmp_path / "device"
    session = Session(root, 0, {})
    session.finish("complete")
    legacy = root / "20261014T113122Z-41fe61f1"
    legacy.mkdir()
    atomic_json(legacy / "manifest.json", {"id": legacy.name, "status": "complete"})
    # Incomplete creation/deletion can leave folders without manifests.
    for name in ("session-000009", "20261014T114449Z-6b7184a0"):
        directory = root / name
        directory.mkdir()
        (directory / "video.h264").write_bytes(b"unfinished recording")
    notes = root / "notes"
    notes.mkdir()
    (notes / "keep.txt").write_text("keep")
    local_copy = tmp_path / "downloaded"
    local_copy.mkdir()
    atomic_json(local_copy / "manifest.json", {"id": "downloaded"})
    (root / "session-000010").symlink_to(local_copy, target_is_directory=True)

    assert delete_sessions(root) == 4
    assert get_sessions(root) == []
    assert delete_sessions(root) == 0
    assert (notes / "keep.txt").read_text() == "keep"
    assert (local_copy / "manifest.json").exists()
    next_number = json.loads((root / ".session-counter.json").read_text())["last_number"] + 1
    new = Session(root, 0, {})
    new.finish("complete")
    assert new.number == next_number
    assert new.number > 9


@pytest.fixture
def device_api(tmp_path, monkeypatch):
    from oakapp.main import Recorder, make_handler

    monkeypatch.setenv("RECORDER_RESERVE_MB", "0")
    recorder = Recorder(tmp_path / "device")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(recorder))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", recorder
    finally:
        recorder.stop.set()
        server.shutdown()
        server.server_close()


def test_sessions_discover_link_local_after_lan_address_changes(device_api, monkeypatch, capsys):
    import flight_host

    base, recorder = device_api
    session = Session(recorder.root, 0, {})
    session.finish("complete")
    recorder.status["device_id"] = "3511165575"
    discovery = {"items": [{"serial_number": "3511165575", "port": 9998,
                             "ip_addresses": ["10.12.222.29", "169.254.239.12"], "ncm_ips": []}]}

    def discover(command, **kwargs):
        assert command[1:] == ["device", "list", "--format", "json"]
        return SimpleNamespace(stdout=json.dumps(discovery))

    calls = []

    def local_request(address, path, **kwargs):
        calls.append((address, path))
        if address != "http://169.254.239.12:8088":
            raise OSError(101, "Network is unreachable")
        return request_json(base, path, **kwargs)

    monkeypatch.setattr(flight_host.subprocess, "run", discover)
    monkeypatch.setattr(flight_host, "request_json", local_request)
    monkeypatch.setattr(sys, "argv", ["flight_host.py", "sessions"])
    flight_host.main()
    output = capsys.readouterr()
    assert "000001" in output.out and "Finished" in output.out
    assert "Using recorder at http://169.254.239.12:8088" in output.err
    assert calls[-1] == ("http://169.254.239.12:8088", "/api/sessions")


def test_discovery_does_not_choose_between_multiple_recorders(monkeypatch):
    import flight_host

    items = [{"serial_number": str(number), "ip_addresses": [f"169.254.1.{number}"]} for number in (1, 2)]
    monkeypatch.setattr(flight_host.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps({"items": items})))
    monkeypatch.setattr(flight_host, "request_json", lambda base, *a, **k: {"state": "recording", "device_id": base})
    with pytest.raises(RuntimeError, match="Multiple recorders found.*--device"):
        resolve_device()
    # An explicit address works without discovery and keeps a custom API port.
    monkeypatch.setattr(flight_host.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected discovery"))
    assert resolve_device("169.254.239.12:8089") == "http://169.254.239.12:8089"


def test_discovery_explains_missing_link_and_can_find_stopped_app(monkeypatch):
    import flight_host

    def unreachable(*args, **kwargs):
        raise OSError(101, "Network is unreachable")

    discovery = {"items": []}
    monkeypatch.setattr(flight_host.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(discovery)))
    monkeypatch.setattr(flight_host, "request_json", unreachable)
    with pytest.raises(RuntimeError, match="Link-Local Only.*--device"):
        resolve_device()
    discovery["items"].append({"serial_number": "3511165575", "ip_addresses": ["169.254.239.12"]})
    assert resolve_device(for_start=True) == "http://169.254.239.12:8088"


def test_cli_download_uses_direct_local_http_despite_proxy_settings(device_api, tmp_path):
    base, recorder = device_api
    flight = fixture_flight(recorder.root, cameras=1)
    output = tmp_path / "downloaded"
    # A fresh process ensures urllib sees these settings before its first request.
    env = os.environ.copy()
    for name in list(env):
        if name.lower().endswith("_proxy"):
            env.pop(name)
    env.update(http_proxy="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9", no_proxy="")
    result = subprocess.run(
        [sys.executable, "flight_host.py", "download", "1", "--device", base,
         "--output", str(output), "--raw-only"],
        env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "SHA-256 verified" in result.stdout
    for source in flight.rglob("*"):
        if source.is_file():
            assert (output / flight.name / source.relative_to(flight)).read_bytes() == source.read_bytes()


def test_wipe_api_stops_capture_and_waits_for_all_writers(device_api, monkeypatch):
    import oakapp.main as device

    base, recorder = device_api
    previous = Session(recorder.root, 0, {})
    previous.finish("complete")
    fake_device = SimpleNamespace(getDeviceId=lambda: "test", getConnectedCameraFeatures=lambda: [])
    monkeypatch.setattr(device.dai, "Device", lambda *_: nullcontext(fake_device))
    monkeypatch.setattr(device.dai, "Pipeline", lambda *_: nullcontext(None))
    writer_started = threading.Event()
    release_writer = threading.Event()

    def sensor():
        writer_started.set()
        recorder.stop.wait()
        release_writer.wait()
        recorder.session.telemetry.json({"time_us": 1, "pressure_pa": 100000})

    monkeypatch.setattr(recorder, "record_sensor", sensor)
    worker = threading.Thread(target=recorder.run)
    worker.start()
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            assert writer_started.wait(5)
            current = recorder.session.path
            future = executor.submit(request_json, base, "/api/sessions", method="DELETE")
            assert recorder.stop.wait(5)
            assert not recorder.finished.is_set()
            assert not future.done()
            assert previous.path.exists() and current.exists()
            release_writer.set()
            assert future.result(timeout=5) == {"deleted_sessions": 2, "state": "stopped"}
        finally:
            recorder.stop.set()
            release_writer.set()
            worker.join(timeout=5)
    assert not worker.is_alive()
    assert request_json(base, "/api/sessions") == []
    status = request_json(base, "/api/status")
    assert status["state"] == "stopped"
    assert status["flight_id"] is None and status["session_number"] is None
    assert "sensor" not in status and "frames" not in status
    assert request_json(base, "/api/sessions", method="DELETE")["deleted_sessions"] == 0


def test_wipe_api_refuses_to_delete_until_writers_finish(device_api, monkeypatch):
    base, recorder = device_api
    session = Session(recorder.root, 0, {})
    session.finish("error")
    recorder.status.update(state="error", error="Worker is still shutting down")
    wipe = recorder.wipe_sessions
    monkeypatch.setattr(recorder, "wipe_sessions", lambda: wipe(timeout=0))
    with pytest.raises(HTTPError) as caught:
        request_json(base, "/api/sessions", method="DELETE")
    assert caught.value.code == 409
    assert "no sessions were deleted" in caught.value.read().decode()
    assert session.path.exists()


@pytest.mark.parametrize("answer", ["", "no", None])
def test_wipe_cli_cancellation_does_not_contact_device(monkeypatch, capsys, answer):
    import flight_host

    def confirm(_):
        if answer is None:
            raise EOFError
        return answer

    monkeypatch.setattr(sys, "argv", ["flight_host.py", "wipe"])
    monkeypatch.setattr("builtins.input", confirm)
    monkeypatch.setattr(flight_host, "request_json", lambda *args, **kwargs: pytest.fail("Contacted device before confirmation"))
    flight_host.main()
    assert "Wipe cancelled" in capsys.readouterr().out


@pytest.mark.parametrize("extra_args", [[], ["--yes"]])
def test_wipe_cli_confirmed_deletes_device_sessions(device_api, monkeypatch, capsys, extra_args):
    import flight_host

    base, recorder = device_api
    session = Session(recorder.root, 0, {})
    session.finish("complete")
    recorder.finished.set()
    monkeypatch.setattr(sys, "argv", ["flight_host.py", "wipe", "--device", base, *extra_args])

    def confirm(_):
        assert not extra_args, "--yes must skip the prompt"
        return "yes"

    monkeypatch.setattr("builtins.input", confirm)
    flight_host.main()
    assert not session.path.exists()
    assert "Deleted 1 session(s)" in capsys.readouterr().out


def test_process_kill_and_unfinished_journal_are_recoverable(tmp_path):
    script = '''
import sys,time
from pathlib import Path
from oakapp.recorder_storage import Session,CameraJournal
s=Session(sys.argv[1],0,{"fps":30})
j=CameraJournal(s.path/"cameras"/"CAM_A")
p=b"\\x00\\x00\\x01\\x67SPS\\x00\\x00\\x01\\x68PPS\\x00\\x00\\x01\\x65FRAME"
j.append(p,100,1,320,240)
s.telemetry.json({"time_us":100,"pressure_pa":100000})
with (j.directory/"000000.h264").open("ab",buffering=0) as f:f.write(b"unfinished video payload")
with (j.directory/"frames.jsonl").open("ab",buffering=0) as f:f.write(b'{"unfinished":')
with (s.path/"telemetry.jsonl").open("ab",buffering=0) as f:f.write(b'{"unfinished":')
print(s.path,flush=True)
time.sleep(60)
'''
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        directory = Path(process.stdout.readline().strip())
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
    recover_sessions(tmp_path)
    assert json.loads((directory / "manifest.json").read_text())["status"] == "interrupted"
    frames = list(verified_frames(directory / "cameras" / "CAM_A"))
    assert len(frames) == 1
    assert b"unfinished" not in frames[0][1]
    assert len(list(read_jsonl(directory / "telemetry.jsonl"))) == 1


def test_checksum_detects_committed_frame_damage(tmp_path):
    journal = CameraJournal(tmp_path / "CAM_A")
    packet = b"\x00\x00\x01\x65frame"
    journal.append(packet, 0, 0, 320, 240)
    journal.close()
    (tmp_path / "CAM_A" / "000000.h264").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="corrupt"):
        list(verified_frames(tmp_path / "CAM_A"))


def test_full_resolution_five_camera_playback_and_original_timestamps(tmp_path):
    directory = fixture_flight(tmp_path)
    result = prepare_flight(directory)
    assert len(result["cameras"]) == 5
    assert len(list(read_jsonl(directory / "telemetry.jsonl"))) == 20
    for camera in result["cameras"]:
        with av.open(str(directory / "videos" / (camera["name"] + ".mp4"))) as video:
            frames = list(video.decode(video=0))
        assert len(frames) == 12
        assert all((frame.width, frame.height) == (camera["width"], camera["height"]) for frame in frames)
        assert camera["missing_frames"] == 1
        times = [float(frame.pts * frame.time_base) for frame in frames]
        assert times[6] - times[5] == pytest.approx(1 / 30 + 0.05, abs=0.001)
        assert times[0] == pytest.approx(0, abs=0.001)
    assert (directory / "telemetry.csv").exists()


def test_resume_transfer_and_playback_range_requests(tmp_path, monkeypatch):
    from oakapp.main import make_handler
    monkeypatch.delenv("RECORDER_API_TOKEN", raising=False)

    root = tmp_path / "device"
    root.mkdir()
    flight = fixture_flight(root, cameras=1)

    class FakeRecorder:
        pass

    recorder = FakeRecorder()
    recorder.root = root
    recorder.status = {"state": "stopped", "flight_id": flight.name}
    recorder.stop = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(recorder))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert request_json(base, "/api/status")["state"] == "stopped"
        sessions = request_json(base, "/api/sessions")
        assert select_sessions(sessions, "1")[0]["id"] == flight.name
        assert select_sessions(sessions, flight.name) == select_sessions(sessions, "1")
        with pytest.raises(ValueError, match="number"):
            select_sessions(sessions, "0")
        snapshot = request_json(base, f"/api/flights/{flight.name}/files")
        info = next(item for item in snapshot["files"] if item["path"].endswith(".h264"))
        download = tmp_path / "host"
        partial = download / (info["path"] + ".download")
        partial.parent.mkdir(parents=True)
        partial.write_bytes((flight / info["path"]).read_bytes()[:10])
        download_file(base, flight.name, info, download)
        assert (download / info["path"]).read_bytes() == (flight / info["path"]).read_bytes()
        with pytest.raises(ValueError):
            download_file(base, flight.name, {"path": "../escape", "size": 1, "sha256": ""}, download)
    finally:
        server.shutdown()
        server.server_close()
    prepare_flight(flight)
    server = ThreadingHTTPServer(("127.0.0.1", 0), playback_handler(flight))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        from urllib.request import Request, urlopen
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(Request(base + "/videos/CAM_A.mp4", headers={"Range": "bytes=0-99"})) as response:
            assert response.status == 206
            assert len(response.read()) == 100
        assert request_json(base, "/api/playback")["cameras"][0]["width"] == 320
    finally:
        server.shutdown()
        server.server_close()
