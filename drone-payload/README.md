# Edge-AI Drone Payload with OAK 4 FFC

> [!NOTE]
> This is a **showcase project** built for a Luxonis demo. The code is **unmaintained** and is shared for reference only.
> For maintained, minimal examples of OAK features, see [oak-examples](https://github.com/luxonis/oak-examples).

**Used for:** The *OAK 4 FFC on a Drone* showcase: an OAK 4 FFC mounted on an F450 quadcopter records both downward cameras together with barometric altitude, pressure and temperature during flight. A host tool downloads sessions and plays them back in sync in the browser.

**Device:** **OAK 4 FFC** with 2× OG05B10 global-shutter camera modules and a SparkFun BMP581 pressure sensor (Qwiic, I²C), powered directly from the drone's 3S LiPo battery.

[![OAK 4 FFC on a Drone: Edge AI and Custom Sensors](https://img.youtube.com/vi/2Cx0MPVmgLc/maxresdefault.jpg)](https://www.youtube.com/watch?v=2Cx0MPVmgLc)

| | |
|---|---|
| Blog post | [Mounting the OAK 4 FFC on an F450 drone](https://discuss.luxonis.com/blog/7074-mounting-the-oak-4-ffc-on-an-f450-drone) |
| Video | [YouTube](https://www.youtube.com/watch?v=2Cx0MPVmgLc) |
| Docs | [Edge-AI Drone Payload with OAK 4 FFC](https://docs.luxonis.com/overview/industry-examples/robotics/oak4_ffc_drone) |

## OAK4 drone recorder

The OAK App discovers supported cameras on CAM_A–CAM_E and records compressed
video at each camera's native resolution and 30 FPS. It logs BMP581 temperature,
pressure, and barometric height relative to the start of each recording.
Every camera stream is rotated 180° before compression to compensate for the
upside-down camera mounting. Saved videos retain their native dimensions.

The host tool lists sessions, downloads their grouped video and sensor data,
and provides synchronized browser playback.

### Host setup

The existing `.venv` is ready to use. To recreate it:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If Python's `ensurepip` is unavailable, use the system pip:

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv install pip -r requirements.txt
```

The project-local `./oakctl` launcher uses `.tools/oakctl` and its local device
configuration for app management. For a fresh checkout, obtain the executable
from the [official oakctl installation instructions](https://docs.luxonis.com/software-v3/oak-apps/oakctl)
and place it at `.tools/oakctl`.

### Recording and download

Run commands from this directory. The host tool discovers the recorder on the
local network using `./oakctl`, including its link-local address when offline.
It prints the selected address. Use `--device IP[:PORT]` to choose a device
explicitly; the recorder API port defaults to `8088`. Transfer uses direct plain
HTTP and bypasses host proxy settings. No internet connection is needed to list,
download, prepare, or play recordings.

```bash
.venv/bin/python flight_host.py run        # Finish any current session and start a new one
.venv/bin/python flight_host.py status     # Camera and sensor status
.venv/bin/python flight_host.py sessions   # Newest session first
.venv/bin/python flight_host.py download   # Choose a session from the list
```

For a direct Ethernet connection without a router/DHCP server, set the host's
Ethernet IPv4 method to **Link-Local Only**, or assign an unused address such as
`169.254.10.112` with netmask `255.255.0.0` (no gateway or DNS needed). The camera
falls back to its own `169.254.x.x` address, as described in the
[OAK4 networking guide](https://docs.luxonis.com/hardware/platform/deploy/oak4-deployment-guide/oak4-getting-started#No-DHCP).
The old LAN address `10.12.222.29` is also tried if discovery cannot find a running
recorder. If multiple recorders are found, choose one using `--device`.

To inspect discovery or use a known link-local address directly:

```bash
./oakctl device list --format json
.venv/bin/python flight_host.py sessions --device 169.254.239.12
.venv/bin/python flight_host.py download --device 169.254.239.12
```

The link-local address above is an example; automatic discovery follows address
changes without editing the script. Listing and downloading use the recorder's
HTTP API and do not require device adoption or a cloud login.

Session numbers are permanent and independent of the device clock and Internet
access. `download 6` selects session `000006` even as newer sessions are added.
New folders use names such as `session-000006`. Device timestamps are displayed
for reference and may be incorrect offline. Older recordings retain their
original folders and can still be selected by their permanent numbers.

To finish and download the current session:

```bash
.venv/bin/python flight_host.py download latest --stop
```

`stop` ends capture while keeping download available; `run` starts a new session.
Downloading a completed session leaves ongoing capture running. Interrupted
downloads resume when repeated, and files are verified with SHA-256.
`download all` selects all sessions; `--raw-only` skips playback preparation.
Device recordings are retained after download.

To permanently delete all sessions from the device and free their storage:

```bash
.venv/bin/python flight_host.py wipe
```

The command asks for confirmation, stops any active capture, and waits for all
recording workers to finish before deleting the session folders. Add `--yes` to
skip the prompt. Downloaded copies on your computer are kept, and permanent
session numbering continues from the previous counter. The device stays stopped;
use `run` to start recording again. This requires recorder app version 1.0.7 or
later on the device; build and install it using the instructions below.

### Playback

Use the folder printed by the download command:

```bash
.venv/bin/python flight_host.py play recordings/SESSION_ID
```

The browser opens at `http://127.0.0.1:8090`. All camera previews and the height,
pressure, and temperature graphs fit together without page scrolling. Drag the
timeline or click a graph to seek every stream together. Double-click a camera
for fullscreen inspection.

Use **Start** and **End**, then **Apply range**, to display a selected interval
across every camera and graph. Enter seconds (for example `12.5`) or `m:ss.sss`
(`1:02.500`), measured from the start of the recording. Playback stops at the
selected end; **Back to start** returns to the selected start. **Full session**
restores the original interval.

Videos retain native resolution and original frame timing. H.264 is remuxed to
MP4; MJPEG is converted to H.264 for browser playback. Original compressed data
remains in the downloaded folder.

Each download contains `manifest.json`, `telemetry.jsonl`, `events.jsonl`,
camera segments and frame journals under `cameras/`, and a transfer manifest.
Preparation adds `videos/*.mp4`, `telemetry.csv`, and `playback.json`. To prepare
a raw download later:

```bash
.venv/bin/python flight_host.py prepare recordings/SESSION_ID
```

### OAK App package

Source and device dependencies are in `oakapp/`. Build the package:

```bash
mkdir -p dist
./oakctl app build oakapp -d 10.12.222.29 --output dist
```

The package is `dist/com-ffcdrone-flight-recorder_1-0-8.oakapp`. To install it
and enable device boot activation:

```bash
./oakctl app install dist/com-ffcdrone-flight-recorder_1-0-8.oakapp \
  -d 10.12.222.29 --keep-others --enable true --start true
```

The installed recorder starts at device boot. Each startup scans all five camera
ports and begins a new numbered session. Unsupported native-resolution/FPS
combinations report an error. If no camera is detected, sensor logging continues
with `sensor_only` status.

The SparkFun BMP581 uses I2C bus 4, address `0x47`, at 10 Hz. The first second
of valid readings establishes the pressure reference:

```text
height_m = 44330 × (1 − (pressure_pa / starting_pressure_pa)^0.190294957)
```

Settings in `oakapp/oakapp.toml` include FPS, sensor rate, I2C bus/address, API
port, segment duration, and free-space reserve. `RECORDER_BITRATE` sets the H.264
bitrate; zero selects encoder defaults. Device dependencies are listed in
`oakapp/requirements.txt`.

Recordings and the session counter use persistent `OAKAGENT_STORAGE_SHARED`
storage. Compressed frame bytes are synced before their frame journal entries;
sensor samples and counter updates are also synced. After restart, unfinished
sessions are marked interrupted, and playback recovers committed frames while
ignoring incomplete tails. Capture stops at the 1 GiB free-space reserve, keeping
download available. Physical power-cut behavior depends on the storage
controller; process termination and incomplete-write recovery are tested.

### Tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
```

Tests cover clock-independent numbering, legacy recordings, interrupted session
creation, abrupt writer termination, damaged frames, native-resolution playback,
original timestamps, local device discovery, direct HTTP downloads with host
proxies configured, resumable downloads, browser video ranges, and confirmed
device wipes that wait for recording workers and preserve session numbering.
