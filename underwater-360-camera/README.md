# Underwater 360° Camera with OAK 4 FFC

> [!NOTE]
> This is a **showcase project** built for a Luxonis demo. The code is **unmaintained** and is shared for reference only.
> For maintained, minimal examples of OAK features, see [oak-examples](https://github.com/luxonis/oak-examples).

**Used for:** The *Underwater 360° Camera* showcase: an OAK 4 FFC with five wide-FOV cameras sealed in a watertight enclosure, filmed in a water tank and in shallow sea water from a canoe. This app records every camera to the device on boot, so the rig can film without a host connection.

**Device:** **OAK 4 FFC** devkit with 5× OAK-FFC OV9282 W (global-shutter, wide FOV) camera modules on CAM_A–CAM_E, inside a Blue Robotics 4" watertight enclosure, powered over PoE.

[![Building an Underwater 360° Camera with Luxonis OAK 4 FFC](https://img.youtube.com/vi/tCZqND1Ejtg/maxresdefault.jpg)](https://www.youtube.com/watch?v=tCZqND1Ejtg)

| | |
|---|---|
| Blog post | [Building an underwater 360° camera with OAK 4 FFC](https://discuss.luxonis.com/blog/7071-building-an-underwater-360-camera-with-oak-4-ffc) |
| Video | [YouTube](https://www.youtube.com/watch?v=tCZqND1Ejtg) |
| Docs | [Underwater 360° Camera with OAK 4 FFC](https://docs.luxonis.com/overview/industry-examples/smart-cities-and-transportation/underwater_360_camera) |

## OAK4 FFC boot recorder

This OAK App detects cameras with `Device.getConnectedCameraFeatures()` (as in
`depthai-core/utilities/cam_test.py`) and records every detected socket as H.264 video in a Matroska (`.mkv`) file.
Matroska stores camera timestamps so VLC plays at the recorded speed.
It uses one UTC start timestamp for the session, for example
`20260924T122306.849710Z_cama.mkv` through
`20260924T122306.849710Z_came.mkv`. Files are opened exclusively, so a later
boot cannot overwrite an earlier session.

The tested OAK4-FFC has five OV9282 cameras, CAM_A–CAM_E. RVC4 supports three
concurrent ISP streams and two additional raw streams. CAM_A–C feed the H.264
encoders directly. CAM_D–E use the raw outputs; the app converts their packed
RAW10 frames to grayscale NV12 before passing them to hardware H.264 encoders.
The app defaults to 20 FPS.

CAM_D–E get software image processing and startup exposure calibration, enabled
by default. The conversion retains all ten sensor bits, subtracts the OV9282 black
level (64), applies gamma 2.2, and maps pixels to video luma range 16–235. This
restores midtone and shadow detail that was missing from the old linear raw
conversion. Small previews from CAM_A–C supply a smoothed median brightness
target; each raw camera calibrates independently for the first ten seconds,
then locks its exposure and digital gain for the rest of the recording. Later
changes in the scene or reference cameras do not adjust these locked settings.
Calibration runs again when the recorder restarts. Use
`--raw-exposure-settle-seconds` to change the startup calibration duration.
During calibration, updates are limited to twice per second with a deadband.

Raw cameras start at 10 ms (`--raw-exposure-us`) and requested ISO 400
(`--raw-iso`). On the tested RVC4, raw-only exposure changes work, but ISO remains
100 despite higher requests. During startup calibration, the controller increases exposure first,
up to 33 ms or one frame period, then uses at most 8× digital gain. Gain can make
noise more visible in very dark scenes; it cannot recover light the sensor did
not capture. A highlight guard limits exposure increases when a substantial area
is already saturated. Different views may still have different contrast and
sharpness; software processing does not reproduce every ISP denoising operation.

Use `--no-raw-auto-exposure` to keep the requested raw exposure fixed. Tone
correction remains active; `--raw-gamma` and `--raw-black-level` override its
sensor-specific defaults. The log reports each raw camera's measured luma,
requested exposure, digital gain, calibration/lock status, and shared target
every ten seconds. It also logs when each raw camera locks its settings.

Recordings go to `OAKAGENT_STORAGE_SHARED/recordings` on the device
(`/app-storage/shared/recordings` in the tested container). Shared storage
persists across app restarts and package updates. The app preserves a minimum
of 5 GiB free device space and stops writing once its recordings reach 50 GiB.
It waits for space to be freed; it never deletes recordings automatically.
Set `OAK_MAX_RECORDING_GB`, `OAK_MIN_FREE_GB`, or `--fps` to suit your storage
and recording duration. At 20 FPS, recordings use storage faster than at 10 FPS.

### Test from a host

Use the existing virtual environment, or install `requirements.txt` into one,
then run:

```sh
source venv/bin/activate
python3 main.py --device 10.12.222.29 --duration 10 --output-dir ./recordings
```

`--device` is optional when DepthAI finds only one camera device. The OAK App
uses the camera inside its own container automatically. Keep the cameras under the intended
lighting during the ten-second startup calibration; the settings then stay fixed. These changes apply to new recordings; existing
videos are not rewritten. An already installed boot recorder needs rebuilding
and reinstalling to use the new processing.

Image processing and exposure tests do not require connected cameras:

```sh
python3 -m unittest -v test_image_quality
```

### Install for automatic recording on boot

On a host with `oakctl` and access to the OAK4:

```sh
oakctl app build .
oakctl app install ./com-example-underwater-ffc-recorder_1-0-8.oakapp
oakctl app list
oakctl app enable APP_ID
oakctl app start APP_ID
```

`oakctl app install` may already enable and start the app. Check the `E` marker
and `running` status in `oakctl app list`. An enabled app starts on camera boot. The recorder runs under its own supervisor
without the base image’s cloud connection or WebRTC services, so recording does
not require internet access.
Use `oakctl app logs APP_ID --no-follow --tail 30` to check the recording paths.

### Retrieve a full camera session

The app must be running for `oakctl app exec` to access its shared storage.
From this repository on a host with `oakctl` installed:

```sh
python3 retrieve.py --device 10.12.222.29
python3 retrieve.py --device 10.12.222.29 --session 2
python3 retrieve.py --device 10.12.222.29 --latest
python3 retrieve.py --device 10.12.222.29 --delete-session 2 --dry-run
python3 retrieve.py --device 10.12.222.29 --delete-session 2
```

The tool automatically selects the only running app on the device. If more than
one app is running, pass its app ID before the options.

The first command lists one row per session, newest first, with a readable UTC
start time, its cameras, and total size. `--session 2` downloads every camera
from row 2; `--latest` downloads every camera from row 1. Both create one
`downloads/TIMESTAMP` folder per session. Use `--dry-run` with either selection
to preview it, or `--output-dir PATH` to choose a destination. The tool never
selects an individual camera. `--delete-session` previews the entire group,
then asks before deleting all of its files on device. It refuses to delete a
session with open recording files. Empty files from early failed starts are
omitted from the list.
If the newest session is still recording, its files are copied up to their
listed size; retrieve that session after it closes for a complete copy.

An offline cold boot may reset the OAK clock to 1970 or an older date. In that case the recorder
uses the last saved recording time as a lower-bound estimate, marks new file
names with `_offline`, and keeps each new session unique. The list labels these
times as estimates; actual UTC start time needs a clock source on the device.
Older files made while the clock read 1970 appear as `clock reset` sessions.

New `.mkv` files play directly in VLC at their recorded frame rate. Older
`.h264` files created by versions 1.0.2 and earlier were recorded at 10 FPS
but contain no timing data. After retrieving an older group, give each file
10 FPS timestamps without re-encoding:

```sh
ffmpeg -r 10 -i downloads/TIMESTAMP/FILE.h264 -c copy downloads/TIMESTAMP/FILE.mkv
```

Keep the original until you have checked the converted video.

[Luxonis OAK App storage and boot management](https://docs.luxonis.com/software-v3/oak-apps/oakctl) ·
[RVC4 camera stream limits](https://docs.luxonis.com/hardware/platform/rvc/rvc4)
