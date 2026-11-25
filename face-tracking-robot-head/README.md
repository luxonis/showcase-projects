# Face-Tracking Robot Head with OAK 4 FFC

> [!NOTE]
> This is a **showcase project** built for a Luxonis demo. The code is **unmaintained** and is shared for reference only.
> For maintained, minimal examples of OAK features, see [oak-examples](https://github.com/luxonis/oak-examples).

**Used for:** The *Face-Tracking Robot Head* showcase: a 3D-printed humanoid face on a 2-axis servo platform that turns to follow a person's face. Person/face detection, a high-resolution face crop and stereo depth (live point cloud) run on the device, which steers the servos over serial.

**Device:** **OAK 4 FFC** devkit with 2× IMX586 autofocus camera modules (stereo pair), connected over USB to a servo control board (firmware in `robot_head_arduino/`), powered by PoE+.

[![Building a Humanoid Head with OAK4 FFC](https://img.youtube.com/vi/E6ewyC0bJPo/maxresdefault.jpg)](https://www.youtube.com/watch?v=E6ewyC0bJPo)

| | |
|---|---|
| Blog post | [Building a face-tracking robot head with the OAK4 FFC](https://discuss.luxonis.com/blog/7078-building-a-face-tracking-robot-head-with-the-oak4-ffc) |
| Video | [YouTube](https://www.youtube.com/watch?v=E6ewyC0bJPo) |
| Docs | [Face-Tracking Robot Head with OAK 4 FFC](https://docs.luxonis.com/overview/industry-examples/robotics/face_tracking_robot_head) |

## Robot Head

Arduino servo control and OAK camera tracking. Run these commands from this directory.

Set up Python dependencies and upload the Arduino Uno firmware (once):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt platformio
pio run -d robot_head_arduino -t upload --upload-port /dev/ttyUSB0
```

Connect the Arduino and OAK camera, then run one script at a time:

```bash
python scripts/arm.py                                             # Type commands such as: 1 90
python scripts/arm2.py                                            # Smooth control with arrows/WASD; q quits
python scripts/person_follow.py --device 10.12.222.68             # Face tracking
python scripts/focused_person_follow.py --device-ip 10.12.222.68  # Focused tracking + point cloud
python scripts/cloud.py --device-ip 10.12.222.68                  # Stereo depth + point cloud
```

Replace the camera IP as needed. Servo scripts default to `/dev/ttyUSB0`; use
`--port` for `arm2.py`, `--serial-port` for `focused_person_follow.py`, or edit
the port constant in the other servo scripts.

Open <http://localhost:8082> for the camera views. Add `--no-servos` to
`focused_person_follow.py` to run detection without the Arduino, and
`--duration SECONDS` for a timed trial run.

Depth and the point cloud use the `StereoDepth` node with the two IMX586
cameras on CAM_A and CAM_D, which must be calibrated as a stereo pair.
`scripts/oakapp.toml` packages `focused_person_follow.py` as an OAK App to run
on the device itself.
