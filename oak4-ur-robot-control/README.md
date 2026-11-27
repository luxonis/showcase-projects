# OAK 4 → UR Robot Arm Control

> [!NOTE]
> This is a **showcase project** built for a Luxonis demo. The code is **unmaintained** and is shared for reference only.
> For maintained, minimal examples of OAK features, see [oak-examples](https://github.com/luxonis/oak-examples).

**Used for:** The *OAK 4 → UR robot arm* showcase: the Universal Robots ROS 2 driver runs **on the OAK 4 camera** as an OAK App and controls a UR arm directly, with no separate industrial PC. The camera tracks a person's hand (MediaPipe palm detection + depth) and the arm follows it; with the camera on the end effector it can also segment and grasp an object held in the hand (MobileSAM + Track-Anything, RG2 gripper).

**Device:** **OAK4-LITE-TOF** (an OAK4-D R7 also worked), mounted on the end effector of a **UR7e** (PolyScope X) with an **OnRobot RG2** gripper on tool digital output 0. A simulated UR5e/UR7e (URSim in Docker) was used for development.

<!-- TODO: add the YouTube thumbnail once the video is published:
[![<video title>](https://img.youtube.com/vi/<id>/maxresdefault.jpg)](https://www.youtube.com/watch?v=<id>) -->

| | |
|---|---|
| Blog post | TODO: not published yet |
| Video | TODO: not published yet |
| Docs | TODO: no docs.luxonis.com page yet |

## Overview

Run the [Universal Robots ROS 2 driver](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/)
**on an OAK4 camera** and control a UR arm directly from it — no separate industrial PC.
The first target is a simulated UR5e (URSim in Docker); the same app drives a real arm by
switching the config profile.

```
 LAN
 ┌────────────────────────────┐   29999 dashboard, 30001-30004 primary/RTDE   ┌─────────────────────────┐
 │ OAK4 (oakctl app)          │ ────────────────────────────────────────────▶ │ UR controller           │
 │  ros:kilted + ur driver    │                                               │  URSim in Docker (sim)  │
 │  robot_prep  (dashboard)   │ ◀──────────────────────────────────────────── │  or real UR7e/UR5e/UR5  │
 │  motion_demo / hand_follow│   50001-50004 reverse / trajectory / script   │                         │
 └────────────────────────────┘                                               └─────────────────────────┘
```

### Restart checklist (our setup)

Everything needed to bring the demo back up. Verified working on 2026-11-06.

| What | Value |
|---|---|
| Camera | OAK4-LITE-TOF R1, serial `361589140`, IP **`10.12.222.78`** (check: `oakctl device info -d 10.12.222.78`). Earlier: OAK4-D R7 at `10.12.222.229` — same camera sockets, both work |
| Host (URSim, Docker, ROS 2 Kilted) | IP **`10.12.222.225`** (check: `hostname -I`, use the `10.12.222.x` one) |
| Dev VM (runs `oakctl`, this repo) | NATed `192.168.122.x` — no Docker here; URSim runs on the host |
| Robot model | UR5e (URSim e-series); target arm: UR7e with PolyScope X (see [PolyScope X](#polyscope-x-ur7e--ur12e--ur15)) |
| PolyScope GUI | `http://localhost:6080/vnc.html` on the host |
| Hand view | **`http://10.12.222.78:8090/`** (only while the app runs with `CONTROL_MODE=hand`) |

If an IP changed, update `ROBOT_IP` (host) and `CAMERA_IP` in `config/sim.env`, or pass
`ROBOT_IP=...` on the deploy command line.

#### 1. Host: start URSim

```bash
docker rm -f ursim 2>/dev/null
ros2 run ur_client_library start_ursim.sh -m ur5e \
  -f "-p 6080:6080 -p 29999:29999 -p 30001-30004:30001-30004"
```

Leave it running (Ctrl+C stops the simulator). Do **not** publish port 5900 — it is taken on
the host and the start fails with `failed to bind host port 0.0.0.0:5900`.

#### 2. PolyScope (one-time setup, persists in `~/.ursim`)

Open `http://localhost:6080/vnc.html` on the host, then:

```
Connect
Power ON  (bottom-left red button -> ON -> START, until it shows Normal) -> Exit
Installation -> URCaps -> External Control -> Host IP = 10.12.222.78, Port = 50002
Program -> New -> Empty Program
Program -> URCaps -> External Control
Save -> external_control.urp
```

After a URSim restart you usually only need **Power ON** (or let `AUTO_POWER_ON=true` do it)
and check the Host IP is still the camera IP.

#### 3. Dev VM: deploy to the camera

```bash
cd ~/Documents/Git/oak4-ur-robot-control

# hand following (camera NOT on the arm -> mirror mode)
./scripts/deploy.sh sim CONTROL_MODE=hand HEADLESS_MODE=false --detach

# or: scripted waypoint loop
./scripts/deploy.sh sim CONTROL_MODE=waypoints HEADLESS_MODE=false --detach

# or: driver only, drive from MoveIt on the host
./scripts/deploy.sh sim CONTROL_MODE=none HEADLESS_MODE=false --detach
```

The first build after a change in `oakapp.toml` / `requirements.txt` takes several minutes
(ROS + DepthAI install on the device); code-only changes rebuild in seconds. Wait until
`oakctl app list -d 10.12.222.78` shows `running` and the camera listens on 50002:

```bash
timeout 3 bash -c '</dev/tcp/10.12.222.78/50002' && echo "driver ready"
```

#### 4. PolyScope: press Play

`Program` with the External Control node → **Play** (bottom right) → *Play from beginning*.
Only press Play **after** step 3 — otherwise PolyScope shows
`connection to remote PC at 10.12.222.78:50002 could not be established`.
Every redeploy restarts the driver, so press Play again after each deploy.

#### 5. Use it

- **Hand mode:** open `http://10.12.222.78:8090/` on the host, hold an open palm ~60 cm in
  front of the camera. Overlay states: `waiting` (no hand yet) → `following` → `holding`
  (hand lost, arm stays) → `returning home` (after 5 s) → `waiting`. The arm moves **only while
  a palm is detected**.
- **Waypoint mode:** the arm loops home → left → left_low → right → right_low → home.
- **MoveIt (CONTROL_MODE=none):** on the host
  `ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e`, drag the marker,
  *Plan & Execute*.

#### 6. Watch / debug

```bash
oakctl app list -d 10.12.222.78                       # running / stopped / building
oakctl app logs 000 -d 10.12.222.78                   # follow logs (Ctrl+C to quit)
oakctl app logs 000 -d 10.12.222.78 --no-follow | grep -v Overrun | tail -30
```

Robot state from any LAN machine via the Dashboard Server:

```bash
timeout 5 bash -c 'exec 3<>/dev/tcp/10.12.222.225/29999; head -1 <&3; for c in robotmode safetystatus programState; do echo $c >&3; head -1 <&3; done'
```

Expected: `Robotmode: RUNNING`, `Safetystatus: NORMAL`, `PLAYING ...`.

#### 7. Stop

```bash
oakctl app stop 000 -d 10.12.222.78     # camera app (arm stops receiving commands)
# host: Ctrl+C in the start_ursim.sh terminal
```

#### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `failed to bind host port 0.0.0.0:5900` | Leave `-p 5900:5900` out (step 1), then `docker rm -f ursim` and retry |
| PolyScope: `connection to remote PC at 10.12.222.78:50002 could not be established` | Camera app not running yet / was stopped. Deploy (step 3), wait for 50002, then Play |
| Hand mode: arm does not move | No palm detected (overlay says `waiting`/`holding`). Show an open palm facing the camera, ~40–100 cm away, good lighting |
| Arm stops after a redeploy | Driver restarted → press Play again in PolyScope |
| `oakctl ...: Login token expired ... not a TTY` | Run any `oakctl` command once in an interactive terminal and log in (e.g. after a camera reboot) |
| `oakctl app logs/stop` hang, `Failed to wait for RPC response` | Agent stuck (seen after hanging `oakctl app exec`). Avoid `app exec`; reboot the camera with `oakctl device reboot -d 10.12.222.78` from an interactive terminal, then log in again |
| Many `controller_manager ... Overrun` warnings | Expected: the OAK4 has no real-time kernel. Harmless for these demos |
| Robot program won't start in headless mode | Use the URCap flow (`HEADLESS_MODE=false` + Play), which is what we verified |

### Layout

| Path | What it is |
|---|---|
| `host/start_ursim.sh` | Starts URSim (UR5e) in Docker on an x86 LAN machine and publishes the robot ports on its LAN IP |
| `oakapp/` | The OAK App built and run on the camera with `oakctl` |
| `oakapp/entrypoint.sh` | Robot prep → `ur_control.launch.py` → motion node |
| `oakapp/ur_oak_control/arm.py` | `URArm` — small API over the driver (`wait_until_ready`, `move_joints`). Import this from perception code. |
| `oakapp/ur_oak_control/motion_demo.py` | Scripted joint-space motion through `waypoints.yaml` |
| `oakapp/ur_oak_control/hand_follow.py` | Hand following: DepthAI palm tracking → Cartesian target → IK → streamed joint commands |
| `oakapp/ur_oak_control/hand_tracker.py` | DepthAI pipeline: palm detection (MediaPipe) + stereo depth on the OAK4 |
| `oakapp/ur_oak_control/follow_controller.py` | Control law (mirror / servo), workspace clamp, speed limit, hold / return home |
| `oakapp/ur_oak_control/kinematics.py` | UR forward / inverse kinematics from DH parameters |
| `oakapp/hand_follow.yaml` | Hand-follow tuning: home pose, workspace, speeds, gains |
| `oakapp/ur_oak_control/robot_prep.py` | Reachability check; Dashboard Server steps (power on, brake release, load/play) when the controller has one — skipped on PolyScope X |
| `oakapp/ur_oak_control/activate_controller.py` | Activates `scaled_joint_trajectory_controller` if the driver leaves it inactive |
| `config/sim.env`, `config/real.env` | Deployment profiles |
| `scripts/deploy.sh` | `oakctl app run` with a profile |

### Quick start (simulation)

Requirements: an OAK4 on the LAN (`oakctl` installed on your machine) and an x86 Linux host
with Docker and ROS 2 **Kilted** on the same LAN for URSim (and optionally MoveIt/RViz).
The camera app also uses Kilted — keep host and camera on the same ROS distro so they can
talk over DDS.

1. **Start URSim** on the host with the upstream script from `ur_client_library`
   (`sudo apt install ros-kilted-ur-client-library`):

   ```bash
   ros2 run ur_client_library start_ursim.sh -m ur5e \
     -f "-p 6080:6080 -p 29999:29999 -p 30001-30004:30001-30004"
   ```

   Port 5900 (raw VNC) is left out on purpose: it is often taken on the host (e.g. by a
   libvirt/QEMU VM display) and fails with `failed to bind host port 0.0.0.0:5900`. The browser
   GUI on 6080 is enough; add `-p 5901:5900` if you want a VNC client. After a failed start, run
   `docker rm -f ursim` before retrying.

   `-f` publishes the robot ports **and** the PolyScope GUI on all host interfaces, so the camera
   reaches the robot via the host's LAN IP and you can open the GUI from another machine at
   `http://<HOST_LAN_IP>:6080/vnc.html`. On the host itself,
   `http://192.168.56.101:6080/vnc.html?host=192.168.56.101&port=6080` works too.
   Without ROS on the host, `./host/start_ursim.sh` does the same with plain Docker.

2. **Set `ROBOT_IP`** in `config/sim.env` to the host's **LAN IP** (e.g. `hostname -I`), not the
   container IP `192.168.56.101` the script prints — the camera cannot reach Docker's internal
   network. Set `CAMERA_IP` too if your camera differs.

3. **Deploy to the camera:**

   ```bash
   ./scripts/deploy.sh sim
   ```

   IPs change between sessions/networks, so you can also pass them per run:
   `./scripts/deploy.sh sim ROBOT_IP=<HOST_LAN_IP>` (any `KEY=VALUE` overrides the profile).

   The first build installs ROS 2 Kilted + the UR driver on the device (several minutes).
   The app then:
   1. waits for the robot ports, powers the simulated robot on and releases the brakes,
   2. starts the driver in headless mode (it sends the URScript program to the robot itself),
   3. activates `scaled_joint_trajectory_controller` if the driver left it inactive,
   4. loops through the poses in `oakapp/waypoints.yaml`.

   Watch the arm move in PolyScope. Logs: `oakctl app logs 000 -d <CAMERA_IP>`.
   Debug inside the app: `oakctl app exec 000 -d <CAMERA_IP> bash -- -c 'source /opt/ros/kilted/setup.bash && ros2 control list_controllers'`.

4. Stop: `oakctl app stop 000 -d <CAMERA_IP>` and Ctrl+C the URSim script.

#### Test without any robot

`./scripts/deploy.sh sim ROBOT_IP=127.0.0.1 MOCK_HARDWARE=true MOTION_LOOPS=2` runs the full
camera-side pipeline (driver, controller activation, trajectories) against ros2_control mock
hardware. Expect `Goal reached, success!` and `motion demo finished` in the logs.

#### Alternative: External Control URCap (proven manual flow)

If headless mode does not start the program, use the URCap flow instead — this is also how a
real arm is usually run:

1. Deploy with `./scripts/deploy.sh sim HEADLESS_MODE=false`.
2. In PolyScope: **Power ON** the robot (or keep `AUTO_POWER_ON=true`).
3. **Installation → URCaps → External Control → Host IP** = camera IP (port 50002).
4. **Program → URCaps → External Control**, then press **Play** (bottom right).
   Save the program as `external_control.urp` and later automate step 4 with
   `AUTO_PLAY_PROGRAM=external_control.urp`.

#### Control from MoveIt / RViz on the host

The driver on the camera is a normal ROS 2 node on the LAN. To plan motions interactively:

```bash
./scripts/deploy.sh sim CONTROL_MODE=none                 # camera: driver only
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e   # host
```

Drag the interactive marker and press **Plan & Execute**. Both sides need the same
`ROS_DOMAIN_ID` and must allow UDP multicast (DDS discovery) on the LAN.

### Configuration

All variables go to the app as environment variables (`oakctl app run --env`).

| Variable | Default | Meaning |
|---|---|---|
| `ROBOT_IP` | — | UR controller / URSim IP |
| `UR_TYPE` | `ur5e` | `ur7e`, `ur12e`, `ur5e`, `ur10e`, `ur5` (CB3), ... |
| `HEADLESS_MODE` | `true` | `true`: driver sends the program. `false`: use the External Control URCap program |
| `AUTO_POWER_ON` | `false` | Power on + brake release via Dashboard Server (sim profile sets `true`; ignored on PolyScope X) |
| `AUTO_PLAY_PROGRAM` | — | URCap mode: program to `load` + `play` once the driver is up (PolyScope 5 only) |
| `MOCK_HARDWARE` | `false` | `true`: no robot needed — driver runs with `use_mock_hardware` (pipeline test) |
| `REVERSE_IP` | auto | IP the robot uses to reach the camera (driver auto-detects) |
| `KINEMATICS_PARAMS_FILE` | — | Robot calibration file (real arm) |
| `CONTROL_MODE` | `waypoints` | `waypoints` (scripted loop), `hand` (follow a hand), `none` (driver only) |
| `FOLLOW_MODE` | `mirror` | Hand mode: `mirror` (camera not on the arm) or `servo` (camera on the tool) |
| `EYE_IN_HAND_SIM` | `false` | `servo` with the camera on the desk (URSim): emulate a camera that moves with the tool |
| `WEB_VIEW_PORT` | `8090` | Hand mode debug view `http://<CAMERA_IP>:8090/` (`0` = off) |
| `MOTION_LOOPS` | `0` | Passes through the waypoints, `0` = forever |
| `MAX_JOINT_SPEED` | `0.5` | rad/s cap used to time trajectory segments |
| `WAYPOINTS_FILE` | `/app/waypoints.yaml` | Waypoint file |

### Hand following (camera on the end effector)

`CONTROL_MODE=hand` makes the tool follow a hand seen by the OAK4: hand up → tool up,
hand left/right → tool left/right, hand closer/farther → tool backs off/approaches.

```bash
./scripts/deploy.sh sim CONTROL_MODE=hand
```

Open `http://<CAMERA_IP>:8090/` to see the camera view with the detected palm, current
state and offset. Hold your palm open towards the camera ~60 cm away.

How it works:

1. `hand_tracker.py` runs MediaPipe palm detection on the OAK4 and reads the palm distance
   from stereo depth aligned to the colour camera.
2. `follow_controller.py` turns the palm position into a tool offset from the home pose,
   expressed in the frame of a camera mounted on the flange (x = image right, y = image down,
   z = optical axis). It applies a deadband, smoothing, a ±20/15/15 cm workspace box and a
   0.15 m/s speed limit, holds position when the hand is lost and returns home after 5 s.
3. `kinematics.py` solves IK for that pose (orientation kept at home) and the node streams the
   joints to `scaled_joint_trajectory_controller` at 20 Hz.

Modes (`FOLLOW_MODE`):

- `mirror` — for the simulation, where the camera is **not** on the arm. The tool offset is
  proportional to the hand's offset from the image centre / 0.6 m reference distance.
- `servo` — for a camera really mounted on the tool. The arm moves until the hand is back in
  the image centre at the reference distance (closed-loop visual servoing). With a fixed camera
  this would just drift to the workspace limit, so use it only eye-in-hand — or with
  `EYE_IN_HAND_SIM=true` (below).

#### Depth sources: stereo, ToF, fused (demo view)

The tracker measures the palm distance with every depth source the camera has, all aligned
to the RGB image: **stereo** (classic `StereoDepth` by default, see below), **ToF** (OAK4 LITE TOF) and **fused**
(per pixel: ToF up close, stereo far, blended in between, see `depth_fusion.py`).
The selected one drives the arm.

`http://<camera-ip>:8090/` shows two streams with buttons under them:

| | |
|---|---|
| RGB (left, `/rgb` full window) | wider camera view (`view_size`, `[1280, 960]` on the real profile) with the HUD and video labels (`view_labels`, e.g. "NO PC - everything runs on the {device}") |
| Depth (right, `/depth` full window) | selected source, fixed colour range (`depth_view_range_m`), no-depth hatched, palm distance and a live graph of all three sources |
| Buttons / keys | `1` stereo, `2` ToF, `3` fused, `h` HUD on/off, `l` labels on/off |

`DEPTH_SOURCE=stereo|tof|fused` (deploy override) or `depth_source` in the YAML picks the
start source. Demo: select *Stereo*, bring your hand towards the camera. Close in,
stereo shows **NO DEPTH** and the arm stops backing away. Press `3`, and the arm tracks again.
`POST /source/<name>` switches the source from a script.

`stereo_engine` picks the stereo engine. `classic` (default) is the hardware `StereoDepth` node on
`stereo_size` mono images (default 640×400), tuned with `stereo_preset` (default `ROBOTICS`).
It runs cooler than `neural`, the `NeuralDepth` network on the NPU (`neural_depth_model`), which
gives denser depth but heated the camera to ~80 °C chip / 72 °C skin within 15 minutes in our
soak test, with render and control-loop times growing by 50–75 %.

#### Camera on the end effector (eye-in-hand)

The tool keeps your palm centred in the image at `reference_depth` (0.6 m): move the hand
right and the tool follows right until the palm is centred again, step back and the tool
approaches. Orientation stays at the home orientation (translation only).

**Try it in URSim** with the camera still on the desk:

```bash
./scripts/deploy.sh sim UR_TYPE=ur7e HEADLESS_MODE=false CONTROL_MODE=hand \
  FOLLOW_MODE=servo EYE_IN_HAND_SIM=true
```

then press Play in PolyScope. The palm the fixed camera sees is re-projected into a virtual
camera that sits on the simulated tool (same pose as the real camera at home, moved by the
current offset). The web view shows the real detection (green) and what the tool camera
would see (cyan circle); the arm moves until the cyan circle is near the centre cross.

**On the real arm:**

1. Mount the camera on the flange looking along the tool z axis (out of the flange).
2. Find the image rotation vs. `tool0`: at the home pose,
   jog the tool +x in the *Tool* frame on the pendant and see where the scene moves in the
   web view. Image content moving left ⇒ camera x = tool x ⇒ `camera_rotation_deg: 0`;
   otherwise try 90 / 180 / 270 in `oakapp/hand_follow.yaml` until it matches.
3. Check `home_joints_deg` leaves room around the arm (it points the tool horizontally away
   from the base) and keep `workspace` and `max_tcp_speed` small for the first run.
4. `./scripts/deploy.sh real CONTROL_MODE=hand` (`real.env` already sets `FOLLOW_MODE=servo`),
   press Play, keep a hand on the e-stop. Do **not** set `EYE_IN_HAND_SIM` on the real arm.

Tune everything in `oakapp/hand_follow.yaml`. If the camera is mounted rotated on the flange,
set `camera_rotation_deg`.

#### Real UR7e, camera on the end effector (our lab setup)

Camera `192.168.65.172` (OAK4-LITE-TOF), UR7e `192.168.65.242` (no port 29999 → PolyScope X).
`config/real.env` selects `CONTROL_MODE=hand`, `GRIPPER=none`, `oakapp/hand_follow_real.yaml` (servo, `grip_mode: "off"`,
±30 cm, 0.25 m/s, `home_joints_deg: current`, no return-home) and `DRY_RUN=true`.

1. Pendant: External Control URCapX Host IP = `192.168.65.172`, port 50002. Jog the arm to a
   safe start pose with the camera looking into free space. That pose becomes the follow
   home — the app makes **no** initial move.
2. Stop other apps that use the camera (`oakctl app list -d 192.168.65.172`).
3. **Dry run** (never commands the arm): `./scripts/deploy.sh real`, open
   `http://192.168.65.172:8090/`. Hold a palm to the image right: the logged / overlaid
   `base offset` must point from the tool towards your hand. If not, change
   `camera_rotation_deg` (90 / 180 / 270) and redeploy.
4. **Live:** `./scripts/deploy.sh real DRY_RUN=false`, wait for port 50002, press Play on the
   pendant, hand on the e-stop. The arm moves only while a palm is detected and holds when it
   is lost.

#### Grasping what the person holds (real UR7e + RG2)

While following the hand, the arm looks for an object **in** the hand and grasps it
(`object:` / `grasp:` in `oakapp/hand_follow_real.yaml`, code in `ur_oak_control/held_object.py`
and `grasp.py`):

1. **Selection cue** (every frame, no model): next to the detected palm, the pixels at the
   hand's depth (`depth_band_m`) that are not skin coloured form the object. Blobs that run out
   of the search region (sleeve, table) are ignored. Once the blob holds still for
   `select_frames` frames:
2. **MobileSAM** segments it (include point inside the blob, exclude point on the skin); masks
   that are mostly skin or miss the blob are rejected.
3. **Track-Anything (XMem)** follows the mask from then on - the hand may leave the image.
   Both models run on the HTP through ONNX Runtime QNN (`oakapp/track_anything/`, vendored
   from oak-examples `apps/onnxruntime-track-anything`, Apache 2.0). The ~294 MB of models are
   downloaded at image build; the first start compiles them for QNN (a few minutes).
   `object.tracker: cue` (or `OBJECT_TRACKER=cue`) skips the models and uses the cue blob every
   frame; this is also the automatic fallback if QNN does not load.
4. **Grasp:** the arm servos the object to the **grasp target** (where an object between the
   RG2 jaws appears in the camera image, and at what depth), first from `pregrasp_m` farther
   back, then straight in; inside `tol_xy` / `tol_z` for `close_frames` frames the gripper
   closes and the arm holds. `curl -X POST http://192.168.65.172:8090/object/release` opens it;
   objects are then ignored for `regrip_cooldown_s`. Without a grasp target the arm only keeps
   the object centred at `approach_depth_m` and never closes.

The web view shows the cue (grey brackets, green include / red exclude point), the tracked
mask outline (yellow, red when gripped) and the grasp target (magenta). `GET /object` returns
the tracker status, observation and grasp state; `GET /object/mask.jpg` the mask;
`POST /object/clear` drops the target.

**Calibrate the grasp target** (arm does not move, gripper live, auto-grip off):

```bash
./scripts/deploy.sh real DRY_RUN=true GRIPPER_IN_DRY_RUN=true GRASP_AUTO_GRIP=false
```

1. Press Play on the pendant (tool outputs go through the driver).
2. Show the object in your hand until it is tracked (yellow outline), then hold it between
   the open jaws where a good grip would be, still for ~1 s.
3. `curl -X POST http://192.168.65.172:8090/gripper/close` - records the object position just
   before closing (`GET /object` -> `last_mark`).
4. Grip good? `curl -X POST .../object/target/last` uses it live; copy `x`, `y`, `depth_m` into
   `grasp.grasp_target` to keep it. `curl -X POST .../gripper/open` and repeat a few times.

Keep your fingers at the back of the object - the RG2 closes with `force_n`.

**Drop into a box:** with `drop.joints_deg` set, a grasp is followed by: wait `settle_s` ->
joint move through `via_deg` to the drop pose (`joint_speed`) -> open -> back the same way to
the follow home -> follow the hand again. Record the poses by hand-guiding the arm while the
app sends no commands:

```bash
./scripts/deploy.sh real DRY_RUN=true            # driver connected, arm not commanded
# freedrive the arm to a pose on the way (optional), then:
curl -X POST http://192.168.65.172:8090/drop/via/add
# freedrive it to the drop pose above the box, then:
curl -X POST http://192.168.65.172:8090/drop/record
curl -s http://192.168.65.172:8090/arm             # -> copy "drop" into drop: in the YAML
```

The moves are plain joint-space moves: check the path from the grasp area to the box is free
(add via poses if not). `POST /drop/run` starts the sequence by hand while an object is gripped.

### Moving to a real arm

The camera-side app is identical; only `config/real.env` changes.

1. **Network:** give the UR control box a static IP on the camera's LAN. From the camera,
   TCP 29999 (PolyScope 5 only) and 30001–30004 on the robot must be reachable, and the robot must reach the
   camera on TCP 50001–50004.
2. **External Control URCap:** (PolyScope X: see [below](#polyscope-x-ur7e--ur12e--ur15)) install it on the pendant
   ([guide](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/install_urcap_e_series.html)),
   set its **Host IP** to the camera IP and port 50002, and create a program containing the
   *External Control* node (e.g. `external_control.urp`).
3. **Calibration (recommended):** extract the robot's factory calibration with
   `ur_calibration` and mount/copy it into the app; set `KINEMATICS_PARAMS_FILE`.
4. **Profile:** set `ROBOT_IP` and `UR_TYPE` in `config/real.env`. The profile keeps
   `AUTO_POWER_ON=false`, `HEADLESS_MODE=false`, `MOTION_LOOPS=1`, `MAX_JOINT_SPEED=0.2`.
5. **Check the waypoints** against the real workspace, turn the pendant speed slider down, then:

   ```bash
   ./scripts/deploy.sh real
   ```

   and press **Play** on the External Control program on the pendant (or set
   `AUTO_PLAY_PROGRAM`). The driver also works in headless mode on a real e-series arm, but the
   robot must then be in *Remote Control* mode.

#### PolyScope X (UR7e / UR12e / UR15)

Newer arms such as the UR7e can ship with **PolyScope X** instead of the classic PolyScope 5.
The ROS 2 driver supports both; the differences that matter here:

| | PolyScope 5 (e-series, CB3) | PolyScope X |
|---|---|---|
| Dashboard Server (TCP 29999) | yes | **no** (Robot REST API from 10.11 instead, not used by this app) |
| External Control plugin | `.urcap` ([guide](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/install_urcap_e_series.html)) | **URCapX** `external-control-X.Y.Z.urcapx` from [releases](https://github.com/UniversalRobots/Universal_Robots_ExternalControl_URCapX/releases) |
| Ports the camera needs | 29999, 30001–30004 | 30001–30004 |
| `AUTO_POWER_ON`, `AUTO_PLAY_PROGRAM` | work | ignored — use the pendant |

Setup on the pendant:

1. **Settings → Security → Services:** enable **Primary Client Interface** and **RTDE**.
2. **System Manager → URCaps → +URCap** (admin password), install the External Control URCapX
   from a USB stick, reboot.
3. **Application → External Control:** set the host IP to the camera IP (`CAMERA_IP`), keep
   the port at 50002.
4. Create a program with the **External Control** node.
5. Profile: `config/real.env` already has `UR_TYPE=ur7e`, `HEADLESS_MODE=false`.
   `./scripts/deploy.sh real`, wait until the camera listens on 50002, then press **Play** on the
   pendant (use *Update program* on the node if it asks).

`robot_prep` detects PolyScope X by the missing port 29999 and logs
`no Dashboard Server ... assuming PolyScope X`; that is expected. UR7e/UR12e have the same
geometry as UR5e/UR10e (`kinematics.py`); the long-reach UR7e-920 / UR12e-1300 are not covered.
UR flags External Control on PolyScope X as still maturing, so check the controller's
PolyScope X version against the URCapX release notes.

> Safety: the app commands real hardware. Keep the e-stop within reach, verify every waypoint,
> and do not enable `AUTO_POWER_ON` on a real arm unless the cell is guarded.

### Extending

`URArm` in `oakapp/ur_oak_control/arm.py` is the integration point. A perception node using
DepthAI on the same camera can compute targets and call:

```python
arm = URArm(node)
arm.wait_until_ready()
arm.move_joints([target_joints], max_joint_speed=0.3)
```

For Cartesian targets, add MoveIt (`ur_moveit_config`) or use the driver's URScript /
`forward_position_controller` interfaces. Because the app runs with host networking, standard
ROS 2 tooling on the LAN (RViz, `ros2 topic echo`) sees the driver's topics when
`ROS_DOMAIN_ID` matches.

### Notes

- The driver warns that the OAK4 kernel is not a low-latency / real-time kernel. Short
  trajectories work fine; for tight control loops see
  [real-time setup](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_client_library/doc/real_time.html).
- If `apt-get` fails during the build with DNS errors, add a `resolv.conf` to `oakapp/`
  (e.g. `nameserver 8.8.8.8`) and a `{ type = "COPY", source = "resolv.conf", target = "/etc/resolv.conf" }`
  step at the start of `prepare_container`.
- The base image is pulled from Docker Hub by the device; if you hit anonymous pull
  rate limits, run `oakctl registry login https://registry-1.docker.io`.
