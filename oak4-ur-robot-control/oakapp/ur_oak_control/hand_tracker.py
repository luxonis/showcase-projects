"""Palm detection + stereo / ToF depth on the OAK4, running in a background thread.

Produces the most recent HandObservation (normalised image position of the palm
and its distance from the camera) and the frames for the web view.

Depth comes from up to three sources, all aligned to the colour frame:
  stereo   classic StereoDepth (default, cooler) or neural stereo (NeuralDepth)
  tof      the ToF sensor (OAK4 LITE TOF), if the device has one
  fused    stereo + ToF, by `fusion_engine`:
             heuristic  per-pixel rule on the app CPU (depth_fusion.py)
             neural     depthai's ToFStereoFusion node (beta, RVC4): ToF + NeuralDepth through a
                        fusion network; stereo / tof are then its own neural-depth / ToF stages.
                        fusion_tof_near: ToF overrides it up close (heuristic rule, network = "stereo")
             vpp        Vpp node: ToF painted as a virtual pattern onto the rectified stereo pair,
                        then classic StereoDepth (stereo = the same pair without the pattern)
The palm depth of every source is computed each frame (HandObservation.depths); the one
named by `depth_source` becomes depth_m - switchable at run time from the web view.

Grip detection publishes a debounced hand_closed (True = close the gripper, False = open,
None = not known yet). grip_mode:
  twist    wrist twist, from the palm detector's own rotation angle (no extra model):
           |twist from upright| < twist_open_deg -> open, > twist_close_deg -> closed.
           "Upright" is twist_reference_deg, or learned from the first frames the hand is seen.
  fingers  MediaPipe hand landmarker on the rotated hand region, counts extended fingers
           (fist -> closed). A landmark failure only disables grip detection.
  off      no grip detection
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import depthai as dai
import numpy as np
from depthai_nodes.node import ParsingNeuralNetwork

from ur_oak_control.depth_fusion import SOURCES, fuse, palm_depths

MODEL_YAML = "/app/depthai_models/mediapipe_palm_detection.{platform}.yaml"
LM_MODEL_YAML = "/app/depthai_models/mediapipe_hand_landmarker.{platform}.yaml"
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]   # (tip, PIP) of index, middle, ring, pinky
EXTENDED_RATIO = 1.15   # finger extended if |wrist-tip| > ratio * |wrist-PIP|


@dataclass
class HandObservation:
    x: float            # 0..1, left -> right in the image
    y: float            # 0..1, top -> bottom in the image
    depth_m: float      # distance along the optical axis, nan if unknown
    confidence: float
    stamp: float        # time.monotonic()
    depths: dict | None = None   # {source: (depth_m, valid_fraction)} for stereo / tof / fused


DEPTH_STALE_S = 0.5      # a depth map older than this counts as missing
FUSION_ENGINES = ("heuristic", "neural", "vpp")


class HandTracker(threading.Thread):
    def __init__(self, fps: int = 15, min_confidence: float = 0.6, size: int = 640,
                 grip_mode: str = "off", grip_debounce: int = 3,
                 twist_open_deg: float = 30.0, twist_close_deg: float = 60.0,
                 twist_reference_deg: float | None = None, view_size=960,
                 depth_source: str = "stereo", stereo_engine: str = "classic",
                 stereo_preset: str = "ROBOTICS", stereo_size=(640, 400),
                 neural_depth_model: str = "NEURAL_DEPTH_LARGE", tof: str | bool = "auto",
                 tof_profile: str = "MID_RANGE", tof_fps: float | None = None,
                 tof_confidence: float = 0.0, tof_min_m: float = 0.10,
                 fuse_near_m: float = 0.35, fuse_far_m: float = 0.8,
                 fusion_engine: str = "heuristic", neural_fusion_confidence: float = 0.3,
                 fusion_tof_near: bool = True, held=None,
                 vpp_size=(640, 400)):
        super().__init__(daemon=True, name="hand-tracker")
        self.fps = fps
        self.min_confidence = min_confidence
        self.size = size
        self.latest: HandObservation | None = None
        # view_size: int -> square view (same crop as detection); [w, h] -> e.g. full FOV 4:3
        self.view_wh = ((int(view_size), int(view_size)) if isinstance(view_size, (int, float))
                        else (int(view_size[0]), int(view_size[1])))
        self.view_map = (1.0, 0.0, 1.0, 0.0)   # det-normalised -> view-normalised: ax, bx, ay, by
        self.view_frame = None                 # latest high-res BGR frame for the web view
        # depth sources
        if depth_source not in SOURCES:
            raise ValueError(f"depth_source must be one of {SOURCES}, got {depth_source!r}")
        if stereo_engine not in ("neural", "classic"):
            raise ValueError(f"stereo_engine must be neural or classic, got {stereo_engine!r}")
        if not hasattr(dai.node.StereoDepth.PresetMode, stereo_preset):
            raise ValueError(f"unknown stereo_preset {stereo_preset!r}")
        if fusion_engine not in FUSION_ENGINES:
            raise ValueError(f"fusion_engine must be one of {FUSION_ENGINES}, got {fusion_engine!r}")
        self.fusion_engine = fusion_engine     # may fall back to heuristic (no ToF / no RVC4)
        self.neural_fusion_confidence = float(neural_fusion_confidence)
        # neural / vpp: ToF still wins up close (< fuse_near_m), blended into the device-fused
        # depth up to fuse_far_m - the fusion network alone leans on stereo even at hand range
        self.fusion_tof_near = bool(fusion_tof_near)
        self.held = held                       # held_object.HeldObject or None (grasping off)
        self.commands: deque = deque(maxlen=20)  # (kind, arg) from the web view -> control loop
        self.vpp_size = (int(vpp_size[0]), int(vpp_size[1]))
        self.depth_source = depth_source       # switched live from the web view
        self.stereo_engine = stereo_engine
        self.neural_depth_model = neural_depth_model
        self.stereo_preset = stereo_preset
        self.stereo_size = (int(stereo_size[0]), int(stereo_size[1]))
        self.tof_mode = str(tof).lower()       # auto | true | false
        self.tof_profile = tof_profile
        self.tof_fps = tof_fps
        self.tof_confidence = tof_confidence
        self.tof_min_mm = 1000 * tof_min_m
        self._tof_cf = None
        self.fuse_near_mm, self.fuse_far_mm = 1000 * fuse_near_m, 1000 * fuse_far_m
        self.available_sources: list[str] = []
        # uint16 mm, aligned to colour; "fused" only for the on-device engines (neural / vpp)
        self.depth_frames: dict = {"stereo": None, "tof": None, "fused": None}
        self._depth_stamps: dict = {"stereo": 0.0, "tof": 0.0, "fused": 0.0}
        self.history: deque = deque(maxlen=600)  # (t, {source: depth_m}) for the depth graph
        self.device_name = "OAK4"
        self._ticks = {k: deque(maxlen=400) for k in ("det", "stereo", "tof", "fused")}   # /debug/perf rates
        self.ctrl_perf: dict = {}              # control loop timing, set by hand_follow
        self.sysinfo: dict | None = None       # chip temperature / CPU load (SystemLogger, 1 Hz)
        self.sysinfo_stamp = 0.0
        self.show_hud = True                   # web view toggles (clean shots for video)
        self.show_labels = True
        self.det: dict | None = None           # latest detection snapshot for the web view
        self.hud: dict = {}                    # controller state for the web view
        self.status_lines: list[str] = []   # drawn on the web view, set by the controller
        self.virtual_point: tuple[float, float] | None = None  # eye-in-hand sim marker (normalised)
        self.focal_norm: tuple[float, float] = (0.8, 0.8)      # fx, fy / image size; set from calibration
        self.error: str | None = None
        # grip detection (hand landmarks)
        if grip_mode not in ("off", "twist", "fingers"):
            raise ValueError(f"grip_mode must be off, twist or fingers, got {grip_mode!r}")
        self.grip_mode = grip_mode
        self.detect_grip = grip_mode == "fingers"   # landmark model needed
        self.grip_debounce = grip_debounce
        self.grip_detail = ""                  # e.g. "twist +12 deg" / "4 fingers", web view
        self.twist_open_deg = twist_open_deg
        self.twist_close_deg = twist_close_deg
        self.twist_reference = twist_reference_deg
        self.twist: float | None = None
        self._ang_vec: np.ndarray | None = None
        self._calib: list[float] = []
        self.hand_closed: bool | None = None   # debounced: True fist, False open
        self.extended: int | None = None       # extended fingers in the last landmark result
        self.grip_status = "off" if grip_mode == "off" else "ready"   # shown in the web view
        self._raw_closed: bool | None = None
        self._raw_count = 0
        self._best_rect = None
        self._lm_pending_since: float | None = None
        self._lm_m_inv = None
        self._lm_points = None                 # landmarks in normalised image coords (drawing)
        self.ready = threading.Event()
        self.pipeline_up = threading.Event()   # set once the pipeline has run a few seconds
        self.restarts = 0
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    MAX_RESTARTS = 5

    def run(self) -> None:
        """Run the pipeline; if the device side crashes or the pipeline stops, reconnect and
        start it again (the arm holds meanwhile: observations go stale). Only after
        MAX_RESTARTS failures in a row does the error reach the controller."""
        failures = 0
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._run()
                reason = "pipeline stopped"
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            try:
                dev, self._device = getattr(self, "_device", None), None
                if dev is not None:
                    dev.close()
            except Exception:
                pass
            if self._stop_event.is_set():
                return
            failures = 0 if time.monotonic() - t0 > 60 else failures + 1
            if failures > self.MAX_RESTARTS:
                self.error = reason          # surfaced to the controller, which stops the arm
                self.ready.set()
                return
            self.restarts += 1
            print(f"[hand_tracker] {reason} - restarting the camera pipeline in 3 s "
                  f"({failures}/{self.MAX_RESTARTS})", flush=True)
            self.latest, self.det = None, None
            self._stop_event.wait(3.0)

    def _run(self) -> None:
        device = dai.Device()
        self._device = device                 # closed by run() before a restart
        platform = device.getPlatform().name
        try:
            self.device_name = device.getProductName() or device.getDeviceName() or "OAK4"
        except Exception:
            pass
        try:
            k = device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, self.size, self.size)
            self.focal_norm = (k[0][0] / self.size, k[1][1] / self.size)
        except Exception as exc:  # uncalibrated device: keep the rough default
            print(f"[hand_tracker] no colour intrinsics ({exc}), using focal {self.focal_norm}", flush=True)
        features = device.getConnectedCameraFeatures()
        print("[hand_tracker] cameras: " + ", ".join(
            f"{f.socket.name} {f.sensorName} {[t.name for t in f.supportedTypes]}" for f in features), flush=True)
        tof_socket = next((f.socket for f in features if dai.CameraSensorType.TOF in f.supportedTypes), None)
        sockets = {f.socket for f in features}
        has_stereo = {dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C} <= sockets \
            and tof_socket not in (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
        use_tof = tof_socket is not None and self.tof_mode != "false"
        if self.tof_mode == "true" and tof_socket is None:
            raise RuntimeError("tof: true but the device has no ToF sensor")

        archive = dai.NNArchive(dai.getModelFromZoo(
            dai.NNModelDescription.fromYamlFile(MODEL_YAML.format(platform=platform))))
        frame_type = dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i

        with dai.Pipeline(device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            color = cam.requestOutput((self.size, self.size), frame_type, fps=self.fps)
            # higher resolution (optionally wider) view of the same camera, only for the web view
            view = (cam.requestOutput(self.view_wh, frame_type, fps=self.fps)
                    if self.view_wh != (self.size, self.size) else None)

            resize = pipeline.create(dai.node.ImageManip)
            w, h = archive.getInputWidth(), archive.getInputHeight()
            resize.setMaxOutputFrameSize(w * h * 3)
            resize.initialConfig.setOutputSize(w, h, mode=dai.ImageManipConfig.ResizeMode.STRETCH)
            resize.initialConfig.setFrameType(frame_type)
            color.link(resize.inputImage)
            palm_nn = pipeline.create(ParsingNeuralNetwork).build(resize.out, archive)

            q_depth = {}
            q_tof_raw = None
            if self.fusion_engine != "heuristic" and not (has_stereo and use_tof and platform == "RVC4"):
                print(f"[hand_tracker] fusion_engine {self.fusion_engine} needs stereo + ToF on RVC4 "
                      f"- using heuristic fusion", flush=True)
                self.fusion_engine = "heuristic"
            if self.fusion_engine == "neural":
                outs, tof_raw = self._build_neural_fusion(pipeline, color)
            elif self.fusion_engine == "vpp":
                outs, tof_raw = self._build_vpp(pipeline, tof_socket, color)
            else:
                outs, tof_raw = {}, None
                if has_stereo:
                    outs["stereo"] = self._build_stereo(pipeline, platform, color)
                else:
                    print("[hand_tracker] no stereo pair (CAM_B + CAM_C) - stereo depth off", flush=True)
                if use_tof:
                    outs["tof"], tof_raw = self._build_tof(pipeline, tof_socket, color)
                else:
                    print("[hand_tracker] no ToF sensor (or tof: false) - ToF depth off", flush=True)
            for name, out in outs.items():
                q_depth[name] = out.createOutputQueue(maxSize=1, blocking=False)
            if tof_raw is not None:
                q_tof_raw = tof_raw.createOutputQueue(maxSize=1, blocking=False)   # /debug/depth only
            self.available_sources = [s for s in ("stereo", "tof") if s in q_depth]
            if len(self.available_sources) == 2:
                self.available_sources.append("fused")
            if self.available_sources and self.depth_source not in self.available_sources:
                print(f"[hand_tracker] depth source {self.depth_source} not available, "
                      f"using {self.available_sources[0]}", flush=True)
                self.depth_source = self.available_sources[0]

            lm = self._build_landmarker(pipeline, platform) if self.detect_grip else None

            q_sys = None
            try:   # chip temperature / CPU load for the web view (1 Hz)
                syslog = pipeline.create(dai.node.SystemLogger)
                syslog.setRate(1.0)
                q_sys = syslog.out.createOutputQueue(maxSize=1, blocking=False)
            except Exception as exc:
                print(f"[hand_tracker] no system logger ({exc}) - no temperature in the view", flush=True)

            q_det = palm_nn.out.createOutputQueue(maxSize=1, blocking=False)
            q_color = color.createOutputQueue(maxSize=1, blocking=False)
            q_view = view.createOutputQueue(maxSize=1, blocking=False) if view is not None else None

            pipeline.start()
            t_started = time.monotonic()
            print(f"[hand_tracker] pipeline running on {platform} ({self.device_name}) at {self.fps} fps, "
                  f"depth sources {self.available_sources} (fusion: {self.fusion_engine}), "
                  f"using {self.depth_source}", flush=True)
            self.ready.set()

            color_frame = None
            color_msg = None
            mapped = view is None
            while pipeline.isRunning() and not self._stop_event.is_set():
                det_msg = q_det.get()
                now = time.monotonic()
                if not self.pipeline_up.is_set() and now - t_started > 5.0:
                    self.pipeline_up.set()       # e.g. held-object models may load on the NPU now
                self._ticks["det"].append(now)
                for name, q in q_depth.items():
                    d = q.tryGet()
                    if d is not None:
                        f = d.getFrame()
                        if name == "tof" and self.tof_min_mm > 0:
                            # below ~10 cm the VD55H1 mostly reports noise (dark surfaces, the desk
                            # right under the lens) - drop it so it never reads as "very close"
                            f = np.where(f < self.tof_min_mm, 0, f).astype(np.uint16)
                        self.depth_frames[name] = f
                        self._depth_stamps[name] = now
                        self._ticks[name].append(now)
                if q_sys is not None:
                    m = q_sys.tryGet()
                    if m is not None:
                        self._sysinfo_update(m)
                if q_tof_raw is not None:
                    r = q_tof_raw.tryGet()
                    if r is not None:
                        self.depth_frames["tof_raw"] = r.getFrame()
                c = q_color.tryGet()
                new_color = c is not None
                if c is not None:
                    color_msg, color_frame = c, c.getCvFrame()
                if q_view is not None:
                    v = q_view.tryGet()
                    if v is not None:
                        self.view_frame = v.getCvFrame()
                        if not mapped and color_msg is not None:
                            self._map_view(color_msg, v)
                            mapped = True
                elif color_frame is not None:
                    self.view_frame = color_frame
                if self.held is None or self.held.hand_needed:
                    self._process(det_msg)
                elif self.det is not None:       # object tracked by XMem: hand ignored, skip its work
                    self.det, self.latest = None, None
                if self.held is not None and new_color:
                    try:
                        det = self.det
                        fresh = det is not None and now - det["stamp"] < 0.3
                        palm = det["rect"][:4] if fresh else None
                        self.held.process(color_frame, now, palm,
                                          det["depth"] if fresh else float("nan"), self.depth_crop)
                    except Exception as exc:   # never let object grasping stop hand following
                        import traceback
                        print(f"[hand_tracker] held object failed, grasping off: "
                              f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", flush=True)
                        self.held = None
                if lm is not None:
                    try:
                        self._landmarks(lm, frame_type, color_frame)
                    except Exception as exc:  # never let grip detection stop following
                        print(f"[hand_tracker] hand landmarks failed, grip detection off: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                        lm, self.hand_closed, self.grip_status = None, None, "error"

    def _build_stereo(self, pipeline, platform, color):
        """Stereo depth aligned to the colour detection frame."""
        if self.stereo_engine == "neural":
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=self.fps)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=self.fps)
            stereo = pipeline.create(dai.node.NeuralDepth).build(
                left.requestFullResolutionOutput(), right.requestFullResolutionOutput(),
                getattr(dai.DeviceModelZoo, self.neural_depth_model))
            print(f"[hand_tracker] stereo: NeuralDepth {self.neural_depth_model}", flush=True)
        else:
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth).build(
                left.requestOutput(self.stereo_size, fps=self.fps),
                right.requestOutput(self.stereo_size, fps=self.fps),
                presetMode=getattr(dai.node.StereoDepth.PresetMode, self.stereo_preset),
            )
            print(f"[hand_tracker] stereo: classic StereoDepth {self.stereo_preset} "
                  f"{self.stereo_size[0]}x{self.stereo_size[1]}", flush=True)
            if platform != "RVC4":
                color.link(stereo.inputAlignTo)
                return stereo.depth
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        color.link(align.inputAlignTo)
        return align.outputAligned

    def _align_to_color(self, pipeline, out, color, on_host: bool):
        align = pipeline.create(dai.node.ImageAlign)
        if on_host:
            align.setRunOnHost(True)
        out.link(align.input)
        color.link(align.inputAlignTo)
        return align.outputAligned

    def _build_neural_fusion(self, pipeline, color):
        """depthai's ToFStereoFusion (beta, RVC4; examples/python/Beta/tof_neural_fusion.py).
        It owns the ToF + NeuralDepth (MEDIUM) stages; their depth is reused as the tof / stereo
        sources so the web view still compares all three. Everything aligned to colour."""
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=self.fps)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=self.fps)
        fusion = pipeline.create(dai.beta.node.ToFStereoFusion).build(left, right, fps=float(self.fps))
        fusion.initialConfig.confidenceThreshold = self.neural_fusion_confidence
        fusion.initialConfig.setCropToOverlap(False)   # full ToF frame; ImageAlign maps it to colour
        self.stereo_engine, self.neural_depth_model = "neural", "NEURAL_DEPTH_MEDIUM"
        self.tof_profile = "MID_RANGE (fusion)"
        print(f"[hand_tracker] fusion: ToFStereoFusion (ToF MID_RANGE + NeuralDepth MEDIUM), "
              f"confidence >= {self.neural_fusion_confidence}", flush=True)
        outs = {"stereo": self._align_to_color(pipeline, fusion.neuralDepth.depth, color, on_host=False),
                "tof": self._align_to_color(pipeline, fusion.tof.depth, color, on_host=True),
                "fused": self._align_to_color(pipeline, fusion.depth, color, on_host=True)}
        return outs, fusion.tof.depth

    def _build_vpp(self, pipeline, socket, color):
        """ToF-guided stereo (examples/python/Vpp/tof_stereo_vpp.py): ToF depth, aligned to the
        rectified left image, is painted as a virtual pattern onto both images (Vpp), then
        classic StereoDepth. A second StereoDepth on the plain pair is the stereo source."""
        size = self.vpp_size
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        rect = pipeline.create(dai.node.Rectification)
        rect.setRunOnHost(False)
        rect.setOutputSize(*size)
        for cam, inp in ((left, rect.input1), (right, rect.input2)):
            cam.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=self.fps,
                              enableUndistortion=False).link(inp)
        tof = pipeline.create(dai.node.ToF).build(
            boardSocket=socket, profile=getattr(dai.ToFConfig.Profile, self.tof_profile), fps=float(self.fps))
        tof.setOutputUndistortion(False)
        tof_on_rect = pipeline.create(dai.node.ImageAlign)
        tof_on_rect.setRunOnHost(False)
        tof.tofBaseNode.depth.link(tof_on_rect.input)
        rect.output1.link(tof_on_rect.inputAlignTo)

        vpp = pipeline.create(dai.node.Vpp)
        rect.output1.link(vpp.left)
        rect.output2.link(vpp.right)
        tof_on_rect.outputAligned.link(vpp.depth)
        c = vpp.initialConfig
        c.blending = 0.6
        c.distanceGamma = 0.3
        c.maxPatchSize = 3
        c.patchColoringType = dai.VppConfig.PatchColoringType.RANDOM
        c.uniformPatch = False
        c.maxFPS = int(math.ceil(self.fps))
        c.maxNumThreads = 1
        c.injectionParameters.useInjection = False
        c.injectionParameters.textureThreshold = 10.0

        def stereo_node(l_out, r_out):
            st = pipeline.create(dai.node.StereoDepth)
            st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
            st.setExtendedDisparity(True)
            st.setRectification(False)
            l_out.link(st.left)
            r_out.link(st.right)
            return st.depth

        fused = stereo_node(vpp.leftOut, vpp.rightOut)
        plain = stereo_node(rect.output1, rect.output2)
        self.stereo_engine = "classic"
        print(f"[hand_tracker] fusion: VPP (ToF {self.tof_profile} on {socket.name} -> virtual pattern "
              f"-> StereoDepth FAST_ACCURACY) {size[0]}x{size[1]}", flush=True)
        outs = {"stereo": self._align_to_color(pipeline, plain, color, on_host=False),
                "tof": self._align_to_color(pipeline, tof.depth, color, on_host=True),
                "fused": self._align_to_color(pipeline, fused, color, on_host=False)}
        return outs, tof.depth

    def _build_tof(self, pipeline, socket, color):
        """ToF depth aligned to the colour detection frame (alignment on the app CPU, as in
        the depthai ToF align example)."""
        tof = pipeline.create(dai.node.ToF).build(
            boardSocket=socket, profile=getattr(dai.ToFConfig.Profile, self.tof_profile),
            fps=float(self.tof_fps) if self.tof_fps else None)
        depth = tof.depth
        if self.tof_confidence > 0:
            # drop low-amplitude pixels: without this the VD55H1 reports lots of 40-80 mm noise
            cf = pipeline.create(dai.node.ToFDepthConfidenceFilter)
            cf.setRunOnHost(True)
            cf.initialConfig.confidenceThreshold = float(self.tof_confidence)
            tof.depth.link(cf.depth)
            tof.amplitude.link(cf.amplitude)
            self._tof_cf = (cf.initialConfig, cf.inputConfig.createInputQueue())
            depth = cf.filteredDepth
        align = pipeline.create(dai.node.ImageAlign)
        align.setRunOnHost(True)
        depth.link(align.input)
        color.link(align.inputAlignTo)
        print(f"[hand_tracker] ToF on {socket.name}, profile {self.tof_profile}, "
              f"confidence >= {self.tof_confidence}", flush=True)
        return align.outputAligned, tof.depth

    def _sysinfo_update(self, m) -> None:
        """SystemInformation(RVC4) -> self.sysinfo: temp_c (chip average), cpu_temp_c, cpu_pct, mem_pct."""
        try:
            t = getattr(m, "chipTemperature", None)
            info = {"temp_c": float(getattr(t, "average", float("nan"))) if t is not None else float("nan"),
                    "cpu_temp_c": float(getattr(t, "cpuss", getattr(t, "css", float("nan"))))
                    if t is not None else float("nan")}
            cpu = getattr(m, "cpuAvgUsage", None)
            if cpu is None:
                cpu = getattr(getattr(m, "leonCssCpuUsage", None), "average", None)
            if cpu is not None:
                cpu = float(getattr(cpu, "average", cpu))
                info["cpu_pct"] = cpu * 100.0 if cpu <= 1.0 else cpu
            mem = getattr(m, "ddrMemoryUsage", None)
            if mem is not None and getattr(mem, "total", 0):
                info["mem_pct"] = 100.0 * mem.used / mem.total
            self.sysinfo = info
            self.sysinfo_stamp = time.monotonic()
        except Exception as exc:  # never let telemetry break tracking
            if self.sysinfo is None:
                print(f"[hand_tracker] system info not readable: {exc}", flush=True)
            self.sysinfo = {}

    def rate(self, key: str, window: float = 5.0) -> float:
        """Events per second over the last `window` s (det = palm NN results, stereo / tof = depth maps)."""
        now = time.monotonic()
        n = sum(1 for t in list(self._ticks.get(key, ())) if now - t <= window)
        return n / window

    def set_tof_confidence(self, value: float) -> bool:
        """Live retune of the ToF confidence threshold (POST /tune/tof_confidence/<v>)."""
        if self._tof_cf is None:
            return False
        cfg, q = self._tof_cf
        cfg.confidenceThreshold = float(value)
        q.send(cfg)
        self.tof_confidence = float(value)
        print(f"[hand_tracker] ToF confidence threshold -> {value}", flush=True)
        return True

    def depth_stats(self) -> dict:
        """Numbers behind the depth maps (GET /debug/depth): dtype, size, coverage, percentiles."""
        out = {}
        for name, d in list(self.depth_frames.items()):
            if d is None:
                out[name] = None
                continue
            nz = d[d > 0]
            pct = np.percentile(nz, [1, 5, 25, 50, 75, 95, 99]).round(1).tolist() if nz.size else []
            out[name] = {"dtype": str(d.dtype), "shape": list(d.shape), "nonzero": round(nz.size / d.size, 4),
                         "max": float(d.max()), "pct_1_5_25_50_75_95_99": pct}
        # ToF vs stereo alignment check: share of pixels (both 0.2-2.5 m) agreeing within 10 %,
        # as aligned and with ToF mirrored / flipped - the aligned one must clearly win
        st, tf = self.depth_frames.get("stereo"), self.depth_frames.get("tof")
        if st is not None and tf is not None and st.shape == tf.shape:
            agree = {}
            for name, t in (("aligned", tf), ("mirror_x", tf[:, ::-1]), ("flip_y", tf[::-1, :]),
                            ("rot180", tf[::-1, ::-1])):
                ok = (st > 200) & (st < 2500) & (t > 200) & (t < 2500)
                n = int(ok.sum())
                agree[name] = {"pixels": n, "agree_10pct": round(float(
                    (np.abs(st[ok].astype(np.float32) - t[ok]) < 0.1 * st[ok]).mean()), 3) if n else None}
            out["tof_vs_stereo"] = agree
        return out

    def _map_view(self, color_msg, view_msg) -> None:
        """Linear map detection-frame (normalised) -> view-frame (normalised) coordinates, from
        the frames' own transformations (handles a wider view crop than the square detection)."""
        try:
            tc, tv = color_msg.getTransformation(), view_msg.getTransformation()
            (cw, ch), (vw, vh) = tc.getSize(), tv.getSize()
            p = [tc.remapPointTo(tv, dai.Point2f(nx * cw, ny * ch)) for nx, ny in ((0.25, 0.25), (0.75, 0.75))]
            ax = (p[1].x - p[0].x) / vw / 0.5
            ay = (p[1].y - p[0].y) / vh / 0.5
            bx, by = p[0].x / vw - 0.25 * ax, p[0].y / vh - 0.25 * ay
            if not (0.05 < ax < 20 and 0.05 < ay < 20):
                raise ValueError(f"implausible scale {ax:.3f}, {ay:.3f}")
            self.view_map = (ax, bx, ay, by)
            print(f"[hand_tracker] view {vw}x{vh}: detection -> view map "
                  f"x*{ax:.3f}{bx:+.3f}, y*{ay:.3f}{by:+.3f}", flush=True)
        except Exception as exc:
            print(f"[hand_tracker] view mapping failed ({exc}) - assuming the same crop", flush=True)

    def to_view(self, x: float, y: float) -> tuple[float, float]:
        ax, bx, ay, by = self.view_map
        return ax * x + bx, ay * y + by

    def fresh_depth(self, name: str):
        """Latest depth map of a sensor, None if missing or stale."""
        if time.monotonic() - self._depth_stamps.get(name, 0.0) > DEPTH_STALE_S:
            return None
        return self.depth_frames.get(name)

    def depth_crop(self, box_px):
        """Selected depth source (uint16 mm) over a pixel box (x0, y0, x1, y1) of the colour
        detection frame; None if that source has no fresh map."""
        x0, y0, x1, y1 = box_px
        src = self.depth_source

        def crop(name):
            d = self.fresh_depth(name)
            if d is None:
                return None
            if d.shape[:2] != (self.size, self.size):   # maps are aligned to the colour frame
                d = cv2.resize(d, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            return d[y0:y1, x0:x1]

        if src != "fused":
            return crop(src)
        st, tf = crop("stereo"), crop("tof")
        if self.fusion_engine == "heuristic":
            return fuse(st, tf, self.fuse_near_mm, self.fuse_far_mm)[0]
        fu = crop("fused")
        if fu is None or not self.fusion_tof_near:
            return fu
        return fuse(fu, tf, self.fuse_near_mm, self.fuse_far_mm, judge_by_tof=True)[0]

    def box_depth(self, box) -> float:
        """Depth (m, nan if unknown) of the selected source over a normalised box."""
        depths = palm_depths(self.fresh_depth("stereo"), self.fresh_depth("tof"), box,
                             self.fuse_near_mm, self.fuse_far_mm,
                             fused=self.fresh_depth("fused") if self.fusion_engine != "heuristic" else None,
                             tof_near=self.fusion_tof_near)
        src = self.depth_source
        return depths[src][0] if src in self.available_sources else float("nan")

    def set_depth_source(self, name: str) -> bool:
        if name not in self.available_sources:
            return False
        if name != self.depth_source:
            print(f"[hand_tracker] depth source -> {name}", flush=True)
        self.depth_source = name
        return True

    def _process(self, det_msg) -> None:
        best = None
        for det in det_msg.detections:
            if det.confidence >= self.min_confidence and (best is None or det.confidence > best.confidence):
                best = det

        box = None
        self._best_rect = None
        now = time.monotonic()
        if best is None and self.grip_mode == "twist":
            self._ang_vec, self._raw_closed, self._raw_count = None, None, 0
        if best is None:
            self.history.append((now, None))
        if best is not None:
            rect = best.getBoundingBox()
            self._best_rect = rect
            if self.grip_mode == "twist":
                self._twist_update(float(rect.angle))
            cx, cy = rect.center.x, rect.center.y
            bw, bh = rect.size.width, rect.size.height
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            depths = palm_depths(self.fresh_depth("stereo"), self.fresh_depth("tof"), box,
                                 self.fuse_near_mm, self.fuse_far_mm,
                                 fused=self.fresh_depth("fused") if self.fusion_engine != "heuristic" else None,
                                 tof_near=self.fusion_tof_near)
            src = self.depth_source
            depth_m = depths[src][0] if src in self.available_sources else float("nan")
            self.latest = HandObservation(cx, cy, depth_m, float(best.confidence), now, depths)
            self.history.append((now, {s: depths[s][0] for s in self.available_sources}))
            self.det = {"rect": (cx, cy, bw, bh, float(rect.angle)), "x": cx, "y": cy,
                        "depth": depth_m, "depths": depths, "source": src,
                        "conf": float(best.confidence), "stamp": now}

    # Grip detection ----------------------------------------------------
    def _build_landmarker(self, pipeline, platform):
        try:
            archive = dai.NNArchive(dai.getModelFromZoo(
                dai.NNModelDescription.fromYamlFile(LM_MODEL_YAML.format(platform=platform))))
            nn = pipeline.create(dai.node.NeuralNetwork)
            nn.setNNArchive(archive)
            q_in = nn.input.createInputQueue()
            q_out = nn.out.createOutputQueue(maxSize=1, blocking=False)
            self.grip_status = "ready"
            print("[hand_tracker] hand landmarker loaded - grip detection on", flush=True)
            return q_in, q_out, archive.getInputWidth(), archive.getInputHeight()
        except Exception as exc:
            print(f"[hand_tracker] hand landmarker unavailable, grip detection off: {exc}", flush=True)
            self.grip_status = "unavailable"
            return None

    def _landmarks(self, lm, frame_type, color_frame) -> None:
        q_in, q_out, w, h = lm
        res = q_out.tryGet()
        if res is not None:
            self._lm_pending_since = None
            self._classify(res, w, h)
        if self._best_rect is None:
            self._raw_closed, self._raw_count, self._lm_points = None, 0, None
            return
        pending = self._lm_pending_since
        if color_frame is None or (pending is not None and time.monotonic() - pending < 0.5):
            return   # one request in flight at a time
        crop, m_inv = self._crop(color_frame, self._best_rect, w, h)
        img = dai.ImgFrame()
        img.setCvFrame(crop, frame_type)
        q_in.send(img)
        self._lm_m_inv = (m_inv, color_frame.shape[1], color_frame.shape[0])
        self._lm_pending_since = time.monotonic()

    @staticmethod
    def _crop(frame, rect, w: int, h: int):
        """Upright w x h crop of the rotated hand region (MediaPipe landmarker input)."""
        fh, fw = frame.shape[:2]
        c = np.array([rect.center.x * fw, rect.center.y * fh])
        rw, rh = rect.size.width * fw, rect.size.height * fh
        a = math.radians(rect.angle)
        ux = np.array([math.cos(a), math.sin(a)])     # crop x axis in the image
        uy = np.array([-math.sin(a), math.cos(a)])    # crop y axis (down)
        src = np.float32([c - ux * rw / 2 - uy * rh / 2, c + ux * rw / 2 - uy * rh / 2,
                          c + ux * rw / 2 + uy * rh / 2])
        dst = np.float32([[0, 0], [w, 0], [w, h]])
        m = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(frame, m, (w, h)), cv2.invertAffineTransform(m)

    def _debounced(self, raw: bool | None) -> None:
        """Switch hand_closed only after grip_debounce equal results in a row."""
        if raw is None:
            return
        if raw == self._raw_closed:
            self._raw_count += 1
        else:
            self._raw_closed, self._raw_count = raw, 1
        if self._raw_count >= self.grip_debounce and self.hand_closed != raw:
            self.hand_closed = raw
        self.grip_status = ("closed" if self.hand_closed else "open") if self.hand_closed is not None else "?"

    def _twist_update(self, angle_deg: float) -> None:
        a = math.radians(angle_deg)
        v = np.array([math.cos(a), math.sin(a)])
        if self._ang_vec is None:
            self._ang_vec = v
        else:   # smooth on the unit circle (no wrap-around problems)
            m = self._ang_vec + 0.5 * (v - self._ang_vec)
            n = np.linalg.norm(m)
            self._ang_vec = m / n if n > 1e-6 else v
        ang = math.degrees(math.atan2(self._ang_vec[1], self._ang_vec[0]))
        if self.twist_reference is None:
            self._calib.append(ang)
            self.grip_status, self.grip_detail = "calibrating", f" (show open upright hand) {len(self._calib)}/10"
            if len(self._calib) >= 10:
                r = np.radians(self._calib)
                self.twist_reference = math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean()))
                print(f"[hand_tracker] wrist twist reference = {self.twist_reference:.1f} deg "
                      f"(pin it with twist_reference_deg in the YAML)", flush=True)
                self.grip_status, self.grip_detail = "?", ""
            return
        twist = (ang - self.twist_reference + 180.0) % 360.0 - 180.0
        self.twist = twist
        self.grip_detail = f" (twist {twist:+.0f} deg)"
        mag = abs(twist)
        self._debounced(True if mag > self.twist_close_deg else (False if mag < self.twist_open_deg else None))

    def _classify(self, nn_data, w: int, h: int) -> None:
        tensors = {n: np.array(nn_data.getTensor(n, dequantize=True), dtype=np.float32).reshape(-1)
                   for n in nn_data.getAllLayerNames()}
        cands = [t for t in tensors.values() if t.size == 63]
        if not cands:
            raise RuntimeError(f"no 21x3 landmark tensor in {list(tensors)}")
        # image landmarks (pixels or 0..1), not the world ones (metres, |v| < ~0.2)
        pts = max(cands, key=lambda t: float(np.abs(t).max())).reshape(21, 3)[:, :2].copy()
        presence = next((float(t[0]) for n, t in tensors.items()
                         if t.size == 1 and (n == "Identity_1" or "presence" in n.lower()
                                             or "flag" in n.lower())), None)
        if presence is not None and not 0.0 <= presence <= 1.0:
            presence = 1.0 / (1.0 + math.exp(-presence))
        if presence is not None and presence < 0.5:
            self._lm_points = None
            return
        if np.abs(pts).max() <= 1.5:      # normalised -> crop pixels
            pts *= np.array([w, h])
        if self._lm_m_inv is not None:
            m_inv, fw, fh = self._lm_m_inv
            img_pts = pts @ m_inv[:, :2].T + m_inv[:, 2]
            self._lm_points = img_pts / np.array([fw, fh])

        wrist = pts[0]
        ext = sum(np.linalg.norm(pts[tip] - wrist) > EXTENDED_RATIO * np.linalg.norm(pts[pip] - wrist)
                  for tip, pip in FINGERS)
        self.extended = int(ext)
        self.grip_detail = f" ({ext} fingers)"
        self._debounced(True if ext <= 1 else (False if ext >= 3 else None))   # 2 fingers: ambiguous
