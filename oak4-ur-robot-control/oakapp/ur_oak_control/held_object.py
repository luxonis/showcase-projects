"""Find and track the object a person holds, for the arm to grasp.

Selection cue (no model, every camera frame while nothing is tracked)
  The held object is the only thing next to the detected palm (within region_scale x the palm
  box) at the hand's distance (+- depth_band_m) whose colour is not skin (YCrCb skin range).
  Compact blobs only: a component that runs out of the search region (sleeve, table, body)
  is ignored. The blob must stay put for select_frames frames.

Tracking (`tracker` in the YAML)
  xmem  The cue blob prompts MobileSAM (include point inside the blob, exclude point on the
        skin) on that frame; the mask is checked (size, little skin, overlaps the cue) and
        Track-Anything / XMem follows it from then on - the hand may leave the picture.
        Both models run on the HTP through ONNX Runtime QNN (track_anything/, vendored from
        oak-examples apps/onnxruntime-track-anything). If they cannot load, falls back to cue.
  cue   The cue blob itself, every frame (needs the palm in view the whole time).

The result is an ObjectObservation (mask centroid, box, median depth over the mask) in the
colour detection frame - the same frame the depth maps are aligned to.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from ur_oak_control.grasp import Grasp, GraspConfig, ObjectObservation, target_dict

XMEM_W, XMEM_H = 576, 320           # Track-Anything input; the square frame is letterboxed
PAD_BGR = (104, 116, 124)           # ImageNet mean, as MobileSAM's padding


@dataclass
class HeldObjectConfig:
    enabled: bool = True
    tracker: str = "xmem"               # xmem | cue
    region_scale: float = 2.6           # search square = this x the palm box (longer side)
    depth_band_m: float = 0.15          # object depth within this of the hand depth
    skin_cr: tuple = (133, 180)         # generic YCrCb skin range (until / outside the learned one)
    skin_cb: tuple = (77, 135)
    skin_learn_frames: int = 30         # learn YOUR skin colour from the empty palm, then freeze
    skin_tol: float = 12.0              # learned skin = generic & within this of the learned Cr / Cb
    skin_ref: tuple | None = None       # [Cr, Cb] to start with a known skin colour (no learning)
    min_area_rel: float = 0.10          # blob area vs palm box area
    max_area_rel: float = 4.0
    select_frames: int = 5              # cue stable this many frames -> select (MobileSAM)
    select_jitter: float = 0.06         # max cue centroid motion between those frames (normalised)
    min_mask_px: int = 150              # tracked mask smaller than this (640 px frame) = lost
    lost_frames: int = 8                # consecutive lost frames -> drop the target
    max_skin_overlap: float = 0.35      # MobileSAM mask: at most this share of skin pixels
    min_cue_overlap: float = 0.25       # ... and must cover this share of the cue blob

    @classmethod
    def from_dict(cls, d: dict | None) -> "HeldObjectConfig":
        d = dict(d or {})
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.skin_cr, cfg.skin_cb = tuple(cfg.skin_cr), tuple(cfg.skin_cb)
        if cfg.skin_ref is not None:
            cfg.skin_ref = (float(cfg.skin_ref[0]), float(cfg.skin_ref[1]))
        if cfg.tracker not in ("xmem", "cue"):
            raise ValueError(f"object.tracker must be xmem or cue, got {cfg.tracker!r}")
        return cfg


def skin_mask(bgr: np.ndarray, cfg: HeldObjectConfig, ref=None) -> np.ndarray:
    """Skin pixels: the generic YCrCb range, narrowed to +-skin_tol around ref = (Cr, Cb) - the
    learned colour of this person's palm - once known (a red / orange object is often inside the
    generic range but not near the real skin colour)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lo_cr, hi_cr, lo_cb, hi_cb = cfg.skin_cr[0], cfg.skin_cr[1], cfg.skin_cb[0], cfg.skin_cb[1]
    if ref is not None:
        t = cfg.skin_tol
        lo_cr, hi_cr = max(lo_cr, ref[0] - t), min(hi_cr, ref[0] + t)
        lo_cb, hi_cb = max(lo_cb, ref[1] - t), min(hi_cb, ref[1] + t)
    m = cv2.inRange(ycc, (0, int(lo_cr), int(lo_cb)), (255, int(hi_cr), int(hi_cb)))
    return m > 0


def palm_skin_sample(bgr: np.ndarray, depth_mm, palm, hand_depth_m: float, cfg) -> tuple | None:
    """Median (Cr, Cb) of the generic-skin pixels in the inner half of the palm box at hand depth."""
    fh, fw = bgr.shape[:2]
    cx, cy, pw, ph = palm
    x0, x1 = int(max(0, (cx - pw / 4) * fw)), int(min(fw, (cx + pw / 4) * fw))
    y0, y1 = int(max(0, (cy - ph / 4) * fh)), int(min(fh, (cy + ph / 4) * fh))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = bgr[y0:y1, x0:x1]
    ok = skin_mask(crop, cfg)
    d = depth_mm((x0, y0, x1, y1))
    if d is not None and hand_depth_m == hand_depth_m:
        ok &= (d > 50) & (np.abs(d.astype(np.float32) - 1000.0 * hand_depth_m) < 1000.0 * cfg.depth_band_m)
    if ok.sum() < 150:
        return None
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    return float(np.median(ycc[..., 1][ok])), float(np.median(ycc[..., 2][ok]))


def find_held_object(bgr: np.ndarray, depth_mm, palm, hand_depth_m: float,
                     cfg: HeldObjectConfig, skin_ref=None, debug: dict | None = None) -> dict | None:
    """The held-object blob next to the palm (see module docstring), or None.

    bgr: colour detection frame. depth_mm: (x0, y0, x1, y1) px -> uint16 mm crop of the
    selected depth source aligned to it (or None). palm: (cx, cy, w, h) normalised."""
    fh, fw = bgr.shape[:2]
    cx, cy, pw, ph = palm
    side = cfg.region_scale * max(pw * fw, ph * fh)
    x0, y0 = int(max(0, cx * fw - side / 2)), int(max(0, cy * fh - side / 2))
    x1, y1 = int(min(fw, cx * fw + side / 2)), int(min(fh, cy * fh + side / 2))
    if debug is not None:
        debug.clear()
        debug.update(time=time.monotonic(), palm=palm, hand_depth_m=hand_depth_m,
                     region=(x0, y0, x1, y1), comps=[], result="")
    if x1 - x0 < 16 or y1 - y0 < 16:
        return _why(debug, "search region too small")
    crop = bgr[y0:y1, x0:x1]
    d = depth_mm((x0, y0, x1, y1))
    if d is None:
        return _why(debug, "no fresh depth map")
    skin = skin_mask(crop, cfg, skin_ref)
    if debug is not None:
        debug.update(crop=crop.copy(), skin=skin, depth=d.copy())
    if hand_depth_m != hand_depth_m:          # palm depth unknown: median depth of the skin
        sd = d[skin & (d > 50)]
        if sd.size < 50:
            return _why(debug, "palm depth unknown and too little skin with depth")
        hand_depth_m = float(np.median(sd)) / 1000.0
    band = 1000.0 * cfg.depth_band_m
    near = (d > 50) & (np.abs(d.astype(np.float32) - 1000.0 * hand_depth_m) < band)
    skin_d = cv2.dilate(skin.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    cand = (near & ~skin_d).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if debug is not None:
        debug.update(near=near, cand=cand.astype(bool), hand_depth_used_m=hand_depth_m)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    palm_px = pw * fw * ph * fh
    H, W = cand.shape
    # a border only counts where the search region was not cut by the image edge
    edges = (x0 > 0, y0 > 0, x1 < fw, y1 < fh)     # left, top, right, bottom
    best = None
    for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1 if n > 1 else []:
        bx, by, bw, bh, area = (int(v) for v in stats[i])
        touches = ((bx == 0 and edges[0]) or (by == 0 and edges[1])
                   or (bx + bw >= W and edges[2]) or (by + bh >= H and edges[3]))
        why = ("too small" if area < cfg.min_area_rel * palm_px else
               "too big" if area > cfg.max_area_rel * palm_px else
               "touches the search border (sleeve / table / body?)" if touches else "")
        if debug is not None and len(debug["comps"]) < 8:
            debug["comps"].append({"box": [bx, by, bw, bh], "area_vs_palm": round(area / max(1.0, palm_px), 2),
                                   "rejected": why or None})
        if why == "too small":
            break
        if why:
            continue
        best = i
        break
    if best is None:
        return _why(debug, "no candidate blob (non-skin at hand depth)" if n <= 1 else "all blobs rejected")
    comp = (labels == best).astype(np.uint8)
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
    hand = (skin & near).astype(np.uint8)
    exclude = None
    if hand.sum() > 50:
        hd = cv2.distanceTransform(hand, cv2.DIST_L2, 5)
        ey, ex = np.unravel_index(int(np.argmax(hd)), hd.shape)
        exclude = ((x0 + ex) / fw, (y0 + ey) / fh)
    bx, by, bw, bh, area = stats[best]
    _why(debug, "object found")
    full = np.zeros((fh, fw), bool)
    full[y0:y1, x0:x1] = comp > 0
    return {"x": (x0 + cents[best][0]) / fw, "y": (y0 + cents[best][1]) / fh,
            "w": bw / fw, "h": bh / fh, "area": area / (fw * fh),
            "box": ((x0 + bx) / fw, (y0 + by) / fh, (x0 + bx + bw) / fw, (y0 + by + bh) / fh),
            "include": ((x0 + ix) / fw, (y0 + iy) / fh), "exclude": exclude,
            "mask": full, "region": (x0 / fw, y0 / fh, x1 / fw, y1 / fh),
            "hand_depth_m": hand_depth_m}


def _why(debug, reason: str):
    if debug is not None:
        debug["result"] = reason
    return None


def letterbox(bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Square-ish frame -> 576x320 XMem input without stretching. Returns (image, (x, y, w, h))."""
    fh, fw = bgr.shape[:2]
    s = min(XMEM_W / fw, XMEM_H / fh)
    w, h = int(round(fw * s)), int(round(fh * s))
    x, y = (XMEM_W - w) // 2, (XMEM_H - h) // 2
    out = np.empty((XMEM_H, XMEM_W, 3), np.uint8)
    out[:] = PAD_BGR
    out[y:y + h, x:x + w] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return out, (x, y, w, h)


def letterbox_mask(mask: np.ndarray, lb) -> np.ndarray:
    x, y, w, h = lb
    out = np.zeros((XMEM_H, XMEM_W), bool)
    out[y:y + h, x:x + w] = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return out


def unletterbox_mask(mask: np.ndarray, lb, size) -> np.ndarray:
    x, y, w, h = lb
    return cv2.resize(mask[y:y + h, x:x + w].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0


class HeldObject:
    """Cue selection + (optionally) XMem tracking + the grasp state machine (used by HandTracker)."""

    def __init__(self, cfg: HeldObjectConfig, grasp_cfg: GraspConfig, log=print):
        self.cfg = cfg
        self.grasp_cfg = grasp_cfg
        self.grasp = Grasp(grasp_cfg)
        self.log = log
        self.latest: ObjectObservation | None = None
        self.mask: np.ndarray | None = None       # current object mask (detection frame)
        self.cue: dict | None = None              # latest cue blob (overlay)
        self.history: deque = deque(maxlen=60)    # ObjectObservation, calibration marks
        self.marks: list[dict] = []
        self.mode = cfg.tracker
        self.status = "searching" if self.mode == "cue" else "loading models"
        self.error: str | None = None
        self.timings: dict = {}
        self._stable = 0
        self._last_cue = None
        self.skin_ref = cfg.skin_ref                # learned (Cr, Cb) of the palm, None = generic
        self._skin_samples: list = []
        self.debug: dict = {}                       # last cue internals (GET /object/debug)
        self._suppress_until = 0.0
        self._lock = threading.Lock()
        self._frame = None                        # (seq, bgr, stamp) for the worker
        self._seq = 0
        self._select = None                       # pending selection request
        self._tracking = False
        self._selecting = False                   # selection queued / running in the worker
        self._models_ready = self.mode == "cue"
        self._epoch = 0                           # bumped by clear(); stale worker results are dropped
        self._depth_fn = None
        self._stop = threading.Event()
        self.models_go = threading.Event()        # set by the caller once the camera pipeline is up
        if self.mode == "xmem":
            threading.Thread(target=self._worker, daemon=True, name="xmem").start()

    @property
    def hand_needed(self) -> bool:
        """False while XMem tracks an object: the hand is not used then (saves CPU)."""
        return not (self.mode == "xmem" and self._tracking)

    # ---- called from the HandTracker thread, every new colour frame --------------------
    def process(self, bgr: np.ndarray, now: float, palm, hand_depth_m: float, depth_fn) -> None:
        """palm: (cx, cy, w, h) normalised or None. depth_fn(px box) -> mm crop (selected source)."""
        self._depth_fn = depth_fn
        if self._tracking:
            with self._lock:
                self._seq += 1
                self._frame = (self._seq, bgr, now)
            return
        cue = None
        if palm is not None and now >= self._suppress_until:
            cue = find_held_object(bgr, depth_fn, palm, hand_depth_m, self.cfg, self.skin_ref, self.debug)
            if cue is None and self.skin_ref is None:     # empty hand: learn its skin colour
                s = palm_skin_sample(bgr, depth_fn, palm, hand_depth_m, self.cfg)
                if s is not None:
                    self._skin_samples.append(s)
                    if len(self._skin_samples) >= self.cfg.skin_learn_frames:
                        a = np.array(self._skin_samples)
                        self.skin_ref = (round(float(np.median(a[:, 0])), 1), round(float(np.median(a[:, 1])), 1))
                        self._skin_samples = []
                        self.log(f"[held_object] skin colour learned: Cr {self.skin_ref[0]}, Cb "
                                 f"{self.skin_ref[1]} (+-{self.cfg.skin_tol}); pin it with object.skin_ref")
        prev, self.cue, self._last_cue = self._last_cue, cue, cue
        if cue is None:
            self._stable = 0
            if self.mode == "cue":
                self.mask = None
            return
        moved = prev is None or max(abs(cue["x"] - prev["x"]), abs(cue["y"] - prev["y"])) > self.cfg.select_jitter
        self._stable = 1 if moved else self._stable + 1
        if self.mode == "cue":
            self._commit(self._observe(cue["mask"], now, 1.0), cue["mask"], self._epoch, "tracking (cue)")
        elif self._stable >= self.cfg.select_frames and self._models_ready and not self._selecting:
            with self._lock:
                if self._tracking:
                    return
                self._selecting = True
                self._select = {"bgr": bgr.copy(), "cue": cue, "stamp": now, "epoch": self._epoch}
                self.status = "selecting"

    def _observe(self, mask: np.ndarray, stamp: float, conf: float) -> ObjectObservation | None:
        """Observation (centroid, box, median depth) of a mask; None if it is too small."""
        ys, xs = np.nonzero(mask)
        if xs.size < self.cfg.min_mask_px:
            return None
        fh, fw = mask.shape
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        z = float("nan")
        d = self._depth_fn((x0, y0, x1, y1)) if self._depth_fn else None
        if d is not None:
            m = cv2.erode(mask[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if m.sum() < 20:
                m = mask[y0:y1, x0:x1]
            v = d[m]
            v = v[(v > 50) & (v < 4000)]
            if v.size >= max(10, 0.15 * m.sum()):
                z = float(np.median(v)) / 1000.0
        return ObjectObservation(float(xs.mean()) / fw, float(ys.mean()) / fh, z, (x1 - x0) / fw,
                                 (y1 - y0) / fh, xs.size / float(fw * fh), stamp, conf)

    def _commit(self, obs, mask, epoch: int, status: str) -> bool:
        """Publish a result - only if no clear() happened since the work for it started."""
        with self._lock:
            if epoch != self._epoch:
                return False
            if obs is not None:
                self.mask, self.latest = mask, obs
                self.history.append(obs)
            self.status = status
            return True

    # ---- control commands -------------------------------------------------------------
    def clear(self, suppress_s: float = 0.0, reason: str = "") -> None:
        """Drop the target; ignore objects for suppress_s (e.g. right after a release)."""
        with self._lock:
            self._epoch += 1
            self._tracking = False
            self._select = None
            self._frame = None
            self.latest, self.mask, self.cue, self._stable = None, None, None, 0
            self._suppress_until = time.monotonic() + suppress_s
            if self._models_ready:
                self.status = "searching"
        if reason:
            self.log(f"[held_object] target cleared ({reason})")

    def relearn_skin(self) -> None:
        self.skin_ref, self._skin_samples = None, []
        self.log("[held_object] relearning the skin colour - show the empty palm")

    def debug_info(self) -> dict:
        d = self.debug
        out = {"skin_ref": self.skin_ref, "skin_learning": f"{len(self._skin_samples)}/{self.cfg.skin_learn_frames}"
               if self.skin_ref is None else "done", "skin_tol": self.cfg.skin_tol}
        if not d:
            return {**out, "result": "no palm seen yet"}
        out.update({k: d.get(k) for k in ("result", "palm", "hand_depth_m", "hand_depth_used_m", "region", "comps")})
        out["age_s"] = round(time.monotonic() - d["time"], 2)
        crop, skin = d.get("crop"), d.get("skin")
        if crop is not None:
            ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            cand = d.get("cand")
            def stats(m):
                if m is None or not m.any():
                    return None
                return {"px": int(m.sum()), "Cr": round(float(np.median(ycc[..., 1][m])), 1),
                        "Cb": round(float(np.median(ycc[..., 2][m])), 1)}
            out["skin_px"] = stats(skin)
            out["candidate_px"] = stats(cand)
            near = d.get("near")
            out["near_px"] = int(near.sum()) if near is not None else None
            out["region_px"] = int(crop.shape[0] * crop.shape[1])
        return out

    def debug_image(self):
        """2x2 panel of the search region: colour | skin (white) / near-depth (grey) | candidates | depth."""
        d = self.debug
        crop = d.get("crop")
        if crop is None:
            return None
        h, w = crop.shape[:2]
        def gray(m):
            return cv2.cvtColor((m.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR) if m is not None else np.zeros_like(crop)
        sk = np.zeros((h, w), np.uint8)
        if d.get("near") is not None:
            sk[d["near"]] = 90
        if d.get("skin") is not None:
            sk[d["skin"]] = 255
        dep = d.get("depth")
        dep = cv2.applyColorMap(255 - np.clip((dep.astype(np.float32) - 100) / 900 * 255, 0, 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO) if dep is not None else np.zeros_like(crop)
        cand = gray(d.get("cand"))
        img = crop.copy()
        for c in d.get("comps", []):
            x, y, bw, bh = c["box"]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255) if c["rejected"] else (0, 255, 0), 2)
        top = np.hstack([img, cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)])
        bot = np.hstack([cand, dep])
        panel = np.vstack([top, bot])
        cv2.putText(panel, str(d.get("result", ""))[:60], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return panel

    def stop(self) -> None:
        self._stop.set()

    # ---- XMem worker ------------------------------------------------------------------
    def _worker(self) -> None:
        # Loading QNN sessions while the DepthAI pipeline starts its own networks on the same
        # NPU crashed the device - wait until the pipeline has been running for a while.
        while not self.models_go.wait(0.5):
            if self._stop.is_set():
                return
        try:
            from track_anything.mobilesam import MobileSamSelector
            from track_anything.model_assets.prepare import ensure_models
            from track_anything.sessions import QnnSessionManager
            from track_anything.xmem import XMemTracker

            t0 = time.monotonic()
            root = ensure_models(lambda m: self.log(f"[held_object] {m}"), self._stop)
            sessions = QnnSessionManager(root, lambda m: self.log(f"[held_object] {m}"))
            selector = MobileSamSelector(sessions)
            XMemTracker.load_models(sessions)
            selector.load_models()
            self.status, self._models_ready = "searching", True
            self.log(f"[held_object] MobileSAM + XMem ready on QNN in {time.monotonic() - t0:.0f} s")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"[held_object] track-anything unavailable, using the cue tracker: {self.error}\n"
                     + traceback.format_exc())
            self.mode, self.status, self._models_ready = "cue", "searching", True
            return

        tracker, lb, last_seq, lost = None, None, 0, 0
        while not self._stop.is_set():
            with self._lock:
                req, self._select = self._select, None
                frame = self._frame
            try:
                if req is not None:
                    try:
                        if req["epoch"] == self._epoch:
                            tracker, lb = self._start(req, selector, sessions, XMemTracker)
                            last_seq, lost = 0, 0
                    finally:
                        self._selecting = False
                    continue
                if tracker is None or not self._tracking or frame is None or frame[0] == last_seq:
                    self._stop.wait(0.005)
                    continue
                epoch = self._epoch
                seq, bgr, stamp = frame
                last_seq = seq
                img, _ = letterbox(bgr)
                m, conf = tracker.step(img)
                self.timings = {k: round(v, 1) for k, v in tracker.timings.items()}
                if epoch != self._epoch:
                    continue
                mask = unletterbox_mask(m, lb, (bgr.shape[1], bgr.shape[0]))
                obs = self._observe(mask, stamp, conf)
                if obs is None:
                    lost += 1
                    if self._commit(None, None, epoch, f"lost {lost}/{self.cfg.lost_frames}") \
                            and lost >= self.cfg.lost_frames:
                        tracker = None
                        self.clear(reason="object lost")
                else:
                    lost = 0
                    self._commit(obs, mask, epoch, "tracking")
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.log(f"[held_object] XMem failed: {self.error}\n{traceback.format_exc()}")
                tracker = None
                self.clear(suppress_s=1.0)

    def _start(self, req, selector, sessions, XMemTracker):
        """MobileSAM on the selection frame, check the mask, seed XMem."""
        t0 = time.perf_counter()
        bgr, cue = req["bgr"], req["cue"]
        points = [{"x": cue["include"][0], "y": cue["include"][1], "label": 1}]
        if cue["exclude"] is not None:
            points.append({"x": cue["exclude"][0], "y": cue["exclude"][1], "label": 0})
        mask, score = selector.select(selector.encode_selection(bgr), points)
        n = int(mask.sum())
        skin = float((mask & skin_mask(bgr, self.cfg, self.skin_ref)).sum()) / max(1, n)
        cover = float((mask & cue["mask"]).sum()) / max(1, int(cue["mask"].sum()))
        too_big = n > 6 * cue["mask"].sum() + 2000
        ms = (time.perf_counter() - t0) * 1000
        if n < self.cfg.min_mask_px or skin > self.cfg.max_skin_overlap or cover < self.cfg.min_cue_overlap or too_big:
            self.log(f"[held_object] selection rejected: {n} px, skin {skin:.2f}, covers cue {cover:.2f}, "
                     f"score {score:.2f} ({ms:.0f} ms)")
            with self._lock:
                if req["epoch"] == self._epoch:
                    self.status, self._stable = "searching", 0
                    self._suppress_until = time.monotonic() + 1.0
            return None, None
        img, lb = letterbox(bgr)
        tracker = XMemTracker(sessions)
        tracker.initialize(img, letterbox_mask(mask, lb))
        with self._lock:
            if req["epoch"] != self._epoch:
                return None, None
            self._tracking = True
        self._commit(self._observe(mask, req["stamp"], score), mask, req["epoch"], "tracking")
        self.log(f"[held_object] object selected: {n} px, skin {skin:.2f}, covers cue {cover:.2f}, "
                 f"score {score:.2f}, MobileSAM {ms:.0f} ms - tracking with XMem")
        return tracker, lb

    # ---- calibration / web view -------------------------------------------------------
    def mark(self, reason: str, window_s: float = 0.7) -> dict | None:
        """Median object position over the last window_s (calibration snapshot)."""
        now = time.monotonic()
        recent = [o for o in self.history if now - o.stamp <= window_s]
        if not recent:
            return None
        zs = [o.depth_m for o in recent if o.depth_m == o.depth_m]
        m = {"x": round(float(np.median([o.x for o in recent])), 4),
             "y": round(float(np.median([o.y for o in recent])), 4),
             "depth_m": round(float(np.median(zs)), 4) if zs else None,
             "w": round(float(np.median([o.w for o in recent])), 4),
             "h": round(float(np.median([o.h for o in recent])), 4),
             "frames": len(recent), "depth_frames": len(zs), "reason": reason,
             "time": time.strftime("%H:%M:%S")}
        self.marks.append(m)
        del self.marks[:-20]
        return m

    def set_target(self, m: dict | None) -> bool:
        if m is None or m.get("depth_m") is None:
            return False
        self.grasp_cfg.grasp_target = target_dict(m)
        return True

    def snapshot(self) -> dict:
        o = self.latest
        age = time.monotonic() - o.stamp if o else None
        g = self.grasp
        return {"object": None if o is None else {
                    "x": round(o.x, 4), "y": round(o.y, 4),
                    "depth_m": round(o.depth_m, 4) if o.depth_m == o.depth_m else None,
                    "w": round(o.w, 4), "h": round(o.h, 4), "area": round(o.area, 4),
                    "confidence": round(o.confidence, 3), "age_s": round(age, 2)},
                "tracker": self.mode, "status": self.status, "error": self.error,
                "xmem_ms": self.timings,
                "cue": None if self.cue is None else {k: self.cue[k] for k in ("x", "y", "include", "exclude",
                                                                                  "hand_depth_m")},
                "grasp_state": g.state, "detail": g.detail, "grasp_target": self.grasp_cfg.grasp_target,
                "auto_grip": self.grasp_cfg.auto_grip, "marks": self.marks[-5:],
                "last_mark": self.marks[-1] if self.marks else None}
