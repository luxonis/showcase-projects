"""Stereo + ToF depth fusion, palm depth per source and the depth-view colouring.

All depth maps are uint16 millimetres aligned to the same colour frame, 0 = no depth.

Fusion rule (per pixel), chosen to be easy to explain on camera:
  only one sensor has depth      -> that sensor
  both have depth, agree         -> blend; the weight moves from ToF (close, where stereo is
                                    weakest) to stereo (far, where ToF gets noisy) between
                                    near_mm and far_mm, judged by the ToF distance
  both have depth, disagree >25% -> ToF if stereo says it is close, else stereo
"""

from __future__ import annotations

import cv2
import numpy as np

SOURCES = ("stereo", "tof", "fused")
MIN_MM, MAX_MM = 50, 4000          # plausible depth for a hand in front of the camera
DISAGREE_FRAC = 0.25               # both sensors valid but this far apart -> pick one, no blend

# Which sensor a fused pixel came from (for the "mix" readout)
SRC_NONE, SRC_STEREO, SRC_TOF, SRC_BLEND = 0, 1, 2, 3


def _valid(d: np.ndarray | None, shape) -> np.ndarray:
    if d is None:
        return np.zeros(shape, bool)
    return (d >= MIN_MM) & (d <= MAX_MM)


def fuse(stereo: np.ndarray | None, tof: np.ndarray | None,
         near_mm: float = 350.0, far_mm: float = 800.0,
         judge_by_tof: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns (fused depth uint16 mm, source map uint8). Either input may be None.

    judge_by_tof: on strong disagreement trust ToF when ToF (not stereo) says it is close - for a
    device-fused "stereo" input, which always has a value (classic stereo just goes empty up close).

    Whole-array arithmetic (no boolean fancy indexing) - several times faster on the camera CPU.
    """
    ref = stereo if stereo is not None else tof
    if ref is None:
        return None, None
    if stereo is not None and tof is not None and stereo.shape != tof.shape:
        tof = cv2.resize(tof, (stereo.shape[1], stereo.shape[0]), interpolation=cv2.INTER_NEAREST)
    zeros = np.zeros(ref.shape[:2], np.float32)
    sf = stereo.astype(np.float32) if stereo is not None else zeros
    tf = tof.astype(np.float32) if tof is not None else zeros
    vs = (sf >= MIN_MM) & (sf <= MAX_MM)
    vt = (tf >= MIN_MM) & (tf <= MAX_MM)
    w_tof = np.clip((far_mm - tf) * (1.0 / max(1.0, far_mm - near_mm)), 0.0, 1.0)
    # Strong disagreement: one of them is wrong. Up close (by stereo) trust ToF, else stereo -
    # so stray ToF noise never overrides a good stereo surface and vice versa.
    disagree = np.abs(tf - sf) > DISAGREE_FRAC * np.maximum(tf, sf)
    w_tof = np.where(disagree, ((tf if judge_by_tof else sf) < near_mm).astype(np.float32), w_tof)
    w_tof = np.where(vs & vt, w_tof, vt.astype(np.float32))      # only one valid -> that one
    any_valid = vs | vt
    out = np.where(any_valid, w_tof * tf + (1.0 - w_tof) * sf, 0.0).astype(np.uint16)
    src = np.where(w_tof >= 0.999, SRC_TOF, np.where(w_tof <= 0.001, SRC_STEREO, SRC_BLEND)).astype(np.uint8)
    src[~any_valid] = SRC_NONE
    return out, src


def roi(frame: np.ndarray | None, box) -> np.ndarray | None:
    """Inner half of a normalised (x0, y0, x1, y1) box - avoids background at the box edges."""
    if frame is None:
        return None
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) / 4, (y1 - y0) / 4
    c0, c1 = int(max(0, (x0 + mx) * fw)), int(min(fw, (x1 - mx) * fw))
    r0, r1 = int(max(0, (y0 + my) * fh)), int(min(fh, (y1 - my) * fh))
    if c1 <= c0 or r1 <= r0:
        return None
    return frame[r0:r1, c0:c1]


def palm_depth(patch: np.ndarray | None, min_valid: float = 0.15) -> tuple[float, float]:
    """(median depth m or nan, fraction of valid pixels) of a palm depth patch.

    Depth counts as known only if at least `min_valid` of the palm has depth - a few stray
    pixels must not look like a measurement when the sensor has really lost the hand.
    """
    if patch is None or patch.size == 0:
        return float("nan"), 0.0
    ok = (patch >= MIN_MM) & (patch <= MAX_MM)
    frac = float(ok.mean())
    if frac < min_valid or ok.sum() < 10:
        return float("nan"), frac
    return float(np.median(patch[ok])) / 1000.0, frac


def palm_depths(stereo, tof, box, near_mm: float, far_mm: float, fused=None,
                tof_near: bool = False) -> dict:
    """{source: (depth_m, valid_frac)} for the palm box, all three sources.

    `fused`: a fused depth map from an on-device engine (neural / vpp); None -> fuse() here.
    `tof_near`: re-fuse that map with ToF by the rule above (ToF up close, blend, device-fused far)."""
    ps, pt = roi(stereo, box), roi(tof, box)
    if ps is not None and pt is not None and ps.shape != pt.shape:
        pt = cv2.resize(pt, (ps.shape[1], ps.shape[0]), interpolation=cv2.INTER_NEAREST)
    if fused is None:
        pf = fuse(ps, pt, near_mm, far_mm)[0]
    else:
        pf = roi(fused, box)
        if tof_near:
            if pf is not None and pt is not None and pf.shape != pt.shape:
                pt = cv2.resize(pt, (pf.shape[1], pf.shape[0]), interpolation=cv2.INTER_NEAREST)
            pf = fuse(pf, pt, near_mm, far_mm, judge_by_tof=True)[0]
    return {"stereo": palm_depth(ps), "tof": palm_depth(pt), "fused": palm_depth(pf)}


_HATCH: dict = {}


def _hatch(w: int, h: int) -> np.ndarray:
    """Dark diagonal stripes for "no depth here" (cached per size)."""
    if (w, h) not in _HATCH:
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.empty((h, w, 3), np.uint8)
        img[:] = (22, 22, 24)
        img[((xx + yy) // max(2, w // 120)) % 4 == 0] = (48, 48, 54)
        _HATCH[(w, h)] = img
    return _HATCH[(w, h)]


def colorize(depth: np.ndarray | None, size: tuple[int, int], min_mm: float, max_mm: float) -> np.ndarray:
    """Fixed-range depth colouring (near = warm, far = cool), no-depth pixels hatched dark grey.

    A fixed range keeps colours stable while the hand moves; the hatch makes "this sensor
    sees nothing here" obvious on video. Colours at the depth map's own resolution, then one
    nearest-neighbour upscale (cheap on the camera CPU).
    """
    w, h = size
    if depth is None:
        return _hatch(w, h).copy()
    t = (depth.astype(np.float32) - min_mm) * (255.0 / max(1.0, max_mm - min_mm))
    idx = 255 - np.clip(t, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(idx, cv2.COLORMAP_TURBO)
    invalid = cv2.inRange(depth, MIN_MM, MAX_MM)          # 255 = valid
    col = cv2.resize(col, (w, h), interpolation=cv2.INTER_NEAREST)
    invalid = cv2.bitwise_not(cv2.resize(invalid, (w, h), interpolation=cv2.INTER_NEAREST))
    cv2.copyTo(_hatch(w, h), invalid, col)
    return col


def source_mix(src: np.ndarray | None) -> tuple[float, float]:
    """Share of fused pixels that came (mostly) from ToF and from stereo, of the valid ones."""
    if src is None:
        return 0.0, 0.0
    counts = np.bincount(src.ravel(), minlength=4)
    n = int(counts[1:].sum())
    if n == 0:
        return 0.0, 0.0
    tof = float(counts[SRC_TOF]) + 0.5 * float(counts[SRC_BLEND])
    return tof / n, 1.0 - tof / n
