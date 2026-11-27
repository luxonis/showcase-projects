"""Web-view rendering: rotated camera view with a clean HUD over the hand detection.

Everything is drawn AFTER rotating the frame (so text stays upright); detection
coordinates are normalised in the unrotated detection image and mapped here.
"""

from __future__ import annotations

import glob
import math
import time

import cv2
import numpy as np

from ur_oak_control.depth_fusion import colorize, fuse, source_mix

# BGR palette
WHITE = (245, 245, 245)
MUTED = (175, 175, 175)
PANEL = (28, 24, 22)
GREEN = (120, 220, 90)
RED = (80, 80, 240)
RED_DIM = (70, 70, 170)
GREEN_DIM = (90, 170, 70)
AMBER = (40, 180, 250)
CYAN = (230, 200, 60)
GREY = (120, 120, 120)
FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA


class View:
    """Maps normalised detection-image coords into the rotated output image."""

    def __init__(self, rotation: int, w: int, h: int, m=(1.0, 0.0, 1.0, 0.0)):
        self.rot = rotation % 360
        self.w, self.h = w, h   # size AFTER rotation
        self.m = m              # detection -> this frame (normalised, before rotation): ax, bx, ay, by

    def pt(self, x: float, y: float) -> tuple[int, int]:
        ax, bx, ay, by = self.m
        x, y = ax * x + bx, ay * y + by
        r = self.rot
        if r == 90:      # image rotated clockwise
            x, y = 1 - y, x
        elif r == 180:
            x, y = 1 - x, 1 - y
        elif r == 270:
            x, y = y, 1 - x
        return int(round(x * self.w)), int(round(y * self.h))

    def vec(self, dx: float, dy: float) -> tuple[float, float]:
        r = self.rot
        if r == 90:
            return -dy, dx
        if r == 180:
            return -dx, -dy
        if r == 270:
            return dy, -dx
        return dx, dy


def rotate(img, rotation: int):
    r = rotation % 360
    if r == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if r == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _rounded(img, p0, p1, r, color):
    x0, y0 = p0
    x1, y1 = p1
    r = max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1, AA)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1, AA)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, AA)


def _text(img, txt, org, scale, color, thick=1):
    cv2.putText(img, txt, org, FONT, scale, color, thick, AA)


def _text_w(txt, scale, thick=1):
    return cv2.getTextSize(txt, FONT, scale, thick)[0][0]


def _deg_text(img, value, org, scale, color):
    """'+12°' - Hershey fonts have no degree sign, so draw a small ring."""
    txt = f"{value:+.0f}"
    _text(img, txt, org, scale, color)
    w = _text_w(txt, scale)
    r = max(2, int(4 * scale))
    cv2.circle(img, (org[0] + w + r + 2, org[1] - int(16 * scale)), r, color, 1, AA)
    return w + 2 * r + 4


def _chip(img, layer, txt, x, y, s, color):
    """Coloured pill with dark text; returns its right edge."""
    tw = _text_w(txt, 0.5 * s, 1)
    h = int(22 * s)
    _rounded(img, (x, y), (x + tw + int(16 * s), y + h), h // 2, color)
    _text(img, txt, (x + int(8 * s), y + int(16 * s)), 0.5 * s, (25, 25, 25), 1)
    return x + tw + int(16 * s)


def _corner_brackets(img, corners, color, frac=0.22, thick=2):
    n = len(corners)
    for i in range(n):
        p = np.array(corners[i], float)
        for j in (i - 1, (i + 1) % n):
            q = np.array(corners[j], float)
            e = p + (q - p) * frac
            cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in e), color, thick, AA)


def _dashed(img, p0, p1, color, dash=10, thick=1):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = np.linalg.norm(p1 - p0)
    if d < 1:
        return
    u = (p1 - p0) / d
    t = 0.0
    while t < d:
        a = p0 + u * t
        b = p0 + u * min(t + dash, d)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, thick, AA)
        t += 2 * dash


def _arc(img, c, r, a0, a1, color, thick):
    """Arc from a0 to a1 (degrees, cv2 convention: clockwise from +x, y down)."""
    if a1 < a0:
        a0, a1 = a1, a0
    cv2.ellipse(img, c, (r, r), 0, a0, a1, color, thick, AA)


def render(frame, tracker, rotation: int = 0) -> np.ndarray:
    img = rotate(frame, rotation).copy()
    h, w = img.shape[:2]
    s = w / 960.0
    view = View(rotation, w, h, getattr(tracker, "view_map", (1.0, 0.0, 1.0, 0.0)))
    hud = dict(getattr(tracker, "hud", {}) or {})
    show_hud = getattr(tracker, "show_hud", True)
    base = img.copy()
    layer = img.copy()   # translucent fills go here, blended once where they were drawn
    post = []            # text that sits on a translucent pill: drawn after the blend
    rois = []            # regions with translucent fills - only these get blended (cheap at 1920x1440)

    det = tracker.det
    fresh = det is not None and time.monotonic() - det["stamp"] < 0.4
    closed = tracker.hand_closed
    state_col = RED if closed else (GREEN if closed is not None else CYAN)

    # centre crosshair
    cx, cy = w // 2, h // 2
    g, L = int(6 * s), int(18 * s)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (cx + dx * g, cy + dy * g), (cx + dx * L, cy + dy * L), WHITE, 1, AA)

    if fresh:
        rcx, rcy, rw, rh, ang = det["rect"]
        a = math.radians(ang)
        ux = np.array([math.cos(a), math.sin(a)])
        uy = np.array([-math.sin(a), math.cos(a)])
        c = np.array([rcx, rcy])
        corners = [c + sx * ux * rw / 2 + sy * uy * rh / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pc = [view.pt(*p) for p in corners]
        _corner_brackets(img, pc, state_col, thick=max(2, int(2 * s)))

        palm = view.pt(det["x"], det["y"])
        _dashed(img, (cx, cy), palm, WHITE, dash=int(8 * s))
        cv2.circle(layer, palm, int(11 * s), state_col, -1, AA)
        cv2.circle(img, palm, int(11 * s), WHITE, max(1, int(2 * s)), AA)

        # landmarks (fingers mode)
        pts = getattr(tracker, "_lm_points", None)
        if pts is not None:
            for x, y in pts:
                cv2.circle(img, view.pt(x, y), max(2, int(3 * s)), state_col, -1, AA)

        # twist gauge around the palm
        ref = tracker.twist_reference
        vec = tracker._ang_vec
        if tracker.grip_mode == "twist" and vec is not None:
            box_px = max(abs(pc[0][0] - pc[2][0]), abs(pc[0][1] - pc[2][1]))
            R = int(np.clip(0.33 * box_px, 55 * s, 170 * s))
            t = max(2, int(5 * s))

            def heading(dx, dy):   # image direction -> cv2 arc angle in the rotated view
                vx, vy = view.vec(dx, dy)
                return math.degrees(math.atan2(vy, vx))

            if ref is not None:
                ra = math.radians(ref)
                h_ref = heading(math.sin(ra), -math.cos(ra))   # "up" of the upright hand
                o, cl = tracker.twist_open_deg, tracker.twist_close_deg
                _arc(layer, palm, R, h_ref - 180, h_ref + 180, GREY, max(1, t - 2))   # track
                _arc(layer, palm, R, h_ref - o, h_ref + o, GREEN_DIM, t)              # open zone
                _arc(layer, palm, R, h_ref + cl, h_ref + 180, RED_DIM, t)             # close zones
                _arc(layer, palm, R, h_ref - 180, h_ref - cl, RED_DIM, t)
                tick = view.vec(math.sin(ra), -math.cos(ra))
                p_ref = (int(palm[0] + tick[0] * (R + 10 * s)), int(palm[1] + tick[1] * (R + 10 * s)))
                cv2.line(img, (int(palm[0] + tick[0] * (R - 10 * s)), int(palm[1] + tick[1] * (R - 10 * s))),
                         p_ref, WHITE, max(1, int(2 * s)), AA)
            # needle: current hand "up" direction
            cs, sn = float(vec[0]), float(vec[1])
            nx, ny = view.vec(sn, -cs)
            tip = (int(palm[0] + nx * R), int(palm[1] + ny * R))
            cv2.line(img, palm, tip, state_col, max(2, int(3 * s)), AA)
            cv2.circle(img, tip, max(4, int(7 * s)), state_col, -1, AA)
            cv2.circle(img, tip, max(4, int(7 * s)), WHITE, 1, AA)
            if tracker.twist is not None:
                sc = 0.6 * s
                tw = _text_w(f"{tracker.twist:+.0f}", sc) + int(12 * s)
                d = R + 34 * s
                lx = int(palm[0] + nx * d) - tw // 2
                ly = int(palm[1] + ny * d) + int(8 * s)
                _rounded(layer, (lx - int(9 * s), ly - int(21 * s)), (lx + tw + int(7 * s), ly + int(8 * s)),
                         int(14 * s), PANEL)
                post.append(lambda im, lx=lx, ly=ly, sc=sc: _deg_text(im, tracker.twist, (lx, ly), sc, WHITE))
        else:
            R = int(40 * s)

        # depth / confidence tag under the palm
        z = det["depth"]
        tag = f"{z:.2f} m  |  {det['conf']:.2f}" if z == z else f"z ?  |  {det['conf']:.2f}"
        tw = _text_w(tag, 0.5 * s)
        tx, ty = palm[0] - tw // 2, palm[1] + R + int(34 * s)
        _rounded(layer, (tx - int(8 * s), ty - int(17 * s)), (tx + tw + int(8 * s), ty + int(7 * s)),
                 int(11 * s), PANEL)
        post.append(lambda im, tag=tag, tx=tx, ty=ty: _text(im, tag, (tx, ty), 0.5 * s, WHITE))
        e = R + int(130 * s) + tw // 2
        rois.append((palm[0] - e, palm[1] - e, palm[0] + e, palm[1] + e))

    vp = getattr(tracker, "virtual_point", None)
    if vp is not None:
        cv2.circle(img, view.pt(*vp), int(10 * s), CYAN, max(1, int(2 * s)), AA)

    _draw_object(img, layer, post, rois, tracker, view, s)

    # HUD panel (top left)
    pad, lh = int(14 * s), int(30 * s)
    rows = 4
    pw = int(370 * s)
    if show_hud:
        _rounded(layer, (int(12 * s), int(12 * s)), (int(12 * s) + pw, int(12 * s) + pad * 2 + lh * rows),
                 int(14 * s), PANEL)
        rois.append((0, 0, int(16 * s) + pw, int(16 * s) + pad * 2 + lh * rows))
    labels = _labels(tracker) if getattr(tracker, "show_labels", True) else []
    label_rows, label_box = _layout_labels(layer, labels, w, h, s)
    if label_box:
        rois.append(label_box)
    # cv2 only (releases the GIL - keeps the 50 Hz control loop smooth)
    for x0, y0, x1, y1 in rois:
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        lr, br, ir = layer[y0:y1, x0:x1], base[y0:y1, x0:x1], img[y0:y1, x0:x1]
        diff = cv2.cvtColor(cv2.absdiff(lr, br), cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)
        cv2.copyTo(cv2.addWeighted(lr, 0.62, ir, 0.38, 0), mask, ir)
        br[:] = lr          # done: an overlapping region must not blend these pixels again
    for draw in post:
        draw(img)
    for draw in label_rows:
        draw(img)
    if not show_hud:
        return img
    _sys_chip(img, tracker, s)

    x0, y = int(12 * s) + pad, int(12 * s) + pad
    live = hud.get("live")
    xe = _chip(img, None, "LIVE" if live else "DRY RUN", x0, y, s, RED if live else AMBER)
    mode = str(hud.get("mode", "")).upper()
    _text(img, mode, (xe + int(12 * s), y + int(16 * s)), 0.5 * s, WHITE)
    xd = xe + int(12 * s) + _text_w(mode, 0.5 * s) + int(10 * s)
    cv2.circle(img, (xd, y + int(11 * s)), max(2, int(3 * s)), MUTED, -1, AA)
    _text(img, str(hud.get("state", "")).upper(), (xd + int(10 * s), y + int(16 * s)), 0.5 * s, WHITE)
    y += lh
    off = hud.get("offset")
    if off is not None:
        _text(img, "OFFSET", (x0, y + int(16 * s)), 0.45 * s, MUTED)
        _text(img, f"x {off[0] * 100:+5.1f}   y {off[1] * 100:+5.1f}   z {off[2] * 100:+5.1f} cm",
              (x0 + int(78 * s), y + int(16 * s)), 0.5 * s, WHITE)
    y += lh
    gs = hud.get("gripper")
    gcol = RED if gs is True else (GREEN if gs is False else GREY)
    _text(img, "GRIPPER", (x0, y + int(16 * s)), 0.45 * s, MUTED)
    xe = _chip(img, None, "CLOSED" if gs is True else ("OPEN" if gs is False else "--"), x0 + int(78 * s), y, s, gcol)
    detail = hud.get("grip_detail", "")
    if detail:
        _text(img, detail, (xe + int(10 * s), y + int(16 * s)), 0.45 * s, MUTED)
    y += lh
    src = getattr(tracker, "depth_source", None)
    if src:
        _text(img, "DEPTH", (x0, y + int(16 * s)), 0.45 * s, MUTED)
        xe = _chip(img, None, source_name(tracker, src), x0 + int(78 * s), y, s, SOURCE_COL[src])
        z = det["depth"] if fresh else float("nan")
        _text(img, f"{z:.2f} m" if z == z else ("no depth" if fresh else "--"),
              (xe + int(10 * s), y + int(16 * s)), 0.5 * s, WHITE if z == z else RED)
    return img


# ---------------------------------------------------------------------------------------
# Video labels and the depth view

STEREO_COL = (240, 170, 50)    # BGR blue
TOF_COL = (50, 150, 255)       # orange
FUSED_COL = (110, 225, 80)     # green
SOURCE_COL = {"stereo": STEREO_COL, "tof": TOF_COL, "fused": FUSED_COL}
SOURCE_SUB = {"stereo": "2 cameras", "tof": "time-of-flight sensor",
              "fused": "stereo + ToF, fused per pixel"}
FUSION_SUB = {"neural": "ToF + neural stereo, fusion network",
              "vpp": "ToF-guided stereo (virtual pattern)"}

DEFAULT_LABELS = [
    "NO PC - everything runs on the {device}",
    "UR ROS 2 driver on the camera",
    "Hand AI on-device",
    "Depth: {source}",
]


OBJECT_COL = (40, 225, 255)    # BGR yellow
CUE_COL = (200, 200, 200)
TARGET_COL = (255, 90, 230)    # magenta


def _draw_object(img, layer, post, rois, tracker, view, s) -> None:
    """Held object: mask outline + depth, the selection cue, the grasp target and the error line."""
    ho = getattr(tracker, "held", None)
    if ho is None:
        return
    g = ho.grasp_cfg.grasp_target
    if g is not None:      # grasp target: magenta ring + cross ("object goes here")
        gp = view.pt(g["x"], g["y"])
        r = int(16 * s)
        cv2.circle(img, gp, r, TARGET_COL, max(1, int(2 * s)), AA)
        cv2.line(img, (gp[0] - r, gp[1]), (gp[0] + r, gp[1]), TARGET_COL, 1, AA)
        cv2.line(img, (gp[0], gp[1] - r), (gp[0], gp[1] + r), TARGET_COL, 1, AA)
        _text(img, f"GRASP {g['depth_m']:.2f} m", (gp[0] + r + int(6 * s), gp[1] - int(6 * s)),
              0.45 * s, TARGET_COL)
    now = time.monotonic()
    o, mask = ho.latest, ho.mask
    tracked = o is not None and mask is not None and now - o.stamp < 0.5
    cue = ho.cue
    if not tracked and cue is not None:     # selection cue: dashed-ish grey box + include point
        x0, y0, x1, y1 = cue["box"]
        p0, p1 = view.pt(x0, y0), view.pt(x1, y1)
        corners = [p0, (p1[0], p0[1]), p1, (p0[0], p1[1])]
        _corner_brackets(img, corners, CUE_COL, thick=max(1, int(2 * s)))
        cv2.circle(img, view.pt(*cue["include"]), max(3, int(5 * s)), GREEN, -1, AA)
        if cue["exclude"] is not None:
            cv2.circle(img, view.pt(*cue["exclude"]), max(3, int(5 * s)), RED, -1, AA)
        _text(img, ho.status.upper(), (min(p0[0], p1[0]), max(p0[1], p1[1]) + int(20 * s)), 0.45 * s, CUE_COL)
    if not tracked:
        return
    state = ho.grasp.state
    col = RED if state == "gripped" else OBJECT_COL
    # mask outline, mapped point by point into the (rotated, possibly wider) view
    small = cv2.resize(mask.astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if len(c) < 3:
            continue
        pts = np.array([view.pt((p[0][0] + 0.5) / 160, (p[0][1] + 0.5) / 160) for p in c], np.int32)
        cv2.polylines(img, [pts], True, col, max(2, int(3 * s)), AA)
    c = view.pt(o.x, o.y)
    cv2.circle(img, c, max(3, int(6 * s)), col, -1, AA)
    if g is not None and state == "approach":
        _dashed(img, c, view.pt(g["x"], g["y"]), TARGET_COL, dash=int(8 * s))
    z = f"{o.depth_m:.2f} m" if o.depth_m == o.depth_m else "z ?"
    tag = f"OBJECT {z}" + ("" if state == "idle" else f" | {state.upper()}")
    tw = _text_w(tag, 0.5 * s)
    by = max(view.pt(o.x - o.w / 2, o.y - o.h / 2)[1], view.pt(o.x + o.w / 2, o.y + o.h / 2)[1])
    tx, ty = c[0] - tw // 2, by + int(28 * s)
    _rounded(layer, (tx - int(8 * s), ty - int(17 * s)), (tx + tw + int(8 * s), ty + int(7 * s)),
             int(11 * s), PANEL)
    rois.append((tx - int(10 * s), ty - int(20 * s), tx + tw + int(10 * s), ty + int(10 * s)))
    post.append(lambda im, tag=tag, tx=tx, ty=ty: _text(im, tag, (tx, ty), 0.5 * s, col))


def source_name(tracker, src: str) -> str:
    if src == "stereo":
        return "NEURAL STEREO" if getattr(tracker, "stereo_engine", "classic") == "neural" else "STEREO"
    return {"tof": "ToF", "fused": "FUSED"}.get(src, src.upper())


def _labels(tracker) -> list[tuple[str, tuple]]:
    src = getattr(tracker, "depth_source", "stereo")
    out = []
    for lbl in getattr(tracker, "labels", None) or DEFAULT_LABELS:
        txt = str(lbl).replace("{device}", getattr(tracker, "device_name", "OAK4"))
        col = FUSED_COL
        if "{source}" in txt:
            txt = txt.replace("{source}", source_name(tracker, src))
            col = SOURCE_COL.get(src, FUSED_COL)
        out.append((txt, col))
    return out


MIX_SC = 0.8          # text scale of the fused ToF / stereo share row (x view scale)
MIX_BAR_W = 240       # px at 960: width of the split bar between the two percentages


def _mix_row_w(s: float) -> int:
    """Width of the fused share row (widest case, 100 % on both labels)."""
    sc = MIX_SC * s
    return (_text_w("ToF 100%", sc, 2) + _text_w("100% stereo", sc, 2)
            + int(MIX_BAR_W * s) + 2 * int(14 * s))


def _mix_row(im, x: int, y: int, s: float, mix: tuple[float, float]) -> None:
    """Fused share row: 'ToF 63%  [orange|blue]  37% stereo', top-left corner at (x, y).
    The bar is split in proportion to the share of fused pixels each sensor supplied."""
    sc, gap, bh = MIX_SC * s, int(14 * s), int(22 * s)
    tof, stereo = mix
    valid = tof + stereo > 0
    tof_pct = round(100 * tof) if valid else None
    st_pct = 100 - tof_pct if valid else None
    l_txt = f"ToF {tof_pct}%" if valid else "ToF -"
    r_txt = f"{st_pct}% stereo" if valid else "- stereo"
    base = y + bh // 2 + int(10 * s)                      # text baseline, centred on the bar
    l_w = _text_w("ToF 100%", sc, 2)                      # fixed slot: the bar doesn't jump
    _text(im, l_txt, (x + l_w - _text_w(l_txt, sc, 2), base), sc, TOF_COL if valid else GREY, 2)
    bx0 = x + l_w + gap
    bx1 = bx0 + int(MIX_BAR_W * s)
    if valid:
        split = bx0 + int(round((bx1 - bx0) * tof))
        if split > bx0:
            cv2.rectangle(im, (bx0, y), (split, y + bh), TOF_COL, -1)
        if split < bx1:
            cv2.rectangle(im, (split, y), (bx1, y + bh), STEREO_COL, -1)
    else:
        cv2.rectangle(im, (bx0, y), (bx1, y + bh), GREY, 1, AA)
    _text(im, r_txt, (bx1 + gap, base), sc, STEREO_COL if valid else GREY, 2)


def _layout_labels(layer, labels, w, h, s) -> list:
    """Lower-third tags, bottom left: translucent pill + accent bar. Fills go on `layer`,
    the returned callables draw the text after the blend."""
    sc, ph, gap = 0.62 * s, int(38 * s), int(10 * s)
    x = int(22 * s)
    y = h - int(22 * s) - len(labels) * (ph + gap) + gap
    draws = []
    box = None
    if labels:
        box = (0, y - 4, x + max(_text_w(t, sc) for t, _ in labels) + int(48 * s), h)
    for txt, col in labels:
        tw = _text_w(txt, sc)
        _rounded(layer, (x, y), (x + tw + int(40 * s), y + ph), int(10 * s), PANEL)
        draws.append(lambda im, x=x, y=y, txt=txt, col=col: (
            cv2.rectangle(im, (x + int(10 * s), y + int(9 * s)), (x + int(15 * s), y + ph - int(9 * s)), col, -1),
            _text(im, txt, (x + int(26 * s), y + ph // 2 + int(8 * s)), sc, WHITE)))
        y += ph + gap
    return draws, box


def render_depth(tracker, rotation: int = 0, size: int = 960, min_mm: float = 150.0,
                 max_mm: float = 1500.0, window_s: float = 8.0) -> np.ndarray:
    """Depth stream: the selected source, palm distance, and a live graph of all sources."""
    src = tracker.depth_source
    # per-pixel work at DEPTH_WORK px (fusion, colouring), one cheap upscale to `size`
    ws = min(DEPTH_WORK, size)

    def small(d):
        return None if d is None else cv2.resize(d, (ws, ws), interpolation=cv2.INTER_NEAREST)

    st_raw, tf_raw = tracker.fresh_depth("stereo"), tracker.fresh_depth("tof")
    engine = getattr(tracker, "fusion_engine", "heuristic")
    fu_raw = tracker.fresh_depth("fused") if engine != "heuristic" else None
    key = (src, id(st_raw), id(tf_raw), id(fu_raw), size, rotation, min_mm, max_mm)
    if _DEPTH_CACHE.get("key") == key:          # no new depth map since the last frame
        coloured, mix = _DEPTH_CACHE["img"], _DEPTH_CACHE["mix"]
    else:
        st, tf = small(st_raw), small(tf_raw)
        mix = None
        if src == "fused" and engine != "heuristic" and getattr(tracker, "fusion_tof_near", False):
            # device-fused depth as the far component, ToF up close; mix = ToF / device share
            depth, srcmap = fuse(small(fu_raw), tf, tracker.fuse_near_mm, tracker.fuse_far_mm,
                                 judge_by_tof=True)
            mix = source_mix(srcmap)
        elif src == "fused" and engine != "heuristic":
            depth = small(fu_raw)               # fused on the device - no per-pixel source map
        elif src == "fused":
            depth, srcmap = fuse(st, tf, tracker.fuse_near_mm, tracker.fuse_far_mm)
            mix = source_mix(srcmap)
        else:
            depth = st if src == "stereo" else tf
        coloured = rotate(colorize(depth, (size, size), min_mm, max_mm), rotation)
        _DEPTH_CACHE.update(key=key, img=coloured, mix=mix)
    img = coloured.copy()
    h, w = img.shape[:2]
    s = w / 960.0
    view = View(rotation, w, h)
    post = []
    col = SOURCE_COL.get(src, WHITE)

    # palm box + its distance from the selected source
    det = tracker.det
    if det is not None and time.monotonic() - det["stamp"] < 0.4:
        rcx, rcy, rw, rh, ang = det["rect"]
        a = math.radians(ang)
        ux = np.array([math.cos(a), math.sin(a)])
        uy = np.array([-math.sin(a), math.cos(a)])
        c = np.array([rcx, rcy])
        pc = [view.pt(*(c + sx * ux * rw / 2 + sy * uy * rh / 2))
              for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        _corner_brackets(img, pc, WHITE, thick=max(2, int(3 * s)))
        z, frac = det["depths"].get(src, (float("nan"), 0.0))
        txt = f"{z:.2f} m" if z == z else "NO DEPTH"
        sc = 1.1 * s
        tw = _text_w(txt, sc, 2)
        top = min(p[1] for p in pc)
        cxp = int(sum(p[0] for p in pc) / 4)
        tx, ty = cxp - tw // 2, max(int((190 if src == "fused" else 130) * s), top - int(18 * s))
        _panel(img, (tx - int(14 * s), ty - int(36 * s)), (tx + tw + int(14 * s), ty + int(12 * s)),
               int(14 * s), PANEL if z == z else RED_DIM)
        post.append(lambda im, txt=txt, tx=tx, ty=ty, sc=sc: _text(im, txt, (tx, ty), sc, WHITE, 2))

    # title (top left)
    title = source_name(tracker, src)
    sub = SOURCE_SUB.get(src, "")
    if src == "fused":
        sub = FUSION_SUB.get(getattr(tracker, "fusion_engine", "heuristic"), sub)
    if src == "stereo" and getattr(tracker, "stereo_engine", "classic") == "neural":
        sub = "2 cameras + neural network"
    tsc, ssc = 1.0 * s, 0.5 * s
    bw = max(_text_w(title, tsc, 2), _text_w(sub, ssc)) + int(44 * s)
    bottom = int(92 * s)
    if mix is not None:                      # fused: big ToF / stereo share row under the title
        bw = max(bw, _mix_row_w(s) + int(44 * s))
        bottom = int(152 * s)
        post.append(lambda im: _mix_row(im, int(30 * s), int(104 * s), s, mix))
    _panel(img, (int(14 * s), int(14 * s)), (int(14 * s) + bw, bottom), int(14 * s), PANEL)
    post.append(lambda im: (
        cv2.rectangle(im, (int(26 * s), int(28 * s)), (int(32 * s), int(78 * s)), col, -1),
        _text(im, title, (int(44 * s), int(56 * s)), tsc, WHITE, 2),
        _text(im, sub, (int(44 * s), int(80 * s)), ssc, MUTED)))

    # colour scale (right edge)
    bx0, by0, by1 = w - int(36 * s), int(120 * s), int(h * 0.58)
    bar = _colour_bar(int(12 * s), by1 - by0)
    img[by0:by1, bx0:bx0 + bar.shape[1]] = bar
    for frac_y, mm in ((0.0, min_mm), (1.0, max_mm)):
        lbl = f"{mm / 1000:.2f}"
        post.append(lambda im, lbl=lbl, fy=frac_y: _text(
            im, lbl, (bx0 - _text_w(lbl, 0.45 * s) - int(6 * s), int(by0 + fy * (by1 - by0)) + int(5 * s)),
            0.45 * s, WHITE))

    # graph panel (bottom): palm distance from every source over the last window_s seconds
    gx0, gx1 = int(14 * s), w - int(14 * s)
    gy0, gy1 = h - int(250 * s), h - int(14 * s)
    _panel(img, (gx0, gy0), (gx1, gy1), int(14 * s), PANEL)
    px0, px1 = gx0 + int(64 * s), gx1 - int(20 * s)
    py0, py1 = gy0 + int(58 * s), gy1 - int(18 * s)
    zmax = max_mm / 1000.0

    def ypix(z):
        return int(py1 - (min(max(z, 0.0), zmax) / zmax) * (py1 - py0))

    for zt in np.arange(0.0, zmax + 1e-6, 0.25 if zmax <= 1.6 else 0.5):
        yy = ypix(zt)
        cv2.line(img, (px0, yy), (px1, yy), (70, 66, 64), 1, AA)
        _text(img, f"{zt:.2f}", (gx0 + int(12 * s), yy + int(5 * s)), 0.42 * s, MUTED)
    ref = (getattr(tracker, "hud", {}) or {}).get("reference_depth")
    if ref:
        _dashed(img, (px0, ypix(ref)), (px1, ypix(ref)), WHITE, dash=int(8 * s))
        _text(img, "follow distance", (px1 - _text_w("follow distance", 0.42 * s), ypix(ref) - int(6 * s)),
              0.42 * s, WHITE)

    now = time.monotonic()
    hist = [(t, v) for t, v in list(tracker.history) if now - t <= window_s]
    order = [x for x in tracker.available_sources if x != src] + [src]   # selected one on top
    for name in order:
        pts, seg = [], []
        last_t = None
        for t, vals in hist:
            z = vals.get(name, float("nan")) if vals else float("nan")
            if z != z or (last_t is not None and t - last_t > 0.4):
                if len(seg) > 1:
                    pts.append(seg)
                seg = []
            if z == z:
                x = int(px1 - (now - t) / window_s * (px1 - px0))
                seg.append((x, ypix(z)))
            last_t = t
        if len(seg) > 1:
            pts.append(seg)
        c = SOURCE_COL[name]
        thick = max(2, int((4 if name == src else 2) * s))
        if name != src:
            c = tuple(int(0.65 * v) for v in c)
        for seg in pts:
            cv2.polylines(img, [np.array(seg, np.int32)], False, c, thick, AA)

    # legend with the current values (dropouts in red)
    lx, ly = gx0 + int(20 * s), gy0 + int(36 * s)
    cur = det["depths"] if det is not None and now - det["stamp"] < 0.4 else {}
    for name in tracker.available_sources:
        z = cur.get(name, (float("nan"), 0))[0]
        nm = source_name(tracker, name)
        sel = name == src
        cv2.circle(img, (lx + int(7 * s), ly - int(7 * s)), int(7 * s), SOURCE_COL[name], -1, AA)
        if sel:
            cv2.circle(img, (lx + int(7 * s), ly - int(7 * s)), int(11 * s), WHITE, max(1, int(2 * s)), AA)
        _text(img, nm, (lx + int(28 * s), ly), 0.55 * s, WHITE if sel else MUTED, 2 if sel else 1)
        val = f"{z:.2f} m" if z == z else ("NO DEPTH" if cur else "--")
        xv = lx + int(28 * s) + _text_w(nm, 0.55 * s, 2) + int(10 * s)
        _text(img, val, (xv, ly), 0.55 * s, RED if (cur and z != z) else WHITE, 2 if sel else 1)
        lx = xv + _text_w(val, 0.55 * s, 2) + int(36 * s)

    for draw in post:
        draw(img)
    if getattr(tracker, "show_hud", True):
        _sys_chip(img, tracker, s, top=int(14 * s), right=w - int(70 * s), view="depth")
    return img


_CLK = {"t": 0.0, "pct": None}


def cpu_clock_pct():
    """Lowest core clock as % of its max (100 = no throttling), from sysfs, cached 1 s."""
    now = time.monotonic()
    if now - _CLK["t"] > 1.0:
        _CLK["t"] = now
        vals = []
        for d in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq"):
            try:
                vals.append(int(open(f"{d}/scaling_cur_freq").read()) / int(open(f"{d}/cpuinfo_max_freq").read()))
            except (OSError, ValueError, ZeroDivisionError):
                pass
        _CLK["pct"] = 100.0 * min(vals) if vals else None
    return _CLK["pct"]


def _sys_chip(img, tracker, s: float, top: int | None = None, right: int | None = None,
              view: str = "rgb") -> None:
    """Top-right readout. Line 1: chip temperature + CPU load (amber > 70 C, red > 85 C).
    Line 2: live speed - this view's output fps and the AI / stereo / ToF rates, so a recording
    shows by itself whether performance changes with temperature."""
    h, w = img.shape[:2]
    sc = 0.55 * s
    pad = int(12 * s)
    right = w - int(14 * s) if right is None else right
    top = int(14 * s) if top is None else top
    info = getattr(tracker, "sysinfo", None)
    if info and time.monotonic() - getattr(tracker, "sysinfo_stamp", 0.0) > 5.0:
        info = None
    # line 2 text
    vf = getattr(tracker, "view_fps", None)
    parts = []
    if vf is not None:
        parts.append(f"VIEW {vf(view):.1f} fps")
    rate = getattr(tracker, "rate", None)
    if rate is not None:
        parts += [f"AI {rate('det'):.0f}", f"STEREO {rate('stereo'):.0f}", f"ToF {rate('tof'):.0f}"]
        if getattr(tracker, "fusion_engine", "heuristic") != "heuristic":
            parts.append(f"FUSED {rate('fused'):.0f}")
    perf_txt = "   ".join(parts)
    sc2 = 0.48 * s
    pw = _text_w(perf_txt, sc2) if perf_txt else 0
    # line 1 text
    temp = info.get("temp_c", float("nan")) if info else float("nan")
    cpu = info.get("cpu_pct") if info else None
    t_txt = f"{temp:.0f}" if temp == temp else "--"
    clk = cpu_clock_pct()
    c_txt = (f"CPU {cpu:.0f}%" if cpu is not None else "") + (f"   CLK {clk:.0f}%" if clk is not None else "")
    ring = max(2, int(3 * s))
    tw = _text_w(t_txt, sc, 2) + 2 * ring + 6 + _text_w("C", sc, 2)
    cw = _text_w(c_txt, sc) if c_txt else 0
    line1 = tw + (int(16 * s) + cw if c_txt else 0) if info else 0
    if not line1 and not pw:
        return
    width = pad + max(line1, pw) + pad
    rows = (1 if line1 else 0) + (1 if pw else 0)
    x0, y0, x1, y1 = right - width, top, right, top + int(10 * s) + rows * int(28 * s)
    _panel(img, (x0, y0), (x1, y1), int(12 * s), PANEL)
    base_y = y0 + int(27 * s)
    if line1:
        col = RED if temp > 85 else (AMBER if temp > 70 else WHITE)
        x = x1 - pad - line1                       # right-aligned
        _text(img, t_txt, (x, base_y), sc, col, 2)
        x += _text_w(t_txt, sc, 2) + ring + 3
        cv2.circle(img, (x, base_y - int(12 * s)), ring, col, max(1, int(s)), AA)
        _text(img, "C", (x + ring + 3, base_y), sc, col, 2)
        if c_txt:
            _text(img, c_txt, (x1 - pad - cw, base_y), sc, AMBER if (clk is not None and clk < 90) else MUTED)
        base_y += int(28 * s)
    if pw:
        _text(img, perf_txt, (x1 - pad - pw, base_y - int(2 * s)), sc2, WHITE)


DEPTH_WORK = 480     # px: depth view fusion / colouring resolution (upscaled to depth_view_size)
_DEPTH_CACHE: dict = {}
_BAR: dict = {}


def _colour_bar(w: int, h: int) -> np.ndarray:
    if (w, h) not in _BAR:
        lut = cv2.applyColorMap(np.arange(255, -1, -1, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_TURBO)
        _BAR[(w, h)] = cv2.resize(lut, (w, h), interpolation=cv2.INTER_LINEAR)
    return _BAR[(w, h)]


def _panel(img, p0, p1, r, color, alpha: float = 0.72) -> None:
    """Translucent rounded panel, blended only inside its own rectangle (cheap)."""
    h, w = img.shape[:2]
    x0, y0 = max(0, int(p0[0])), max(0, int(p0[1]))
    x1, y1 = min(w, int(p1[0])), min(h, int(p1[1]))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    _rounded(over, (int(p0[0]) - x0, int(p0[1]) - y0), (int(p1[0]) - x0, int(p1[1]) - y0), r, color)
    cv2.addWeighted(over, alpha, roi, 1.0 - alpha, 0, dst=roi)
