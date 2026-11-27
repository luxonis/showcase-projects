"""Tiny web view: http://<camera-ip>:<port>/

  /            both streams side by side + depth-source buttons (stereo / ToF / fused)
  /rgb         RGB stream only (full window - for screen recording)
  /depth       depth stream only (full window)
  /rgb.mjpg    /depth.mjpg   raw MJPEG streams (/stream.mjpg = /rgb.mjpg, older links)
  /state       JSON: current depth source, available sources, toggles
  POST /source/<stereo|tof|fused>   switch the depth source - also what the arm follows
  POST /toggle/<hud|labels>         show / hide the HUD panel or the video labels
  POST /tune/tof_confidence/<0..1>  live ToF confidence threshold (tof_confidence in the YAML)
  GET  /debug/depth                 dtype / coverage / percentiles of every depth map
  GET  /debug/perf                  rates (palm NN, depth maps), render ms / fps, control loop, CPU clocks
  GET  /object                      JSON: held object (tracker status, observation), grasp state /
                                    target, calibration marks
  GET  /object/mask.jpg             current object mask (white) / selection cue (grey)
  POST /gripper/<open|close>        manual gripper command (close also records an object mark)
  POST /object/mark                 record the current object position as a calibration mark
  POST /object/target/last          use the last mark as the grasp target (until redeploy)
  POST /object/release              open the gripper after an automatic grasp
  POST /object/clear                forget the tracked object (select again)
  GET  /object/debug                why the selection cue does / does not find an object (JSON)
  GET  /object/debug.jpg            search region: colour | skin (white) + hand depth (grey) |
                                    candidate pixels | depth
  POST /object/skin/relearn         learn the skin colour again (show the empty palm)
  POST /object/auto_grip/<on|off>   enable / disable closing automatically
  GET  /arm                         JSON: current joints (deg), follow home, drop settings
  POST /drop/record                 current joints -> drop pose (hand-guide the arm there first)
  POST /drop/via/add | /drop/via/clear   add the current joints as a via pose / remove all
  POST /drop/run                    run the drop sequence now (only while an object is gripped)
  POST /drop/<on|off>               drop after a grasp, or just hold the object

Keyboard on every page: 1 stereo, 2 ToF, 3 fused, h HUD, l labels.
"""

from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from ur_oak_control.overlay import render, render_depth

_STYLE = """
:root{--bg:#0e0f11;--panel:#17191c;--line:#2a2d31;--text:#eceef0;--muted:#9aa0a6;
--stereo:#32aaf0;--tof:#ff9632;--fused:#50e16e}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
font:15px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
"""

PAGE_BOTH = ("""<!doctype html><html><head><meta charset="utf-8"><title>OAK4 hand follow</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>""" + _STYLE + """
body{display:flex;flex-direction:column;min-height:100%}
.views{flex:1;display:flex;gap:12px;padding:12px;align-items:center;justify-content:center;min-height:0}
.views figure{margin:0;flex:1 1 0;min-width:0;display:flex;flex-direction:column;align-items:center;gap:6px}
.views img{max-width:100%;max-height:calc(100vh - 120px);border-radius:10px;display:block;background:#000}
figcaption{color:var(--muted);font-size:13px}figcaption a{color:inherit}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:center;padding:0 12px 14px}
.seg{display:inline-flex;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:4px;gap:4px}
.seg button,.tog{font:600 15px/1 inherit;font-family:inherit;color:var(--muted);background:transparent;
border:0;border-radius:9px;padding:12px 18px;cursor:pointer;display:flex;align-items:center;gap:8px}
.seg button i{width:10px;height:10px;border-radius:50%;display:inline-block;background:var(--c)}
.seg button.on{background:var(--c);color:#111}.seg button.on i{background:#111}
.seg button:disabled{opacity:.3;cursor:not-allowed}
.tog{background:var(--panel);border:1px solid var(--line)}.tog.on{color:var(--text);border-color:var(--muted)}
.sys{color:var(--muted);font:600 14px/1 ui-monospace,monospace;font-variant-numeric:tabular-nums;min-width:12ch}
.sys.warm{color:#f0b429}.sys.hot{color:#ff5d5d}
kbd{font:12px ui-monospace,monospace;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:1px 5px}
@media (max-width:800px){.views{flex-direction:column}.views img{max-height:none}}
</style></head><body>
<div class="views">
<figure><img src="/rgb.mjpg" alt="RGB camera view"><figcaption>RGB &middot; <a href="/rgb">full window</a></figcaption></figure>
<figure><img src="/depth.mjpg" alt="Depth view"><figcaption>Depth &middot; <a href="/depth">full window</a></figcaption></figure>
</div>
<div class="bar">
<div class="seg" role="group" aria-label="Depth source">
<button data-src="stereo" style="--c:var(--stereo)"><i></i>%STEREO_LABEL% <kbd>1</kbd></button>
<button data-src="tof" style="--c:var(--tof)"><i></i>ToF <kbd>2</kbd></button>
<button data-src="fused" style="--c:var(--fused)"><i></i>Fused <kbd>3</kbd></button>
</div>
<button class="tog" data-tog="hud">HUD <kbd>h</kbd></button>
<button class="tog" data-tog="labels">Labels <kbd>l</kbd></button>
<span id="sys" class="sys"></span>
</div>
""" + "%SCRIPT%" + "</body></html>")

PAGE_ONE = ("""<!doctype html><html><head><meta charset="utf-8"><title>OAK4 %TITLE%</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>""" + _STYLE + """
body{display:flex;align-items:center;justify-content:center;overflow:hidden}
img{max-width:100vw;max-height:100vh;display:block}
</style></head><body><img src="%SRC%" alt="%TITLE%">""" + "%SCRIPT%" + "</body></html>")

SCRIPT = """<script>
const keys={'1':'stereo','2':'tof','3':'fused'};
function apply(s){document.querySelectorAll('[data-src]').forEach(b=>{
  b.classList.toggle('on',b.dataset.src===s.source);b.disabled=!s.available.includes(b.dataset.src);});
  document.querySelectorAll('[data-tog]').forEach(b=>b.classList.toggle('on',!!s[b.dataset.tog]));
  const el=document.getElementById('sys');if(el&&s.sys){const t=s.sys.temp_c,c=s.sys.cpu_pct;
  el.textContent=(t!=null&&isFinite(t)?Math.round(t)+' \u00b0C':'')+(c!=null?'  CPU '+Math.round(c)+' %':'');
  el.className='sys'+(t>85?' hot':t>70?' warm':'');}}
async function post(p){try{const r=await fetch(p,{method:'POST'});if(r.ok)apply(await r.json());}catch(e){}}
async function poll(){try{apply(await (await fetch('/state')).json());}catch(e){}}
document.querySelectorAll('[data-src]').forEach(b=>b.onclick=()=>post('/source/'+b.dataset.src));
document.querySelectorAll('[data-tog]').forEach(b=>b.onclick=()=>post('/toggle/'+b.dataset.tog));
addEventListener('keydown',e=>{if(e.repeat||e.ctrlKey||e.metaKey||e.altKey)return;
  if(keys[e.key])post('/source/'+keys[e.key]);else if(e.key==='h')post('/toggle/hud');
  else if(e.key==='l')post('/toggle/labels');});
poll();setInterval(poll,700);
</script>"""


class _Renderer:
    """Renders + JPEG-encodes in its own thread, only while somebody is watching."""

    def __init__(self, name: str, draw, fps: float = 12.0, quality: int = 85):
        self.name, self.draw, self.period, self.quality = name, draw, 1.0 / fps, quality
        self.jpeg: bytes | None = None
        self.clients = 0
        self.ms_ema = 0.0                       # render + JPEG encode time
        self.produced: list = []                # timestamps of produced frames (rate)
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name=f"web-{name}").start()

    def _run(self) -> None:
        last_err = None
        while True:
            t0 = time.monotonic()
            if self.clients > 0:
                try:
                    img = self.draw()
                    if img is not None:
                        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                        if ok:
                            self.jpeg = buf.tobytes()
                            done = time.monotonic()
                            ms = 1000 * (done - t0)
                            self.ms_ema = ms if self.ms_ema == 0 else 0.8 * self.ms_ema + 0.2 * ms
                            self.produced = [t for t in self.produced if done - t <= 5.0] + [done]
                    last_err = None
                except Exception as exc:  # a drawing bug must never take the app down
                    msg = f"{type(exc).__name__}: {exc}"
                    if msg != last_err:
                        print(f"[web_view] {self.name} render error: {msg}", flush=True)
                    last_err = msg
            time.sleep(max(0.01, self.period - (time.monotonic() - t0)))


def start_web_view(tracker, port: int, rotation: int = 0, fps: float = 12.0,
                   depth_size: int = 960, depth_range_m=(0.15, 1.5)) -> ThreadingHTTPServer:
    def draw_rgb():
        frame = tracker.view_frame
        return None if frame is None else render(frame, tracker, rotation)

    def draw_depth():
        if tracker.view_frame is None:
            return None
        return render_depth(tracker, rotation, depth_size,
                            1000 * float(depth_range_m[0]), 1000 * float(depth_range_m[1]))

    renderers = {"/rgb.mjpg": _Renderer("rgb", draw_rgb, fps),
                 "/depth.mjpg": _Renderer("depth", draw_depth, fps)}
    renderers["/stream.mjpg"] = renderers["/rgb.mjpg"]

    def view_fps(name: str) -> float:     # shown on the stream itself (overlay._sys_chip)
        rd = renderers.get(f"/{name}.mjpg")
        now = time.monotonic()
        return sum(1 for t in list(rd.produced) if now - t <= 5.0) / 5.0 if rd else 0.0

    tracker.view_fps = view_fps
    pages = {
        "/": PAGE_BOTH.replace("%SCRIPT%", SCRIPT).replace(
            "%STEREO_LABEL%", "Neural stereo" if tracker.stereo_engine == "neural" else "Stereo"),
        "/rgb": PAGE_ONE.replace("%SRC%", "/rgb.mjpg").replace("%TITLE%", "RGB").replace("%SCRIPT%", SCRIPT),
        "/depth": PAGE_ONE.replace("%SRC%", "/depth.mjpg").replace("%TITLE%", "depth").replace("%SCRIPT%", SCRIPT),
    }

    def cpu_freqs() -> dict:
        """Current / max clock per core in MHz (throttling shows as cur << max under load)."""
        out = {}
        import glob
        for d in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq")):
            try:
                cur = int(open(f"{d}/scaling_cur_freq").read()) // 1000
                mx = int(open(f"{d}/cpuinfo_max_freq").read()) // 1000
                out[d.split("/")[-2]] = [cur, mx]
            except (OSError, ValueError):
                pass
        return out

    def perf() -> dict:
        now = time.monotonic()
        r = {name: {"ms": round(rd.ms_ema, 1),
                    "fps": round(sum(1 for t in rd.produced if now - t <= 5.0) / 5.0, 1),
                    "clients": rd.clients}
             for name, rd in (("rgb", renderers["/rgb.mjpg"]), ("depth", renderers["/depth.mjpg"]))}
        return {"det_fps": round(tracker.rate("det"), 1), "stereo_fps": round(tracker.rate("stereo"), 1),
                "tof_fps": round(tracker.rate("tof"), 1),
                "fused_fps": round(tracker.rate("fused"), 1), "fusion_engine": tracker.fusion_engine, "render": r,
                "ctrl": getattr(tracker, "ctrl_perf", {}), "cpu_mhz": cpu_freqs(),
                "sys": {k: (v if v == v else None) for k, v in (getattr(tracker, "sysinfo", None) or {}).items()},
                "source": tracker.depth_source}

    def state() -> dict:
        return {"source": tracker.depth_source, "available": tracker.available_sources,
                "hud": tracker.show_hud, "labels": tracker.show_labels,
                "sys": {k: (v if v == v else None) for k, v in (getattr(tracker, "sysinfo", None) or {}).items()}}

    def drop_post(parts) -> tuple[int, dict]:
        drop = getattr(tracker, "drop", None)
        if drop is None:
            return 404, {"error": "drop not available"}
        q = getattr(tracker, "arm_joints", None)
        deg = [round(math.degrees(v), 2) for v in q] if q else None
        if parts in (["drop", "record"], ["drop", "via", "add"]):
            if deg is None:
                return 409, {"error": "no joint states from the driver"}
            if parts[1] == "record":
                drop["joints_deg"] = deg
            else:
                drop["via_deg"].append(deg)
            print(f"[web_view] drop {'pose' if parts[1] == 'record' else 'via pose'} recorded: {deg} deg "
                  f"(copy into drop: in the YAML to keep it)", flush=True)
            return 200, {"drop": drop}
        if parts == ["drop", "via", "clear"]:
            drop["via_deg"] = []
            return 200, {"drop": drop}
        if parts == ["drop", "run"]:
            tracker.commands.append(("drop", None))
            return 200, {"drop": "requested"}
        if len(parts) == 2 and parts[1] in ("on", "off"):
            drop["enabled"] = parts[1] == "on"
            return 200, {"drop": drop}
        return 404, {"error": "unknown command"}

    def object_post(parts) -> tuple[int, dict]:
        """Gripper / held-object commands; gripper ones run in the control loop (tracker.commands)."""
        ho = tracker.held
        cmds = tracker.commands
        if parts[0] == "drop":
            return drop_post(parts)
        if parts == ["gripper", "close"] or parts == ["gripper", "open"]:
            mark = ho.mark(f"gripper {parts[1]}") if (ho and parts[1] == "close") else None
            cmds.append(("gripper", parts[1] == "close"))
            return 200, {"gripper": parts[1], "mark": mark}
        if ho is None:
            return 404, {"error": "grasping off"}
        if parts == ["object", "mark"]:
            m = ho.mark("manual")
            return (200 if m else 409), {"mark": m}
        if parts == ["object", "target", "last"]:
            m = ho.marks[-1] if ho.marks else None
            ok = ho.set_target(m)
            if ok:
                print(f"[web_view] grasp target set from mark: {ho.grasp_cfg.grasp_target}", flush=True)
            return (200 if ok else 409), {"grasp_target": ho.grasp_cfg.grasp_target}
        if parts == ["object", "release"]:
            cmds.append(("release", None))
            return 200, {"release": True}
        if parts == ["object", "skin", "relearn"]:
            ho.relearn_skin()
            return 200, {"skin_ref": None}
        if parts == ["object", "clear"]:
            ho.clear(reason="web view")
            return 200, {"cleared": True}
        if len(parts) == 3 and parts[:2] == ["object", "auto_grip"] and parts[2] in ("on", "off"):
            ho.grasp_cfg.auto_grip = parts[2] == "on"
            return 200, {"auto_grip": ho.grasp_cfg.auto_grip}
        return 404, {"error": "unknown command"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep app logs clean
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int = 200) -> None:
            self._send(code, "application/json", json.dumps(state()).encode())

        def do_POST(self):
            parts = self.path.strip("/").split("/")
            if len(parts) == 2 and parts[0] == "source":
                self._json(200 if tracker.set_depth_source(parts[1]) else 409)
            elif len(parts) == 3 and parts[:2] == ["tune", "tof_confidence"]:
                try:
                    ok = tracker.set_tof_confidence(float(parts[2]))
                except ValueError:
                    ok = False
                self._send(200 if ok else 400, "application/json",
                           json.dumps({"tof_confidence": tracker.tof_confidence}).encode())
            elif parts[0] in ("gripper", "object", "drop"):
                code, body = object_post(parts)
                self._send(code, "application/json", json.dumps(body).encode())
            elif len(parts) == 2 and parts[0] == "toggle" and parts[1] in ("hud", "labels"):
                attr = "show_" + parts[1]
                setattr(tracker, attr, not getattr(tracker, attr))
                self._json()
            else:
                self.send_error(404)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in pages:
                self._send(200, "text/html; charset=utf-8", pages[path].encode())
                return
            if path == "/state":
                self._json()
                return
            if path == "/debug/perf":
                self._send(200, "application/json", json.dumps(perf()).encode())
                return
            if path == "/arm":
                q = getattr(tracker, "arm_joints", None)
                body = {"joints_deg": [round(math.degrees(v), 2) for v in q] if q else None,
                        "home_deg": getattr(tracker, "arm_home", None),
                        "drop": getattr(tracker, "drop", None)}
                self._send(200, "application/json", json.dumps(body).encode())
                return
            if path == "/object":
                ho = tracker.held
                self._send(200 if ho else 404, "application/json",
                           json.dumps(ho.snapshot() if ho else {"error": "grasping off"}).encode())
                return
            if path == "/object/debug":
                ho = tracker.held
                self._send(200 if ho else 404, "application/json",
                           json.dumps(ho.debug_info() if ho else {"error": "grasping off"}).encode())
                return
            if path == "/object/debug.jpg":
                ho = tracker.held
                img = ho.debug_image() if ho else None
                if img is None:
                    self.send_error(404)
                    return
                self._send(200, "image/jpeg", cv2.imencode(".jpg", img)[1].tobytes())
                return
            if path == "/object/mask.jpg":
                ho = tracker.held
                if ho is None:
                    self.send_error(404)
                    return
                img = np.zeros((tracker.size, tracker.size), np.uint8)
                cue, mask = ho.cue, ho.mask
                if cue is not None:
                    img[cue["mask"]] = 110
                if mask is not None:
                    img[mask] = 255
                self._send(200, "image/jpeg", cv2.imencode(".jpg", img)[1].tobytes())
                return
            if path == "/debug/depth":
                self._send(200, "application/json", json.dumps(tracker.depth_stats()).encode())
                return
            renderer = renderers.get(path)
            if renderer is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            with renderer.lock:
                renderer.clients += 1
            try:
                while True:
                    jpeg = renderer.jpeg
                    if jpeg is not None and jpeg is not last:
                        last = jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with renderer.lock:
                    renderer.clients -= 1

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="web-view").start()
    return server
