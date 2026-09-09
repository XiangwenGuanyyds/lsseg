"""Browser tool for the first round of occlusion annotation.

Run: python tools/annotate_occlusion_round1.py <scene>
Use --review to revisit occluded instances, or --port to change the server port.
Scenes: pig, frame, stream, mix.

Records are saved under local_record/annotation_records/<scene>/first_round/:
    occluded.val.json, occluded.test.json
    not_occluded.val.json, not_occluded.test.json

Each click saves the labels for the current split. Already labelled instances
are skipped when continuing a session. Restart clears the not-occluded records
and keeps the occluded records. The consensus tool reads these files directly.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PER_SCENE_DIR = PROJECT_ROOT / "dataset" / "per_scene"
RAW_DIR       = PROJECT_ROOT / "dataset" / "raw" / "pigs" / "pig"
ANNOTATION_DIR = PROJECT_ROOT / "local_record" / "annotation_records"

# val.coco.json's images live under raw/valid/, test.coco.json's under raw/test/.
SPLIT_TO_RAW_SUBDIR = {"val": "valid", "test": "test"}


# ============================================================
# Disk helpers
# ============================================================
def load_id_set(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return {int(x) for x in json.load(f)}


def save_id_set(path: Path, ids: set) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(ids), f)


def load_scene_records(scene: str) -> list:
    """Read val.coco.json + test.coco.json of one scene, flatten into per-ann
    records preserving order (val first, then test)."""
    scene_dir = PER_SCENE_DIR / scene
    if not scene_dir.is_dir():
        raise SystemExit(f"[occlusion_round1] scene dir not found: {scene_dir}")

    records = []
    for split in ("val", "test"):
        json_path = scene_dir / f"{split}.coco.json"
        if not json_path.exists():
            raise SystemExit(f"annotation file not found: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        image_map = {img["id"]: img for img in data["images"]}
        for ann in data["annotations"]:
            img = image_map[ann["image_id"]]
            # segmentation: COCO polygon list form (list[list[float]]), each
            # inner list flat [x1,y1,x2,y2,...] in original-image coords. RLE
            # form (dict with counts/size) isn't drawn in JS — fall through
            # as-is and the canvas overlay simply skips it.
            records.append({
                "ann_id":       int(ann["id"]),
                "image_id":     int(img["id"]),
                "bbox":         [float(v) for v in ann["bbox"]],   # COCO xywh
                "segmentation": ann.get("segmentation", []),
                "file_name":    img["file_name"],
                "split":        split,
                "image_w":      int(img["width"]),
                "image_h":      int(img["height"]),
            })
    return records


# ============================================================
# Session state
# ============================================================
class AppState:

    def __init__(self, scene: str, review_mode: bool):
        self.scene = scene
        self.review_mode = review_mode
        self.records = load_scene_records(scene)

        record_dir = ANNOTATION_DIR / scene / "first_round"
        record_dir.mkdir(parents=True, exist_ok=True)
        self.occ_paths = {
            "val":  record_dir / "occluded.val.json",
            "test": record_dir / "occluded.test.json",
        }
        self.state_paths = {
            "val":  record_dir / "not_occluded.val.json",
            "test": record_dir / "not_occluded.test.json",
        }

        for split in ("val", "test"):
            expected = {r["ann_id"] for r in self.records if r["split"] == split}
            occluded, clean = self._occluded(split), self._viewed(split)
            if not expected:
                raise SystemExit(f"no instances found for {split}")
            if (occluded & clean) or ((occluded | clean) - expected):
                raise SystemExit(f"check first-round records for {split}: overlapping or unknown instance IDs")

        if review_mode:
            self.queue = [r for r in self.records
                          if r["ann_id"] in self._occluded(r["split"])]
        else:
            self.queue = list(self.records)

        self.cursor = 0
        if not review_mode:
            self._advance_to_unseen()

    # --- file IO (re-read every time so disk is the source of truth) -----
    def _occluded(self, split):
        return load_id_set(self.occ_paths[split])

    def _viewed(self, split):
        return load_id_set(self.state_paths[split])

    def _save_occluded(self, split, ids):
        save_id_set(self.occ_paths[split], ids)

    def _save_viewed(self, split, ids):
        save_id_set(self.state_paths[split], ids)

    # --- queue navigation ------------------------------------------------
    def _advance_to_unseen(self):
        while self.cursor < len(self.queue):
            r = self.queue[self.cursor]
            judged = self._occluded(r["split"]) | self._viewed(r["split"])
            if r["ann_id"] in judged:
                self.cursor += 1
            else:
                break

    # --- read snapshot for /state ---------------------------------------
    def snapshot(self):
        if self.cursor >= len(self.queue):
            current = None
        else:
            r = self.queue[self.cursor]
            split, ann_id = r["split"], r["ann_id"]
            occ, viewed = self._occluded(split), self._viewed(split)
            if ann_id in occ:
                cur_state = "occluded"
            elif ann_id in viewed:
                cur_state = "viewed"
            else:
                cur_state = "unseen"
            current = {
                **r,
                "current_state": cur_state,
                "position":      self.cursor + 1,
                "queue_total":   len(self.queue),
            }
        return {
            "scene":      self.scene,
            "mode":       "review" if self.review_mode else "annotation",
            "current":    current,
            "n_total_records":   len(self.records),
            "n_occluded":        sum(len(self._occluded(s)) for s in ("val", "test")),
            "n_viewed":          sum(len(self._viewed(s))   for s in ("val", "test")),
        }

    # --- actions ---------------------------------------------------------
    def _current_record(self):
        if self.cursor >= len(self.queue):
            return None
        return self.queue[self.cursor]

    def mark_occluded(self):
        r = self._current_record()
        if r is None:
            return
        split, ann_id = r["split"], r["ann_id"]
        occ, viewed = self._occluded(split), self._viewed(split)
        occ.add(ann_id)
        viewed.discard(ann_id)
        self._save_occluded(split, occ)
        self._save_viewed(split, viewed)
        self.cursor += 1
        if not self.review_mode:
            self._advance_to_unseen()

    def mark_clean(self):
        """Annotation mode 'next', or review mode 'unmark' — both move ann_id
        to the viewed set and advance one step."""
        r = self._current_record()
        if r is None:
            return
        split, ann_id = r["split"], r["ann_id"]
        occ, viewed = self._occluded(split), self._viewed(split)
        occ.discard(ann_id)
        viewed.add(ann_id)
        self._save_occluded(split, occ)
        self._save_viewed(split, viewed)
        self.cursor += 1
        if not self.review_mode:
            self._advance_to_unseen()

    def just_next(self):
        """Review mode 'next' — no state change, just advance."""
        if self.cursor < len(self.queue):
            self.cursor += 1

    def previous(self):
        """Step one ann_id back in the queue. No state change. No skip."""
        if self.cursor > 0:
            self.cursor -= 1
        elif self.cursor >= len(self.queue) and len(self.queue) > 0:
            self.cursor = len(self.queue) - 1

    def restart(self):
        """Annotation-mode only: clear viewed sets (occluded untouched), reset
        cursor to first unseen."""
        for s in ("val", "test"):
            self._save_viewed(s, set())
        self.cursor = 0
        if not self.review_mode:
            self._advance_to_unseen()


# ============================================================
# HTTP handler
# ============================================================
APP: AppState = None  # set in main()


class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # quiet default access log

    # ----- routing ------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/state":
            self._send_json(APP.snapshot())
        elif path.startswith("/image/"):
            self._serve_image(path)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/action":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self.send_error(400, "bad json")
            return
        action = req.get("action")
        dispatch = {
            "mark_occluded":   APP.mark_occluded,
            "mark_clean":      APP.mark_clean,
            "unmark_occluded": APP.mark_clean,   # alias for review-mode UI
            "next":            APP.just_next,
            "previous":        APP.previous,
            "restart":         APP.restart,
        }
        fn = dispatch.get(action)
        if fn is None:
            self.send_error(400, f"unknown action: {action}")
            return
        fn()
        self._send_json(APP.snapshot())

    # ----- helpers ------------------------------------------------------
    def _send_bytes(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj):
        self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json")

    def _serve_image(self, path: str):
        # /image/<split>/<filename>
        parts = path.split("/")
        if len(parts) != 4:
            self.send_error(404)
            return
        _, _, split, filename = parts
        filename = unquote(filename)
        subdir = SPLIT_TO_RAW_SUBDIR.get(split)
        if subdir is None:
            self.send_error(404)
            return
        file_path = RAW_DIR / subdir / filename
        if not file_path.exists():
            self.send_error(404)
            return
        ext = file_path.suffix.lower().lstrip(".")
        content_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        }.get(ext, "application/octet-stream")
        with open(file_path, "rb") as f:
            self._send_bytes(f.read(), content_type)


# ============================================================
# Single-page UI
# ============================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>occ annotator</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px; color: #222; }
    h2 { margin: 0 0 8px 0; }
    #header { display: flex; justify-content: space-between; align-items: baseline; gap: 20px; }
    #counters { font-size: 14px; color: #555; }
    #main { display: flex; gap: 24px; align-items: flex-start; margin-top: 12px; }
    #full-container { position: relative; }
    #full-image { display: block; border: 1px solid #ccc; }
    #bbox-overlay { position: absolute; border: 3px solid red; pointer-events: none; box-sizing: border-box; }
    #crop-canvas { display: block; border: 1px solid #ccc; image-rendering: pixelated; }
    #crop-label, #full-label { font-size: 12px; color: #888; margin-bottom: 4px; }
    #actions { margin-top: 18px; }
    button { font-size: 15px; padding: 8px 18px; margin-right: 8px; cursor: pointer; }
    button:disabled { opacity: .5; cursor: wait; }
    .primary { background: #d33; color: white; border: none; }
    .secondary { background: #eee; }
    .restart { background: #444; color: white; border: none; margin-left: 30px; }
    .badge { padding: 2px 8px; border-radius: 4px; font-weight: 600; }
    .badge-occluded { background: #fee; color: #c00; }
    .badge-viewed   { background: #ffeed1; color: #b06; }
    .badge-unseen   { background: #e7f5e7; color: #050; }
    .badge-split-val  { background: #e0ecff; color: #1452a3; }
    .badge-split-test { background: #fff1d6; color: #a86412; }
    #done-banner { font-size: 22px; color: #888; margin: 30px 0; }
  </style>
</head>
<body>

<div id="header">
  <h2 id="title">…</h2>
  <div id="counters">…</div>
</div>

<div id="main">
  <div>
    <div id="full-label">原图（红框 = 当前实例 bbox）</div>
    <div id="full-container">
      <img id="full-image" src="">
      <div id="bbox-overlay"></div>
    </div>
  </div>
  <div>
    <div id="crop-label">bbox 抠图</div>
    <canvas id="crop-canvas" width="0" height="0"></canvas>
  </div>
</div>

<div id="done-banner" style="display:none">— 队列已结束 —</div>

<div id="actions"></div>

<script>
const FULL_MAX_W  = 1000;  // 原图显示宽度上限（超过等比缩小，bbox 框跟着缩）
const FULL_MAX_H  = 700;   // 原图显示高度上限

let state = null;
let busy = false;

function setBusy(value) {
  busy = value;
  document.querySelectorAll('#actions button').forEach(button => button.disabled = value);
}

function showError() {
  document.getElementById('actions').style.display = 'none';
  alert('保存或加载失败，请刷新页面核对当前进度后继续。');
}

async function refresh() {
  const res = await fetch("/state");
  state = await res.json();
  await render();
}

async function action(name) {
  if (busy) return;
  setBusy(true);
  try {
    const res = await fetch("/action", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action: name}),
    });
    if (!res.ok) throw new Error('Request failed');
    state = await res.json();
    await render();
    setBusy(true);
    await new Promise(resolve => setTimeout(resolve, 300));
  } catch (error) {
    showError();
  } finally {
    setBusy(false);
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

async function render() {
  document.getElementById("title").textContent =
    `场景 [${state.scene}] · ${state.mode === "review" ? "查看遮挡" : "标注"}模式`;

  const cur = state.current;
  const main = document.getElementById("main");
  const banner = document.getElementById("done-banner");
  const actions = document.getElementById("actions");
  const counters = document.getElementById("counters");

  // counter line
  const occN = state.n_occluded, viewN = state.n_viewed, total = state.n_total_records;
  if (state.mode === "annotation") {
    counters.innerHTML =
      `已判定 ${occN + viewN}/${total}` +
      `（遮挡 ${occN} · 非遮挡 ${viewN}）` +
      (cur ? ` · 当前位置 ${cur.position}/${cur.queue_total}` : "");
  } else {
    counters.innerHTML =
      `共 ${occN} 个遮挡` +
      (cur ? ` · 当前位置 ${cur.position}/${cur.queue_total}` : "");
  }

  if (!cur) {
    main.style.display = "none";
    banner.style.display = "block";
    actions.innerHTML = `<button class="secondary" onclick="action('previous')">上一张</button>`;
    return;
  }

  main.style.display = "flex";
  banner.style.display = "none";

  // ---- load image once, reuse for both <img> and canvas ----
  const url = `/image/${cur.split}/${encodeURIComponent(cur.file_name)}`;
  const img = await loadImage(url);

  // full image — fixed display size; bbox overlay scaled by the same factor
  const fullImg = document.getElementById("full-image");
  fullImg.src = url;
  const fullScale = Math.min(FULL_MAX_W / img.width, FULL_MAX_H / img.height, 1);
  fullImg.style.width  = Math.round(img.width  * fullScale) + "px";
  fullImg.style.height = Math.round(img.height * fullScale) + "px";
  const overlay = document.getElementById("bbox-overlay");
  const [bx, by, bw, bh] = cur.bbox;
  overlay.style.left   = (bx * fullScale) + "px";
  overlay.style.top    = (by * fullScale) + "px";
  overlay.style.width  = (bw * fullScale) + "px";
  overlay.style.height = (bh * fullScale) + "px";

  // crop canvas — 1:1 with original image pixels (no scaling)
  const canvas = document.getElementById("crop-canvas");
  const ctx = canvas.getContext("2d");
  canvas.width  = Math.round(bw);
  canvas.height = Math.round(bh);
  ctx.drawImage(img, bx, by, bw, bh, 0, 0, canvas.width, canvas.height);

  // Overlay GT mask on crop. COCO segmentation is list[list[float]] in
  // original-image coords ([x1,y1,x2,y2,...] flat). Translate to crop-local
  // coords via (p - bbox_origin); no scaling because canvas is 1:1.
  const segs = cur.segmentation;
  if (Array.isArray(segs)) {
    ctx.fillStyle   = "rgba(60, 200, 120, 0.40)";
    ctx.strokeStyle = "rgba(20, 140, 70, 0.95)";
    ctx.lineWidth   = 1.5;
    for (const poly of segs) {
      if (!Array.isArray(poly) || poly.length < 6) continue;
      ctx.beginPath();
      for (let i = 0; i < poly.length; i += 2) {
        const px = poly[i]     - bx;
        const py = poly[i + 1] - by;
        if (i === 0) ctx.moveTo(px, py);
        else         ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  }

  // split + state badges appended to title
  const labelMap = {occluded: "已标遮挡", viewed: "已标非遮挡", unseen: "未判定"};
  document.getElementById("title").innerHTML =
    `场景 [${state.scene}] · ${state.mode === "review" ? "查看遮挡" : "标注"}模式 ` +
    `<span class="badge badge-split-${cur.split}">${cur.split}</span> ` +
    `<span class="badge badge-${cur.current_state}">${labelMap[cur.current_state]}</span>`;

  // buttons
  if (state.mode === "annotation") {
    actions.innerHTML = `
      <button class="primary"   onclick="action('mark_occluded')">标记遮挡</button>
      <button class="secondary" onclick="action('mark_clean')">下一张（非遮挡）</button>
      <button class="secondary" onclick="action('previous')">上一张</button>
      <button class="restart"   onclick="confirmRestart()">重新开始</button>
    `;
  } else {
    actions.innerHTML = `
      <button class="primary"   onclick="action('unmark_occluded')">改为非遮挡</button>
      <button class="secondary" onclick="action('next')">下一张</button>
      <button class="secondary" onclick="action('previous')">上一张</button>
    `;
  }
}

function confirmRestart() {
  if (busy) return;
  if (confirm("确定要重新开始吗？\n所有'非遮挡'判定会被清空（已标遮挡保留）。")) {
    action("restart");
  }
}

refresh().catch(showError);
</script>
</body>
</html>
"""


# ============================================================
# Entrypoint
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Browser-based occlusion annotator.")
    parser.add_argument("scene", choices=["pig", "frame", "stream", "mix"],
                        help="Which per-scene directory to annotate.")
    parser.add_argument("--review", action="store_true",
                        help="Review-occluded mode: walk currently-occluded items "
                             "and optionally unmark.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global APP
    APP = AppState(args.scene, args.review)

    for record in APP.records:
        image_path = RAW_DIR / SPLIT_TO_RAW_SUBDIR[record["split"]] / record["file_name"]
        if not image_path.is_file():
            raise SystemExit(f"image not found: {image_path}")

    print(f"[occlusion_round1] scene  = {args.scene}")
    print(f"[occlusion_round1] mode   = {'review' if args.review else 'annotation'}")
    print(f"[occlusion_round1] queue  = {len(APP.queue)} items")
    print(f"[occlusion_round1] occ    = {APP.occ_paths['val']}")
    print(f"[occlusion_round1]          {APP.occ_paths['test']}")
    print(f"[occlusion_round1] state  = {APP.state_paths['val']}")
    print(f"[occlusion_round1]          {APP.state_paths['test']}")
    print(f"[occlusion_round1] open   = http://localhost:{args.port}/")

    httpd = HTTPServer(("localhost", args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[occlusion_round1] bye")


if __name__ == "__main__":
    main()
