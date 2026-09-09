"""Browser tool for the second round of occlusion annotation.

Run: python tools/annotate_occlusion_round2.py <scene>
Labels and event logs are saved in local_record/annotation_records/<scene>/v2/.
This tool reads and saves only the second round.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_SCENE_DIR = PROJECT_ROOT / "dataset" / "per_scene"
RAW_DIR = PROJECT_ROOT / "dataset" / "raw" / "pigs" / "pig"
ANNOTATION_DIR = PROJECT_ROOT / "local_record" / "annotation_records"
SPLITS = ("val", "test")
SPLIT_TO_RAW_SUBDIR = {"val": "valid", "test": "test"}
LABELS = {"occluded", "not_occluded"}
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    temporary.replace(path)


@dataclass(frozen=True)
class AnnotationPaths:
    scene: str
    dataset_dir: Path
    root: Path

    @classmethod
    def from_scene(cls, scene: str) -> "AnnotationPaths":
        dataset_dir = PER_SCENE_DIR / scene
        if not dataset_dir.is_dir():
            raise SystemExit(f"dataset directory not found: {dataset_dir}")
        return cls(scene, dataset_dir, ANNOTATION_DIR / scene / "v2")

    def labels(self, split: str) -> Path:
        return self.root / f"labels.{split}.json"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"


def load_records(dataset_dir: Path, category_id: int = 2) -> list[dict]:
    """Load pig instances from val and test annotations."""
    records = []
    for split in SPLITS:
        annotation_path = dataset_dir / f"{split}.coco.json"
        if not annotation_path.exists():
            raise SystemExit(f"annotation file not found: {annotation_path}")
        data = load_json(annotation_path)
        images = {int(image["id"]): image for image in data["images"]}
        for annotation in data["annotations"]:
            if int(annotation.get("category_id", -1)) != category_id:
                continue
            image = images[int(annotation["image_id"])]
            records.append({
                "ann_id": int(annotation["id"]),
                "image_id": int(image["id"]),
                "bbox": [float(value) for value in annotation["bbox"]],
                "segmentation": annotation.get("segmentation", []),
                "file_name": image["file_name"],
                "split": split,
                "image_w": int(image["width"]),
                "image_h": int(image["height"]),
            })
        if not any(record["split"] == split for record in records):
            raise SystemExit(f"no category {category_id} instances found for {split}")
    return records


def records_by_split(records: list[dict]) -> dict[str, list[dict]]:
    return {
        split: [record for record in records if record["split"] == split]
        for split in SPLITS
    }


def write_label_file(path: Path, *, scene: str, split: str,
                     labels: dict[int, str]) -> None:
    invalid = set(labels.values()) - LABELS
    if invalid:
        raise ValueError(f"invalid labels: {sorted(invalid)}")
    atomic_write_json(path, {
        "schema_version": SCHEMA_VERSION,
        "dataset": scene,
        "round": "v2",
        "split": split,
        "source": "blind second annotation",
        "updated_at": utc_now(),
        "labels": {str(key): labels[key] for key in sorted(labels)},
    })


def read_label_file(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    document = load_json(path)
    labels = {int(key): value for key, value in document.get("labels", {}).items()}
    invalid = set(labels.values()) - LABELS
    if invalid:
        raise SystemExit(f"invalid labels in {path}: {sorted(invalid)}")
    return labels


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


class AnnotationApp:
    def __init__(self, paths: AnnotationPaths, records: list[dict]):
        self.paths = paths
        self.queue = records
        for split, rows in records_by_split(records).items():
            expected = {record["ann_id"] for record in rows}
            existing = read_label_file(paths.labels(split))
            if not expected:
                raise SystemExit(f"no instances found for {split}")
            if set(existing) - expected:
                raise SystemExit(f"unknown instance IDs in second round/{split}; check existing records")
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.cursor = 0
        self._advance_to_unlabeled()

    def _labels(self, split: str) -> dict[int, str]:
        return read_label_file(self.paths.labels(split))

    def _advance_to_unlabeled(self) -> None:
        while self.cursor < len(self.queue):
            record = self.queue[self.cursor]
            if record["ann_id"] not in self._labels(record["split"]):
                break
            self.cursor += 1

    def current(self):
        return self.queue[self.cursor] if self.cursor < len(self.queue) else None

    def set_label(self, label: str) -> None:
        if label not in LABELS:
            raise ValueError(label)
        record = self.current()
        if record is None:
            return
        split = record["split"]
        labels = self._labels(split)
        previous = labels.get(record["ann_id"])
        labels[record["ann_id"]] = label
        write_label_file(
            self.paths.labels(split), scene=self.paths.scene,
            split=split, labels=labels,
        )
        append_event(self.paths.events, {
            "timestamp": utc_now(),
            "split": split,
            "ann_id": record["ann_id"],
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "previous_label": previous,
            "new_label": label,
        })
        self.cursor += 1
        self._advance_to_unlabeled()

    def previous(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1
        elif self.cursor >= len(self.queue) and self.queue:
            self.cursor = len(self.queue) - 1

    def next_labeled(self) -> None:
        record = self.current()
        if record is None:
            return
        if record["ann_id"] not in self._labels(record["split"]):
            raise ValueError("the current instance must be labeled before it can be skipped")
        self.cursor += 1

    def snapshot(self) -> dict:
        current = self.current()
        counts = {}
        for split in SPLITS:
            labels = self._labels(split)
            counts[split] = {
                "labeled": len(labels),
                "occluded": sum(value == "occluded" for value in labels.values()),
                "not_occluded": sum(value == "not_occluded" for value in labels.values()),
            }
        if current is not None:
            output = dict(current)
            output["current_label"] = self._labels(current["split"]).get(current["ann_id"])
            output["position"] = self.cursor + 1
            output["queue_total"] = len(self.queue)
        else:
            output = None
        return {
            "dataset": self.paths.scene,
            "current": output,
            "queue_total": len(self.queue),
            "counts": counts,
        }


APP: AnnotationApp | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/state":
            self._send_json(APP.snapshot())
        elif path.startswith("/image/"):
            self._serve_image(path)
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/action":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            action = request.get("action")
            if action == "occluded":
                APP.set_label("occluded")
            elif action == "not_occluded":
                APP.set_label("not_occluded")
            elif action == "previous":
                APP.previous()
            elif action == "next":
                APP.next_labeled()
            else:
                raise ValueError(f"unknown action: {action}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(APP.snapshot())

    def _serve_image(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) != 4:
            self.send_error(404)
            return
        split, filename = parts[2], unquote(parts[3])
        if Path(filename).name != filename:
            self.send_error(400)
            return
        raw_subdir = SPLIT_TO_RAW_SUBDIR.get(split)
        image_path = RAW_DIR / raw_subdir / filename if raw_subdir else None
        if image_path is None or not image_path.is_file():
            self.send_error(404)
            return
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
            image_path.suffix.lower(), "application/octet-stream"
        )
        self._send(image_path.read_bytes(), mime)

    def _send_json(self, value, status: int = 200) -> None:
        self._send(json.dumps(value).encode("utf-8"), "application/json", status)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTML_PAGE = r"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Occlusion review</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px; color: #202124; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: baseline; }
    h2 { margin: 0; font-size: 20px; }
    #status { color: #555; font-size: 14px; }
    #labels { margin-top: 8px; color: #555; min-height: 22px; }
    #workspace { display: flex; gap: 22px; align-items: flex-start; margin-top: 12px; }
    #full-wrap { position: relative; }
    #full { display: block; border: 1px solid #aaa; }
    #box { position: absolute; border: 3px solid #d22; box-sizing: border-box; pointer-events: none; }
    #crop { display: block; border: 1px solid #aaa; image-rendering: pixelated; }
    .caption { color: #777; font-size: 12px; margin-bottom: 4px; }
    #actions { margin-top: 16px; }
    button { padding: 9px 18px; margin-right: 8px; font-size: 15px; cursor: pointer; }
    button:disabled { opacity: .5; cursor: wait; }
    .occ { color: white; background: #b42318; border: 1px solid #b42318; }
    .clean { background: #eef6ee; border: 1px solid #7a9d7a; }
    .nav { background: #f3f3f3; border: 1px solid #aaa; }
    #done { display: none; margin-top: 30px; color: #666; font-size: 20px; }
  </style>
</head>
<body>
<header><h2 id="title">Occlusion review</h2><div id="status"></div></header>
<div id="labels"></div>
<div id="workspace">
  <div><div class="caption">Full image</div><div id="full-wrap"><img id="full"><div id="box"></div></div></div>
  <div><div class="caption">Current instance and ground-truth mask</div><canvas id="crop"></canvas></div>
</div>
<div id="actions"></div><div id="done">Review queue completed.</div>
<script>
const MAX_W = 1000, MAX_H = 700;
let state;
let busy = false;
function setBusy(value) {
  busy = value;
  document.querySelectorAll('#actions button').forEach(button => button.disabled = value);
}
function showError() {
  document.getElementById('actions').style.display = 'none';
  alert('Save or image loading failed. Refresh the page to check progress before continuing.');
}
const getImage = url => new Promise((resolve, reject) => {
  const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = url;
});
async function refresh() { state = await (await fetch('/state')).json(); await render(); }
async function act(action) {
  if (busy) return;
  setBusy(true);
  try {
    const response = await fetch('/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action})});
    const value = await response.json();
    if (!response.ok) { alert(value.error || 'Save failed'); return; }
    state = value;
    await render();
    setBusy(true);
    await new Promise(resolve => setTimeout(resolve, 300));
  } catch (error) {
    showError();
  } finally {
    setBusy(false);
  }
}
async function render() {
  const current = state.current;
  const totals = Object.values(state.counts).reduce((a, x) => a + x.labeled, 0);
  document.getElementById('title').textContent = `${state.dataset} · second-round annotation`;
  document.getElementById('status').textContent = `Labeled ${totals}/${state.queue_total}` + (current ? ` · Position ${current.position}/${current.queue_total}` : '');
  if (!current) {
    document.getElementById('workspace').style.display = 'none'; document.getElementById('actions').style.display = 'none'; document.getElementById('done').style.display = 'block'; return;
  }
  document.getElementById('workspace').style.display = 'flex'; document.getElementById('actions').style.display = 'block'; document.getElementById('done').style.display = 'none';
  const labels = document.getElementById('labels');
  labels.textContent = current.current_label ? `Current label: ${current.current_label}` : '';
  const url = `/image/${current.split}/${encodeURIComponent(current.file_name)}`;
  const image = await getImage(url); const scale = Math.min(MAX_W/image.width, MAX_H/image.height, 1);
  const full = document.getElementById('full'); full.src = url; full.style.width = Math.round(image.width*scale)+'px'; full.style.height = Math.round(image.height*scale)+'px';
  const [x,y,w,h] = current.bbox; const box = document.getElementById('box'); box.style.left=x*scale+'px'; box.style.top=y*scale+'px'; box.style.width=w*scale+'px'; box.style.height=h*scale+'px';
  const canvas = document.getElementById('crop'); canvas.width=Math.max(1,Math.round(w)); canvas.height=Math.max(1,Math.round(h)); const ctx=canvas.getContext('2d'); ctx.drawImage(image,x,y,w,h,0,0,canvas.width,canvas.height);
  if (Array.isArray(current.segmentation)) { ctx.fillStyle='rgba(40,180,100,.38)'; ctx.strokeStyle='rgba(10,120,60,.95)'; ctx.lineWidth=1.5; for (const poly of current.segmentation) { if (!Array.isArray(poly)||poly.length<6) continue; ctx.beginPath(); for(let i=0;i<poly.length;i+=2){ const px=poly[i]-x,py=poly[i+1]-y; if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);} ctx.closePath();ctx.fill();ctx.stroke(); } }
  document.getElementById('actions').innerHTML = `<button class="occ" onclick="act('occluded')">Occluded</button><button class="clean" onclick="act('not_occluded')">Not occluded</button><button class="nav" onclick="act('previous')">Previous</button><button class="nav" onclick="act('next')">Next</button>`;
}
refresh().catch(showError);
</script>
</body></html>
"""


def serve(paths: AnnotationPaths, records: list[dict], port: int) -> None:
    global APP
    for record in records:
        image_path = RAW_DIR / SPLIT_TO_RAW_SUBDIR[record["split"]] / record["file_name"]
        if not image_path.is_file():
            raise SystemExit(f"image not found: {image_path}")
    APP = AnnotationApp(paths, records)
    print(f"dataset : {paths.scene}")
    print(f"queue   : {len(APP.queue)}")
    print(f"output  : {paths.root}")
    server = HTTPServer(("localhost", port), Handler)
    print(f"open    : http://localhost:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=["pig", "frame", "stream", "mix"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    paths = AnnotationPaths.from_scene(args.scene)
    records = load_records(paths.dataset_dir)
    serve(paths, records, args.port)


if __name__ == "__main__":
    main()
