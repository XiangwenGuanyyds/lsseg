"""Browse dataset images and consensus occlusion groups.

Run: python tools/view_dataset.py mix
"""

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "per_scene"
IMAGE_ROOT = PROJECT_ROOT / "dataset" / "raw" / "pigs" / "pig"
SPLITS = {"train": "train", "val": "valid", "test": "test"}
GROUPS = {"all", "occluded", "not_occluded"}


class DatasetViewer:
    def __init__(self, name, dataset_root=DATASET_ROOT, image_root=IMAGE_ROOT):
        self.name = name
        self.directory = dataset_root / name
        if Path(name).name != name or name in {".", ".."} or not self.directory.is_dir():
            raise ValueError(f"Dataset not found: {name} (expected a directory under {dataset_root})")
        self.image_root = image_root
        self.records = []
        self.splits = []
        for split in SPLITS:
            path = self.directory / f"{split}.coco.json"
            if not path.is_file():
                continue
            with path.open() as handle:
                coco = json.load(handle)
            images = {image["id"]: image for image in coco["images"]}
            self.splits.append(split)
            for ann in coco["annotations"]:
                if ann["category_id"] != 2:
                    continue
                image = images[ann["image_id"]]
                self.records.append({
                    "index": len(self.records), "split": split,
                    "ann_id": ann["id"], "image_id": image["id"],
                    "file_name": image["file_name"],
                    "bbox": ann["bbox"], "segmentation": ann.get("segmentation", []),
                })
        if not self.records:
            raise ValueError(f"No pig instances found in dataset: {name}")

    def page(self, split="all", group="all", index=0):
        if split != "all" and split not in self.splits:
            raise ValueError("Invalid split or group")
        if group not in GROUPS:
            raise ValueError("Invalid split or group")
        selected_splits = self.splits if split == "all" else [split]
        result = {"dataset": self.name, "splits": self.splits, "total": 0,
                  "image_count": 0, "position": 0, "current": None,
                  "message": "", "commands": []}
        if group != "all":
            selected_splits = [s for s in selected_splits if s in {"val", "test"}]
            if not selected_splits:
                result["message"] = "Consensus groups are available for validation and test splits only."
                return result
            groups = {}
            missing = []
            for current_split in selected_splits:
                labels = {}
                expected = {r["ann_id"] for r in self.records if r["split"] == current_split}
                for label in ("occluded", "not_occluded", "ambiguous"):
                    path = self.directory / "occlusion_review" / "consensus" / f"{label}.{current_split}.json"
                    if not path.is_file():
                        missing.append(path.name)
                        continue
                    with path.open() as handle:
                        values = json.load(handle)
                    if not isinstance(values, list):
                        raise ValueError(f"Invalid consensus ID list: {path.name}")
                    labels[label] = {int(value) for value in values}
                if len(labels) == 3:
                    occluded = labels["occluded"]
                    clean = labels["not_occluded"]
                    ambiguous = labels["ambiguous"]
                    overlap = (occluded & clean) | (occluded & ambiguous) | (clean & ambiguous)
                    if overlap or (occluded | clean | ambiguous) != expected:
                        raise ValueError(f"Consensus labels do not match the {current_split} split.")
                    groups[current_split] = labels[group]
            if missing:
                result["message"] = (
                    "Consensus labels were not detected or are incomplete. "
                    "Complete both annotation rounds, then generate the consensus labels. "
                    "Select the group again after running these commands.\nMissing: " + ", ".join(missing)
                )
                result["commands"] = [
                    f"python tools/annotate_occlusion_round1.py {self.name}",
                    f"python tools/annotate_occlusion_round2.py {self.name}",
                    f"python tools/annotate_occlusion_consensus.py {self.name}",
                ]
                return result
        records = []
        for record in self.records:
            if record["split"] not in selected_splits:
                continue
            if group != "all" and record["ann_id"] not in groups[record["split"]]:
                continue
            records.append(record)
        result["total"] = len(records)
        result["image_count"] = len({(r["split"], r["image_id"]) for r in records})
        if records:
            index = max(0, min(index, len(records) - 1))
            result["position"] = index + 1
            result["current"] = records[index]
        else:
            result["message"] = "No instances in this selection."
        return result

    def image_path(self, index):
        if index < 0 or index >= len(self.records):
            raise ValueError("Invalid image index")
        record = self.records[index]
        directory = (self.image_root / SPLITS[record["split"]]).resolve()
        path = (directory / record["file_name"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Invalid image path")
        return path


class Handler(BaseHTTPRequestHandler):
    timeout = 15

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        args = parse_qs(url.query)
        viewer = self.server.viewer
        try:
            if url.path == "/":
                self.send_body(HTML.encode(), "text/html; charset=utf-8")
            elif url.path == "/state":
                split = args.get("split", ["all"])[0]
                group = args.get("group", ["all"])[0]
                index = int(args.get("index", ["0"])[0])
                value = viewer.page(split, group, index)
                self.send_body(json.dumps(value).encode(), "application/json")
            elif url.path == "/image":
                path = viewer.image_path(int(args.get("index", ["-1"])[0]))
                if not path.is_file():
                    raise FileNotFoundError(f"Image not found: {path.name}")
                mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                self.send_body(path.read_bytes(), mime)
            else:
                self.send_error(404)
        except (ValueError, OSError, KeyError) as error:
            self.send_body(json.dumps({"error": str(error)}).encode(), "application/json", 400)

    def send_body(self, body, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


HTML = r"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset viewer</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f6;color:#242a2b;font:15px Arial,sans-serif;letter-spacing:0}
header{padding:18px 24px;background:#fff;border-bottom:1px solid #cdd3d3;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
h1{font-size:22px;margin:0}#count{color:#596367;font-size:14px;font-variant-numeric:tabular-nums}
main{padding:20px 24px}.toolbar{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin-bottom:18px}
label{display:flex;flex-direction:column;gap:6px;font-size:13px}select,input,button{font:inherit;height:36px;border:1px solid #a9b4b5;background:#fff;border-radius:4px;color:inherit;padding:0 10px}
select{max-width:100%}button{cursor:pointer}button:hover{background:#e4eeeb}button:disabled{opacity:.45;cursor:default}
.navigation{display:flex;align-items:center;gap:8px;margin-left:auto}input{width:84px;text-align:center}#total{min-width:55px;font-variant-numeric:tabular-nums}
#message{background:#fff6da;border-left:3px solid #b97b13;padding:16px;white-space:pre-line;line-height:1.5}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;margin:12px 0 0}#workspace{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:20px}
figure{margin:0;min-width:0}figcaption{font-size:14px;font-weight:600;margin-bottom:10px}.surface{height:560px;max-height:65vh;background:#e5e9e8;display:flex;align-items:center;justify-content:center;overflow:hidden}
canvas{display:block;width:100%;height:100%;object-fit:contain}#details{font-size:13px;color:#505c60;overflow-wrap:anywhere;line-height:1.6;margin-top:14px}
[hidden]{display:none!important}@media(max-width:760px){header{padding:16px}main{padding:16px}.navigation{margin-left:0;flex-wrap:wrap}#workspace{grid-template-columns:1fr}.surface{height:340px;max-height:50vh}.toolbar{gap:12px}}
</style></head>
<body><header><h1 id="title">Dataset viewer</h1><span id="count"></span></header>
<main><div class="toolbar">
<label>Split<select id="split"><option value="all">All splits</option></select></label>
<label>Group<select id="group"><option value="all">All instances</option><option value="occluded">Occluded (consensus)</option><option value="not_occluded">Not occluded (consensus)</option></select></label>
<div class="navigation"><button id="previous">Previous</button><label>Instance<input id="position" type="number" min="1" value="1" aria-label="Instance number"></label><span id="total">/ 0</span><button id="next">Next</button></div>
</div><div id="message" hidden><span id="messageText"></span><pre id="commands" hidden></pre></div>
<div id="workspace" hidden><figure><figcaption>Full image</figcaption><div class="surface"><canvas id="full"></canvas></div></figure>
<figure><figcaption>Current instance and ground-truth mask</figcaption><div class="surface"><canvas id="crop"></canvas></div></figure></div>
<div id="details"></div></main>
<script>
const $ = id => document.getElementById(id);
let busy = false, state = null, initialised = false;
function controls() {
  $('split').disabled = $('group').disabled = busy;
  $('position').disabled = busy || !state?.current;
  $('previous').disabled = busy || !state?.current || state.position <= 1;
  $('next').disabled = busy || !state?.current || state.position >= state.total;
}
function message(text, commands=[]) {
  $('message').hidden = false;
  $('messageText').textContent = text;
  $('commands').textContent = commands.join('\n');
  $('commands').hidden = !commands.length;
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Image could not be loaded. Check the dataset image files.'));
    img.src = url;
  });
}

async function show(index=0) {
  if (busy) return;
  busy = true;
  controls();
  $('workspace').hidden = true;
  $('message').hidden = true;
  $('details').textContent = '';
  try {
    const query = new URLSearchParams({
      split: $('split').value,
      group: $('group').value,
      index
    });
    const response = await fetch('/state?' + query);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to load dataset.');
    state = data;
    if (!initialised) {
      const names = {train:'Training',val:'Validation',test:'Test'};
      data.splits.forEach(split => $('split').add(new Option(names[split],split)));
      initialised = true;
    }
    $('title').textContent = 'Dataset: ' + data.dataset;
    $('count').textContent = `${data.image_count} images · ${data.total} instances`;
    $('position').value = data.position || 1;
    $('position').max = Math.max(1,data.total);
    $('total').textContent = '/ ' + data.total;
    if (!data.current) {
      message(data.message, data.commands);
      return;
    }

    const r = data.current;
    const img = await loadImage('/image?index=' + r.index);
    const full = $('full');
    full.width = img.width;
    full.height = img.height;
    const ctx = full.getContext('2d');
    ctx.drawImage(img,0,0);
    const [x,y,w,h] = r.bbox;
    ctx.strokeStyle = '#f0b12b';
    ctx.lineWidth = Math.max(3,img.width/350);
    ctx.strokeRect(x,y,w,h);

    const crop = $('crop');
    crop.width = Math.max(1,Math.ceil(w));
    crop.height = Math.max(1,Math.ceil(h));
    const c = crop.getContext('2d');
    c.drawImage(img,x,y,w,h,0,0,w,h);
    if (Array.isArray(r.segmentation)) {
      c.fillStyle = 'rgba(0,160,120,.38)';
      c.strokeStyle = '#00875f';
      c.lineWidth = Math.max(1,w/220);
      for (const poly of r.segmentation) {
        if (!Array.isArray(poly) || poly.length < 6) continue;
        c.beginPath();
        for (let i=0; i<poly.length; i+=2) {
          const px = poly[i]-x, py = poly[i+1]-y;
          if (i===0) c.moveTo(px,py);
          else c.lineTo(px,py);
        }
        c.closePath();
        c.fill();
        c.stroke();
      }
    }
    $('details').textContent = `${r.split} · Image ID ${r.image_id} · Annotation ID ${r.ann_id} · ${r.file_name}`;
    $('workspace').hidden = false;
  } catch(error) {
    state = null;
    $('count').textContent = '';
    $('total').textContent = '/ 0';
    message(error.message);
  } finally {
    busy = false;
    controls();
  }
}
$('split').onchange = $('group').onchange = () => show(0);
$('previous').onclick = () => show(state.position-2);
$('next').onclick = () => show(state.position);
$('position').onchange = () => {
  const position = Number.parseInt($('position').value,10) || 1;
  show(Math.max(0,position-1));
};
show();
</script></body></html>
"""


def open_browser(url):
    try:
        opened = webbrowser.open_new_tab(url)
    except (webbrowser.Error, OSError):
        opened = False
    if not opened:
        print(f"Could not open a browser automatically. Open this address: {url}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset directory name, for example mix")
    args = parser.parse_args()
    try:
        viewer = DatasetViewer(args.dataset)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except OSError as error:
        parser.exit(1, f"Error: Could not start the local viewer: {error}\n")
    server.viewer = viewer
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Dataset: {viewer.name}", flush=True)
    print(f"Open: {url}", flush=True)
    try:
        Thread(target=open_browser, args=(url,), daemon=True).start()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
