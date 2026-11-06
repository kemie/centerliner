// Runs the Python engine off the page's main thread, so the page (and its spinner) stays responsive.
// Messages out: {type:"progress", text, step, steps} | {type:"ready"} | {type:"fail", error} | {type:"result", id, json}
import { initWasm, Resvg } from "https://cdn.jsdelivr.net/npm/@resvg/resvg-wasm@2.6.2/index.mjs";
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.mjs";

const VERSION = new URL(import.meta.url).searchParams.get("v") || "0";
const ENGINE_FILES = ["centerline_exact", "centerline", "exact_edges", "exact_arcs", "exact_snap", "exact_ovals", "common", "webconvert"];
const STEPS = 5;
const progress = (step, text) => postMessage({ type: "progress", step, steps: STEPS, text });
const tick = () => new Promise((r) => setTimeout(r, 0));

let py = null, convertFn = null;

async function boot() {
  progress(1, "Loading the SVG renderer…");
  await initWasm(fetch("https://cdn.jsdelivr.net/npm/@resvg/resvg-wasm@2.6.2/index_bg.wasm"));
  globalThis.renderSvg = (svg, width) => new Resvg(svg, { fitTo: { mode: "width", value: width } }).render().asPng();

  progress(2, "Starting Python…");
  py = await loadPyodide();

  progress(3, "Loading the maths libraries (the big download, only slow on a first visit)…");
  await py.loadPackage(["numpy", "scipy", "scikit-image", "pillow", "shapely", "micropip"]);

  progress(4, "Loading path tools…");
  await py.pyimport("micropip").install("svgpathtools");

  progress(5, "Starting the converter…");
  py.FS.mkdirTree("/engine");
  for (const f of ENGINE_FILES) {
    const r = await fetch(new URL(`engine/${f}.py`, import.meta.url).href + "?v=" + VERSION);  // version from the page: new version = fresh files
    if (!r.ok) throw new Error(`could not load engine/${f}.py`);
    py.FS.writeFile(`/engine/${f}.py`, await r.text());
  }
  py.runPython(`
import sys, types, js
shim = types.ModuleType("resvg_py")  # the desktop renderer's name, answered by the in-browser renderer
def svg_to_bytes(svg_path=None, svg_string=None, width=None, **kw):
    if svg_string is None:
        svg_string = open(svg_path, encoding="utf-8").read()
    return js.renderSvg(svg_string, int(width)).to_bytes()
shim.svg_to_bytes = svg_to_bytes
sys.modules["resvg_py"] = shim
sys.path.insert(0, "/engine")
import webconvert
`);
  convertFn = py.pyimport("webconvert").convert;
  postMessage({ type: "ready" });
}

// Only the newest request is worth computing: while one runs, newer ones replace older queued ones.
let latest = null, running = false;
async function drain() {
  running = true;
  while (latest) {
    const m = latest; latest = null;
    let json;
    try { json = convertFn(m.svg, m.radius); }
    catch (e) { json = JSON.stringify({ error: "Could not convert this file. It may be unsupported." }); console.error(e); }
    postMessage({ type: "result", id: m.id, json });
    await tick();
  }
  running = false;
}
onmessage = (e) => {
  if (e.data.type !== "convert") return;
  latest = e.data;
  if (convertFn && !running) setTimeout(drain, 0);
};
const queuedBeforeReady = () => { if (latest && !running) setTimeout(drain, 0); };

boot().then(queuedBeforeReady).catch((e) => { console.error(e); postMessage({ type: "fail", error: String(e) }); });
