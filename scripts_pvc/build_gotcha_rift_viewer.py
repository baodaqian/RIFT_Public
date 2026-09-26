#!/usr/bin/env python3
"""Pack an ``export_gotcha_rift_pointcloud_pvc.py`` directory into one self-contained 3D viewer page (A64).

Voxels are stored as grid indices (uint8 x3) plus sqrt-energy quantized to uint16 against the set's peak; the
mesh surface samples as uint16 positions over the region cube. Everything is inlined as base64, and three.js
0.128 (build + OrbitControls) loads from jsdelivr.

    python scripts_pvc/build_gotcha_rift_viewer.py EXPORT_DIR OUT.html
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps

# Rows of the selector: (row label, description, [(chip, set label)]); '{e}' in a chip becomes the set's epoch.
ROWS = [
    ('Measured data', 'A^H y of every TRAIN pulse, 12.5 cm voxels: what the data alone images',
     [('raw', 'data_raw'), ('v2 phase', 'data_v2')]),
    ('T5b', '524k point cap · uncorrected data · trilinear densify', [('ep 10', 'T5b_ep10'), ('ep 16', 'T5b_ep16'), ('ep 24', 'T5b_ep24')]),
    ('M5b', '1M point cap · uncorrected data', [('ep 10', 'M5b_ep10'), ('ep 16', 'M5b_ep16'), ('ep 24', 'M5b_ep24')]),
    ('CM5b', '1M point cap · first-round (b56) phase correction', [('ep 10', 'CM5b_ep10'), ('ep 16', 'CM5b_ep16'), ('ep 24', 'CM5b_ep24')]),
    ('M5c · CM5c', '1M cap · half step (target 0.5) · uncorrected / b56', [('M5c 24', 'M5c_ep24'), ('CM5c 24', 'CM5c_ep24')]),
    ('T5c · I5', '524k cap · uncorrected · half step / render-preserving start', [('T5c 24', 'T5c_ep24'), ('I5 24', 'I5_ep24')]),
    ('D2', 'SH degree 2 · 524k cap · uncorrected', [('ep 10', 'D2_ep10'), ('ep 24', 'D2_ep24'), ('ep 40', 'D2_ep40')]),
    ('F5full', 'all 305 IDs · v2 phase correction · 1M cap · 40 epochs (running)',
     [('ep 10', 'F5full_ep10'), ('ep 16', 'F5full_ep16'), ('ep 24', 'F5full_ep24'), ('now · ep {e}', 'F5full_now')]),
    ('F5c', 'as F5full with the half step (target 0.5) · running', [('now · ep {e}', 'F5c_now')]),
    ('Gain twins', 'F5full twins: A pooled gain (extended to 40) · B learnable gain · D curvature start (12 epochs)',
     [('A · ep {e}', 'F5gA_now'), ('B · ep {e}', 'F5gB_now'), ('D · ep {e}', 'F5gD_now')]),
    ('Extended to 70', 'CM5b and CM5c (b56 data) and D2, continued with the curvature guard',
     [('CM5b · ep {e}', 'CM5bx70_now'), ('CM5c · ep {e}', 'CM5cx70_now'), ('D2 · ep {e}', 'D2x70_now')]),
]


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def main(export_dir, out):
    export_dir = Path(export_dir)
    index = json.loads((export_dir / 'index.json').read_text())
    extent = index['extent_m']
    sets, payload = {}, {}
    for info in index['sets']:
        data = np.load(export_dir / f"{info['label']}.npy").astype(np.float64)
        voxel = info['voxel_m']
        ijk = np.rint((data[:, :3] + extent) / voxel - 0.5).astype(int)
        grid = int(round(2 * extent / voxel))
        if ijk.min() < 0 or ijk.max() >= grid or grid > 255:
            raise ValueError(f"{info['label']}: voxel indices outside the grid")
        amp = np.sqrt(np.clip(data[:, 3], 0, None))
        q = np.round(amp / amp.max() * 65535).astype('<u2')
        payload[info['label']] = dict(ijk=b64(ijk.astype(np.uint8)), q=b64(q))
        meta = {k: info.get(k) for k in ('label', 'kind', 'epoch', 'fit_epoch', 'train_rel_mse', 'train_rho',
                                         'val_rel_mse', 'val_rho', 'points_active', 'max_sh_order', 'kept',
                                         'kept_energy_share', 'voxel_m', 'on_car', 'cic_best')}
        meta.update(grid=grid, count=int(len(q)))
        sets[info['label']] = meta
    rows, names = [], {}
    for row, desc, chips in ROWS:
        kept = []
        for chip, label in chips:
            if label in sets:
                chip = chip.replace('{e}', str(sets[label]['epoch']))
                kept.append((chip, label))
                names[label] = f'{row} · {chip}' if not row.startswith('Measured') else f'Measured data · {chip}'
        if kept:
            rows.append((row, desc, kept))
    surface = np.load(export_dir / 'mesh_surface.npy').astype(np.float64)
    mesh = b64(np.round((surface + extent) / (2 * extent) * 65535).astype('<u2'))
    inferno = (colormaps['inferno'](np.linspace(0, 1, 256))[:, :3] * 255).round().astype(np.uint8)
    config = dict(extent=extent, ground=index['ground_z_m'], rows=rows, names=names, sets=sets,
                  mesh=dict(count=int(len(surface)), stand_in=index['mesh'].get('stand_in')),
                  inferno=b64(inferno))
    html = TEMPLATE.replace('__CONFIG__', json.dumps(config)).replace('__PAYLOAD__', json.dumps(payload)) \
                   .replace('__MESH__', mesh)
    Path(out).write_text(html)
    print(out, f'{len(html) / 1e6:.2f} MB')


TEMPLATE = r'''<title>RIFT Camry Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500&family=Barlow+Semi+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #eceeed; --panel: #f7f8f7; --ink: #15191b; --muted: #58615f; --line: #d1d7d5;
  --accent: #b8480c; --accent-ink: #ffffff; --chip: #e2e6e4; --scope: #050505; --scope-ink: #d7dcda;
  --display: "Barlow Semi Condensed", "Arial Narrow", "Helvetica Neue", Arial, sans-serif;
  --body: "Barlow", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1112; --panel: #151a1b; --ink: #e3e7e6; --muted: #97a19f; --line: #273033;
    --accent: #f98e09; --accent-ink: #1a0d00; --chip: #1e2527; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1112; --panel: #151a1b; --ink: #e3e7e6; --muted: #97a19f; --line: #273033;
  --accent: #f98e09; --accent-ink: #1a0d00; --chip: #1e2527; color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.5 var(--body); }
.wrap { max-width: 1320px; margin: 0 auto; padding-inline: 20px; padding-block: 22px 40px; display: grid; gap: 18px; }
header h1 { font: 600 30px/1.1 var(--display); letter-spacing: .01em; margin: 0; text-wrap: balance; }
header p { margin: 6px 0 0; color: var(--muted); max-width: 72ch; }
.label { font: 600 12px/1 var(--display); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.picker { display: grid; gap: 6px; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.row { display: grid; grid-template-columns: 120px 1fr auto; gap: 12px; align-items: center; padding-block: 5px; border-top: 1px solid var(--line); }
.row:first-of-type { border-top: 0; }
.row .name { font: 600 16px/1.2 var(--display); }
.row .desc { color: var(--muted); font-size: 13.5px; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
.chip { font: 500 13px/1 var(--mono); padding: 7px 10px; border-radius: 6px; border: 1px solid var(--line); background: var(--chip); color: var(--ink); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.chip[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.chip:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stage { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.scope { position: relative; background: var(--scope); border-radius: 10px; overflow: hidden; aspect-ratio: 4 / 3; }
.scope canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.scope .tag { position: absolute; left: 12px; top: 10px; font: 500 12.5px/1.35 var(--mono); color: var(--scope-ink); pointer-events: none; }
.scope .tag b { font: 600 15px/1.2 var(--display); letter-spacing: .02em; display: block; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; }
.controls label { display: flex; gap: 8px; align-items: center; font-size: 14px; }
.controls output { font: 500 13px var(--mono); min-width: 3.5ch; }
.controls .views { display: flex; gap: 6px; }
button { font: 500 13px/1 var(--body); padding: 7px 10px; border-radius: 6px; border: 1px solid var(--line); background: var(--chip); color: var(--ink); cursor: pointer; }
button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
.mips { display: grid; grid-template-columns: repeat(3, auto) 26px; gap: 8px 10px; align-items: end; justify-content: start; overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.mips figure { margin: 0; display: grid; gap: 4px; }
.mips canvas { background: var(--scope); border-radius: 4px; height: 150px; width: auto; image-rendering: auto; }
.mips figcaption { font: 500 12px var(--mono); color: var(--muted); }
.cbar { height: 150px; display: grid; grid-template-rows: auto 1fr auto; font: 500 11px var(--mono); color: var(--muted); }
.cbar div { width: 12px; border-radius: 3px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px 18px; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.stat .v { font: 500 20px/1.2 var(--mono); font-variant-numeric: tabular-nums; }
.stat .s { color: var(--muted); font-size: 12.5px; }
.table { overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font: 13px var(--mono); font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { font: 600 11.5px var(--display); letter-spacing: .07em; text-transform: uppercase; color: var(--muted); }
th:first-child, td:first-child { text-align: left; }
tbody tr { cursor: pointer; }
tbody tr:hover td { background: var(--chip); }
tbody tr[aria-selected="true"] td { color: var(--accent); }
.note { color: var(--muted); font-size: 13px; max-width: 110ch; }
@media (max-width: 760px) {
  .stage { grid-template-columns: 1fr; }
  .row { grid-template-columns: 1fr; gap: 4px; }
  .chips { justify-content: flex-start; }
  .mips { grid-template-columns: repeat(3, auto); }
  .mips canvas { height: 96px; }
  .cbar { display: none; }
}
</style>

<div class="wrap">
  <header>
    <h1>RIFT Camry Viewer</h1>
    <p>GOTCHA Camry HH. Each saved RIFT checkpoint, shown by its full point energy deposited on 6.25 cm voxels (the CIC readout the geometry score uses, at twice its resolution). The registered Camry sits in its own panel, never inside the reconstruction.</p>
  </header>

  <section class="picker" id="picker" aria-label="Checkpoint"></section>

  <div class="stage">
    <div class="scope"><canvas id="recon" aria-label="Reconstruction, rotatable"></canvas><div class="tag" id="tag"></div></div>
    <div class="scope"><canvas id="truth" aria-label="Registered Camry surface, rotatable"></canvas><div class="tag"><b>Camry reference</b>XV20 stand-in mesh, same frame and camera</div></div>
  </div>

  <div class="controls">
    <label for="share">Energy shown <input id="share" type="range" min="10" max="100" step="5" value="70"><output id="shareOut">70%</output></label>
    <label for="size">Voxel size <input id="size" type="range" min="0.3" max="2" step="0.1" value="1"><output id="sizeOut">1.0×</output></label>
    <label for="color">Colour <select id="color"><option value="energy">energy (inferno)</option><option value="height">height</option></select></label>
    <div class="views" role="group" aria-label="Camera">
      <button data-view="iso" aria-pressed="true">3/4</button><button data-view="top">Top</button><button data-view="side">Side</button><button data-view="front">Front</button>
    </div>
    <label for="spin"><input id="spin" type="checkbox"> Rotate</label>
  </div>

  <section class="mips" aria-label="Maximum-intensity projections">
    <figure><canvas id="mipTop"></canvas><figcaption>top (x–y)</figcaption></figure>
    <figure><canvas id="mipSide"></canvas><figcaption>side (x–z)</figcaption></figure>
    <figure><canvas id="mipFront"></canvas><figcaption>front (y–z)</figcaption></figure>
    <div class="cbar" aria-hidden="true"><span>1.0</span><div id="cbar"></div><span>0.2</span></div>
    <figure><canvas id="meshTop"></canvas><figcaption>mesh, top</figcaption></figure>
    <figure><canvas id="meshSide"></canvas><figcaption>mesh, side</figcaption></figure>
    <figure><canvas id="meshFront"></canvas><figcaption>mesh, front</figcaption></figure>
  </section>

  <section class="stats" id="stats" aria-label="Selected checkpoint"></section>

  <div class="table"><table>
    <thead><tr><th>Checkpoint</th><th>Epoch</th><th>Train RelMSE</th><th>Train ρ</th><th>Val RelMSE</th><th>Val ρ</th><th>CIC best F1</th><th>On car</th><th>Median dist.</th><th>Points</th></tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
  <p class="note">MIPs show sqrt(energy), min–max normalised per checkpoint and cut at 0.2, so brightness is shared in scale but not comparable in absolute level across checkpoints. “On car” is the energy share within 12.5 cm of the registered solid (A58); part of the energy below the ground plane is real car–ground double bounce. Val ρ is on the 38 pass-4 units of the 578-unit split (F5full: its own 30-unit split); b56-corrected arms remove pass 4’s own phase offset, so their Val gain is partly alignment. CIC F1 is the best over the threshold sweep against the data-frame mesh (τ 12.5 cm); a dash means not scored. The mesh is a same-generation stand-in registered at 3 cm RMS.</p>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const CONFIG = __CONFIG__;
const PAYLOAD = __PAYLOAD__;
const MESH = "__MESH__";
const E = CONFIG.extent, CROP = [3.0, 1.6, 1.35];
const unb64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const INFERNO = unb64(CONFIG.inferno);
const HEIGHT = ['#104281', '#1c5cab', '#2a78d6', '#5598e7', '#86b6ef', '#b7d3f6', '#e4effc'].map(h => new THREE.Color(h));

function decode(label) {
  const meta = CONFIG.sets[label], raw = PAYLOAD[label];
  const ijk = unb64(raw.ijk), qb = unb64(raw.q), q = new Uint16Array(qb.buffer, qb.byteOffset, meta.count);
  const n = meta.count, pos = new Float32Array(n * 3), amp = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    for (let a = 0; a < 3; a++) pos[3 * k + a] = (ijk[3 * k + a] + 0.5) * meta.voxel_m - E;
    amp[k] = q[k] / 65535;
  }
  let total = 0; for (let k = 0; k < n; k++) total += amp[k] * amp[k];
  const cum = new Float32Array(n); let run = 0;
  for (let k = 0; k < n; k++) { run += amp[k] * amp[k]; cum[k] = run / total * meta.kept_energy_share; }
  return { meta, pos, amp, cum };
}
const cache = {};
const getSet = label => cache[label] || (cache[label] = decode(label));

// Region frame (x front, y left, z up) -> three.js (X = x, Y = z, Z = -y), right-handed either way.
const toThree = (x, y, z) => [x, z, -y];

function makeView(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0x050505);
  const scene = new THREE.Scene();
  const ground = new THREE.GridHelper(6, 12, 0x2a2f31, 0x1a1e20);
  ground.position.y = CONFIG.ground;
  scene.add(ground);
  return { renderer, scene, canvas };
}
const recon = makeView(document.getElementById('recon'));
const truth = makeView(document.getElementById('truth'));
const camera = new THREE.PerspectiveCamera(32, 4 / 3, 0.05, 100);
const target = new THREE.Vector3(0, -0.05, 0);
const controls = [recon, truth].map(v => { const c = new THREE.OrbitControls(camera, v.canvas); c.target.copy(target); c.enableDamping = true; return c; });
const VIEWS = { iso: [5.2, 3.4, -4.6], top: [0, 9.5, 0.001], side: [0, 0.2, -9.5], front: [9.5, 0.2, 0] };
function setView(name) {
  camera.position.set(...VIEWS[name]); camera.lookAt(target); controls.forEach(c => { c.target.copy(target); c.update(); });
  document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === name)));
}
setView('iso');

// Camry surface samples.
(function () {
  const bytes = unb64(MESH), q = new Uint16Array(bytes.buffer, bytes.byteOffset, CONFIG.mesh.count * 3);
  const n = CONFIG.mesh.count, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    const x = q[3 * k] / 65535 * 2 * E - E, y = q[3 * k + 1] / 65535 * 2 * E - E, z = q[3 * k + 2] / 65535 * 2 * E - E;
    pos.set(toThree(x, y, z), 3 * k);
    const c = heightColor(z); col.set([c.r, c.g, c.b], 3 * k);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  truth.scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 0.022, vertexColors: true, sizeAttenuation: true })));
  window.MESH_XYZ = { q, n };
})();

function heightColor(z) {
  const t = Math.min(1, Math.max(0, (z - CONFIG.ground) / 1.6)) * (HEIGHT.length - 1);
  const i = Math.min(HEIGHT.length - 2, Math.floor(t));
  return HEIGHT[i].clone().lerp(HEIGHT[i + 1], t - i);
}
function infernoColor(v) {
  const i = Math.max(0, Math.min(255, Math.round(v * 255)));
  return [INFERNO[3 * i] / 255, INFERNO[3 * i + 1] / 255, INFERNO[3 * i + 2] / 255];
}

let points = null, current = null;
function draw() {
  const set = getSet(current), meta = set.meta;
  const share = +document.getElementById('share').value / 100, mode = document.getElementById('color').value;
  let n = 0; while (n < meta.count && (n === 0 || set.cum[n - 1] < share)) n++;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    pos.set(toThree(set.pos[3 * k], set.pos[3 * k + 1], set.pos[3 * k + 2]), 3 * k);
    if (mode === 'energy') col.set(infernoColor(0.25 + 0.75 * set.amp[k]), 3 * k);
    else { const c = heightColor(set.pos[3 * k + 2]); col.set([c.r, c.g, c.b], 3 * k); }
  }
  if (points) { recon.scene.remove(points); points.geometry.dispose(); points.material.dispose(); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const size = meta.voxel_m * 0.9 * +document.getElementById('size').value;
  const material = new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true, transparent: true,
    opacity: mode === 'energy' ? 0.55 : 0.9, depthWrite: mode !== 'energy',
    blending: mode === 'energy' ? THREE.AdditiveBlending : THREE.NormalBlending });
  points = new THREE.Points(g, material); recon.scene.add(points);
  const shown = n < meta.count ? set.cum[n - 1] : meta.kept_energy_share;
  document.getElementById('tag').innerHTML = `<b>${labelOf(current)}</b>${n.toLocaleString()} voxels · ${(shown * 100).toFixed(0)}% of the energy`;
}

function mip(canvas, set, axes, scale) {
  const [a, b] = axes, px = 0.0625 / scale;
  const w = Math.round(2 * CROP[a] / px), h = Math.round(2 * CROP[b] / px);
  canvas.width = w; canvas.height = h;
  const img = new Float32Array(w * h);
  const n = set ? set.meta.count : 0;
  const half = set ? set.meta.voxel_m / 2 : 0;
  for (let k = 0; k < n; k++) {
    // Each voxel covers its full footprint, so the projection has no gaps at any voxel size.
    const u0 = Math.floor((set.pos[3 * k + a] - half + CROP[a]) / px + 1e-6), u1 = Math.ceil((set.pos[3 * k + a] + half + CROP[a]) / px - 1e-6);
    const v0 = Math.floor((set.pos[3 * k + b] - half + CROP[b]) / px + 1e-6), v1 = Math.ceil((set.pos[3 * k + b] + half + CROP[b]) / px - 1e-6);
    for (let uu = Math.max(0, u0); uu < Math.min(w, u1); uu++) for (let vv = Math.max(0, v0); vv < Math.min(h, v1); vv++) {
      const idx = (h - 1 - vv) * w + uu; if (set.amp[k] > img[idx]) img[idx] = set.amp[k];
    }
  }
  const ctx = canvas.getContext('2d'), out = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const t = (img[i] - 0.2) / 0.8;
    const c = t <= 0 ? [5 / 255, 5 / 255, 5 / 255] : infernoColor(t);
    out.data[4 * i] = c[0] * 255; out.data[4 * i + 1] = c[1] * 255; out.data[4 * i + 2] = c[2] * 255; out.data[4 * i + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}
function meshMip(canvas, axes, scale) {
  const [a, b] = axes, px = 0.0625 / scale, { q, n } = window.MESH_XYZ;
  const w = Math.round(2 * CROP[a] / px), h = Math.round(2 * CROP[b] / px);
  canvas.width = w; canvas.height = h;
  const count = new Float32Array(w * h);
  for (let k = 0; k < n; k++) {
    const x = [q[3 * k] / 65535 * 2 * E - E, q[3 * k + 1] / 65535 * 2 * E - E, q[3 * k + 2] / 65535 * 2 * E - E];
    const u = Math.floor((x[a] + CROP[a]) / px), v = Math.floor((x[b] + CROP[b]) / px);
    if (u >= 0 && u < w && v >= 0 && v < h) count[(h - 1 - v) * w + u] += 1;
  }
  let peak = 0; for (const c of count) peak = Math.max(peak, c);
  const ctx = canvas.getContext('2d'), out = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const g = count[i] ? 70 + 150 * Math.sqrt(count[i] / peak) : 5;
    out.data[4 * i] = g; out.data[4 * i + 1] = g + 4; out.data[4 * i + 2] = g + 3; out.data[4 * i + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}
const MIPS = [['Top', [0, 1]], ['Side', [0, 2]], ['Front', [1, 2]]];
function drawMips() {
  const set = getSet(current);
  MIPS.forEach(([name, axes]) => mip(document.getElementById('mip' + name), set, axes, 2));
}
(function () {
  MIPS.forEach(([name, axes]) => meshMip(document.getElementById('mesh' + name), axes, 2));
  const stops = []; for (let i = 0; i <= 10; i++) { const c = infernoColor(1 - i / 10); stops.push(`rgb(${c.map(v => Math.round(v * 255)).join(',')}) ${i * 10}%`); }
  document.getElementById('cbar').style.background = `linear-gradient(${stops.join(',')})`;
})();

const fmt = (v, d = 3) => v === null || v === undefined ? '—' : Number(v).toFixed(d);
const labelOf = label => CONFIG.names[label] || label;
function stats() {
  const m = CONFIG.sets[current], el = document.getElementById('stats');
  const cells = m.kind === 'voxel_image'
    ? [['Voxels', `${m.voxel_m * 100} cm`, 'A^H y image of the TRAIN data'], ['Energy shown', `${(m.kept_energy_share * 100).toFixed(0)}%`, `top ${m.kept.toLocaleString()} voxels exported`]]
    : [['Train RelMSE / ρ', `${fmt(m.train_rel_mse)} / ${fmt(m.train_rho)}`, `scored at epoch ${m.fit_epoch}`],
       ['Val RelMSE / ρ', `${fmt(m.val_rel_mse)} / ${fmt(m.val_rho)}`, 'pass-4 held-out units'],
       ['CIC best F1', m.cic_best ? fmt(m.cic_best.f1) : '—', m.cic_best ? `threshold ${m.cic_best.threshold}, Chamfer ${Math.round(m.cic_best.chamfer_mm)} mm` : 'not scored'],
       ['On car', m.on_car ? `${(m.on_car.near_share * 100).toFixed(1)}%` : '—', m.on_car ? `median distance ${Math.round(m.on_car.weighted_median_distance_m * 1000)} mm` : ''],
       ['Displaced in height', m.on_car ? `${((m.on_car.off_by_place.above_roof + m.on_car.off_by_place.below_ground) * 100).toFixed(1)}%` : '—', 'above the roof + below ground'],
       ['Points', m.points_active ? m.points_active.toLocaleString() : '—', `max SH order ${m.max_sh_order}`]];
  el.innerHTML = cells.map(([k, v, s]) => `<div class="stat"><div class="label">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');
}
function table() {
  const body = document.getElementById('rows'); body.innerHTML = '';
  for (const [, , chips] of CONFIG.rows) for (const [, l] of chips) {
    const m = CONFIG.sets[l]; if (!m) continue;
    const tr = document.createElement('tr'); tr.dataset.label = l;
    tr.innerHTML = `<td>${labelOf(l)}</td><td>${m.epoch ?? '—'}</td><td>${fmt(m.train_rel_mse)}</td><td>${fmt(m.train_rho)}</td><td>${fmt(m.val_rel_mse)}</td><td>${fmt(m.val_rho)}</td><td>${m.cic_best ? fmt(m.cic_best.f1) : '—'}</td><td>${m.on_car ? (m.on_car.near_share * 100).toFixed(1) + '%' : '—'}</td><td>${m.on_car ? Math.round(m.on_car.weighted_median_distance_m * 1000) + ' mm' : '—'}</td><td>${m.points_active ? m.points_active.toLocaleString() : '—'}</td>`;
    tr.addEventListener('click', () => select(l)); body.appendChild(tr);
  }
}
function picker() {
  const el = document.getElementById('picker');
  el.innerHTML = CONFIG.rows.map(([row, desc, chips]) => `<div class="row"><div class="name">${row}</div><div class="desc">${desc}</div><div class="chips">${chips.filter(([, l]) => CONFIG.sets[l]).map(([c, l]) => `<button class="chip" data-set="${l}" aria-pressed="false">${c}</button>`).join('')}</div></div>`).join('');
  el.querySelectorAll('.chip').forEach(b => b.addEventListener('click', () => select(b.dataset.set)));
}
const ORDER = CONFIG.rows.flatMap(([, , chips]) => chips.map(([, l]) => l)).filter(l => CONFIG.sets[l]);
function select(label) {
  current = label;
  document.querySelectorAll('.chip').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.set === label)));
  document.querySelectorAll('#rows tr').forEach(r => r.setAttribute('aria-selected', String(r.dataset.label === label)));
  draw(); drawMips(); stats();
  try { localStorage.setItem('rift-camry-set', label); } catch (e) {}
}

function resize() {
  for (const v of [recon, truth]) { const r = v.canvas.getBoundingClientRect(); v.renderer.setSize(r.width, r.height, false); }
  const r = recon.canvas.getBoundingClientRect(); camera.aspect = r.width / Math.max(1, r.height); camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function loop() {
  requestAnimationFrame(loop);
  if (document.getElementById('spin').checked && !reduce) {
    const p = camera.position.clone().sub(target); p.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.004); camera.position.copy(target.clone().add(p));
  }
  controls.forEach(c => c.update());
  recon.renderer.render(recon.scene, camera); truth.renderer.render(truth.scene, camera);
}
document.getElementById('share').addEventListener('input', e => { document.getElementById('shareOut').textContent = e.target.value + '%'; draw(); });
document.getElementById('size').addEventListener('input', e => { document.getElementById('sizeOut').textContent = (+e.target.value).toFixed(1) + '×'; draw(); });
document.getElementById('color').addEventListener('change', draw);
document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const i = ORDER.indexOf(current);
  if (e.key === 'ArrowRight') select(ORDER[Math.min(ORDER.length - 1, i + 1)]);
  if (e.key === 'ArrowLeft') select(ORDER[Math.max(0, i - 1)]);
});
picker(); table(); resize();
let start = 'CM5b_ep24'; try { const s = localStorage.getItem('rift-camry-set'); if (s && CONFIG.sets[s]) start = s; } catch (e) {}
select(CONFIG.sets[start] ? start : ORDER[0]); loop();
</script>
'''

if __name__ == '__main__':
    main(*sys.argv[1:])
