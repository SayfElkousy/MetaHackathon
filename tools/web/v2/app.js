// ACL repair designer / viewer (v2).
//
// Rendering-bug fix (see docs/V2_REPORT.md): the old page measured the model's
// bounds with Box3.expandByObject() right after adding a root that was scaled
// x1000 and rotated, but before three.js had updated the root's world matrix.
// The box therefore came out ~1000x too small and the camera was parked ~0.1 mm
// from the centre, inside a bone: an "empty" canvas. Here the geometry is in mm
// (no scaling), every bounds query calls updateMatrixWorld(true) first, near/far
// planes follow the scene size, and rendering is on demand after each change.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const $ = id => document.getElementById(id);
const VIEW_MODE = location.pathname.startsWith('/view');
if (VIEW_MODE) document.body.classList.add('viewmode');
const perf = window.__perf = { t0: performance.now(), steps: [], rebuilds: [], fps: 0 };
const mark = (k) => { perf.steps.push([k, Math.round(performance.now() - perf.t0)]); };

// ------------------------------------------------------------------ state
const S = {
  reg: null, regById: {}, state: null, scene: null, seq: 0, applied: 0, abort: null,
  meshCache: new Map(), objects: new Map(), ckeys: new Map(), matCache: new Map(),
  vis: {}, opac: {}, selection: null, msgs: {},
  stateId: null, firstFramed: false, saveTimer: null, buildTimer: null, colorMode: 'anatomy',
  dirty: false, lastFullHash: null, building: false, region: 'all', undoStack: [], redoStack: [],
  gesture: null, uiKey: '', workerOK: false,
  levelSel: null, levelMode: 'highlight', levelRegion: { 4: 'both', 6: 'both' }, lastEdit: null, layerFixes: null,
  builder: false, savedView: null, compSel: null, labelsOn: false, level0Locked: true,
  groupVis: { layer_braid: { 0: true, 1: true }, layer_channels: { 0: true, 1: true, 2: true } },
};
const DEFAULT_VIS = {
  bone_femur: [true, 0.45], bone_tibia: [true, 0.9], cartilage_femoral: [true, 0.45], cartilage_tibial: [true, 0.6],
  landmark_pcl: [true, 0.35], footprint_tibial: [true, 1], footprint_femoral: [true, 1], intact_reference: [false, 0.15],
  bundle_am_reference: [false, 0.25], bundle_pl_reference: [false, 0.25], injury_region: [true, 0.35],
  native_fascicles: [true, 1], native_fibers: [true, 1], scaffold_yarns: [true, 1], scaffold_fibers: [false, 1],
  centerline: [false, 1], bundle_interface: [false, 0.35], bundle_paths: [false, 1], fit_violations: [true, 1],
  layer_sheath: [true, 0.3], layer_pores: [true, 1], layer_braid: [true, 1], layer_channels: [true, 1], layer_ports: [true, 1],
  layer_coating: [false, 1], layer_coating_sheath: [false, 0.6], layer_grooves: [false, 1], layer_end_tib: [true, 0.55], layer_end_fem: [true, 0.55],
  layer_footprint_tib: [true, 1], layer_footprint_fem: [true, 1], layer_plug_tib: [true, 0.85], layer_plug_fem: [true, 0.85],
  layer_interface: [true, 0.3], layer_anchors: [true, 1],
};
// level colours (summary stripe) and layered-construct component colours
const LV_COL = { 1: '#1fa89c', 2: '#3a8fd0', 3: '#b88a2a', 4: '#d9772b', 5: '#6c7f8a', 6: '#8b5ab0' };
const LCOL = { sheath: 0xcfe4ea, braidAM: 0xe8a33a, braidPL: 0x4f9bd6, chAM: 0xf3e3c6, chPL: 0xd6e6f4, chShared: 0xe5d6f0, lumen: 0x8a7b62,
  coat: 0x2ec4b6, groove: 0x7fd3ff, plug: 0xbfb39e, pore: 0x1d2629, fpOutline: 0xffffff, fpBase: 0xffd27a, over: 0xff2a2a,
  iface: 0x8c73b8, anchor: 0xf2a93b };
const isLayer = (meta) => meta && (meta.id || '').startsWith('layer_');
const CAT_DEFAULT = { retained: [true, 0.22], scaffold_region: [false, 0.22], observed: [true, 0.5], repair_space: [false, 0.16], violation: [true, 1] };
const COL = {
  bone: 0xe6dccb, cartilage: 0x8fb2e6, landmark: 0xb7b7d0, footprintAM: 0xf0a050, footprintPL: 0x5aa0d8,
  retained: 0xd9776d, injury: 0xe03b2f, reference: 0xffb3a8, bundleAM: 0xf0a050, bundlePL: 0x5aa0d8,
  fasAM: 0xe8894a, fasPL: 0x4f8fcf, collide: 0xff2020, centerline: 0xffffff, observed: 0x35c4d8, scaffold: 0xe8c14a,
  repair: 0x6fcfff, interface: 0xffffff, violation: 0xff1a1a, shortfall: 0xff9a1a,
};
const VBITS = { 1: 'outside the ligament envelope', 2: 'outside its repair region (e.g. into the unaffected bundle)', 4: 'beyond the permitted overlap',
  8: 'into bone', 16: 'touching another strand', 32: 'into retained fascicles', 64: 'closer than the requested clearance' };
const vtext = b => Object.entries(VBITS).filter(([k]) => b & Number(k)).map(e => e[1]).join(', ');
const levelObjects = (n) => { const L = S.reg && S.reg.levels.find(x => x.n === n); return L ? L.objects : []; };
const STATUS_COL = { observed: 0x2e9e5b, 'partly observed': 0x7cc08f, assumed: 0x9a9a9a, estimated: 0x9b6fc4, proposed: 0xe0a91f };

function showErr(title, msg, actions) {
  $('overlay').hidden = false; $('ovTitle').textContent = title; $('ovSteps').textContent = msg || '';
  const box = $('ovActions'); box.innerHTML = '';
  (actions || []).forEach(([lab, fn]) => { const b = document.createElement('button'); b.textContent = lab; b.onclick = fn; box.appendChild(b); });
}
function progress(title, lines) { $('overlay').hidden = false; $('ovTitle').textContent = title; $('ovSteps').textContent = lines.join('\n'); $('ovActions').innerHTML = ''; }
function hideOverlay() { $('overlay').hidden = true; }
function badge(t, cls) { const b = $('badge'); b.textContent = t; b.className = 'pill ' + (cls || ''); }

async function api(path, body, signal) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal };
  const r = await fetch(path, opt);
  const ct = r.headers.get('content-type') || '';
  const j = ct.includes('json') ? await r.json() : null;
  if (!r.ok) { const e = new Error((j && j.error) || ('HTTP ' + r.status)); e.status = r.status; e.body = j; throw e; }
  return j;
}
const b64 = (s, T) => { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new T(u.buffer); };

// ------------------------------------------------------------------ 3D setup
let scene, cam, ren, ctl, root, tctl, raycaster, handles = {};
const clipPlanes = { global: new THREE.Plane(new THREE.Vector3(1, 0, 0), 1e6), bone: new THREE.Plane(new THREE.Vector3(-1, 0, 0), 1e6), slabA: null, slabB: null };
let needRender = true;
const invalidate = () => { needRender = true; };

function init3() {
  const host = $('view');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x20282a);
  cam = new THREE.PerspectiveCamera(32, 1, 0.1, 5000);
  ren = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  ren.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  ren.localClippingEnabled = true;
  host.appendChild(ren.domElement);
  ctl = new OrbitControls(cam, ren.domElement);
  ctl.enableDamping = true; ctl.dampingFactor = 0.12;
  ctl.addEventListener('change', invalidate);
  // neutral lighting: hemisphere + soft key + fill (no metallic look)
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8090a0, 1.25));
  const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(80, 140, 120); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.45); fill.position.set(-120, 40, -80); scene.add(fill);
  root = new THREE.Group(); root.rotation.x = -Math.PI / 2; scene.add(root);   // engine Z-up -> three Y-up
  raycaster = new THREE.Raycaster(); raycaster.params.Line.threshold = 0.15;
  // size from the actual container, and again whenever layout changes
  const size = () => {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    ren.setSize(w, h, false); ren.domElement.style.width = w + 'px'; ren.domElement.style.height = h + 'px';
    cam.aspect = w / h; cam.updateProjectionMatrix(); invalidate();
  };
  new ResizeObserver(size).observe(host); size();
  tctl = new TransformControls(cam, ren.domElement);
  tctl.setSize(0.7);
  tctl.addEventListener('dragging-changed', e => { ctl.enabled = !e.value; if (!e.value) onHandleDrop(); });
  tctl.addEventListener('change', invalidate);
  scene.add(tctl);
  ren.domElement.addEventListener('pointerdown', e => { S._down = [e.clientX, e.clientY]; });
  ren.domElement.addEventListener('pointerup', e => { if (S._down && Math.hypot(e.clientX - S._down[0], e.clientY - S._down[1]) < 4) pick(e); });
  ren.domElement.addEventListener('pointermove', hover);
  ren.domElement.addEventListener('pointerleave', () => { $('tooltip').hidden = true; });
  let frames = 0, last = performance.now();
  const loop = () => {
    requestAnimationFrame(loop);
    const moved = ctl.update();
    if (needRender || moved) { ren.render(scene, cam); needRender = false; updateHud(); if (detail.ren) detail.render(); }
    frames++; const now = performance.now(); if (now - last > 1000) { perf.fps = frames * 1000 / (now - last); frames = 0; last = now; }
  };
  loop();
}

function e2t(p) { return new THREE.Vector3(p[0], p[2], -p[1]); }      // engine mm -> three world (root rotation)

function sceneBox(filter) {
  root.updateMatrixWorld(true);                                      // the fix: world matrices before measuring
  const box = new THREE.Box3();
  root.traverse(o => { if ((o.isMesh || o.isLineSegments) && o.visible && (!filter || filter(o))) box.expandByObject(o); });
  return box;
}

function jointBox() {
  // the ACL plus both bone ends: bone bounds clipped to the ligament's height range +/- margin
  const lig = sceneBox(o => ['retained', 'retained_structure', 'scaffold', 'footprint', 'injury', 'reference', 'layer_sheath'].includes(o.userData.category));
  let bones = sceneBox(o => ['bone', 'cartilage'].includes(o.userData.category));
  if (lig.isEmpty()) return bones.isEmpty() ? sceneBox() : bones;
  if (bones.isEmpty()) return lig.expandByScalar(5);
  const b = bones.clone();
  b.min.y = Math.max(b.min.y, lig.min.y - 10); b.max.y = Math.min(b.max.y, lig.max.y + 12);
  // narrow to the intercondylar region around the ligament (both bone ends stay in frame)
  const lc = lig.getCenter(new THREE.Vector3());
  b.min.x = Math.max(b.min.x, lc.x - 30); b.max.x = Math.min(b.max.x, lc.x + 30);
  b.min.z = Math.max(b.min.z, lc.z - 30); b.max.z = Math.min(b.max.z, lc.z + 30);
  return b.union(lig);
}
function setCam(which, keepDist) {
  const side = (S.scene && S.scene.side === 'left') ? -1 : 1;
  const box = jointBox();
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const sz = box.getSize(new THREE.Vector3());
  const r = 0.5 * sz.length();
  const d = keepDist ? cam.position.distanceTo(ctl.target) : r / Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) * 0.95;
  const dirs = {
    anterior: [0, 0, -1], posterior: [0, 0, 1], lateral: [side, 0, 0], medial: [-side, 0, 0], superior: [0, 1, 0.0001],
    q34: [side * 0.62, 0.32, -0.72],
  };
  let v;
  if (which === 'along' && S.scene && S.scene.handles) {
    const a = e2t(S.scene.handles.fp_tib), b = e2t(S.scene.handles.fp_fem);
    root.updateMatrixWorld(true);
    a.applyMatrix4(new THREE.Matrix4()); b.applyMatrix4(new THREE.Matrix4());
    v = b.clone().sub(a).normalize(); c.copy(a.clone().add(b).multiplyScalar(0.5));
  } else v = new THREE.Vector3(...(dirs[which] || dirs.q34)).normalize();
  cam.up.set(0, 1, 0);
  if (which === 'superior' || which === 'along') cam.up.set(0, 0, -1);
  const dist = which === 'along' ? d * 0.75 : d;
  cam.position.copy(c).addScaledVector(v, dist);
  cam.near = Math.max(0.05, dist / 200); cam.far = dist * 20; cam.updateProjectionMatrix();
  ctl.target.copy(c); ctl.update(); invalidate();
}

// ------------------------------------------------------------------ geometry from payload
function disposeObj(o) {
  // geometries are always disposed; materials come from the shared cache and
  // are only disposed when the cache itself is cleared (colour-mode change)
  o.traverse(x => { if (x.geometry) x.geometry.dispose(); if (x.material) (Array.isArray(x.material) ? x.material : [x.material]).forEach(m => { if (!m.userData.cached) m.dispose(); }); });
}
function clearMatCache() { for (const m of S.matCache.values()) m.dispose(); S.matCache.clear(); }

function matFor(meta, opts) {
  const op = Math.round(opts.opacity * 40) / 40;
  const clip = (opts.clip || []).map(p => clipPlanes.global === p ? 'g' : clipPlanes.bone === p ? 'b' : clipPlanes.slabA === p ? 'a' : clipPlanes.slabB === p ? 's' : '?').join('');
  const key = [opts.color, op, opts.vcol ? 1 : 0, clip, opts.line ? 'L' : 'M', S.colorMode, opts.hl ? 'H' : ''].join('|');
  let m = S.matCache.get(key);
  if (m) { m.clippingPlanes = opts.clip; return m; }
  if (opts.line) {
    m = new THREE.LineBasicMaterial({ vertexColors: true, transparent: op < 0.999, opacity: op });
  } else {
    m = new THREE.MeshStandardMaterial({
      color: opts.color, roughness: 0.82, metalness: 0.0, transparent: op < 0.999, opacity: op,
      depthWrite: op >= 0.999, side: THREE.DoubleSide, vertexColors: !!opts.vcol, flatShading: false,
    });
    if (opts.hl) { m.emissive = new THREE.Color(0x5a3a00); m.emissiveIntensity = 0.55; }
  }
  m.clippingPlanes = opts.clip;
  m.userData.cached = true;
  if (S.matCache.size > 400) { const k = S.matCache.keys().next().value; S.matCache.get(k).dispose(); S.matCache.delete(k); }
  S.matCache.set(key, m);
  return m;
}

// ------------------------------------------------------------------ tube geometry worker
let geomWorker = null, geomSeq = 0;
const geomPending = new Map();
function workerCall(msg, transfer) {
  return new Promise((resolve, reject) => {
    try {
      if (!geomWorker) {
        geomWorker = new Worker('/v2/geom_worker.js');
        geomWorker.onmessage = (ev) => {
          const d = ev.data, p = geomPending.get(d.id);
          if (!p) return;
          geomPending.delete(d.id);
          if (d.error) p.reject(new Error(d.error)); else { S.workerOK = true; p.resolve(d); }
        };
        geomWorker.onerror = (e) => { for (const p of geomPending.values()) p.reject(e); geomPending.clear(); };
      }
    } catch (e) { reject(e); return; }
    const id = ++geomSeq;
    geomPending.set(id, { resolve, reject });
    setTimeout(() => { if (geomPending.has(id)) { geomPending.delete(id); reject(new Error('worker timeout')); } }, 15000);
    geomWorker.postMessage(Object.assign({ id }, msg), transfer);
  });
}
async function tubeGeometryAsync(P, offs, R, nsec, colorFn, lidBase) {
  // colors stay on the main thread (they need selection/region state); the
  // frame math and index expansion run in the worker when available
  const nPts = P.length / 3;
  const col = new Float32Array(nPts * 3);
  const c = new THREE.Color();
  for (let l = 0; l < offs.length - 1; l++) {
    for (let j = offs[l]; j < offs[l + 1]; j++) { colorFn(l, j, c); col[3 * j] = c.r; col[3 * j + 1] = c.g; col[3 * j + 2] = c.b; }
  }
  const small = nPts < 4000;
  if (!small) {
    try {
      const Pc = P.slice(), Oc = offs.slice(), Rc = R.slice(), Cc = col.slice();
      const d = await workerCall({ P: Pc.buffer, offs: Oc.buffer, R: Rc.buffer, nsec, col: Cc.buffer, lidBase },
        [Pc.buffer, Oc.buffer, Rc.buffer, Cc.buffer]);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(d.pos), 3));
      g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(d.nor), 3));
      g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(d.vcol), 3));
      g.setAttribute('lid', new THREE.BufferAttribute(new Float32Array(d.lid), 1));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(d.idx), 1));
      g.computeBoundingSphere();
      return g;
    } catch (e) { /* fall through to the in-page builder */ }
  }
  return tubeGeometry(P, offs, R, nsec, colorFn, lidBase);
}

function tubeGeometry(P, offs, R, nsec, colorFn, lidBase) {
  // merged tubes for many polylines; per-vertex color and line id (for picking)
  let nv = 0, ni = 0;
  for (let l = 0; l < offs.length - 1; l++) { const n = offs[l + 1] - offs[l]; if (n < 2) continue; nv += n * nsec; ni += (n - 1) * nsec * 6; }
  const pos = new Float32Array(nv * 3), nor = new Float32Array(nv * 3), col = new Float32Array(nv * 3), lid = new Float32Array(nv);
  const idx = new Uint32Array(ni);
  let vo = 0, io = 0;
  const T = new THREE.Vector3(), N = new THREE.Vector3(), B = new THREE.Vector3(), tmp = new THREE.Vector3(), c = new THREE.Color();
  const cs = [], sn = []; for (let k = 0; k < nsec; k++) { cs.push(Math.cos(2 * Math.PI * k / nsec)); sn.push(Math.sin(2 * Math.PI * k / nsec)); }
  for (let l = 0; l < offs.length - 1; l++) {
    const a = offs[l], n = offs[l + 1] - a; if (n < 2) continue;
    const base = vo;
    for (let i = 0; i < n; i++) {
      const j = a + i, jp = a + Math.min(i + 1, n - 1), jm = a + Math.max(i - 1, 0);
      T.set(P[3 * jp] - P[3 * jm], P[3 * jp + 1] - P[3 * jm + 1], P[3 * jp + 2] - P[3 * jm + 2]).normalize();
      if (i === 0) { tmp.set(0, 0, 1); if (Math.abs(T.z) > 0.9) tmp.set(1, 0, 0); N.crossVectors(T, tmp).normalize(); }
      else { N.addScaledVector(T, -N.dot(T)).normalize(); }
      B.crossVectors(T, N);
      const r = R[j];
      colorFn(l, j, c);
      for (let k = 0; k < nsec; k++) {
        const nx = cs[k] * N.x + sn[k] * B.x, ny = cs[k] * N.y + sn[k] * B.y, nz = cs[k] * N.z + sn[k] * B.z;
        const q = 3 * vo;
        pos[q] = P[3 * j] + r * nx; pos[q + 1] = P[3 * j + 1] + r * ny; pos[q + 2] = P[3 * j + 2] + r * nz;
        nor[q] = nx; nor[q + 1] = ny; nor[q + 2] = nz;
        col[q] = c.r; col[q + 1] = c.g; col[q + 2] = c.b; lid[vo] = lidBase + l; vo++;
      }
    }
    for (let i = 0; i < n - 1; i++) for (let k = 0; k < nsec; k++) {
      const a0 = base + i * nsec + k, a1 = base + i * nsec + (k + 1) % nsec, b0 = a0 + nsec, b1 = a1 + nsec;
      idx[io++] = a0; idx[io++] = b0; idx[io++] = a1; idx[io++] = a1; idx[io++] = b0; idx[io++] = b1;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.setAttribute('lid', new THREE.BufferAttribute(lid, 1));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeBoundingSphere();
  return g;
}

function lineGeometry(P, offs, colorFn) {
  let ns = 0; for (let l = 0; l < offs.length - 1; l++) ns += Math.max(0, offs[l + 1] - offs[l] - 1);
  const pos = new Float32Array(ns * 6), col = new Float32Array(ns * 6), lid = new Float32Array(ns * 2);
  let o = 0; const c = new THREE.Color();
  for (let l = 0; l < offs.length - 1; l++) {
    for (let j = offs[l]; j < offs[l + 1] - 1; j++) {
      for (const jj of [j, j + 1]) { pos.set([P[3 * jj], P[3 * jj + 1], P[3 * jj + 2]], 3 * o); colorFn(l, jj, c); col.set([c.r, c.g, c.b], 3 * o); lid[o] = l; o++; }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.setAttribute('lid', new THREE.BufferAttribute(lid, 1));
  g.computeBoundingSphere(); return g;
}

// Hollow tubes: outer wall, lumen wall (inward normals) and annular end faces, so channel
// entrances are open and the lumen is visible. U gives each section's wide axis (non-circular
// channels keep their orientation); the profile is a superellipse with the given aspect.
function hollowTubeGeometry(P, offs, R, RI, U, nsec, aspect, expo, colOut, colIn, lidBase) {
  let nv = 0, ni = 0;
  for (let l = 0; l < offs.length - 1; l++) { const n = offs[l + 1] - offs[l]; if (n < 2) continue; nv += 2 * n * nsec; ni += 2 * (n - 1) * nsec * 6 + 2 * nsec * 6; }
  const pos = new Float32Array(nv * 3), nor = new Float32Array(nv * 3), col = new Float32Array(nv * 3), lid = new Float32Array(nv), idx = new Uint32Array(ni);
  const sa = Math.sqrt(aspect), e = 2 / expo, px = [], py = [], nx = [], ny = [];
  for (let k = 0; k < nsec; k++) {
    const a = 2 * Math.PI * k / nsec, c = Math.cos(a), s = Math.sin(a);
    px.push(Math.sign(c) * Math.pow(Math.abs(c), e) * sa); py.push(Math.sign(s) * Math.pow(Math.abs(s), e) / sa);
    const gx = c / sa, gy = s * sa, gl = Math.hypot(gx, gy) || 1; nx.push(gx / gl); ny.push(gy / gl);
  }
  const T = new THREE.Vector3(), Uo = new THREE.Vector3(), W = new THREE.Vector3(), cO = new THREE.Color(), cI = new THREE.Color();
  let vo = 0, io = 0;
  for (let l = 0; l < offs.length - 1; l++) {
    const a0 = offs[l], n = offs[l + 1] - a0; if (n < 2) continue;
    const bo = vo, bi = vo + n * nsec;
    for (let i = 0; i < n; i++) {
      const j = a0 + i, jp = a0 + Math.min(i + 1, n - 1), jm = a0 + Math.max(i - 1, 0);
      T.set(P[3 * jp] - P[3 * jm], P[3 * jp + 1] - P[3 * jm + 1], P[3 * jp + 2] - P[3 * jm + 2]).normalize();
      Uo.set(U[3 * j], U[3 * j + 1], U[3 * j + 2]); Uo.addScaledVector(T, -Uo.dot(T));
      if (Uo.lengthSq() < 1e-10) { Uo.set(0, 0, 1); if (Math.abs(T.z) > 0.9) Uo.set(1, 0, 0); Uo.addScaledVector(T, -Uo.dot(T)); }
      Uo.normalize(); W.crossVectors(T, Uo);
      colOut(l, j, cO); colIn(l, j, cI);
      for (let k = 0; k < nsec; k++) {
        for (const [base, r, sg, cc] of [[bo, R[j], 1, cO], [bi, RI[j], -1, cI]]) {
          const v = base + i * nsec + k, q = 3 * v;
          pos[q] = P[3 * j] + r * (px[k] * Uo.x + py[k] * W.x); pos[q + 1] = P[3 * j + 1] + r * (px[k] * Uo.y + py[k] * W.y); pos[q + 2] = P[3 * j + 2] + r * (px[k] * Uo.z + py[k] * W.z);
          nor[q] = sg * (nx[k] * Uo.x + ny[k] * W.x); nor[q + 1] = sg * (nx[k] * Uo.y + ny[k] * W.y); nor[q + 2] = sg * (nx[k] * Uo.z + ny[k] * W.z);
          col[q] = cc.r; col[q + 1] = cc.g; col[q + 2] = cc.b; lid[v] = lidBase + l;
        }
      }
    }
    for (let i = 0; i < n - 1; i++) for (let k = 0; k < nsec; k++) {
      const k1 = (k + 1) % nsec;
      const o0 = bo + i * nsec + k, o1 = bo + i * nsec + k1, o2 = o0 + nsec, o3 = o1 + nsec;
      idx[io++] = o0; idx[io++] = o2; idx[io++] = o1; idx[io++] = o1; idx[io++] = o2; idx[io++] = o3;
      const n0 = bi + i * nsec + k, n1 = bi + i * nsec + k1, n2 = n0 + nsec, n3 = n1 + nsec;
      idx[io++] = n0; idx[io++] = n1; idx[io++] = n2; idx[io++] = n1; idx[io++] = n3; idx[io++] = n2;
    }
    for (const i of [0, n - 1]) for (let k = 0; k < nsec; k++) {           // annular end faces (the entrances)
      const k1 = (k + 1) % nsec, o0 = bo + i * nsec + k, o1 = bo + i * nsec + k1, n0 = bi + i * nsec + k, n1 = bi + i * nsec + k1;
      idx[io++] = o0; idx[io++] = o1; idx[io++] = n0; idx[io++] = o1; idx[io++] = n1; idx[io++] = n0;
    }
    vo += 2 * n * nsec;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.setAttribute('lid', new THREE.BufferAttribute(lid, 1));
  g.setIndex(new THREE.BufferAttribute(idx, 1)); g.computeBoundingSphere();
  return g;
}

function objectColor(meta) {
  if (S.colorMode === 'status') return STATUS_COL[meta.status] || 0x999999;
  if (meta.color) return new THREE.Color(...meta.color).getHex();
  const lc = { layer_sheath: LCOL.sheath, layer_plug: LCOL.plug, layer_pores: LCOL.pore, layer_coating: LCOL.coat, layer_end: 0xd8c4e0,
    layer_interface: LCOL.iface, layer_anchors: LCOL.anchor }[meta.category];
  if (lc) return lc;
  return { bone: COL.bone, cartilage: COL.cartilage, landmark: COL.landmark, retained: COL.retained, injury: COL.injury,
    reference: COL.reference, observed: COL.observed, guide: COL.centerline, repair_space: COL.repair, interface: COL.interface,
    violation: COL.violation }[meta.category] ||
    (meta.id === 'bundle_am_reference' ? COL.bundleAM : meta.id === 'bundle_pl_reference' ? COL.bundlePL : 0xcccccc);
}

function visFor(meta) {
  if (!(meta.id in S.vis)) {
    const d = DEFAULT_VIS[meta.id] || CAT_DEFAULT[meta.category] || [true, 1];
    S.vis[meta.id] = d[0]; S.opac[meta.id] = d[1];
  }
  const compHL = !!(S.compSel && S.compSel.obj === meta.id);
  const o = levelOverride(meta);
  if (o) return [o.vis, o.op == null ? S.opac[meta.id] : o.op, !!o.hl || (compHL && o.vis)];
  return [S.vis[meta.id], S.opac[meta.id], compHL && S.vis[meta.id]];
}
// Level view: the selected level is highlighted (Highlight), shown alone (Isolate), or shown
// opaque with the levels just inside and outside it transparent (Neighbours transparent).
function levelOverride(meta) {
  const n = S.levelSel; if (!n || !S.scene) return null;
  const lv = meta.level || 0, mine = lv === n, sheath = meta.id === 'layer_sheath';
  const own = S.opac[meta.id] == null ? 1 : S.opac[meta.id];
  // highlight emphasizes the selected level but respects explicit show/hide:
  // an object the user hid (e.g. via isolate) stays hidden
  if (S.levelMode === 'highlight') return mine ? { vis: S.vis[meta.id] !== false, op: sheath ? Math.max(own, 0.45) : Math.max(own, 0.85), hl: true } : null;
  if (S.levelMode === 'isolate') return mine ? { vis: true, op: sheath ? 0.6 : 1, hl: false } : { vis: false };
  if (mine) return { vis: true, op: sheath ? 0.6 : 1, hl: true };
  if (lv && Math.abs(lv - n) === 1) return { vis: true, op: 0.14 };
  if (lv) return { vis: false };
  if (['bone', 'cartilage'].includes(meta.category)) return { vis: S.vis[meta.id], op: Math.min(0.22, own) };
  if (meta.category === 'footprint') return null;
  return { vis: false };
}
function refreshLevelView() {
  for (const o of S.objects.values()) { const meta = o.userData.meta; if (!meta) continue; o.visible = visFor(meta)[0]; matOptsFor(o); }
  invalidate();
}

function clipsFor(cat) {
  const out = [];
  if (S.clip && S.clip.on) out.push(clipPlanes.global);
  if (S.cut && S.cut.on && ['bone', 'cartilage', 'landmark'].includes(cat)) out.push(clipPlanes.bone);
  if (S.slab && S.slab.on && clipPlanes.slabA && !['bone', 'cartilage', 'landmark'].includes(cat)) out.push(clipPlanes.slabA, clipPlanes.slabB);
  return out;
}

// Display-only subgroup filtering: concatenate the point ranges of the kept lines.
function gatherPts(arr, offs0, keep, stride) {
  stride = stride || 1;
  let n = 0;
  keep.forEach(l => { n += offs0[l + 1] - offs0[l]; });
  const out = new arr.constructor(n * stride);
  let o = 0;
  keep.forEach(l => {
    const a = offs0[l] * stride, b = offs0[l + 1] * stride;
    out.set(arr.subarray(a, b), o); o += b - a;
  });
  return out;
}
function filterLineMeta(lm0, keep) {
  const lm = {};
  for (const k of Object.keys(lm0 || {})) lm[k] = keep.map(l => lm0[k][l]);
  return lm;
}
// A world-space point hidden by clipping must not be pickable/hoverable: three.js
// raycasts geometry, not clipped fragments, so test the planes explicitly.
function clippedAway(cat, worldPoint) {
  for (const pl of clipsFor(cat)) {
    if (pl.distanceToPoint(worldPoint) < 0) return true;
  }
  return false;
}

async function buildObject(meta) {
  const [vis, op, hl] = visFor(meta);
  const regColor = (i) => { const r = S.scene.regions[i]; const mat = S.state.params['regions.' + r + '.material']; const m = S.reg.materials[mat]; return m ? m.color : [0.9, 0.8, 0.3]; };
  let obj;
  if (meta.kind === 'mesh') {
    let V, F;
    if (meta.cached) ({ V, F } = S.meshCache.get(meta.cache_key));
    else { V = b64(meta.v, Float32Array); F = b64(meta.f, Uint32Array); if (meta.cache_key) S.meshCache.set(meta.cache_key, { V, F }); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(V, 3)); g.setIndex(new THREE.BufferAttribute(F, 1));
    let vcol = false;
    if (meta.attr && meta.attr.mcol && S.colorMode !== 'status' && (meta.category === 'layer_end' || S.colorMode === 'material')) {
      const mc = b64(meta.attr.mcol, Uint8Array); const col = new Float32Array(V.length);
      for (let i = 0; i < V.length / 3; i++) { col[3 * i] = mc[3 * i] / 255; col[3 * i + 1] = mc[3 * i + 1] / 255; col[3 * i + 2] = mc[3 * i + 2] / 255; }
      g.setAttribute('color', new THREE.BufferAttribute(col, 3)); vcol = true;
    } else if (meta.attr && meta.attr.bundle && S.colorMode !== 'status') {
      const bid = b64(meta.attr.bundle, Uint8Array); const col = new Float32Array(V.length);
      const a = new THREE.Color(COL.footprintAM), p = new THREE.Color(COL.footprintPL);
      for (let i = 0; i < bid.length; i++) { const c = bid[i] ? p : a; col.set([c.r, c.g, c.b], 3 * i); }
      g.setAttribute('color', new THREE.BufferAttribute(col, 3)); vcol = true;
    }
    g.computeVertexNormals();
    obj = new THREE.Mesh(g, matFor(meta, { color: vcol ? 0xffffff : objectColor(meta), opacity: op, vcol, clip: clipsFor(meta.category), hl }));
    obj.renderOrder = op < 1 ? 2 : 0;
  } else {
    // AM/PL subgroup visibility (Scaffold Builder): lines whose group is hidden are
    // left out of the built geometry, so they can neither be seen nor picked. The
    // server payload is untouched; this is display-only filtering of decoded lines.
    const P0 = b64(meta.p, Float32Array), offs0 = b64(meta.o, Uint32Array), R0 = b64(meta.r, Float32Array);
    const lm0 = meta.line_meta || {};
    let P = P0, offs = offs0, R = R0, lm = lm0, keep = null;
    {
      const g = S.groupVis[meta.id];
      const grp = g && lm0.group;
      if (g && grp) {
        keep = [];
        for (let l = 0; l < grp.length; l++) if (g[grp[l]] !== false) keep.push(l);
        if (keep.length === grp.length) keep = null;
      }
    }
    const dec = (key, T, stride) => {
      if (!meta[key]) return null;
      const a = b64(meta[key], T);
      return keep ? gatherPts(a, offs0, keep, stride || 1) : a;
    };
    if (keep) {
      P = gatherPts(P0, offs0, keep, 3); R = gatherPts(R0, offs0, keep, 1);
      offs = new Uint32Array(keep.length + 1);
      keep.forEach((l, i) => { offs[i + 1] = offs[i] + (offs0[l + 1] - offs0[l]); });
      lm = filterLineMeta(lm0, keep);
    }
    const nLines = keep ? keep.length : (meta.n_lines || 0);
    const reg = dec('region_pts', Uint8Array);
    const dmg = dec('damage_pts', Uint8Array);
    const viol = dec('viol_pts', Uint8Array);
    const vRed = new THREE.Color(COL.violation), vOra = new THREE.Color(COL.shortfall);
    const am = new THREE.Color(COL.fasAM), pl = new THREE.Color(COL.fasPL), red = new THREE.Color(COL.collide);
    const regCols = (S.scene.regions || []).map((_, i) => new THREE.Color(...regColor(i)));
    const fam = { '+': new THREE.Color(0xe8c14a), '-': new THREE.Color(0x63c7b2), core: new THREE.Color(0xd77fd0), axial: new THREE.Color(0xe8c14a) };
    const statusC = new THREE.Color(STATUS_COL[meta.status] || 0x999999);
    const mcol = dec('mcol_pts', Uint8Array, 3);
    const bAM = new THREE.Color(LCOL.braidAM), bPL = new THREE.Color(LCOL.braidPL);
    const chC = [new THREE.Color(LCOL.chAM), new THREE.Color(LCOL.chPL), new THREE.Color(LCOL.chShared)];
    const sel = S.selection && S.selection.obj === meta.id ? S.selection.line : -1;
    const colorFn = (l, j, c) => {
      if (S.colorMode === 'status') { c.copy(statusC); }
      else if (meta.id.startsWith('native')) {
        c.copy(lm.bundle && lm.bundle[l] ? pl : am);
        if (dmg) c.multiplyScalar(1 - 0.55 * dmg[j] / 255);
        if (meta.id === 'native_fibers') c.lerp(new THREE.Color(0xffffff), 0.35);
        if (lm.bridging && lm.bridging[l]) c.lerp(new THREE.Color(0xffffff), 0.5);
      } else if (meta.id.startsWith('scaffold')) {
        if (S.colorMode === 'family' && lm.family) c.copy(fam[lm.family[l]] || fam['+']);
        else c.copy(reg ? regCols[reg[j]] || regCols[0] : new THREE.Color(COL.scaffold));
        if (meta.id === 'scaffold_fibers') c.lerp(new THREE.Color(0xffffff), 0.3);
        if (lm.collide && lm.collide[l]) c.copy(red);
        if (viol && viol[j]) c.copy(viol[j] & 63 ? vRed : vOra);    // per-sample fit violation
      } else if (isLayer(meta)) {
        if (mcol && S.colorMode === 'material') c.setRGB(mcol[3 * j] / 255, mcol[3 * j + 1] / 255, mcol[3 * j + 2] / 255);
        else if (meta.id === 'layer_braid') c.copy(S.colorMode === 'family' && lm.family ? fam[lm.family[l]] || fam['+'] : (lm.group && lm.group[l] ? bPL : bAM));
        else if (meta.id === 'layer_channels' || meta.id === 'layer_ports') c.copy(lm.group ? chC[lm.group[l]] || chC[0] : chC[0]);
        else if (meta.id === 'layer_coating') c.set(LCOL.coat);
        else if (meta.id === 'layer_grooves') c.set(lm.where && lm.where[l] ? 0xb7e6ff : LCOL.groove);
        else if (meta.id.startsWith('layer_footprint')) c.set(lm.outline && lm.outline[l] ? LCOL.fpBase : LCOL.fpOutline);
        else c.set(0xdddddd);
        if (viol && viol[j]) c.set(LCOL.over);
      } else if (meta.id === 'bundle_paths') c.copy(lm.bundle && lm.bundle[l] ? pl : am);
      else c.set(COL.centerline);
      if (l === sel) c.set(0x00ffff);
    };
    const mode = S.state.display.fiber_mode || 'auto';
    const asTube = meta.kind === 'tubes' || mode === 'tubes' || (mode === 'auto' && nLines <= 300);
    let RI = null, U = null;
    if (meta.hollow) {
      RI = dec('ri_pts', Float32Array); U = dec('u_pts', Float32Array, 3);
      const coat = dec('coat_pts', Uint8Array);
      const lum = new THREE.Color(LCOL.lumen), cc = new THREE.Color(LCOL.coat);
      const inner = (l, j, c) => { if (S.colorMode === 'status') c.copy(statusC); else if (meta.id === 'layer_coating' || (coat && coat[j])) c.copy(cc); else c.copy(lum); if (l === sel) c.set(0x00ffff); };
      const nsec = meta.id === 'layer_coating' ? 10 : (meta.n_lines > 120 ? 10 : 14);
      obj = new THREE.Mesh(hollowTubeGeometry(P, offs, R, RI, U, nsec, meta.aspect || 1, meta.exponent || 2, colorFn, inner, 0),
        matFor(meta, { color: 0xffffff, opacity: op, vcol: true, clip: clipsFor(meta.category), hl }));
    } else if (meta.kind === 'lines' && mode === 'hidden' && !isLayer(meta)) { obj = new THREE.Group(); }
    else if (asTube) {
      const nsec = meta.kind === 'tubes' ? 10 : 5;
      obj = new THREE.Mesh(await tubeGeometryAsync(P, offs, R, nsec, colorFn, 0), matFor(meta, { color: 0xffffff, opacity: op, vcol: true, clip: clipsFor(meta.category), hl }));
    } else {
      const m = matFor(meta, { color: 0xffffff, opacity: op, vcol: true, clip: clipsFor(meta.category), line: true });
      obj = new THREE.LineSegments(lineGeometry(P, offs, colorFn), m);
    }
    obj.userData.lines = { P, offs, meta: lm, viol, RI, U };
  }
  obj.name = meta.id; obj.visible = vis;
  obj.userData.category = meta.category; obj.userData.meta = meta;
  return obj;
}
function matOptsFor(obj) {
  // refresh the cached-material lookup for an object whose content is reused
  const meta = obj.userData.meta, m = obj.material;
  if (!meta || !m) return;
  const isLine = m.isLineBasicMaterial;
  const isPlainMesh = !isLine && meta.kind === 'mesh' && !m.vertexColors;
  const col = isPlainMesh ? objectColor(meta) : (m.color ? m.color.getHex() : 0xffffff);
  const vcol = !!m.vertexColors;
  const [, op, hl] = visFor(meta);
  obj.material = matFor(meta, { color: col, opacity: op == null ? 1 : op, vcol, clip: clipsFor(meta.category), line: isLine, hl });
}

function applyExploded() {
  // Inspection arrangement only: three.js object offsets. Canonical model coordinates
  // and every export are unchanged; e = 0 restores the assembled view exactly.
  const e = Number(S.state.display.exploded || 0);
  $('explodedBadge').hidden = !(e > 0);
  const sx = (S.scene.side === 'left' ? -1 : 1);
  for (const [id, o] of S.objects) {
    const cat = o.userData.category; o.position.set(0, 0, 0);
    if (!e) continue;
    if (id === 'bone_femur' || id === 'cartilage_femoral') o.position.z = e;            // engine +Z (root frame)
    else if (id === 'bone_tibia' || id === 'cartilage_tibial') o.position.z = -e;
    else if (id === 'layer_sheath' || id === 'layer_pores' || id === 'layer_coating_sheath') o.position.x = sx * e * 1.5;
    else if (id === 'layer_braid') o.position.x = sx * e * 0.5;
    else if (id === 'layer_channels' || id === 'layer_ports' || id === 'layer_coating' || id === 'layer_grooves') o.position.x = -sx * e * 0.6;
    else if (id === 'layer_interface') o.position.x = -sx * e * 1.3;
    else if (id === 'layer_end_tib' || id === 'layer_footprint_tib' || id === 'layer_plug_tib') o.position.z = -e * 1.2;
    else if (id === 'layer_end_fem' || id === 'layer_footprint_fem' || id === 'layer_plug_fem') o.position.z = e * 1.2;
    else if (cat === 'scaffold' || cat === 'scaffold_region') o.position.x = sx * e * 0.8;
    else if (cat === 'injury') o.position.x = -sx * e * 0.8;
  }
  const r = $('explodeRange');
  if (r && Number(r.value) !== e) r.value = e;
  invalidate();
}

async function rebuildSceneObjects() {
  // Reuse objects whose content key is unchanged (no geometry rebuild, no GPU
  // re-upload); build the rest, then swap so the old model stays visible until
  // the new one is ready. Replaced geometries are disposed; replaced GPU
  // resources cannot accumulate.
  const uiKey = S.colorMode + '|' + (S.state.display.fiber_mode || 'auto') + '|' + JSON.stringify(S.groupVis);
  const keepSel = S.selection ? S.selection.obj : null;
  const next = new Map(), nextCkeys = new Map();
  let reused = 0, built = 0;
  for (const meta of S.scene.objects) {
    const old = S.objects.get(meta.id);
    if (old && meta.ckey && S.ckeys.get(meta.id) === meta.ckey && S.uiKey === uiKey && meta.id !== keepSel) {
      const [vis] = visFor(meta);
      old.visible = vis;
      old.userData.meta = meta;
      if (old.userData.lines && meta.kind !== 'mesh') { /* same content: keep decoded lines */ }
      matOptsFor(old);
      next.set(meta.id, old); nextCkeys.set(meta.id, meta.ckey); reused++;
      continue;
    }
    const o = await buildObject(meta);
    if (meta.id === keepSel && old) { /* selection highlight rebuilt with the object */ }
    root.add(o); next.set(meta.id, o); nextCkeys.set(meta.id, meta.ckey || ''); built++;
  }
  for (const [id, o] of S.objects) {
    if (next.get(id) !== o) { root.remove(o); disposeObj(o); }
  }
  S.objects = next; S.ckeys = nextCkeys; S.uiKey = uiKey;
  perf.lastReuse = { reused, built };
  applyExploded(); updateSlab(); placeHandles(); refreshLabelAnchors(); invalidate();
}

// ------------------------------------------------------------------ HUD: labels, scale, side
function updateHud() {
  if (!S.scene) return;
  const lab = $('labels'); lab.innerHTML = '';
  const w = ren.domElement.clientWidth, h = ren.domElement.clientHeight;
  const side = S.scene.side === 'left' ? -1 : 1;
  const c = ctl.target.clone(); const c2 = c.clone().project(cam);
  const pts = { Anterior: [0, 0, -1], Posterior: [0, 0, 1], Lateral: [side, 0, 0], Medial: [-side, 0, 0], Proximal: [0, 1, 0], Distal: [0, -1, 0] };
  const R = 0.42 * Math.min(w, h);
  for (const [k, d] of Object.entries(pts)) {
    const p = c.clone().add(new THREE.Vector3(...d).multiplyScalar(10)).project(cam);
    let dx = (p.x - c2.x) * w / 2, dy = -(p.y - c2.y) * h / 2; const L = Math.hypot(dx, dy);
    if (L < 6) continue;                                   // axis points at / away from the viewer
    const el = document.createElement('div'); el.className = 'lab'; el.textContent = k;
    el.style.left = (w / 2 + (c2.x * w / 2) + dx / L * R) + 'px'; el.style.top = (h / 2 - (c2.y * h / 2) + dy / L * R) + 'px';
    lab.appendChild(el);
  }
  if (S.labelsOn && S.labelAnchors) {
    for (const a of S.labelAnchors) {
      const v = a.p.clone().project(cam);
      if (v.z > 1 || v.z < -1) continue;
      const el = document.createElement('div'); el.className = 'clab' + (a.cls ? ' ' + a.cls : ''); el.textContent = a.t;
      el.style.left = ((v.x * 0.5 + 0.5) * w) + 'px'; el.style.top = ((-v.y * 0.5 + 0.5) * h) + 'px';
      lab.appendChild(el);
    }
  }
  // scale bar: pixels per mm at the orbit target
  const dist = cam.position.distanceTo(ctl.target);
  const pxPerMm = ren.domElement.clientHeight / (2 * dist * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)));
  let mm = 10; if (pxPerMm * mm > 160) mm = 5; if (pxPerMm * mm > 160) mm = 2; if (pxPerMm * mm < 40) mm = 20; if (pxPerMm * mm < 40) mm = 50;
  $('scaleLine').style.width = (pxPerMm * mm) + 'px'; $('scaleTxt').textContent = mm + ' mm (at orbit centre)';
  $('sideBadge').textContent = (S.scene.side || 'right').toUpperCase() + ' knee · flexion ' + Number(S.state.params['knee.flexion_deg']).toFixed(0) + '°';
  if (guide && guide.anchorWorld && !$('guideLabel').hidden) placeGuideLabel(guide.anchorWorld);
  else if (!guide && guideAnchor && S.levelSel && !$('guideLabel').hidden) placeGuideLabel(guideAnchor);
  if (!$('detailWrap').hidden && S.detailLevel) updateMag();
}

// ------------------------------------------------------------------ picking + detail
async function pick(ev) {
  const r = ren.domElement.getBoundingClientRect();
  const m = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(m, cam);
  const cands = [...S.objects.values()].filter(o => o.visible);
  const hits = raycaster.intersectObjects(cands, false);
  // hidden geometry cannot be hit (filtered above); clipped-away hits are skipped
  let h = null;
  for (const hh of hits) {
    if (hh.object.visible && !clippedAway(hh.object.userData.category, hh.point)) { h = hh; break; }
  }
  if (!h) { S.selection = null; renderSelection(); return; }
  const o = h.object;
  let line = -1;
  const lidAttr = o.geometry.getAttribute('lid');
  if (lidAttr && h.face) line = Math.round(lidAttr.getX(h.face.a));
  else if (lidAttr && h.index != null) line = Math.round(lidAttr.getX(h.index));
  S.selection = line >= 0 ? { obj: o.name, line, point: h.point.clone() } : null;
  const meta = S.objects.get(o.name).userData.meta;
  if (meta && (meta.level || isLayer(meta))) {
    let group = null;
    if (line >= 0 && o.userData.lines && o.userData.lines.meta.group) group = o.userData.lines.meta.group[line];
    selectComponent({ obj: o.name, group }, { fromScene: true });
  }
  root.remove(o); disposeObj(o);
  const n = await buildObject(meta); root.add(n); S.objects.set(o.name, n);
  applyExploded(); renderSelection(); invalidate();
}

let hoverTick = 0;
function hover(ev) {
  if (ev.buttons !== 0 || !S.scene) { $('tooltip').hidden = true; return; }
  const now = performance.now();
  if (now - hoverTick < 90) return;
  hoverTick = now;
  const r = ren.domElement.getBoundingClientRect();
  const m = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(m, cam);
  const cands = [...S.objects.values()].filter(o => o.visible);
  const hits = raycaster.intersectObjects(cands, false);
  let h = null;
  for (const hh of hits) {
    if (hh.object.visible && !clippedAway(hh.object.userData.category, hh.point)) { h = hh; break; }
  }
  const tip = $('tooltip');
  if (!h) { tip.hidden = true; return; }
  const o = h.object, meta = o.userData.meta || {};
  let extra = '';
  const lidAttr = o.geometry.getAttribute('lid');
  if (lidAttr && o.userData.lines) {
    const l = Math.round(lidAttr.getX(h.face ? h.face.a : h.index));
    const lm = o.userData.lines.meta || {};
    if (lm.group != null && lm.group[l] != null) extra = ['AM', 'PL', 'shared'][lm.group[l]] || '';
    else if (lm.bundle != null && lm.bundle[l] != null) extra = lm.bundle[l] ? 'PL' : 'AM';
    if (lm.strand != null && lm.strand[l] != null) extra += (extra ? ' · ' : '') + 'strand ' + lm.strand[l];
    if (lm.channel != null && lm.channel[l] != null) extra += (extra ? ' · ' : '') + 'channel ' + lm.channel[l];
  }
  tip.innerHTML = `<b>${meta.label || o.name}</b>` + (meta.level ? ` · Level ${meta.level}` : '') + (extra ? ` · ${extra}` : '');
  tip.style.left = (ev.clientX - r.left + 14) + 'px';
  tip.style.top = (ev.clientY - r.top + 10) + 'px';
  tip.hidden = false;
}

function renderSelection() {
  const box = $('selInfo');
  if (!S.selection) { box.textContent = 'Click a fascicle, fiber or yarn in the 3D view.'; return; }
  const o = S.objects.get(S.selection.obj); if (!o) return;
  const lm = o.userData.lines.meta; const l = S.selection.line; const meta = o.userData.meta;
  const rows = [['Object', meta.label], ['Status', meta.status]];
  if (lm.bundle) rows.push(['Bundle', lm.bundle[l] ? 'PL (posterolateral)' : 'AM (anteromedial)']);
  if (lm.fascicle) rows.push(['Fascicle group', lm.fascicle[l]]);
  if (lm.fiber) rows.push(['Representative fiber', lm.fiber[l]]);
  if (lm.yarn) rows.push(['Yarn', lm.yarn[l] + (lm.family ? ' (' + lm.family[l] + ')' : '')]);
  if (lm.collide && lm.collide[l]) rows.push(['Clearance', 'INTERSECTS another yarn']);
  if (lm.unit && meta.units) rows.push(['Repair unit', meta.units[lm.unit[l]] + (lm.layer ? ' · layer ' + (lm.layer[l] + 1) : '')]);
  const vv = o.userData.lines.viol;
  if (vv) { const offs = o.userData.lines.offs; let bits = 0, n = 0; for (let j = offs[l]; j < offs[l + 1]; j++) { bits |= vv[j]; n += vv[j] ? 1 : 0; }
    rows.push(['Fit', bits ? `${n} samples: ${vtext(bits)}` : 'inside its repair space, all clearances met']); }
  if (lm.bridging && lm.bridging[l]) rows.push(['Continuity', 'bridges the tear (residual continuity)']);
  const sm = S.scene.metrics.scaffold;
  if (meta.id.startsWith('scaffold') && sm) {
    const offs = o.userData.lines.offs; const regs = meta.region_pts ? b64(meta.region_pts, Uint8Array) : null;
    if (regs) { const seen = new Set(); for (let j = offs[l]; j < offs[l + 1]; j++) seen.add(S.scene.regions[regs[j]]); rows.push(['Regions along it', [...seen].join(' → ')]); }
  }
  box.innerHTML = '<table class="t">' + rows.map(r => '<tr><th>' + r[0] + '</th><td>' + r[1] + '</td></tr>').join('') + '</table>';
  const kind = meta.id.startsWith('scaffold') ? 'yarn' : 'fascicle';
  const id = kind === 'yarn' ? (lm.yarn ? lm.yarn[l] : null) : (lm.fascicle ? lm.fascicle[l] : null);
  if (id !== null && id !== undefined) {
    const b = document.createElement('button'); b.textContent = 'Show fibers of this ' + kind + ' (detail view)';
    b.onclick = () => openDetail(kind, id); box.appendChild(b);
  }
}

// ------------------------------------------------------------------ Scaffold Builder: components
// Stable component identifiers with structural level, bundle, region and the scene
// objects (or AM/PL line subgroups) they map to. The tree is built from the live
// scene: entries whose objects are absent (e.g. no plugs, no pores) are skipped.
const COMP_DEFS = [
  { header: 'Sheath and inter-bundle zone (Level 5)' },
  { id: 'sheath', short: 'Sheath (ribbon)', level: 5, obj: 'layer_sheath', color: LCOL.sheath, desc: 'Shared outer envelope around both bundles. Outer surface smooth and uncoated.' },
  { id: 'pores', short: 'Sheath pores', level: 5, obj: 'layer_pores', color: LCOL.pore, cls: 'sub', desc: 'Explicit through-wall openings (geometry only; no transport model).' },
  { id: 'iface', short: 'Inter-bundle zone', level: 5, obj: 'layer_interface', color: LCOL.iface, desc: 'Transition band between the mantles. No wall is placed here; channels may cross it. Not load-bearing.' },
  { header: 'AM bundle' },
  { id: 'mantle_am', short: 'AM mantle', level: 4, bundle: 'AM', obj: 'layer_braid', group: 0, color: LCOL.braidAM, desc: 'Braided mantle wrapping the AM-like channel core.' },
  { id: 'core_am', short: 'AM core channels', level: 3, bundle: 'AM', obj: 'layer_channels', group: 0, color: LCOL.chAM, cls: 'sub', desc: 'Hollow fascicle channels of the AM-like core (lumens + walls).' },
  { header: 'PL bundle' },
  { id: 'mantle_pl', short: 'PL mantle', level: 4, bundle: 'PL', obj: 'layer_braid', group: 1, color: LCOL.braidPL, desc: 'Braided mantle wrapping the PL-like channel core.' },
  { id: 'core_pl', short: 'PL core channels', level: 3, bundle: 'PL', obj: 'layer_channels', group: 1, color: LCOL.chPL, cls: 'sub', desc: 'Hollow fascicle channels of the PL-like core (lumens + walls).' },
  { header: 'Shared channels and connections (Level 3)' },
  { id: 'core_shared', short: 'Shared channels', level: 3, obj: 'layer_channels', group: 2, color: LCOL.chShared, desc: 'Channels running in the AM/PL transition band.' },
  { id: 'ports', short: 'Channel ports', level: 3, obj: 'layer_ports', color: LCOL.chShared, cls: 'sub', desc: 'Short hollow cross-connections between neighbouring channels.' },
  { header: 'Insertions and anchors (Level 6)' },
  { id: 'end_tib', short: 'Tibial insertion', level: 6, region: 'tibial', objs: ['layer_end_tib', 'layer_footprint_tib', 'layer_plug_tib'], color: 0xd8c4e0, desc: 'Flared tibial end with graded interface on the tibial footprint.' },
  { id: 'end_fem', short: 'Femoral insertion', level: 6, region: 'femoral', objs: ['layer_end_fem', 'layer_footprint_fem', 'layer_plug_fem'], color: 0xd8c4e0, desc: 'Flared femoral end with graded interface on the femoral footprint.' },
  { id: 'anchors', short: 'Anchors (4)', level: 6, obj: 'layer_anchors', color: LCOL.anchor, desc: 'Schematic markers at the AM/PL attachment-subregion centroids. Not fixation hardware.' },
  { header: 'Surface treatments (Levels 1–2)' },
  { id: 'coating', short: 'Coating', level: 1, objs: ['layer_coating', 'layer_coating_sheath'], color: LCOL.coat, desc: 'Protein coating on channel walls, inner sheath and braid strands as configured. Outer sheath excluded.' },
  { id: 'grooves', short: 'Microgrooves', level: 2, obj: 'layer_grooves', color: LCOL.groove, desc: 'Guidance grooves on channel inner walls and inner sheath. See the detail inspector for true scale.' },
];
const compDef = (id) => COMP_DEFS.find(d => d.id === id);
function compObjs(def) { return def.objs || (def.obj ? [def.obj] : []); }
function compPresent(def) {
  if (!S.scene) return false;
  const ids = new Set(S.scene.objects.map(o => o.id));
  return compObjs(def).some(id => ids.has(id));
}
function compVisible(def) {
  if (def.group != null) return S.vis[def.obj] !== false && S.groupVis[def.obj] && S.groupVis[def.obj][def.group] !== false;
  return compObjs(def).some(id => S.vis[id]);
}
function markersOf() {
  const m = S.scene && S.scene.objects.find(o => o.id === 'layer_anchors');
  return (m && m.markers) || [];
}

function renderCompTree() {
  const box = $('compTree'); if (!box) return;
  box.innerHTML = '';
  if (!S.scene) { box.innerHTML = '<p class="note">No model yet.</p>'; return; }
  const layered = S.state.params['scaffold.arch'] === 'layered';
  if (!layered) box.insertAdjacentHTML('beforeend', '<p class="note">The layered construct is off (gap-filling yarn scaffold selected); structural components appear when it is on.</p>');
  for (const def of COMP_DEFS) {
    if (def.header) { const h = document.createElement('div'); h.className = 'subh'; h.textContent = def.header; box.appendChild(h); continue; }
    if (!compPresent(def)) continue;
    box.appendChild(compRow(def));
    if (def.id === 'anchors') {
      for (const mk of markersOf()) {
        const r = document.createElement('div');
        r.className = 'comprow sub2' + (S.compSel && S.compSel.marker === mk.id ? ' sel' : '') + (S.vis['layer_anchors'] === false ? ' off' : '');
        r.innerHTML = `<span class="dot" style="background:#${new THREE.Color(mk.bundle === 'AM' ? COL.footprintAM : COL.footprintPL).getHexString()}"></span><span class="nm">${mk.label} <span class="lv">${mk.P.map(v => v.toFixed(1)).join(', ')} mm</span></span>`;
        const acts = document.createElement('span'); acts.className = 'acts';
        const fb = document.createElement('button'); fb.textContent = 'focus'; fb.title = 'Move the camera to this anchor';
        fb.onclick = (e) => { e.stopPropagation(); focusMarker(mk); };
        acts.appendChild(fb); r.appendChild(acts);
        r.onclick = () => selectComponent({ obj: 'layer_anchors', marker: mk.id });
        r.title = mk.label + ' — ' + mk.provenance;
        box.appendChild(r);
      }
    }
  }
}
function compRow(def) {
  const r = document.createElement('div');
  const sel = S.compSel && S.compSel.def === def.id;
  r.className = 'comprow' + (def.cls ? ' ' + def.cls : '') + (sel ? ' sel' : '') + (compVisible(def) ? '' : ' off');
  r.innerHTML = `<span class="dot" style="background:#${new THREE.Color(def.color).getHexString()}"></span><span class="nm">${def.short} <span class="lv">L${def.level}${def.bundle ? ' · ' + def.bundle : ''}${def.region ? ' · ' + def.region : ''}</span></span>`;
  const acts = document.createElement('span'); acts.className = 'acts';
  const mk = (t, title, fn) => { const b = document.createElement('button'); b.textContent = t; b.title = title; b.onclick = (e) => { e.stopPropagation(); fn(); }; acts.appendChild(b); };
  mk(compVisible(def) ? 'hide' : 'show', 'Show/hide this component (display only)', () => toggleComp(def));
  mk('isolate', 'Show only this component', () => isolateComp(def));
  mk('focus', 'Move the camera to this component', () => focusComp(def));
  r.appendChild(acts);
  r.onclick = () => selectComponent({ def: def.id, obj: def.obj || (def.objs || [])[0], objs: def.objs, group: def.group != null ? def.group : null });
  r.title = def.desc;
  return r;
}

function selectComponent(sel, opts) {
  opts = opts || {};
  let def = sel.def ? compDef(sel.def) : COMP_DEFS.find(d => !d.header && (d.obj === sel.obj || (d.objs || []).includes(sel.obj)) && (d.group == null || d.group === sel.group));
  if (sel.marker) def = compDef('anchors');
  S.compSel = Object.assign({}, sel, def && { def: def.id });
  if (def && def.level && !opts.noLevel && S.levelSel !== def.level) selectLevel(def.level);
  refreshLevelView(); renderCompTree(); renderCompInfo();
}
function renderCompInfo() {
  const box = $('compInfo'); if (!box) return;
  const cs = S.compSel;
  if (!cs) { box.textContent = 'Nothing selected.'; return; }
  const def = cs.def ? compDef(cs.def) : null;
  const mk = cs.marker && markersOf().find(m => m.id === cs.marker);
  let h = '';
  if (mk) h = `<b>${mk.label}</b> · Level 6<br>Position: <span class="mono">${mk.P.map(v => v.toFixed(2)).join(', ')} mm</span> (model frame)<br>Provenance: ${mk.provenance}<br><span class="muted">Schematic marker, not fixation hardware.</span>`;
  else if (def) {
    const L = (S.reg.levels || []).find(x => x.n === def.level);
    h = `<b>${def.short}</b> · Level ${def.level} — ${L ? L.name : ''}${def.bundle ? ' · ' + def.bundle + ' bundle' : ''}<br>${def.desc}`;
    const notes = compObjs(def).map(id => (S.scene.objects.find(o => o.id === id) || {}).note).filter(Boolean);
    if (notes.length) h += '<br><span class="muted">' + notes[0] + '</span>';
  } else h = 'Nothing selected.';
  box.innerHTML = h;
  if (def && def.level) {
    const b = document.createElement('button'); b.textContent = 'Open Level ' + def.level + ' controls'; b.style.marginTop = '6px';
    b.onclick = () => { showTab('design'); $('pnlLevels').open = true; selectLevel(def.level); };
    box.appendChild(document.createElement('br')); box.appendChild(b);
  }
}

async function refreshObject(id) {
  const o = S.objects.get(id);
  if (!o) return;
  const meta = o.userData.meta;
  root.remove(o); disposeObj(o);
  const n = await buildObject(meta);
  root.add(n); S.objects.set(id, n);
  applyExploded(); refreshLabelAnchors(); invalidate();
}
function toggleComp(def) {
  if (def.group != null) {
    const gv = S.groupVis[def.obj];
    gv[def.group] = gv[def.group] === false;
    if (S.selection && S.selection.obj === def.obj) S.selection = null;
    refreshObject(def.obj).then(() => { renderObjList(); renderCompTree(); renderSelection(); });
    return;
  }
  const on = !compVisible(def);
  compObjs(def).forEach(id => { S.vis[id] = on; });
  refreshLevelView(); renderObjList(); renderCompTree(); refreshLabelAnchors(); saveView();
}
function isolateComp(def) {
  // component isolation replaces any level view mode: the tree selection is the focus
  setLevelMode('highlight');
  if (S.levelSel) selectLevel(null);
  const keep = new Set(compObjs(def));
  for (const meta of S.scene.objects) {
    if (!isLayer(meta)) continue;
    S.vis[meta.id] = keep.has(meta.id);
  }
  const jobs = [];
  if (def.group != null) {
    const gv = S.groupVis[def.obj];
    for (const k of Object.keys(gv)) gv[k] = (Number(k) === def.group);
    jobs.push(refreshObject(def.obj));
  }
  S.selection = null;
  Promise.all(jobs).then(() => { refreshLevelView(); renderObjList(); renderCompTree(); renderCompInfo(); renderSelection(); saveView(); });
}
function groupBox(objId, group) {
  const o = S.objects.get(objId);
  if (!o || !o.userData.lines) return new THREE.Box3();
  const { P, offs, meta } = o.userData.lines;
  const grp = meta.group || [];
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  root.updateMatrixWorld(true);
  for (let l = 0; l < grp.length; l++) {
    if (grp[l] !== group) continue;
    for (let j = offs[l]; j < offs[l + 1]; j++) {
      // object position (explode offset) lives in root space: add before the world transform
      v.set(P[3 * j], P[3 * j + 1], P[3 * j + 2]).add(o.position).applyMatrix4(root.matrixWorld);
      box.expandByPoint(v);
    }
  }
  return box;
}
function compBox(def) {
  root.updateMatrixWorld(true);
  if (def.group != null && def.obj) return groupBox(def.obj, def.group);
  const box = new THREE.Box3();
  compObjs(def).forEach(id => { const o = S.objects.get(id); if (o) box.expandByObject(o); });
  return box;
}
function frameBox(box, dir) {
  if (!box || box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const sz = box.getSize(new THREE.Vector3()).length();
  const d = Math.max(4, (sz / 2) / Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) * 1.2);
  const side = (S.scene && S.scene.side === 'left') ? -1 : 1;
  const v = dir || new THREE.Vector3(side * 0.62, 0.32, -0.72).normalize();
  cam.position.copy(c).addScaledVector(v, d);
  cam.near = Math.max(0.02, d / 300); cam.far = d * 30; cam.updateProjectionMatrix();
  ctl.target.copy(c); ctl.update(); invalidate();
}
function focusComp(def) {
  const box = compBox(def);
  if (box.isEmpty()) return;
  // make sure the target is visible, then frame it with an appropriate orbit target
  if (def.group != null) { S.vis[def.obj] = true; S.groupVis[def.obj][def.group] = true; refreshObject(def.obj); }
  else compObjs(def).forEach(id => { S.vis[id] = true; });
  refreshLevelView(); renderCompTree();
  frameBox(box);
}
function focusMarker(mk) {
  root.updateMatrixWorld(true);
  const w = new THREE.Vector3(...mk.P).applyMatrix4(root.matrixWorld);
  const box = new THREE.Box3(w.clone().addScalar(-2), w.clone().addScalar(2));
  S.vis['layer_anchors'] = true; refreshLevelView(); renderCompTree();
  frameBox(box);
  selectComponent({ obj: 'layer_anchors', marker: mk.id });
}

// ---------------- Scaffold Builder: view mode
const ACL_OVERLAY_IDS = ['retained_tissue', 'retained_tibial_stump', 'retained_femoral_stump', 'intact_reference',
  'bundle_am_reference', 'bundle_pl_reference', 'injury_region', 'native_fascicles', 'native_fibers',
  'bundle_interface', 'bundle_paths'];
const BONE_OVERLAY_IDS = ['bone_femur', 'bone_tibia', 'cartilage_femoral', 'cartilage_tibial', 'landmark_pcl',
  'footprint_tibial', 'footprint_femoral'];
function currentTab() {
  const b = document.querySelector('#tabs button.on');
  return b ? b.dataset.tab : 'design';
}
function setBuilder(on) {
  S.builder = on;
  $('btnViewAnat').classList.toggle('on', !on);
  $('btnViewBuilder').classList.toggle('on', on);
  $('builderBar').hidden = !on;
  if (!S.scene) return;
  if (on) {
    S.savedView = { vis: Object.assign({}, S.vis), opac: Object.assign({}, S.opac), tab: currentTab() };
    for (const meta of S.scene.objects) {
      if (isLayer(meta)) {
        const d = DEFAULT_VIS[meta.id] || [true, 1];
        S.vis[meta.id] = d[0]; S.opac[meta.id] = d[1];
      } else S.vis[meta.id] = false;
    }
    $('ovACL').checked = false; $('ovBone').checked = false;
    setDisplay('exploded', 0);
    showTab('builder');
    renderDisplayControls();
    refreshLevelView(); renderObjList(); renderCompTree();
    fitScaffold();
  } else {
    if (S.savedView) {
      S.vis = Object.assign({}, S.savedView.vis); S.opac = Object.assign({}, S.savedView.opac);
      showTab(S.savedView.tab || 'design'); S.savedView = null;
    }
    refreshLevelView(); renderObjList(); renderCompTree();
  }
  saveView(); refreshLabelAnchors();
}
function applyAclOverlay() {
  // Native ACL tissue is independent of the engineered sheath: its own flag and opacity.
  const on = $('ovACL').checked, op = Number($('ovACLOp').value);
  for (const id of ACL_OVERLAY_IDS) {
    if (!(id in S.vis) && !S.objects.has(id)) continue;
    S.vis[id] = on;
    if (on) S.opac[id] = (id === 'native_fascicles' || id === 'native_fibers') ? Math.min(1, op + 0.35) : op;
  }
  refreshLevelView(); renderObjList(); saveView(); refreshLabelAnchors();
}
function applyBoneOverlay() {
  const on = $('ovBone').checked;
  for (const id of BONE_OVERLAY_IDS) {
    if (!(id in S.vis) && !S.objects.has(id)) continue;
    if (on) { const d = DEFAULT_VIS[id] || [true, 1]; S.vis[id] = d[0]; S.opac[id] = d[1]; }
    else S.vis[id] = false;
  }
  refreshLevelView(); renderObjList(); saveView(); refreshLabelAnchors();
}
function fitScaffold() {
  const box = sceneBox(o => { const m = o.userData.meta; return m && (isLayer(m) || m.category === 'scaffold'); });
  if (!box.isEmpty()) frameBox(box);
}
function restoreBuilderView() {
  if (!S.scene) return;
  for (const meta of S.scene.objects) {
    if (isLayer(meta)) { const d = DEFAULT_VIS[meta.id] || [true, 1]; S.vis[meta.id] = d[0]; S.opac[meta.id] = d[1]; }
  }
  for (const k of Object.keys(S.groupVis)) for (const g of Object.keys(S.groupVis[k])) S.groupVis[k][g] = true;
  S.selection = null; S.compSel = null;
  setDisplay('exploded', 0); applyExploded();
  $('ovACL').checked = false; $('ovBone').checked = false;
  applyAclOverlay(); applyBoneOverlay();
  renderDisplayControls();
  rebuildSceneObjects().then(() => { refreshLevelView(); renderObjList(); renderCompTree(); renderCompInfo(); renderSelection(); fitScaffold(); });
}
function toggleLongCut() {
  // Longitudinal cutaway along the scaffold's long axis (display-only clip preset).
  const btn = $('btnLongCut');
  if (S.clip && S.clip.on && S.clip.long) {
    S.clip = { on: false, axis: 'sagittal', pos: 0, flip: false };
    btn.classList.remove('on');
  } else {
    const box = sceneBox(o => { const m = o.userData.meta; return m && isLayer(m); });
    const c = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
    S.clip = { on: true, axis: 'coronal', pos: -c.z, flip: false, long: true };
    btn.classList.add('on');
  }
  renderClipControls(); updateClip();
}
function refreshLabelAnchors() {
  // Cached world positions for component labels (recomputed on discrete view changes,
  // never per frame). One label per component; never one per channel.
  S.labelAnchors = [];
  if (!S.labelsOn || !S.scene) return;
  root.updateMatrixWorld(true);
  for (const def of COMP_DEFS) {
    if (def.header || !compPresent(def) || !compVisible(def)) continue;
    const box = compBox(def);
    if (box.isEmpty()) continue;
    S.labelAnchors.push({ p: box.getCenter(new THREE.Vector3()), t: `L${def.level} ${def.short}`, cls: def.bundle ? def.bundle.toLowerCase() : '' });
  }
  if (S.vis['layer_anchors'] !== false) {
    for (const mk of markersOf()) {
      S.labelAnchors.push({ p: new THREE.Vector3(...mk.P).applyMatrix4(root.matrixWorld), t: mk.label, cls: mk.bundle.toLowerCase() });
    }
  }
}

const detail = {
  ren: null, scene: null, cam: null, ctl: null, obj: null,
  init() {
    const host = $('detailView');
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color(0x11181a);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x667788, 1.3));
    const k = new THREE.DirectionalLight(0xffffff, 1.0); k.position.set(1, 2, 1); this.scene.add(k);
    this.cam = new THREE.PerspectiveCamera(35, 1, 0.001, 100);
    this.ren = new THREE.WebGLRenderer({ antialias: true }); this.ren.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    host.appendChild(this.ren.domElement);
    this.ctl = new OrbitControls(this.cam, this.ren.domElement); this.ctl.addEventListener('change', invalidate);
    new ResizeObserver(() => { const w = host.clientWidth, h = host.clientHeight; if (!w || !h) return; this.ren.setSize(w, h); this.cam.aspect = w / h; this.cam.updateProjectionMatrix(); invalidate(); }).observe(host);
  },
  render() { if (!$('detailWrap').hidden) { this.ctl.update(); this.ren.render(this.scene, this.cam); } },
};

async function openDetail(kind, id) {
  $('detailWrap').hidden = false; if (!detail.ren) detail.init();
  S.detailLevel = null; S.detailInfo = null; $('detailMag').textContent = '';
  $('detailNote').textContent = 'loading…';
  try {
    const st = JSON.parse(JSON.stringify(S.state));
    let s = st.display.section_s;
    if (S.selection && S.selection.point) {
      // nearest centreline position to the clicked point
      const cl = S.objects.get('centerline');
      if (cl) { const P = cl.userData.lines.P; let best = 1e9, bi = 0; const q = S.selection.point.clone(); root.worldToLocal(q);
        for (let i = 0; i < P.length / 3; i++) { const d = (P[3 * i] - q.x) ** 2 + (P[3 * i + 1] - q.y) ** 2 + (P[3 * i + 2] - q.z) ** 2; if (d < best) { best = d; bi = i; } }
        s = bi / (P.length / 3 - 1); }
    }
    const j = await api('/api/v2/detail', { state: st, selection: { type: kind, id, s, window_mm: 0.4, fibers: 19 } });
    if (detail.obj) { detail.scene.remove(detail.obj); disposeObj(detail.obj); }
    const P = b64(j.p, Float32Array), offs = b64(j.o, Uint32Array), R0 = b64(j.r, Float32Array);
    const R = R0.map(() => j.window_mm / 260);
    const c = new THREE.Color();
    const g = tubeGeometry(P, offs, R, 6, (l, jj, cc) => cc.setHSL(0.08 + 0.5 * (l / Math.max(1, offs.length - 2)), 0.55, 0.62), 0);
    detail.obj = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8 }));
    detail.obj.rotation.x = -Math.PI / 2; detail.scene.add(detail.obj);
    detail.obj.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(detail.obj), ce = box.getCenter(new THREE.Vector3()), rr = box.getSize(new THREE.Vector3()).length() / 2;
    detail.cam.position.copy(ce).add(new THREE.Vector3(rr * 1.6, rr * 0.4, rr * 1.2)); detail.cam.near = rr / 100; detail.cam.far = rr * 50; detail.cam.updateProjectionMatrix();
    detail.ctl.target.copy(ce); detail.ctl.update();
    $('detailNote').textContent = j.note + ' · window ' + j.window_mm.toFixed(1) + ' mm · fibers drawn thicker than true diameter (' +
      (kind === 'yarn' ? j.physical.scaffold_fiber_diameter_um : j.physical.native_fiber_diameter_um) + ' µm)';
    invalidate();
  } catch (e) { $('detailNote').textContent = 'Detail failed: ' + e.message; }
}
$('detailClose').onclick = () => { $('detailWrap').hidden = true; S.detailLevel = null; };

// ------------------------------------------------------------------ handles (landmarks)
const FLEX_A = new THREE.Vector3(0, -6, 24);
function flexR(deg) { const t = THREE.MathUtils.degToRad(-deg); return new THREE.Matrix3().set(1, 0, 0, 0, Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t)); }
function placeHandles() {
  for (const h of Object.values(handles)) { root.remove(h); h.geometry.dispose(); h.material.dispose(); }
  handles = {}; tctl.detach();
  if (VIEW_MODE || !S.scene || !S.handleMode) return;
  const mk = (name, p, color) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.9, 16, 12), new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 }));
    m.position.set(p[0], p[1], p[2]); m.renderOrder = 10; m.name = 'handle_' + name; m.userData.handle = name; root.add(m); handles[name] = m; return m;
  };
  const H = S.scene.handles;
  if (S.handleMode === 'fp_tib') mk('fp_tib', H.fp_tib, 0xffaa00);
  if (S.handleMode === 'fp_fem') mk('fp_fem', H.fp_fem, 0xffaa00);
  if (S.handleMode === 'ctrl') { mk('ctrl0', H.ctrl[0], 0x00ddff); mk('ctrl1', H.ctrl[1], 0x00ddff); }
  const first = Object.values(handles)[0]; if (first) tctl.attach(first);
  ren.domElement.addEventListener('dblclick', handleSwitch);
}
function handleSwitch(ev) {
  if (!S.handleMode) return;
  const r = ren.domElement.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), cam);
  const hit = raycaster.intersectObjects(Object.values(handles), false)[0]; if (hit) tctl.attach(hit.object);
}
function onHandleDrop() {
  const h = tctl.object; if (!h) return;
  gestureStart();
  const side = S.scene.side === 'left' ? -1 : 1;
  const w = h.position.clone(); w.x *= side;                    // back to the right-canonical engine frame
  const name = h.userData.handle;
  if (name === 'fp_tib') setParam('fp.tib.center', [w.x, w.y, w.z].map(v => +v.toFixed(3)));
  else if (name === 'fp_fem') {
    const R = flexR(Number(S.state.params['knee.flexion_deg'])); const loc = w.clone().sub(FLEX_A).applyMatrix3(R.clone().transpose()).add(FLEX_A);
    setParam('fp.fem.center', [loc.x, loc.y, loc.z].map(v => +v.toFixed(3)));
  } else {
    const k = name === 'ctrl0' ? 0 : 1; const base = S.scene.handles.ctrl_base[k];
    const offs = JSON.parse(JSON.stringify(S.state.params['lig.ctrl_offsets']));
    offs[k] = [w.x - side * base[0] * side, w.y - base[1], w.z - base[2]].map(v => +v.toFixed(3));
    offs[k][0] = +(w.x - (side === 1 ? base[0] : -base[0])).toFixed(3);
    setParam('lig.ctrl_offsets', offs, { noUndo: true });
  }
  gestureCommit('drag handle in 3D view');
}

// ------------------------------------------------------------------ builds
function setStatus(text) {
  const el = $('statusLine');
  if (!text) { el.hidden = true; return; }
  el.hidden = false; el.textContent = text;
}
function scheduleBuild(quality, delay) {
  // coalesce rapid slider events: only one trailing build runs, and it always
  // reads the latest value from S.state (no long debounce; numbers update now)
  clearTimeout(S.buildTimer);
  S.buildTimer = setTimeout(() => build(quality), delay == null ? 60 : delay);
  if (S.scene) setStatus(quality === 'preview' ? 'Updating preview…' : 'Finalizing model…');
}
async function build(quality) {
  const seq = ++S.seq;
  S.building = quality;
  if (quality === 'full') clearGuide(true);
  if (S.abort) S.abort.abort();   // cancel the obsolete in-flight job, if any
  const ac = new AbortController(); S.abort = ac;
  badge(quality === 'preview' ? 'preview…' : 'building…', 'busy');
  if (S.scene) setStatus(quality === 'preview' ? 'Updating preview…' : 'Finalizing model…');
  const t0 = performance.now();
  if (!S.scene) progress('Building geometry', ['design: ' + (S.state.case && S.state.case.label), 'server: generating anatomy, injury region, scaffold…']);
  let j;
  try {
    j = await api('/api/v2/build', { state: S.state, quality, have_keys: [...S.meshCache.keys()], seq }, ac.signal);
  } catch (e) {
    if (e.name === 'AbortError') return;
    if (seq !== S.seq) return;
    S.building = null; setStatus(null);
    badge('error', 'fail');
    emit('acl:build-error', { seq, quality, status: e.status, errors: (e.body && e.body.errors) || null, message: String(e.message) });
    if (e.status === 422 && e.body && e.body.errors) {
      S.msgs = {}; e.body.errors.forEach(x => (S.msgs[x.param] = S.msgs[x.param] || []).push({ level: 'fail', text: x.message }));
      refreshMessages();
      if (!S.scene) showErr('The design has invalid values', e.body.errors.map(x => x.param + ': ' + x.message).join('\n'), [['Reset all to imported/default', resetAll], ['Open a demo case', () => openCase('demo_complete')]]);
      else $('timing').textContent = 'not rebuilt: fix the highlighted values';
      return;
    }
    showErr('Could not build the model', String(e.message) + '\nThe previous model stays on screen if there was one.', [['Retry', () => build('full')], ['Open a demo case', () => openCase('demo_complete')]]);
    return;
  }
  if (seq !== S.seq) return;                       // a newer request exists: drop this stale result
  const tServer = performance.now() - t0;
  const t1 = performance.now();
  if (!j.section && S.scene && S.scene.section) j.section = S.scene.section;  // previews keep the previous cut
  S.scene = j;
  setClipPlanes();
  await rebuildSceneObjects();
  if (seq !== S.seq) return;                       // superseded while building 3D: do not publish
  const tGeom = performance.now() - t1;
  S.applied = seq; S.building = null;
  if (!S.firstFramed) { setCam('q34'); S.firstFramed = true; mark('first model framed'); perf.firstBuild = { server_ms: Math.round(tServer), geometry_ms: Math.round(tGeom) }; }
  perf.rebuilds.push({ quality, server_ms: Math.round(tServer), engine_s: j.timings_s.total, geometry_ms: Math.round(tGeom), reuse: perf.lastReuse });
  if (perf.rebuilds.length > 50) perf.rebuilds.shift();
  hideOverlay();
  setStatus(null);
  badge(quality === 'preview' ? 'preview' : 'ready', quality === 'preview' ? 'busy' : 'ok');
  const hits = j.timings_s.cache_hits ? ' · cache ' + Object.entries(j.timings_s.cache_hits).filter(e => e[1]).map(e => e[0]).join(',') : '';
  $('timing').textContent = `${quality} · engine ${j.timings_s.total.toFixed(2)} s · total ${(tServer / 1000).toFixed(2)} s · 3D ${tGeom.toFixed(0)} ms${hits}`;
  S.resolutions = null;
  if (quality === 'full') { S.dirty = false; S.lastFullHash = j.design_hash; }
  updateValidity(quality);
  emit('acl:built', { seq, quality, server_ms: Math.round(tServer), geometry_ms: Math.round(tGeom), engine_s: j.timings_s.total, design_hash: j.design_hash, valid: !!(j.validity ? j.validity.valid : true) });
  S.inactiveLive = null;
  trackChanges();
  if (quality === 'full') S.layerFixes = null;
  collectMessages(); refreshMessages(); renderChecks(); renderObjList(); renderSection(); renderLegend(); renderPerf(); renderSelection(); refreshDependents();
  renderCompTree(); renderCompInfo(); refreshLabelAnchors();
  renderLevelPanels(); if (S.levelSel) placeLevelLabel();
  if (quality === 'preview') scheduleBuild('full', 400);
  else {
    if (j.validity && !j.validity.valid && j.resolution_candidates) loadResolutions(seq);
    if (layerChecks().some(c => c.status === 'fail' && (c.fixes || []).length)) loadLayerFixes(seq);
    if (!$('detailWrap').hidden && S.detailLevel) openLevelDetail(S.detailLevel, true);
  }
}

// ------------------------------------------------------------------ fit validity
function updateValidity(quality) {
  const v = S.scene.validity || { valid: true, fail: 0, failed: [] };
  const vb = $('validBadge'); vb.hidden = false;
  vb.textContent = v.valid ? 'design valid' : `${v.fail} failed check${v.fail > 1 ? 's' : ''}`;
  vb.className = 'pill ' + (v.valid ? 'ok' : 'fail');
  if (quality !== 'full') return;
  if (v.valid) { S.lastValid = { params: JSON.parse(JSON.stringify(S.state.params)), provenance: JSON.parse(JSON.stringify(S.state.provenance)), hash: S.scene.design_hash }; }
  renderBanner();
}
function renderBanner() {
  const v = S.scene.validity || { valid: true };
  const bn = $('fitBanner');
  if (v.valid || VIEW_MODE) { bn.hidden = true; return; }
  bn.hidden = false;
  const items = (v.failed || []).slice(0, 3).map(f => `<div>✖ <b>${f.label}</b>: ${String(f.message || '').slice(0, 160)}</div>`).join('');
  const lay = layerChecks().filter(c => c.status === 'fail');
  bn.innerHTML = lay.length ? `<b>This design has conflicts between levels.</b> Requested values are kept; overflow is drawn in red.${items}`
    : `<b>This design does not fit.</b> Violations are drawn in red; nothing was shrunk or removed to hide them.${items}`;
  const row = document.createElement('div');
  if (lay.length && !VIEW_MODE) {
    const b0 = document.createElement('button'); b0.textContent = `Show in Level ${lay[0].level}`;
    b0.onclick = () => { showTab('design'); $('pnlLevels').open = true; selectLevel(lay[0].level); const c = $('lvcon' + lay[0].level); if (c) c.scrollIntoView({ block: 'center', behavior: 'smooth' }); };
    row.appendChild(b0);
  }
  const b1 = document.createElement('button'); b1.textContent = lay.length ? 'All checks' : 'Show fixes'; b1.onclick = () => showTab('checks');
  row.appendChild(b1);
  if (S.lastValid && S.lastValid.hash !== S.scene.design_hash) {
    const b2 = document.createElement('button'); b2.textContent = 'Restore last valid design'; b2.onclick = restoreLastValid; row.appendChild(b2);
  }
  bn.appendChild(row);
}
function restoreLastValid() {
  if (!S.lastValid) return;
  const changed = Object.keys(S.lastValid.params).filter(k => JSON.stringify(S.lastValid.params[k]) !== JSON.stringify(S.state.params[k]));
  S.state.history.push({ action: 'restore_last_valid', params: changed, t: new Date().toISOString() });
  gestureStart();
  S.state.params = JSON.parse(JSON.stringify(S.lastValid.params)); S.state.provenance = JSON.parse(JSON.stringify(S.lastValid.provenance));
  gestureCommit('restore last valid design');
  renderControls(); afterEdit(false);
}
async function loadResolutions(seq) {
  S.resolutions = 'loading'; renderFitPanel();
  try {
    const j = await api('/api/v2/resolutions', { state: S.state });
    if (seq !== S.seq) return;
    S.resolutions = j.resolutions || [];
  } catch (e) { S.resolutions = { error: e.message }; }
  renderFitPanel();
}
function applyResolution(r) {
  S.state.history.push({ action: 'apply_fix', param: r.param, from: S.state.params[r.param], to: r.value, label: r.label, t: new Date().toISOString() });
  gestureStart(); setParam(r.param, r.value, { noHistory: true, noUndo: true }); gestureCommit('apply fix: ' + r.label); renderControls();
}
function renderFitPanel() {
  const box = $('fitPanel'); if (!box || !S.scene) return;
  const v = S.scene.validity || { valid: true, failed: [] };
  const F = (S.scene.metrics || {}).fit;
  const layeredOn = !!(S.scene.metrics || {}).layered;
  let h = `<div class="fitbox ${v.valid ? 'ok' : 'bad'}"><b>${v.valid ? (layeredOn ? '✔ Design valid: every scaffold level fits its neighbours' : '✔ Design valid: the scaffold fits its repair space') : '✖ Design not valid'}</b>`;
  if (!v.valid) h += '<div class="small">' + (v.failed || []).map(f => `• ${f.label}`).join('<br>') + '</div>';
  if (F && F.strand_volume_mm3 != null) {
    const n = x => x == null ? '—' : Number(x).toFixed(3);
    const rows = [['Strand volume outside the ligament envelope', n(F.volume_outside_envelope_mm3) + ' mm³'],
      ['Strand volume outside its repair region', n(F.volume_outside_repair_space_mm3) + ' mm³'],
      ['Total strand (material) volume', n(F.strand_volume_mm3) + ' mm³'],
      ['Strands crossing into the other bundle', String(F.interface_crossings || 0)],
      ['AM/PL repair-unit contacts', String(F.unit_collisions || 0) + (F.unit_min_clearance_mm != null ? ` (min gap ${n(F.unit_min_clearance_mm)} mm)` : '')],
      ['Samples beyond the permitted overlap', String(F.samples_beyond_overlap || 0)],
      ['Min clearance to ligament boundary', n(F.min_envelope_clearance_mm) + ' mm'],
      ['Min clearance to bone', n(F.min_bone_clearance_mm) + ' mm'],
      ['Clearance shortfalls (warning)', String(F.clearance_shortfall_samples || 0)],
      ['Yarn end ↔ remnant gap', F.end_gap_mm ? `${n(F.end_gap_mm.min)} … ${n(F.end_gap_mm.max)} mm (${F.end_gap_mm.n} ends)` : '—'],
      ['Yarn end misalignment to ligament axis', F.end_misalignment_deg ? `mean ${F.end_misalignment_deg.mean.toFixed(1)}°, max ${F.end_misalignment_deg.max.toFixed(1)}°` : '—'],
      ['Geometric tolerance', F.tolerance_mm + ' mm']];
    (F.units || []).forEach(u => rows.push([`${u.label}: yarns / pieces / braid groups`, `${u.yarns} / ${u.pieces} / ${u.braid_components == null ? 'n/a (aligned)' : u.braid_components}`],
      [`${u.label}: mean occupancy`, (100 * u.occupancy_mean).toFixed(0) + '% of repair-space area']));
    h += '<div class="fitgrid">' + rows.map(([a, b]) => `<span>${a}</span><span class="mono">${b}</span>`).join('') + '</div>';
    h += '<canvas id="occCanvas" width="680" height="220" aria-label="Cross-sectional occupancy along each repair unit"></canvas><div class="note">Occupancy = strand cross-section ÷ repair-space area at each station (a porous scaffold is not expected to fill the space). Dashed: repair-space area (mm², right axis).</div>';
  }
  h += '</div>';
  const RT = (S.scene.metrics || {}).rotation;
  if (RT && RT.rows && RT.rows.length) {
    h += `<div class="fitbox"><b>Total fiber rotation within each unit</b> <span class="muted small">(surgeon-informed range −${RT.limit_deg}° … +${RT.limit_deg}°, measured on the generated paths in a non-twisting frame along each unit's axis)</span><div class="fitgrid">`;
    RT.rows.filter(r => r.n).forEach(r => {
      const ok = r.braid_winding ? r.net_max_abs_deg <= RT.limit_deg + RT.tolerance_deg : r.within_limit;
      const val = r.braid_winding ? `winds up to ${r.net_max_abs_deg.toFixed(1)}°` :
        `${r.net_at_max_deg >= 0 ? '+' : ''}${r.net_at_max_deg.toFixed(1)}° (set ${r.requested_deg >= 0 ? '+' : ''}${Number(r.requested_deg).toFixed(1)}°)`;
      h += `<span>${ok ? '✔' : r.braid_winding ? '⚠' : '✖'} ${r.level.split(':')[0]} — ${r.unit}${r.n > 1 ? ` (${r.n} units)` : ''}</span><span class="mono">${val}</span>`;
    });
    if (layeredOn) ['am', 'pl'].forEach(k => {
      const t = Number(S.state.params[`rf.${k}.yarn_twist_deg`]);
      h += `<span>${Math.abs(t) <= RT.limit_deg ? '✔' : '✖'} braid strand fibres — ${k.toUpperCase()}-like mantle (Level 4 yarn twist)</span><span class="mono">${t >= 0 ? '+' : ''}${t.toFixed(1)}° (set; spread evenly by construction)</span>`;
    });
    h += `</div><div class="note">${RT.note}</div></div>`;
  }
  if (!v.valid && !VIEW_MODE) {
    h += '<div class="fitbox"><b>Explicit resolutions</b> <span class="muted small">(each one was rebuilt and checked; nothing is applied automatically)</span><div id="resList"></div></div>';
  }
  box.innerHTML = h;
  drawOccupancy();
  const rl = $('resList'); if (!rl) return;
  if (S.resolutions === 'loading' || S.resolutions == null) rl.innerHTML = '<div class="small muted">Testing candidate fixes…</div>';
  else if (S.resolutions.error) rl.innerHTML = `<div class="small">Could not test fixes: ${S.resolutions.error}</div>`;
  else if (!S.resolutions.length) rl.innerHTML = '<div class="small">No single-parameter change makes this design fit. Change the anatomy/injury, the architecture, or several scaffold settings.</div>';
  else S.resolutions.forEach(r => {
    const d = document.createElement('div'); d.className = 'res';
    const b = document.createElement('button'); b.textContent = 'Apply'; b.onclick = () => applyResolution(r);
    const sp = document.createElement('span'); sp.textContent = `${r.label}  (${r.param} = ${r.value})`;
    d.append(b, sp); rl.appendChild(d);
  });
}
function drawOccupancy() {
  const cv = $('occCanvas'); const F = (S.scene.metrics || {}).fit; if (!cv || !F || !F.units) return;
  const g = cv.getContext('2d'); g.clearRect(0, 0, cv.width, cv.height);
  const L = 40, R = cv.width - 46, T = 12, B = cv.height - 26;
  let s0 = 1, s1 = 0, amax = 1, omax = 0.2;
  F.units.forEach(u => { u.occupancy_s.forEach(s => { s0 = Math.min(s0, s); s1 = Math.max(s1, s); }); u.repair_area_mm2.forEach(a => amax = Math.max(amax, a)); u.occupancy.forEach(o => omax = Math.max(omax, o)); });
  if (s1 <= s0) s1 = s0 + 0.01; omax = Math.min(1, Math.ceil(omax * 10) / 10);
  const X = s => L + (s - s0) / (s1 - s0) * (R - L), Yo = o => B - o / omax * (B - T), Ya = a => B - a / amax * (B - T);
  g.strokeStyle = '#ccc'; g.strokeRect(L, T, R - L, B - T); g.fillStyle = '#555'; g.font = '11px system-ui';
  g.fillText('occupancy', 2, T + 8); g.fillText((100 * omax).toFixed(0) + '%', 4, T + 22); g.fillText('0', 26, B);
  g.fillText(amax.toFixed(0) + ' mm²', R + 4, T + 10); g.fillText('s = ' + s0.toFixed(2), L, B + 16); g.fillText(s1.toFixed(2), R - 24, B + 16);
  const cols = ['#c0392b', '#2471a3', '#7d3c98'];
  F.units.forEach((u, k) => {
    g.strokeStyle = cols[k % 3]; g.lineWidth = 2; g.setLineDash([]); g.beginPath(); u.occupancy_s.forEach((s, i) => i ? g.lineTo(X(s), Yo(u.occupancy[i])) : g.moveTo(X(s), Yo(u.occupancy[i]))); g.stroke();
    g.lineWidth = 1; g.setLineDash([4, 3]); g.beginPath(); u.occupancy_s.forEach((s, i) => i ? g.lineTo(X(s), Ya(u.repair_area_mm2[i])) : g.moveTo(X(s), Ya(u.repair_area_mm2[i]))); g.stroke(); g.setLineDash([]);
    g.fillStyle = cols[k % 3]; g.fillText(u.label, L + 6 + 120 * k, T + 12);
  });
}

// ------------------------------------------------------------------ editing
function provOf(pid) { return (S.state.provenance || {})[pid] || {}; }
// ------------------------------------------------------------------ undo / redo
// One drag gesture (or one committed edit) is one undo step: the snapshot is
// taken when the gesture starts, the entry is pushed when it commits.
function snapParams() { return { params: JSON.parse(JSON.stringify(S.state.params)), provenance: JSON.parse(JSON.stringify(S.state.provenance)) }; }
function gestureStart() { if (!S.gesture) S.gesture = snapParams(); }
function gestureCommit(label) {
  if (!S.gesture) return;
  const before = S.gesture; S.gesture = null;
  const after = snapParams();
  if (JSON.stringify(before.params) === JSON.stringify(after.params)) return;
  S.undoStack.push({ before, after, label: label || 'edit' });
  if (S.undoStack.length > 60) S.undoStack.shift();
  S.redoStack.length = 0;
  refreshUndoButtons();
}
function applySnap(snap) {
  const changed = Object.keys(snap.params).filter(k => JSON.stringify(snap.params[k]) !== JSON.stringify(S.state.params[k]));
  S.state.params = JSON.parse(JSON.stringify(snap.params));
  S.state.provenance = JSON.parse(JSON.stringify(snap.provenance));
  emit('acl:param', { keys: changed, actor: 'undo' });
}
function doUndo() {
  const e = S.undoStack.pop();
  if (!e) return;
  S.redoStack.push(e);
  applySnap(e.before);
  S.state.history.push({ action: 'undo', label: e.label, t: new Date().toISOString() });
  renderControls(); afterEdit(false); refreshUndoButtons();
}
function doRedo() {
  const e = S.redoStack.pop();
  if (!e) return;
  S.undoStack.push(e);
  applySnap(e.after);
  S.state.history.push({ action: 'redo', label: e.label, t: new Date().toISOString() });
  renderControls(); afterEdit(false); refreshUndoButtons();
}
function refreshUndoButtons() {
  $('btnUndo').disabled = !S.undoStack.length;
  $('btnRedo').disabled = !S.redoStack.length;
  $('btnUndo').title = S.undoStack.length ? 'Undo: ' + S.undoStack[S.undoStack.length - 1].label + ' (Ctrl+Z)' : 'Undo the last parameter edit (Ctrl+Z)';
}
function setParam(pid, value, opts) {
  opts = opts || {};
  const old = S.state.params[pid];
  if (JSON.stringify(old) === JSON.stringify(value)) return;
  if (!opts.noUndo) gestureStart();
  const spec = S.regById[pid] || {};
  if (!S.lastEdit || S.lastEdit.applied) S.lastEdit = { pid, label: spec.label || pid, level: spec.owner_level || 0, before: layerSummary(), applied: false };
  const targets = [pid];
  if (!opts.noPair && spec.pair && [4, 6].includes(spec.owner_level) && S.levelRegion[spec.owner_level] === 'both') targets.push(spec.pair);
  for (const t of targets) {
    S.state.params[t] = JSON.parse(JSON.stringify(value));
    const pv = Object.assign({}, provOf(t));
    if (pv.status !== 'override') pv.previous_status = pv.status;
    if (opts.prov) Object.assign(pv, opts.prov); else { pv.status = 'override'; pv.source = 'manual edit'; }
    S.state.provenance[t] = pv;
  }
  if (!opts.noHistory) S.state.history.push({ action: 'edit', param: pid, from: old, to: value, actor: opts.actor || 'user', t: new Date().toISOString(), ...(opts.txid ? { txid: opts.txid } : {}) });
  S.dirty = true;
  emit('acl:param', { keys: targets, actor: opts.actor || 'user' });
  if (!opts.noBuild) afterEdit(opts.preview);
}
// Events for the workspace (Muse, linked views): parameter changes and build results.
function emit(type, detail) { try { window.dispatchEvent(new CustomEvent(type, { detail })); } catch (e) { /* ignore */ } }
function setDisplay(k, v, rebuild) { S.state.display[k] = v; persist(); if (rebuild) scheduleBuild('preview', 80); }
function afterEdit(preview) {
  updateControlState(); persist();
  scheduleBuild(preview ? 'preview' : 'full', preview ? 40 : 120);
}
function resetDestination(pid) {
  const imp = S.state.imported;
  return imp && pid in imp.params ? 'imported value' : 'template';
}
function resetParam(pid) {
  const imp = S.state.imported;
  const spec = S.regById[pid];
  const val = imp && pid in imp.params ? imp.params[pid] : spec.default;
  const pv = imp && pid in imp.provenance ? imp.provenance[pid] : { status: spec.sources.length ? 'literature' : 'default', source: spec.sources.join(',') || 'engine default' };
  gestureStart();
  S.state.history.push({ action: 'reset', param: pid, from: S.state.params[pid], to: val, t: new Date().toISOString() });
  S.state.params[pid] = JSON.parse(JSON.stringify(val)); S.state.provenance[pid] = JSON.parse(JSON.stringify(pv));
  emit('acl:param', { keys: [pid], actor: 'user' });
  gestureCommit('reset ' + spec.label);
  S.dirty = true;
  renderControls(); afterEdit(false);
}
function resetAll() { gestureStart(); Object.keys(S.regById).forEach(pid => { const imp = S.state.imported; S.state.params[pid] = JSON.parse(JSON.stringify(imp && pid in imp.params ? imp.params[pid] : S.regById[pid].default)); if (imp && pid in imp.provenance) S.state.provenance[pid] = JSON.parse(JSON.stringify(imp.provenance[pid])); });
  S.state.history.push({ action: 'reset_all', t: new Date().toISOString() }); emit('acl:param', { keys: Object.keys(S.regById), actor: 'user' }); gestureCommit('reset all'); S.dirty = true; renderControls(); afterEdit(false); }

// ------------------------------------------------------------------ persistence (tabs, refresh, links)
function persist() {
  try { localStorage.setItem('acl.v2.current', JSON.stringify(S.state)); } catch (e) { /* storage may be unavailable */ }
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(saveRemote, 500);
}
async function saveRemote() {
  try {
    const j = await api('/api/v2/state', { state: S.state });
    S.stateId = j.id;
    const url = new URL(location.href); url.searchParams.set('state', j.id);
    history.replaceState(null, '', url.pathname + url.search);
    if (bc && !VIEW_MODE) bc.postMessage({ type: 'state', id: j.id });
    return j.id;
  } catch (e) { $('timing').textContent = 'could not save to the server store: ' + e.message; return null; }
}
const bc = ('BroadcastChannel' in window) ? new BroadcastChannel('acl-v2') : null;
if (bc && VIEW_MODE) bc.onmessage = async (ev) => {
  if (ev.data && ev.data.type === 'state' && $('follow') && $('follow').checked && ev.data.id !== S.stateId) {
    try { const j = await api('/api/v2/state?id=' + ev.data.id); S.stateId = ev.data.id; S.state = j.state; const url = new URL(location.href); url.searchParams.set('state', ev.data.id); history.replaceState(null, '', url.pathname + url.search); afterLoad(); } catch (e) { /* keep current */ }
  }
};

// ------------------------------------------------------------------ visual guides
let guide = null;
function guideTargets(pid) {
  const spec = S.regById[pid] || {};
  const kind = (spec.help && spec.help.guide) || 'highlight';
  const scope = spec.scope || '';
  let ids;
  if (pid.startsWith('fp.tib')) ids = ['footprint_tibial'];
  else if (pid.startsWith('fp.fem')) ids = ['footprint_femoral'];
  else if (pid.startsWith('bone.')) ids = ['bone_femur', 'bone_tibia', 'cartilage_femoral', 'cartilage_tibial'];
  else if (scope === 'AM region') ids = ['bundle_am_reference', 'native_fascicles'];
  else if (scope === 'PL region') ids = ['bundle_pl_reference', 'native_fascicles'];
  else if (scope === 'injury') ids = ['injury_region', 'retained_tibial_stump', 'retained_femoral_stump', 'retained_tissue'];
  else if (scope === 'scaffold') ids = ['scaffold_yarns', 'scaffold_fibers'];
  else ids = ['intact_reference', 'centerline'];
  if (kind === 'strand') ids = ['scaffold_yarns'];
  if (kind === 'layers') ids = ['scaffold_yarns'];
  if (kind === 'span') ids = ['injury_region', 'retained_tibial_stump', 'retained_femoral_stump'];
  if (kind === 'region') ids = ['scaffold_yarns', 'repair_space', 'repair_space_am', 'repair_space_pl'];
  if (kind === 'arc' && scope === 'scaffold') ids = ['scaffold_yarns'];
  if (kind === 'arc' && scope !== 'scaffold') ids = pid.startsWith('bundle.') ? ['bundle_am_reference', 'bundle_pl_reference', 'bundle_paths'] : ['native_fascicles', 'native_fibers'];
  if ((kind === 'measure_width' || kind === 'measure_thickness')) ids = (pid.startsWith('am.') ? ['bundle_am_reference'] : pid.startsWith('pl.') ? ['bundle_pl_reference'] : ['intact_reference', 'centerline']);
  if (pid.startsWith('fp.')) ids = [pid.includes('.tib.') ? 'footprint_tibial' : 'footprint_femoral'];
  if (spec.owner_level && !spec.legacy) {
    ids = (spec.objects && spec.objects.length ? spec.objects : levelObjects(spec.owner_level));
    if (spec.region === 'tib' || spec.region === 'fem') ids = ids.filter(i => !/_(tib|fem)$/.test(i) || i.endsWith('_' + spec.region));
    return { kind: 'highlight', ids: ids.filter(id => S.objects.has(id)) };
  }
  return { kind, ids: ids.filter(id => S.objects.has(id)) };
}
function centerlinePts() {
  const cl = S.objects.get('centerline');
  return cl ? cl.userData.lines.P : null;
}
function midSectionDims() {
  const sp = S.scene && S.scene.metrics && S.scene.metrics.section_profile;
  if (!sp) return null;
  const k = Math.floor(sp.s.length / 2);
  return { w: sp.width_mm[k], t: sp.thickness_mm[k], s: sp.s[k] };
}
function clearGuide(force) {
  if (guide && guide.pinned && !force) return;
  if (!guide) return;
  if (guide.helper) { scene.remove(guide.helper); guide.helper.geometry.dispose(); guide.helper.material.dispose(); }
  if (guide.group) { root.remove(guide.group); disposeObj(guide.group); }
  $('guideLabel').hidden = true;
  guide = null;
  if (S.levelSel) placeLevelLabel();
}
function placeGuideLabel(world) {
  const el = $('guideLabel');
  const v = world.clone().project(cam);
  const w = ren.domElement.clientWidth, h = ren.domElement.clientHeight;
  el.style.left = ((v.x * 0.5 + 0.5) * w) + 'px';
  el.style.top = ((-v.y * 0.5 + 0.5) * h) + 'px';
  el.hidden = false;
}
function showGuide(pid, opts) {
  opts = opts || {};
  clearGuide(true);
  if (!S.scene || !S.objects.size) return;
  root.updateMatrixWorld(true);
  const spec = S.regById[pid] || {};
  const { kind, ids } = guideTargets(pid);
  const grp = new THREE.Group();
  root.add(grp);
  let anchorWorld = null, label = spec.label || pid;
  const lineMat = (c) => new THREE.LineBasicMaterial({ color: c, depthTest: false, transparent: true });
  // highlight box around the first target (no material changes, no camera move)
  let helper = null;
  const first = ids.map(id => S.objects.get(id)).find(o => o && o.visible !== false);
  if (first) {
    first.updateWorldMatrix(true, true);
    helper = new THREE.BoxHelper(first, 0x0f6f68);
    helper.material.depthTest = false; helper.renderOrder = 20;
    scene.add(helper);
    anchorWorld = new THREE.Box3().setFromObject(first).getCenter(new THREE.Vector3());
  }
  const P = centerlinePts();
  const mid = P ? Math.floor(P.length / 3 / 2) : 0;
  const C = P ? new THREE.Vector3(P[3 * mid], P[3 * mid + 1], P[3 * mid + 2]) : new THREE.Vector3();
  const dims = midSectionDims();
  const val = S.state.params[pid];
  const eng2world = (e) => e.clone().applyMatrix4(root.matrixWorld);
  if ((kind === 'measure_width' || kind === 'measure_thickness') && P && dims) {
    const w = kind === 'measure_width' ? dims.w : dims.t;
    const T = new THREE.Vector3(P[3 * (mid + 2)] - P[3 * (mid - 2)], P[3 * (mid + 2) + 1] - P[3 * (mid - 2) + 1], P[3 * (mid + 2) + 2] - P[3 * (mid - 2) + 2]).normalize();
    let ax = new THREE.Vector3(1, 0, 0);
    if (kind === 'measure_thickness') ax = new THREE.Vector3(0, 1, 0);
    ax.addScaledVector(T, -ax.dot(T)).normalize();
    const a = C.clone().addScaledVector(ax, -w / 2), b = C.clone().addScaledVector(ax, w / 2);
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    grp.add(new THREE.Line(g, lineMat(0x0f6f68)));
    [a, b].forEach(p => { const s = new THREE.Mesh(new THREE.SphereGeometry(0.25, 10, 8), new THREE.MeshBasicMaterial({ color: 0x0f6f68, depthTest: false })); s.position.copy(p); s.renderOrder = 20; grp.add(s); });
    label = `${spec.label}: ${Number(w).toFixed(1)} mm across the mid-section`;
    anchorWorld = eng2world(b);
  } else if (kind === 'arc' && P) {
    const deg = Number(val) || 0;
    const T = new THREE.Vector3(P[3 * (mid + 2)] - P[3 * (mid - 2)], P[3 * (mid + 2) + 1] - P[3 * (mid - 2) + 1], P[3 * (mid + 2) + 2] - P[3 * (mid - 2) + 2]).normalize();
    let N = new THREE.Vector3(0, 0, 1).addScaledVector(T, -T.z).normalize();
    if (N.length() < 0.5) N = new THREE.Vector3(1, 0, 0);
    const B = new THREE.Vector3().crossVectors(T, N).normalize();
    const r = dims ? Math.min(dims.w, dims.t) / 2 + 1.5 : 4;
    const a0 = 0, a1 = THREE.MathUtils.degToRad(Math.max(4, Math.abs(deg)));
    const pts = [];
    for (let i = 0; i <= 32; i++) { const a = a0 + (a1 - a0) * i / 32; pts.push(C.clone().addScaledVector(N, r * Math.cos(a)).addScaledVector(B, r * Math.sin(a) * Math.sign(deg || 1))); }
    grp.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), lineMat(0x0f6f68)));
    const tip = pts[pts.length - 1];
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.35, 1.1, 10), new THREE.MeshBasicMaterial({ color: 0x0f6f68, depthTest: false }));
    cone.position.copy(tip); cone.renderOrder = 20; grp.add(cone);
    label = `${spec.label}: ${deg >= 0 ? '+' : ''}${Number(deg).toFixed(1)}° total end-to-end`;
    anchorWorld = eng2world(tip);
  } else if (kind === 'span' && P) {
    const n = P.length / 3, L = (S.scene.metrics.ligament_length_mm) || 33;
    let cs = S.state.params['injury.center_s'], gap = S.state.params['injury.gap_mm'];
    if (pid.startsWith('injury.am.') || pid.startsWith('injury.pl.')) { const k = pid.split('.')[1]; cs = S.state.params[`injury.${k}.center_s`]; gap = S.state.params[`injury.${k}.gap_mm`]; }
    const f0 = Math.max(0, cs - (gap / 2) / L), f1 = Math.min(1, cs + (gap / 2) / L);
    const i0 = Math.round(f0 * (n - 1)), i1 = Math.round(f1 * (n - 1));
    const a = new THREE.Vector3(P[3 * i0], P[3 * i0 + 1], P[3 * i0 + 2]), b = new THREE.Vector3(P[3 * i1], P[3 * i1 + 1], P[3 * i1 + 2]);
    grp.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), lineMat(0xe03b2f)));
    [a, b].forEach(p => { const s = new THREE.Mesh(new THREE.SphereGeometry(0.4, 10, 8), new THREE.MeshBasicMaterial({ color: 0xe03b2f, depthTest: false })); s.position.copy(p); s.renderOrder = 20; grp.add(s); });
    label = `Missing-tissue interval ≈ ${Number(gap).toFixed(1)} mm around s = ${Number(cs).toFixed(2)}`;
    anchorWorld = eng2world(b);
  } else if (kind === 'footprint') {
    const side = pid.includes('.tib.') ? 'tib' : 'fem';
    const Lm = S.state.params[`fp.${side}.length_mm`], Wm = S.state.params[`fp.${side}.width_mm`];
    label = `${side === 'tib' ? 'Tibial' : 'Femoral'} footprint boundary, ${Number(Lm).toFixed(1)} × ${Number(Wm).toFixed(1)} mm`;
    const h = S.scene.handles && S.scene.handles[side === 'tib' ? 'fp_tib' : 'fp_fem'];
    if (h) anchorWorld = eng2world(new THREE.Vector3(h[0], h[1], h[2]));
  } else if (kind === 'layers') {
    const nl = S.state.params['scaffold.braid_layers'], sp = S.state.params['fit.layer_spacing_mm'];
    label = `${nl} braid layer${nl === 1 ? '' : 's'}${nl > 1 ? `, ${Number(sp).toFixed(2)} mm apart (adjacent layers)` : ''}`;
  } else if (kind === 'strand') {
    const d = S.state.params['scaffold.yarn_diameter_mm'];
    label = `Yarn Ø ${Number(d).toFixed(2)} mm (representative)`;
    if (opts.pinned) {
      const meta = S.objects.get('scaffold_yarns');
      const lm = meta && meta.userData.lines.meta;
      if (lm && lm.yarn && lm.yarn.length) openDetail('yarn', lm.yarn[0]);
      label += ' — magnified detail view (fibers drawn thicker than true size)';
    }
  } else if (kind === 'region') {
    label = 'Repair space: the permitted scaffold volume (dashed boundary in section)';
  }
  guide = { pid, group: grp, helper, anchorWorld, label, pinned: !!opts.pinned };
  const el = $('guideLabel');
  el.textContent = label;
  if (anchorWorld) placeGuideLabel(anchorWorld);
  else el.hidden = true;
  invalidate();
}
function focusGuide(pid) {
  showGuide(pid, { pinned: true });
  if (!guide || !guide.anchorWorld) return;
  const target = guide.anchorWorld.clone();
  const dir = cam.position.clone().sub(ctl.target).normalize();
  const dist = Math.max(18, cam.position.distanceTo(ctl.target) * 0.45);
  cam.position.copy(target).addScaledVector(dir, dist);
  cam.near = Math.max(0.05, dist / 200); cam.far = dist * 20; cam.updateProjectionMatrix();
  ctl.target.copy(target); ctl.update(); invalidate();
}
function highlightForParam(pid, subtle) {
  if (guide && guide.pinned) return;
  const now = performance.now();
  if (guide && guide.pid === pid && now - (guide.t || 0) < 250) return;  // throttle during drags
  showGuide(pid, { pinned: false });
  if (guide) guide.t = now;
}

// ------------------------------------------------------------------ controls UI
function chip(pid) {
  const pv = provOf(pid); const st = pv.status || 'default';
  const tip = [st.toUpperCase(), pv.source && ('source: ' + pv.source), pv.path && ('input: ' + pv.path), pv.confidence != null && ('confidence: ' + pv.confidence),
    pv.uncertainty && ('uncertainty: ' + JSON.stringify(pv.uncertainty)), pv.note, pv.previous_status && ('was: ' + pv.previous_status)].filter(Boolean).join('\n');
  return `<span class="chip ${st.replace(' ', '_')}" title="${tip.replace(/"/g, '&quot;')}">${st}</span>`;
}
const DEPENDENT = () => {
  const p = S.state.params, out = {};
  const sm = p['native.sizing_mode'];
  out[sm === 'count+packing' ? 'native.fascicle_diameter_mm' : sm === 'count+diameter' ? 'native.fascicle_packing' : 'native.fascicle_count'] = 'computed from the other two (sizing mode)';
  out[p['scaffold.braid_mode'] === 'angle' ? 'scaffold.braid_pitch_mm' : 'scaffold.braid_angle_deg'] = 'computed: tan(angle) = 2πR / pitch';
  const na = (cond, ids, why) => { if (cond) ids.forEach(i => (out[i] = why)); };
  na(p['injury.type'] === 'intact', Object.keys(S.regById).filter(k => k.startsWith('injury.') && k !== 'injury.type').concat(Object.keys(S.regById).filter(k => k.startsWith('scaffold.') || k.startsWith('regions.'))), 'not used: injury type is intact');
  na(p['injury.involvement'] !== 'sector' || p['injury.type'] !== 'partial', ['injury.sector_center_deg', 'injury.sector_width_deg', 'injury.depth_fraction'], 'used only for a partial tear with a sector');
  na(p['scaffold.arch'] !== 'braided', ['scaffold.braid_mode', 'scaffold.braid_angle_deg', 'scaffold.braid_pitch_mm', 'scaffold.braid_pattern', 'scaffold.braid_radius_fraction', 'scaffold.core_yarns'], 'braided architecture only');
  na(p['scaffold.arch'] !== 'aligned', ['scaffold.yarn_rotation_deg'], 'aligned architecture only (a braid winds by construction; see Checks)');
  na(p['bundle.rotation_distribution'] !== 'custom', ['bundle.rotation_profile'], 'used when distribution = custom');
  na(Number(p['scaffold.crimp_amplitude_um']) === 0, ['scaffold.crimp_wavelength_um'], 'no effect while the waviness amplitude is 0');
  na(Number(p['native.crimp_amplitude_um']) === 0, ['native.crimp_wavelength_um', 'native.crimp_profile'], 'no effect while the crimp amplitude is 0');
  na(p['scaffold.engagement'] === 'abut', ['scaffold.overlap_tib_mm', 'scaffold.overlap_fem_mm'], 'abut engagement has no overlap');
  ['tib', 'fem'].forEach(s => { na(p['fp.' + s + '.shape'] !== 'crescent', ['fp.' + s + '.curvature_mm'], 'crescent shape only'); na(p['fp.' + s + '.shape'] === 'polygon', ['fp.' + s + '.length_mm', 'fp.' + s + '.width_mm'], 'not used: the boundary polygon defines the shape'); });
  const sm2 = p['lig.section_mode'];
  if (sm2 === 'area_aspect') { out['lig.width_profile'] = 'derived from area, aspect and squareness'; out['lig.thickness_profile'] = 'derived from area, aspect and squareness'; }
  else na(true, ['lig.area_profile', 'lig.aspect_profile'], sm2 === 'width_thickness' ? 'derived: area is computed from the contour (width, thickness, squareness)' : 'derived from the section contours');
  na(sm2 === 'per_bundle' || sm2 === 'imported', ['lig.width_profile', 'lig.thickness_profile', 'lig.shape_profile'], sm2 === 'per_bundle' ? 'derived: the envelope is the union of the AM and PL lobes' : 'derived from the measured contours');
  na(sm2 !== 'per_bundle', ['am.width_profile', 'am.thickness_profile', 'am.shape', 'am.orientation_deg', 'am.shift_profile', 'pl.width_profile', 'pl.thickness_profile', 'pl.shape', 'pl.orientation_deg', 'pl.shift_profile', 'bundle.lobe_overlap_mm'], 'used when the cross-section is defined per bundle');
  na(sm2 === 'per_bundle', ['bundle.am_fraction_mode', 'bundle.am_fraction_profile'], 'derived: per-bundle lobes set the AM share');
  na(sm2 !== 'per_bundle' && p['bundle.am_fraction_mode'] !== 'profile', ['bundle.am_fraction_profile'], 'used when the AM contribution varies along the ligament');
  na(sm2 !== 'imported', ['lig.section_contours'], 'used in imported section mode');
  na(Number(p['lig.contour_irregularity_mm']) === 0, ['lig.contour_lobes', 'lig.contour_seed'], 'no effect while the irregularity is 0');
  if (p['lig.length_mode'] === 'target') out['lig.bow_anterior_mm'] = 'solved so the length meets the target (attachments stay locked to bone)';
  else out['lig.length_target_mm'] = 'current length (from the attachments and centreline)';
  na(p['bundle.structure_mode'] !== 'independent', ['am.fascicle_diameter_scale', 'am.fascicle_rotation_deg', 'am.fiber_rotation_deg', 'am.dispersion_deg', 'am.flare_tib_mm', 'am.flare_fem_mm',
    'pl.fascicle_diameter_scale', 'pl.fascicle_rotation_deg', 'pl.fiber_rotation_deg', 'pl.dispersion_deg', 'pl.flare_tib_mm', 'pl.flare_fem_mm'], 'used with independent per-bundle structure');
  na(p['bundle.structure_mode'] === 'independent', ['native.fascicle_rotation_deg', 'native.fiber_rotation_deg', 'native.dispersion_deg'], 'set per bundle (independent structure)');
  na(p['injury.involvement'] !== 'per_bundle' || p['injury.type'] !== 'partial', ['injury.am.center_s', 'injury.am.gap_mm', 'injury.am.depth_fraction', 'injury.pl.center_s', 'injury.pl.gap_mm', 'injury.pl.depth_fraction'], 'used for a partial tear with per-bundle tears');
  na(p['injury.involvement'] === 'per_bundle' && p['injury.type'] === 'partial', ['injury.center_s', 'injury.gap_mm'], 'set per bundle');
  na(p['fit.partition'] !== 'per_bundle', ['fit.am.overlap_tib_mm', 'fit.am.overlap_fem_mm', 'fit.pl.overlap_tib_mm', 'fit.pl.overlap_fem_mm'], 'used with per-bundle repair units');
  na(p['fit.partition'] === 'per_bundle', ['scaffold.overlap_tib_mm', 'scaffold.overlap_fem_mm'], 'set per repair unit');
  na(p['scaffold.engagement'] === 'abut', ['fit.am.overlap_tib_mm', 'fit.am.overlap_fem_mm', 'fit.pl.overlap_tib_mm', 'fit.pl.overlap_fem_mm'], 'abut engagement has no overlap');
  na(p['scaffold.arch'] !== 'braided', ['scaffold.braid_layers', 'fit.layer_spacing_mm', 'scaffold.core_mode', 'fit.end_transition_mm'], 'braided architecture only');
  na(p['scaffold.arch'] === 'braided' || p['scaffold.aligned_sizing'] !== 'packing', ['scaffold.packing_fraction'], 'aligned yarns sized by packing only');
  na(p['scaffold.arch'] !== 'aligned', ['scaffold.aligned_sizing'], 'aligned architecture only');
  if (p['scaffold.arch'] === 'aligned' && p['scaffold.aligned_sizing'] === 'packing') out['scaffold.yarn_count'] = 'computed from the packing fraction';
  na(p['scaffold.arch'] === 'braided' && p['scaffold.core_mode'] === 'max', ['scaffold.core_yarns'], 'core filling = as many as fit');
  na(Number(p['scaffold.braid_layers']) < 2, ['fit.layer_spacing_mm'], 'used with two or more braid layers');
  const srv = S.inactiveLive || (S.scene && S.scene.inactive) || {};
  for (const [k, why] of Object.entries(srv)) if (!(k in out) && !k.startsWith('native.') && !k.startsWith('lig.')) out[k] = why;
  return out;
};
function derivedValue(pid) {
  const m = S.scene && S.scene.metrics; if (!m) return null;
  const ld = m.layered && m.layered.derived; if (ld && pid in ld) return ld[pid];
  if (pid === 'native.fascicle_diameter_mm') return m.fascicles.diameter_mm;
  if (pid === 'native.fascicle_packing') return m.fascicles.packing;
  if (pid === 'native.fascicle_count') return m.fascicles.count;
  const b0 = m.scaffold && m.scaffold.braid && m.scaffold.braid[0] && m.scaffold.braid[0].layers[0];
  if (pid === 'scaffold.braid_pitch_mm' && b0) return b0.pitch_mm;
  if (pid === 'scaffold.braid_angle_deg' && b0) return b0.angle_deg;
  const sp = m.section_profile, mid = sp ? Math.floor(sp.s.length / 2) : 0;
  if (pid === 'lig.width_profile' && sp) return sp.width_mm[mid];
  if (pid === 'lig.thickness_profile' && sp) return sp.thickness_mm[mid];
  if (pid === 'lig.area_profile' && sp) return sp.area_mm2[mid];
  if (pid === 'lig.aspect_profile' && sp) return sp.aspect[mid];
  if (pid === 'bundle.am_fraction_profile' && sp) return sp.am_fraction[mid];
  if (pid === 'lig.length_target_mm' && m.length) return m.length.length_mm;
  if (pid === 'lig.bow_anterior_mm' && m.length) return m.length.bow_anterior_mm;
  if (pid === 'scaffold.yarn_count' && m.scaffold && m.scaffold.units && m.scaffold.units[0] && m.scaffold.units[0].sizing) return m.scaffold.units[0].sizing.count;
  return null;
}

function renderControls() {
  const q = ($('paramSearch').value || '').toLowerCase();
  const match = p => !q || (p.label + ' ' + p.id + ' ' + (p.component || '')).toLowerCase().includes(q);
  // ---- Anatomy and Pose panel: knee pose, tear definition, reference anatomy, native tissue
  const box = $('controlsAnat'); box.innerHTML = '';
  for (const g of S.reg.groups.filter(g => g.panel === 'anatomy')) {
    const ps = S.reg.params.filter(p => p.group === g.id && match(p));
    if (!ps.length) continue;
    const basic = q ? ps : ps.filter(p => p.level === 'basic');
    const adv = q ? [] : ps.filter(p => p.level !== 'basic');
    const showAdv = (S.showAdv || {})[g.id];
    const det = document.createElement('details'); det.className = 'grp'; det.open = !!q || (S.openGroups || {})[g.id];
    det.dataset.group = g.id; det.ontoggle = () => { (S.openGroups = S.openGroups || {})[g.id] = det.open; };
    det.innerHTML = `<summary>${g.label} <span class="muted small">(${ps.length}${adv.length && !showAdv ? ', ' + basic.length + ' shown' : ''})</span></summary>`;
    const sub = document.createElement('div'); sub.className = 'sub'; det.appendChild(sub);
    let lastSub = null;
    const addP = (p) => {
      if (p.subgroup !== lastSub) { const h = document.createElement('div'); h.className = 'subh'; h.textContent = p.subgroup || ''; if (p.subgroup) sub.appendChild(h); lastSub = p.subgroup; }
      sub.appendChild(controlFor(p));
    };
    basic.forEach(addP);
    if (adv.length) {
      if (showAdv) adv.forEach(addP);
      const t = document.createElement('button'); t.className = 'advToggle';
      t.textContent = showAdv ? 'Hide advanced settings' : `Show advanced settings (${adv.length})`;
      t.setAttribute('aria-expanded', showAdv ? 'true' : 'false');
      t.onclick = () => { (S.showAdv = S.showAdv || {})[g.id] = !showAdv; renderControls(); };
      sub.appendChild(t);
    }
    box.appendChild(det);
  }
  if (q) $('pnlAnat').open = !!box.children.length;
  // ---- Scaffold Levels panel: six expandable levels, each owning its controls
  renderLevels(q, match);
  updateControlState(); refreshMessages(); renderLevelPanels(); refreshL0Lock();
}

const REG_LABEL = { am: 'AM-like', pl: 'PL-like', tib: 'Tibial', fem: 'Femoral' };
function renderLevels(q, match) {
  const box = $('controls'); box.innerHTML = '';
  const layered = S.state.params['scaffold.arch'] === 'layered';
  for (const L of S.reg.levels) {
    const all = S.reg.params.filter(p => p.owner_level === L.n);
    const ps = all.filter(match);
    if (q && !ps.length) continue;
    const det = document.createElement('details'); det.className = 'lvl'; det.dataset.level = L.n; det.style.setProperty('--lvc', LV_COL[L.n]);
    det.open = q ? true : S.levelSel === L.n;
    det.innerHTML = `<summary><span class="lvn">L${L.n}</span><b>${L.name}</b><span class="lvt">${L.title}</span><span class="pill tiny lvst" id="lvst${L.n}"></span></summary>`;
    det.addEventListener('toggle', () => {
      if (det.open && S.levelSel !== L.n) selectLevel(L.n);
      else if (!det.open && S.levelSel === L.n) selectLevel(null);
    });
    const body = document.createElement('div'); body.className = 'lvbody'; det.appendChild(body);
    body.insertAdjacentHTML('beforeend', `<div class="lvmeta">Modifies: <b>${L.component}</b>${L.n === 6 ? ' · <span id="lvsrc"></span>' : ''}</div>`);
    if (!layered) body.insertAdjacentHTML('beforeend', `<div class="lvmeta">The construct is the gap-filling yarn scaffold (Level 5 → Scaffold construct); this level's layered-construct controls are inactive and its yarn controls are listed below.</div>`);
    // region selector: AM-like / PL-like / both (Level 4), tibial / femoral / both (Level 6)
    let reg = null;
    if (L.n === 4 || L.n === 6) {
      reg = S.levelRegion[L.n];
      const opts = L.n === 4 ? ['am', 'pl', 'both'] : ['tib', 'fem', 'both'];
      const rs = document.createElement('div'); rs.className = 'regsel'; rs.innerHTML = `<span class="lbl">${L.n === 4 ? 'Region' : 'End'}:</span>`;
      opts.forEach(o => {
        const b = document.createElement('button'); b.className = 'seg' + (reg === o ? ' on' : ''); b.textContent = o === 'both' ? (L.n === 4 ? 'Both (linked)' : 'Both ends (linked)') : REG_LABEL[o];
        b.title = o === 'both' ? 'Show one set of controls; an edit applies to both regions' : 'Edit only this region';
        b.onclick = () => { S.levelRegion[L.n] = o; renderControls(); }; rs.appendChild(b);
      });
      body.appendChild(rs);
    }
    body.insertAdjacentHTML('beforeend', `<div class="lvsum" id="lvsum${L.n}"></div><div id="lvchg${L.n}"></div><div class="lvcon" id="lvcon${L.n}"></div>`);
    const cur = document.createElement('div'); cur.className = 'sub'; body.appendChild(cur);
    const leg = document.createElement('details'); leg.className = 'legacy'; leg.open = !layered;
    leg.innerHTML = `<summary>Gap-filling yarn scaffold controls owned by this level (used when the construct is braided or aligned)</summary>`;
    const legBox = document.createElement('div'); leg.appendChild(legBox);
    let lastSub = null, lastLegSub = null;
    for (const p of ps) {
      if (reg && p.region) {
        if (reg !== 'both' && p.region !== reg) continue;
        if (reg === 'both' && p.pair && (p.region === 'pl' || p.region === 'fem')) continue;
      }
      const tgt = p.legacy ? legBox : cur;
      const last = p.legacy ? lastLegSub : lastSub;
      if (p.subgroup !== last) { const h = document.createElement('div'); h.className = 'subh'; h.textContent = p.subgroup || ''; if (p.subgroup) tgt.appendChild(h); if (p.legacy) lastLegSub = p.subgroup; else lastSub = p.subgroup; }
      const c = controlFor(p);
      if (reg === 'both' && p.pair) {
        const lab = c.querySelector('label'); lab.textContent += L.n === 4 ? ' (AM-like + PL-like)' : ' (both ends)';
        const pv = S.state.params[p.pair];
        if (JSON.stringify(pv) !== JSON.stringify(S.state.params[p.id])) {
          const d = document.createElement('div'); d.className = 'desc'; d.textContent = `${REG_LABEL[p.region === 'am' ? 'pl' : 'fem']} currently ${fmtV(pv)} — an edit here sets both.`; c.appendChild(d);
        }
      }
      tgt.appendChild(c);
    }
    if (legBox.children.length) body.appendChild(leg);
    body.insertAdjacentHTML('beforeend', `<div id="lvcalc${L.n}"></div>`);
    const act = document.createElement('div'); act.className = 'lvact';
    const mk = (t, title, fn) => { const b = document.createElement('button'); b.textContent = t; b.title = title; b.onclick = fn; act.appendChild(b); };
    mk('↺ Reset level', `Reset every Level ${L.n} control to its imported value or template`, () => resetLevel(L.n));
    mk('Show/hide', `Show or hide every Level ${L.n} component (display only)`, () => toggleLevel(L.n));
    mk('Isolate', `Show only Level ${L.n} (display only)`, () => isolateLevel(L.n));
    mk('Focus', `Move the camera to Level ${L.n}`, () => focusLevel(L.n));
    mk('Cross-section', 'Show a cross-section with this level emphasised', () => { selectLevel(L.n); openLvSection(true); });
    mk('Magnified detail', 'True-scale detail of this level', () => { selectLevel(L.n); openLevelDetail(L.n); });
    if (!VIEW_MODE) body.appendChild(act);
    box.appendChild(det);
  }
}

function basisOf(pid) {
  const pv = provOf(pid), spec = S.regById[pid] || {};
  if (pv.basis) return pv.basis;
  if (['measured', 'observed', 'inferred'].includes(pv.status)) return 'measured';
  return spec.basis || 'assumed';
}
function levelInfo(p) {
  const d = document.createElement('div'); d.className = 'lvinfo';
  const comp = document.createElement('span'); comp.innerHTML = `Modifies: <b>${p.component || '—'}</b>`; d.appendChild(comp);
  if ((p.neighbors || []).length) {
    const nb = document.createElement('span'); nb.textContent = 'May affect: '; d.appendChild(nb);
    p.neighbors.forEach(([n, why]) => {
      const c = document.createElement('span'); c.className = 'lvchip'; c.style.borderColor = LV_COL[n]; c.textContent = 'L' + n;
      c.title = `Level ${n}: ${why}`; c.onclick = (e) => { e.preventDefault(); selectLevel(n); const det = document.querySelector(`#controls details.lvl[data-level="${n}"]`); if (det) det.scrollIntoView({ block: 'start', behavior: 'smooth' }); };
      d.appendChild(c);
    });
  }
  const bsel = document.createElement('select'); bsel.title = 'Is this value measured, assumed or experimental? (recorded with the design; does not change the geometry)';
  ['measured', 'assumed', 'experimental'].forEach(o => { const op = document.createElement('option'); op.value = o; op.textContent = o; bsel.appendChild(op); });
  bsel.value = basisOf(p.id); bsel.setAttribute('aria-label', p.label + ' basis');
  bsel.onchange = () => {
    const pv = Object.assign({}, provOf(p.id)); pv.basis = bsel.value; S.state.provenance[p.id] = pv;
    S.state.history.push({ action: 'basis', param: p.id, to: bsel.value, t: new Date().toISOString() }); persist(); updateControlState();
  };
  const bl = document.createElement('span'); bl.textContent = 'Value is'; const bt = document.createElement('span'); bt.className = 'basis ' + basisOf(p.id); bt.textContent = basisOf(p.id);
  d.append(bl, bsel);
  return d;
}

function copyDerivedOnModeSwitch(pid, val) {
  // switching a calculated quantity to user-set keeps the geometry: the calculated value is copied in
  const L = S.scene && S.scene.metrics && S.scene.metrics.layered; if (!L) return;
  const D = L.derived || {}, DP = L.derived_profiles || {}, P = S.state.params;
  const put = (k, v) => {
    if (v == null || !(k in P)) return;
    P[k] = typeof v === 'number' ? +Number(v).toFixed(3) : v;
    const pv = Object.assign({}, provOf(k)); if (pv.status !== 'override') pv.previous_status = pv.status;
    pv.status = 'override'; pv.source = 'copied from the calculated value when switching to user-set'; S.state.provenance[k] = pv;
  };
  if (pid === 'sheath.size_mode' && val === 'profiles') { put('sheath.width_profile', DP['sheath.width_profile']); put('sheath.thickness_profile', DP['sheath.thickness_profile']); }
  if (pid === 'rf.size_mode' && val === 'manual') ['am', 'pl'].forEach(b => ['envelope_width_mm', 'envelope_thickness_mm'].forEach(f => put(`rf.${b}.${f}`, D[`rf.${b}.${f}`])));
  if (pid === 'ch.count_mode' && val === 'count') { put('ch.am_count', D['ch.am_count']); put('ch.pl_count', D['ch.pl_count']); }
  const m = pid.match(/^end\.(tib|fem)\.(size_mode|flare_mode)$/);
  if (m && val === 'manual') {
    const k = `end.${m[1]}.`, sides = S.levelRegion[6] === 'both' ? ['tib', 'fem'] : [m[1]];
    sides.forEach(sd => { const kk = `end.${sd}.`; (m[2] === 'size_mode' ? ['length_mm', 'width_mm'] : ['flare_width_mm', 'flare_thickness_mm']).forEach(f => put(kk + f, D[kk + f])); });
    void k;
  }
}
async function refreshInactive() {
  try { const j = await api('/api/v2/inactive', { state: { params: S.state.params } }); S.inactiveLive = j.inactive; updateControlState(); } catch (e) { /* keep the last map */ }
}
function resetLevel(n) {
  if (!confirm(`Reset every Level ${n} control to its imported value or template?`)) return;
  const imp = S.state.imported;
  gestureStart();
  S.reg.params.filter(p => p.owner_level === n).forEach(p => {
    S.state.params[p.id] = JSON.parse(JSON.stringify(imp && p.id in imp.params ? imp.params[p.id] : p.default));
    S.state.provenance[p.id] = JSON.parse(JSON.stringify(imp && p.id in imp.provenance ? imp.provenance[p.id] : { status: p.sources.length ? 'literature' : 'default', source: p.sources.join(',') || 'engine default' }));
  });
  S.state.history.push({ action: 'reset_level', level: n, t: new Date().toISOString() });
  S.lastEdit = { pid: null, label: `reset Level ${n}`, level: n, before: layerSummary(), applied: false };
  gestureCommit(`reset Level ${n}`); S.dirty = true; renderControls(); afterEdit(false);
}

// ---- level actions: show/hide, isolate, focus (all display-only)
function toggleLevel(n) {
  const ids = levelObjects(n);
  const on = !ids.some(id => S.vis[id]);
  ids.forEach(id => { S.vis[id] = on; });
  refreshLevelView(); renderObjList(); renderCompTree(); refreshLabelAnchors(); saveView();
}
function isolateLevel(n) {
  setLevelMode('highlight');
  if (S.levelSel) selectLevel(null);
  const keep = new Set(levelObjects(n));
  for (const meta of S.scene.objects) {
    if (!isLayer(meta)) continue;
    S.vis[meta.id] = keep.has(meta.id);
  }
  refreshLevelView(); renderObjList(); renderCompTree(); refreshLabelAnchors(); saveView();
}
function focusLevel(n) {
  const ids = levelObjects(n);
  ids.forEach(id => { S.vis[id] = true; });
  refreshLevelView(); renderCompTree();
  root.updateMatrixWorld(true);
  const box = new THREE.Box3();
  ids.forEach(id => { const o = S.objects.get(id); if (o) box.expandByObject(o); });
  frameBox(box);
}

// ---- level selection, status, consequences, calculated outputs, change tracking
function selectLevel(n) {
  S.levelSel = n || null;
  document.querySelectorAll('#controls details.lvl').forEach(d => { const want = Number(d.dataset.level) === S.levelSel; if (d.open !== want) d.open = want; });
  const L = S.reg.levels.find(x => x.n === S.levelSel);
  $('lvlHint').textContent = L ? `Level ${L.n} (${L.name}) selected: ${{ highlight: 'highlighted in the model', isolate: 'shown alone', neighbors: 'opaque, with the neighbouring levels transparent' }[S.levelMode]}. Close it to clear the selection.`
    : 'Open a level to select it: it is highlighted in the 3D model and its adjustments appear below.';
  refreshLevelView(); placeLevelLabel();
  if (!$('lvSecWrap').hidden) renderSection();
  if (!$('detailWrap').hidden && S.detailLevel && S.levelSel && S.detailLevel !== S.levelSel) openLevelDetail(S.levelSel);
}
function placeLevelLabel() {
  if (!S.levelSel || !S.scene) { if (!guide) $('guideLabel').hidden = true; return; }
  const ids = levelObjects(S.levelSel).filter(id => S.objects.has(id));
  root.updateMatrixWorld(true);
  const box = new THREE.Box3(); ids.forEach(id => { const o = S.objects.get(id); if (o && o.visible) box.expandByObject(o); });
  const L = S.reg.levels.find(x => x.n === S.levelSel);
  const el = $('guideLabel');
  if (box.isEmpty()) { el.hidden = true; return; }
  const sm = ((S.scene.metrics.layered || {}).summary || {})[String(S.levelSel)] || {};
  el.textContent = `Level ${L.n} · ${L.name}` + (sm.channels ? ` — ${sm.channels} hollow channels` : '');
  guideAnchor = box.getCenter(new THREE.Vector3()); placeGuideLabel(guideAnchor);
}
let guideAnchor = null;
function layerSummary() { const m = S.scene && S.scene.metrics && S.scene.metrics.layered; return m ? JSON.parse(JSON.stringify(m.summary || {})) : null; }
const fmtKey = k => k.replace(/_(mm|um|nm|deg)$/, ' ($1)').replace(/_/g, ' ').replace('(um)', '(µm)');
const fmtNum = v => v == null ? '—' : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)) : String(v);
function layerChecks() { return ((S.scene && S.scene.checks) || []).filter(c => c.category === 'layer'); }
function renderLevelPanels() {
  if (!S.scene || !S.reg) return;
  const LM = S.scene.metrics && S.scene.metrics.layered;
  const layered = S.state.params['scaffold.arch'] === 'layered';
  const checks = layerChecks();
  for (const L of S.reg.levels) {
    const n = L.n, st = $('lvst' + n);
    const mine = checks.filter(c => c.levels && c.levels.includes(n) && c.status !== 'pass');
    const nf = mine.filter(c => c.status === 'fail').length, nw = mine.filter(c => c.status === 'warn').length;
    if (st) { st.textContent = !layered ? 'yarn scaffold' : !LM ? 'inactive' : nf ? `${nf} conflict${nf > 1 ? 's' : ''}` : nw ? `${nw} warning${nw > 1 ? 's' : ''}` : 'ok';
      st.className = 'pill tiny lvst ' + (!LM ? '' : nf ? 'fail' : nw ? 'warn' : 'ok'); }
    const sum = $('lvsum' + n);
    if (sum) {
      const sm = LM && LM.summary && LM.summary[String(n)];
      sum.textContent = sm ? Object.entries(sm).map(([k, v]) => `${fmtKey(k)}: ${fmtNum(v)}`).join(' · ') : (layered ? 'No layered construct (intact ligament or no repair needed).' : '');
      sum.hidden = !sum.textContent;
    }
    const src = n === 6 && $('lvsrc');
    if (src && LM && LM.footprint_source) src.innerHTML = ['tib', 'fem'].map(sd => { const f = LM.footprint_source[sd]; return `${REG_LABEL[sd]} footprint: <b title="${f.note}">${f.source}</b>`; }).join(' · ');
    renderConsequences(n, mine);
    renderCalc(n, LM);
    renderChange(n, LM);
  }
}
function renderConsequences(n, list) {
  const box = $('lvcon' + n); if (!box) return;
  box.innerHTML = '';
  if (!S.scene.metrics.layered) return;
  if (!list.length) { box.innerHTML = '<div class="cf ok">✔ No conflicts involving this level.</div>'; return; }
  list.forEach(c => {
    const d = document.createElement('div'); d.className = 'cf ' + c.status;
    const icon = c.status === 'fail' ? '✖' : c.status === 'warn' ? '⚠' : 'ℹ';
    const others = (c.levels || []).filter(x => x !== n);
    d.innerHTML = `<b>${icon} ${c.label.replace(/^L\d [^:]*: /, '')}</b>${c.message}` +
      (c.level !== n ? ` <i>(owned by Level ${c.level}; shown here because it involves this level)</i>` : others.length ? ` <i>(also involves Level ${others.join(', ')})</i>` : '');
    if ((c.fixes || []).length && !VIEW_MODE) {
      const fx = document.createElement('div'); fx.className = 'fx';
      c.fixes.forEach(f => {
        const b = document.createElement('button'); b.textContent = (f.level !== n ? `L${f.level}: ` : '') + f.label;
        b.title = 'Sets only: ' + Object.entries(f.set).map(([k, v]) => `${k} = ${fmtV(v)}`).join(', ');
        b.onclick = () => applyLayerFix(f);
        const v = (Array.isArray(S.layerFixes) ? S.layerFixes : []).find(r => r.conflict === c.id && r.label === f.label);
        const sp = document.createElement('span'); sp.className = 'vf';
        if (v) sp.textContent = v.verified ? '✔ tested: resolves it' + (v.new_conflicts && v.new_conflicts.length ? ' (new: ' + v.new_conflicts.join(', ') + ')' : '') : v.verified === false ? '✖ tested: ' + v.result : '';
        else if (c.status === 'fail' && S.layerFixes === 'loading') sp.textContent = 'testing…';
        const w = document.createElement('span'); w.append(b, sp); fx.appendChild(w);
      });
      d.appendChild(fx);
    }
    box.appendChild(d);
  });
}
function renderCalc(n, LM) {
  const box = $('lvcalc' + n); if (!box) return;
  if (!LM) { box.innerHTML = ''; return; }
  const rows = (LM.calculated || {})[String(n)] || [], asm = (LM.assumptions || {})[String(n)] || [];
  const tr = r => `<tr><td>${r.label}</td><td class="v">${fmtNum(r.value)} ${r.units || ''}</td><td class="b">${r.basis}</td></tr>`;
  box.innerHTML = (rows.length ? `<div class="subh">Calculated outputs (from the geometry; not validated performance)</div><table class="calc">${rows.map(tr).join('')}</table>` : '') +
    (asm.length ? `<div class="subh">Entered assumptions (inputs, not results)</div><table class="calc">${asm.map(tr).join('')}</table>` : '');
}
function renderChange(n, LM) {
  const box = $('lvchg' + n); if (!box) return;
  const e = S.lastShown;
  if (!e || !LM) { box.innerHTML = ''; return; }
  const d = e.diffs[String(n)];
  if (!d || !d.length) { box.innerHTML = ''; return; }
  const own = e.level === n;
  box.innerHTML = `<div class="lvchg">${own ? `<b>Your last edit</b> (${e.label}) changed:` : `<span class="nb">↳ Neighbour consequence</span> of editing Level ${e.level} (${e.label}):`} ${d.join(' · ')}</div>`;
}
function trackChanges() {
  // compare the layered summary before the last edit with the current build (all levels)
  const e = S.lastEdit; if (!e || e.applied || !S.scene || !S.scene.metrics.layered || !e.before) return;
  const now = S.scene.metrics.layered.summary || {}, diffs = {};
  for (const k of Object.keys(now)) {
    const a = e.before[k] || {}, b = now[k] || {};
    const ch = Object.keys(b).filter(x => JSON.stringify(fmtNum(a[x])) !== JSON.stringify(fmtNum(b[x]))).map(x => `${fmtKey(x)} ${fmtNum(a[x])} → ${fmtNum(b[x])}`);
    if (ch.length) diffs[k] = ch;
  }
  e.applied = true; S.lastShown = { label: e.label, level: e.level, diffs };
}
function applyLayerFix(f) {
  S.state.history.push({ action: 'apply_fix', set: f.set, label: f.label, t: new Date().toISOString() });
  gestureStart();
  for (const [k, v] of Object.entries(f.set)) setParam(k, v, { noUndo: true, noHistory: true, noPair: true });
  S.lastEdit = Object.assign(S.lastEdit || {}, { label: 'fix: ' + f.label });
  gestureCommit('apply fix: ' + f.label); renderControls();
}
async function loadLayerFixes(seq) {
  S.layerFixes = 'loading'; renderLevelPanels();
  try { const j = await api('/api/v2/layer_fixes', { state: S.state }); if (seq !== S.seq) return; S.layerFixes = j.fixes || []; }
  catch (e) { S.layerFixes = []; }
  renderLevelPanels();
}

function gotoRegion(which) {
  S.region = which;
  ['All', 'AM', 'PL'].forEach(r => $('rg' + r).classList.toggle('on', r.toLowerCase() === which));
  const note = $('regionNote');
  if (which === 'all') { note.textContent = ''; return; }
  const gid = which === 'am' ? 'am' : 'pl';
  note.textContent = which === 'am' ? 'Editing the AM (anteromedial) region' : 'Editing the PL (posterolateral) region';
  $('pnlAnat').open = true;
  const det = document.querySelector(`#tab-design details[data-group="${gid}"]`);
  if (det) {
    det.open = true; (S.openGroups = S.openGroups || {})[gid] = true;
    det.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }
  if (S.state.params['bundle.structure_mode'] !== 'independent')
    note.textContent += ' — per-bundle structure is shared, so these controls are inactive until structure is independent';
}

function refreshL0Lock() {
  const b = $('btnL0Lock'); if (!b) return;
  b.textContent = S.level0Locked ? 'Patient inputs locked' : 'Patient inputs unlocked';
  b.classList.toggle('on', !S.level0Locked);
  const c = S.state && S.state.case;
  $('demoNote').textContent = c && c.synthetic
    ? 'Demonstration dataset: every value is synthetic unless its chip says measured/observed.'
    : 'Patient-derived values are marked measured/observed; template values stay editable.';
}
function sourceClass(pid) {
  const st = provOf(pid).status || 'default';
  if (st === 'measured' || st === 'observed') return 'MRI (measured)';
  if (st === 'literature') return 'literature';
  if (st === 'inferred' || st === 'estimated') return 'derived';
  if (st === 'override' || st === 'proposed') return 'design (edited)';
  return 'design (template default)';
}
function valueSource(pid) {
  const st = provOf(pid).status || 'default';
  if (st === 'override') return 'manual edit';
  if (st === 'measured' || st === 'observed') return 'imported measurement';
  if (st === 'inferred' || st === 'estimated') return 'inferred input';
  if (st === 'missing' || st === 'awaiting_review') return 'template default (value missing — needs review)';
  return 'template default';
}
function controlFor(p) {
  const div = document.createElement('div'); div.className = 'ctl'; div.dataset.pid = p.id;
  const v = S.state.params[p.id];
  const unit = p.units ? `<span class="unit">${p.units}</span>` : '';
  const dest = resetDestination(p.id);
  div.innerHTML = `<div class="top"><span class="ovr" hidden title="Manually overridden"></span><label>${p.label}</label>${unit}<span class="chipbox">${chip(p.id)}</span><button class="rst" title="Reset to ${dest}">↺ ${dest === 'imported value' ? 'imported' : 'template'}</button></div>`;
  div.querySelector('.rst').onclick = () => resetParam(p.id);
  div.querySelector('.top').insertBefore(Object.assign(document.createElement('span'), { className: 'vtag' }), div.querySelector('.chipbox'));
  if (p.owner_level) div.appendChild(levelInfo(p));
  const body = document.createElement('div'); div.appendChild(body);
  if (p.kind === 'scalar' || p.kind === 'int') {
    const row = document.createElement('div'); row.className = 'top';
    const rng = document.createElement('input'); rng.type = 'range'; rng.min = p.min; rng.max = p.max; rng.step = p.step || (p.kind === 'int' ? 1 : 0.01); rng.value = v;
    const num = document.createElement('input'); num.type = 'number'; num.min = p.min; num.max = p.max; num.step = p.step || 'any'; num.value = v;
    rng.setAttribute('aria-label', p.label); num.setAttribute('aria-label', p.label + ' value');
    const conv = x => p.kind === 'int' ? Math.round(Number(x)) : Number(x);
    rng.oninput = () => { num.value = rng.value; setParam(p.id, conv(rng.value), { preview: true, noHistory: true }); highlightForParam(p.id, false); };
    rng.onchange = () => { S.state.history.push({ action: 'edit', param: p.id, to: conv(rng.value), actor: 'user', t: new Date().toISOString() }); gestureCommit(p.label + ' = ' + rng.value + (p.units ? ' ' + p.units : '')); clearGuide(); scheduleBuild('full', 50); };
    rng.onfocus = () => highlightForParam(p.id, true);
    rng.onblur = () => clearGuide();
    num.onfocus = () => highlightForParam(p.id, true);
    num.onblur = () => clearGuide();
    num.onchange = () => {
      const x = Number(num.value);
      if (!isFinite(x) || (p.min != null && x < p.min) || (p.max != null && x > p.max)) {
        S.msgs[p.id] = [{ level: 'fail', text: `${num.value} is outside ${p.min}–${p.max} ${p.units}; not applied` }]; refreshMessages();
        num.value = S.state.params[p.id];
        return;
      }
      rng.value = x; setParam(p.id, conv(x)); gestureCommit(p.label + ' = ' + x + (p.units ? ' ' + p.units : ''));
    };
    row.appendChild(rng); row.appendChild(num); body.appendChild(row);
  } else if (p.kind === 'enum') {
    const sel = document.createElement('select'); p.options.forEach(o => { const op = document.createElement('option'); op.value = o; op.textContent = o; sel.appendChild(op); });
    sel.value = v; sel.setAttribute('aria-label', p.label);
    sel.onchange = () => {
      // switching which variable is independent keeps the geometry: copy the derived value in
      gestureStart();
      if (p.id === 'native.sizing_mode' && S.scene) { const m = S.scene.metrics.fascicles; S.state.params['native.fascicle_diameter_mm'] = +m.diameter_mm.toFixed(3); S.state.params['native.fascicle_packing'] = +Math.min(0.9, m.packing).toFixed(3); S.state.params['native.fascicle_count'] = m.count; }
      copyDerivedOnModeSwitch(p.id, sel.value);
      if (p.id === 'scaffold.braid_mode' && S.scene && S.scene.metrics.scaffold && S.scene.metrics.scaffold.braid) { const b = S.scene.metrics.scaffold.braid; S.state.params['scaffold.braid_pitch_mm'] = +Math.min(80, Math.max(2, b.pitch_mm)).toFixed(2); S.state.params['scaffold.braid_angle_deg'] = +b.angle_deg.toFixed(2); }
      setParam(p.id, sel.value, { noUndo: true }); gestureCommit(p.label + ' = ' + sel.value); renderControls(); refreshInactive();
    };
    sel.onfocus = () => highlightForParam(p.id, true);
    sel.onblur = () => clearGuide();
    body.appendChild(sel);
  } else if (p.kind === 'profile') body.appendChild(curveEditor(p));
  else if (p.kind === 'point3' || p.kind === 'offsets') {
    const pts = p.kind === 'point3' ? [v] : v;
    pts.forEach((pt, k) => {
      const row = document.createElement('div'); row.className = 'top';
      ['x', 'y', 'z'].forEach((ax, a) => {
        const n = document.createElement('input'); n.type = 'number'; n.step = 0.1; n.value = pt[a]; n.title = ax + ' (mm)'; n.style.width = '74px';
        n.onchange = () => { const cur = JSON.parse(JSON.stringify(S.state.params[p.id])); const tgt = p.kind === 'point3' ? cur : cur[k]; tgt[a] = Number(n.value); setParam(p.id, cur); };
        row.append(ax, n);
      });
      body.appendChild(row);
    });
    if (!VIEW_MODE) {
      const b = document.createElement('button'); const mode = p.id === 'fp.tib.center' ? 'fp_tib' : p.id === 'fp.fem.center' ? 'fp_fem' : 'ctrl';
      b.textContent = S.handleMode === mode ? 'Done dragging' : 'Drag in 3D view';
      b.onclick = () => { S.handleMode = S.handleMode === mode ? null : mode; placeHandles(); renderControls(); invalidate(); };
      body.appendChild(b);
      const d = document.createElement('div'); d.className = 'desc'; d.textContent = 'Frame: right-knee model frame (mm); femoral values in the femur frame at extension. Points are snapped onto bone.'; body.appendChild(d);
    }
  } else if (p.kind === 'polygon2') body.appendChild(polygonEditor(p));
  else if (p.kind === 'points3') {
    const n = (v || []).length; const d = document.createElement('div'); d.className = 'desc';
    d.textContent = n ? `${n} imported centreline points (model frame). Edits: control handles and bow apply on top.` : 'None: the centreline is a spline through the attachments.';
    body.appendChild(d);
    if (n) { const b = document.createElement('button'); b.textContent = 'Use spline instead'; b.onclick = () => setParam(p.id, []); body.appendChild(b); }
  } else if (p.kind === 'composition') body.appendChild(compositionEditor(p));
  const msg = document.createElement('div'); msg.className = 'msgbox'; div.appendChild(msg);
  if (p.description) { const d = document.createElement('div'); d.className = 'desc'; d.textContent = p.description + (p.sources.length ? ' [' + p.sources.join(', ') + ']' : ''); div.appendChild(d); }
  div.appendChild(helpBlock(p));
  return div;
}
function baSVG(guide) {
  // small before/after schematic (secondary to the live model, always labeled)
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'ba'); svg.setAttribute('viewBox', '0 0 300 44');
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Schematic: smaller value on the left, larger on the right');
  const T = (x, y, s, c) => { const e = document.createElementNS(NS, 'text'); e.textContent = s; e.setAttribute('x', x); e.setAttribute('y', y); e.setAttribute('font-size', '10'); e.setAttribute('fill', c || '#5c6a69'); svg.appendChild(e); };
  const R = (x, y, w, h, fill, stroke) => { const e = document.createElementNS(NS, 'rect'); e.setAttribute('x', x); e.setAttribute('y', y); e.setAttribute('width', w); e.setAttribute('height', h); e.setAttribute('fill', fill); e.setAttribute('stroke', stroke); svg.appendChild(e); };
  const C = (x, y, r, fill, stroke) => { const e = document.createElementNS(NS, 'circle'); e.setAttribute('cx', x); e.setAttribute('cy', y); e.setAttribute('r', r); e.setAttribute('fill', fill); e.setAttribute('stroke', stroke); svg.appendChild(e); };
  T(8, 14, '− smaller'); T(232, 14, '+ larger'); T(8, 40, 'schematic — see the live model');
  if (guide === 'arc') {
    const A = (x, a0, a1) => { const e = document.createElementNS(NS, 'path'); const r = 12; const p = (a) => (x + r * Math.cos(a)) + ',' + (24 + r * Math.sin(a)); e.setAttribute('d', `M${p(a0)} A${r},${r} 0 0 1 ${p(a1)}`); e.setAttribute('fill', 'none'); e.setAttribute('stroke', '#0f6f68'); e.setAttribute('stroke-width', '2'); svg.appendChild(e); const d = document.createElementNS(NS, 'circle'); d.setAttribute('cx', x); d.setAttribute('cy', 24); d.setAttribute('r', 1.6); d.setAttribute('fill', '#0f6f68'); svg.appendChild(d); };
    A(60, -0.35, 0.35); A(215, -1.1, 1.1);
  } else if (guide === 'span') {
    R(30, 20, 40, 8, '#e03b2f33', '#e03b2f'); R(180, 20, 90, 8, '#e03b2f33', '#e03b2f');
  } else if (guide === 'footprint') {
    const E = (x, w) => { const e = document.createElementNS(NS, 'ellipse'); e.setAttribute('cx', x); e.setAttribute('cy', 24); e.setAttribute('rx', w); e.setAttribute('ry', 8); e.setAttribute('fill', '#f0a05033'); e.setAttribute('stroke', '#c06a2a'); svg.appendChild(e); };
    E(55, 18); E(215, 34);
  } else if (guide === 'layers') {
    R(35, 18, 50, 3, '#e8c14a', '#888'); R(35, 26, 50, 3, '#3fb39b', '#888');
    R(190, 14, 50, 3, '#e8c14a', '#888'); R(190, 30, 50, 3, '#3fb39b', '#888');
  } else if (guide === 'strand') {
    C(55, 24, 6, '#e8c14a55', '#888'); C(215, 24, 11, '#e8c14a55', '#888'); T(232, 40, 'magnified');
  } else if (guide === 'region') {
    R(35, 16, 40, 16, '#6fcfff33', '#1f8fcf'); R(190, 12, 60, 24, '#6fcfff33', '#1f8fcf');
  } else {
    R(35, 16, 40, 16, '#d9776d33', '#b55045'); R(190, 12, 60, 24, '#d9776d33', '#b55045');
  }
  return svg;
}
function helpBlock(p) {
  const det = document.createElement('details'); det.className = 'help';
  const sum = document.createElement('summary'); sum.textContent = 'What does this change?'; det.appendChild(sum);
  const dl = document.createElement('dl'); det.appendChild(dl);
  const h = p.help || {};
  const row = (t, v) => { const dt = document.createElement('dt'); dt.textContent = t; const dd = document.createElement('dd'); dd.textContent = v; dl.append(dt, dd); return dd; };
  row('What it controls', h.what || p.description || p.label);
  row('Increase', h.increase || '—');
  row('Decrease', h.decrease || '—');
  row('Where to look', h.look || '—');
  row('Units', p.units || (p.kind === 'enum' ? 'choice' : '—'));
  row('Specification key', p.id);
  row('Source', sourceClass(p.id));
  row('Affects', { geometry: 'geometry (rebuilds the model)', visualization: 'visualization only (no rebuild)', metadata: 'metadata only (recorded; no simulated effect)' }[p.affects] || p.affects || 'geometry');
  row('Scope', p.scope || '—');
  if (p.owner_level) {
    row('Owner', `Level ${p.owner_level} — ${(S.reg.levels.find(L => L.n === p.owner_level) || {}).name || ''}`);
    row('Modifies', p.component || '—');
    row('Neighbour levels', (p.neighbors || []).map(([n, why]) => `L${n}: ${why}`).join('; ') || 'none directly');
    row('Basis', basisOf(p.id) + ' (measured, assumed or experimental)');
  } else row('Panel', 'Anatomy and Pose (reference anatomy, not a scaffold control)');
  row('Dependencies', h.depends || '—');
  const vs = row('Value source', valueSource(p.id)); vs.className = 'vsrc';
  if (h.guide) dl.appendChild(baSVG(h.guide));
  const r = document.createElement('div'); r.className = 'hrow'; dl.appendChild(r);
  const b1 = document.createElement('button'); b1.textContent = 'Show on model';
  b1.title = 'Temporarily highlight the affected structure with a visual guide (camera stays put)';
  b1.onclick = () => showGuide(p.id);
  const b2 = document.createElement('button'); b2.textContent = 'Focus camera';
  b2.title = 'Move the camera to the highlighted structure (explicit action)';
  b2.onclick = () => focusGuide(p.id);
  r.append(b1, b2);
  return det;
}

function curveEditor(p) {
  const wrap = document.createElement('div');
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg'); svg.setAttribute('class', 'curve'); svg.setAttribute('viewBox', '0 0 300 110');
  svg.setAttribute('aria-label', p.label + ' curve editor (drag points)');
  wrap.appendChild(svg);
  const info = document.createElement('div'); info.className = 'desc'; wrap.appendChild(info);
  const get = () => { const v = S.state.params[p.id]; return Array.isArray(v) ? { s: S.reg.s_knots.slice(), v: v.slice(), knots: true } : { s: v.s.slice(), v: v.v.slice(), knots: false }; };
  const X = s => 14 + s * 272, Y = val => 98 - (val - p.min) / (p.max - p.min) * 88, V = y => p.min + (98 - y) / 88 * (p.max - p.min);
  const draw = () => {
    const d = get(); svg.innerHTML = '';
    const ax = document.createElementNS(svgNS, 'path'); ax.setAttribute('d', 'M14 98H286M14 10V98'); ax.setAttribute('stroke', '#ccd'); svg.appendChild(ax);
    const ln = document.createElementNS(svgNS, 'polyline'); ln.setAttribute('points', d.s.map((s, i) => X(s) + ',' + Y(d.v[i])).join(' '));
    ln.setAttribute('fill', 'none'); ln.setAttribute('stroke', '#0f6f68'); ln.setAttribute('stroke-width', '2'); svg.appendChild(ln);
    const lab = (t, x, y, a) => { const e = document.createElementNS(svgNS, 'text'); e.textContent = t; e.setAttribute('x', x); e.setAttribute('y', y); e.setAttribute('font-size', '9'); e.setAttribute('fill', '#667'); if (a) e.setAttribute('text-anchor', a); svg.appendChild(e); };
    lab('tibia', 14, 108); lab('femur', 286, 108, 'end'); lab(p.max + ' ' + p.units, 16, 9); lab(p.min, 16, 96);
    d.s.forEach((s, i) => {
      const c = document.createElementNS(svgNS, 'circle'); c.setAttribute('class', 'pt'); c.setAttribute('cx', X(s)); c.setAttribute('cy', Y(d.v[i])); c.setAttribute('r', 5);
      c.onpointerdown = ev => {
        c.setPointerCapture(ev.pointerId);
        const move = e2 => {
          const r = svg.getBoundingClientRect(); const y = (e2.clientY - r.top) / r.height * 110;
          const val = Math.min(p.max, Math.max(p.min, V(y))); const dd = get(); dd.v[i] = +val.toFixed(3);
          const nv = dd.knots ? dd.v : { s: dd.s, v: dd.v };
          S.state.params[p.id] = nv; draw(); info.textContent = `s = ${s.toFixed(2)}: ${val.toFixed(2)} ${p.units}`;
          setParam(p.id, JSON.parse(JSON.stringify(nv)), { preview: true, noHistory: true });
        };
        const up = () => { c.onpointermove = null; c.onpointerup = null; S.state.history.push({ action: 'edit', param: p.id, to: S.state.params[p.id], actor: 'user', t: new Date().toISOString() }); gestureCommit(p.label); scheduleBuild('full', 50); };
        c.onpointermove = move; c.onpointerup = up;
      };
      c.innerHTML = `<title>s=${s.toFixed(2)}: ${d.v[i]} ${p.units}</title>`; svg.appendChild(c);
    });
    info.textContent = (d.knots ? 'Drag points to change the value along the ligament (local edit).' : `${d.s.length} imported samples — spatial detail kept. Drag to edit.`);
  };
  draw(); return wrap;
}

function polygonEditor(p) {
  const wrap = document.createElement('div');
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg'); svg.setAttribute('class', 'poly'); svg.setAttribute('viewBox', '-15 -12 30 24');
  svg.setAttribute('aria-label', p.label + ' boundary editor');
  const side = p.id.split('.')[1];
  const info = document.createElement('div'); info.className = 'desc';
  info.textContent = 'Boundary in the footprint tangent chart (mm, long axis →). Drag vertices; double-click an edge to add, Alt-click a vertex to remove.';
  const draw = () => {
    svg.innerHTML = '';
    const pts = S.state.params[p.id] || [];
    const g = document.createElementNS(svgNS, 'path'); g.setAttribute('d', 'M-15 0H15M0 -12V12'); g.setAttribute('stroke', '#dde'); g.setAttribute('stroke-width', '0.1'); svg.appendChild(g);
    if (pts.length >= 3) {
      const pg = document.createElementNS(svgNS, 'polygon'); pg.setAttribute('points', pts.map(q => q[0] + ',' + (-q[1])).join(' '));
      pg.setAttribute('fill', '#f0a05055'); pg.setAttribute('stroke', '#c0392b'); pg.setAttribute('stroke-width', '0.15');
      pg.ondblclick = ev => { const r = svg.getBoundingClientRect(); const x = (ev.clientX - r.left) / r.width * 30 - 15, y = -((ev.clientY - r.top) / r.height * 24 - 12);
        let best = 0, bd = 1e9; pts.forEach((q, i) => { const n = pts[(i + 1) % pts.length]; const mx = (q[0] + n[0]) / 2, my = (q[1] + n[1]) / 2; const d = (mx - x) ** 2 + (my - y) ** 2; if (d < bd) { bd = d; best = i; } });
        const np = pts.slice(); np.splice(best + 1, 0, [+x.toFixed(2), +y.toFixed(2)]); setParam(p.id, np); draw(); };
      svg.appendChild(pg);
    }
    pts.forEach((q, i) => {
      const c = document.createElementNS(svgNS, 'circle'); c.setAttribute('class', 'pt'); c.setAttribute('cx', q[0]); c.setAttribute('cy', -q[1]); c.setAttribute('r', 0.55);
      c.onpointerdown = ev => {
        if (ev.altKey) { const np = pts.slice(); np.splice(i, 1); setParam(p.id, np); draw(); return; }
        c.setPointerCapture(ev.pointerId);
        c.onpointermove = e2 => { const r = svg.getBoundingClientRect(); const x = (e2.clientX - r.left) / r.width * 30 - 15, y = -((e2.clientY - r.top) / r.height * 24 - 12);
          const np = JSON.parse(JSON.stringify(S.state.params[p.id])); np[i] = [+x.toFixed(2), +y.toFixed(2)]; S.state.params[p.id] = np; c.setAttribute('cx', x); c.setAttribute('cy', -y); };
        c.onpointerup = () => { c.onpointermove = null; setParam(p.id, JSON.parse(JSON.stringify(S.state.params[p.id]))); gestureCommit(p.label); draw(); };
      };
      svg.appendChild(c);
    });
  };
  wrap.appendChild(svg); wrap.appendChild(info);
  const b = document.createElement('button'); b.textContent = 'Start from the current oval/crescent';
  b.onclick = () => { const L = S.state.params['fp.' + side + '.length_mm'], W = S.state.params['fp.' + side + '.width_mm']; const pts = [];
    for (let k = 0; k < 12; k++) { const a = 2 * Math.PI * k / 12; pts.push([+(L / 2 * Math.cos(a)).toFixed(2), +(W / 2 * Math.sin(a)).toFixed(2)]); }
    S.state.params['fp.' + side + '.shape'] = 'polygon'; setParam(p.id, pts); renderControls(); };
  wrap.appendChild(b); draw(); return wrap;
}

function compositionEditor(p) {
  const wrap = document.createElement('div'); wrap.className = 'comp';
  const v = S.state.params[p.id] || {};
  const tot = Object.values(v).reduce((a, b) => a + Number(b), 0) || 1;
  (p.materials || Object.keys(S.reg.materials)).forEach(mat => {
    const lab = document.createElement('span'); lab.innerHTML = `<i class="sw" style="background:rgb(${S.reg.materials[mat].color.map(x => Math.round(x * 255)).join(',')})"></i>${mat}`;
    const n = document.createElement('input'); n.type = 'number'; n.min = 0; n.max = 1; n.step = 0.05; n.value = v[mat] || 0; n.setAttribute('aria-label', mat + ' fraction');
    const norm = document.createElement('span'); norm.className = 'mono small'; norm.textContent = ((v[mat] || 0) / tot * 100).toFixed(0) + '%';
    n.onchange = () => { const cur = Object.assign({}, S.state.params[p.id]); const x = Number(n.value);
      if (!(x >= 0)) { S.msgs[p.id] = [{ level: 'fail', text: 'fractions must be ≥ 0' }]; refreshMessages(); return; }
      if (x > 0) cur[mat] = x; else delete cur[mat]; setParam(p.id, cur); gestureCommit(p.label); renderControls(); };
    wrap.append(lab, n, norm);
  });
  const b = document.createElement('button'); b.textContent = 'Normalize to 1'; b.style.gridColumn = '1/4';
  b.onclick = () => { const cur = S.state.params[p.id]; const t = Object.values(cur).reduce((a, c) => a + Number(c), 0); if (t > 0) { const o = {}; for (const k in cur) o[k] = +(cur[k] / t).toFixed(4); setParam(p.id, o); renderControls(); } };
  wrap.appendChild(b);
  const d = document.createElement('div'); d.className = 'desc'; d.style.gridColumn = '1/4'; d.textContent = `Total = ${tot.toFixed(3)}${Math.abs(tot - 1) > 1e-3 ? ' — must be 1: use Normalize (colours and exports use normalized fractions)' : ' ✔'}. Design metadata; no material model is evaluated.`; if (Math.abs(tot - 1) > 1e-3) d.style.color = 'var(--warn)'; wrap.appendChild(d);
  return wrap;
}

function updateControlState() {
  if (!S.reg) return;
  const dep = DEPENDENT();
  document.querySelectorAll('#tab-design .ctl').forEach(div => {
    const pid = div.dataset.pid; const why = dep[pid];
    const spec = S.regById[pid] || {};
    const pv = provOf(pid);
    const locked = S.level0Locked && spec.owner_level === 0 && ['measured', 'observed', 'inferred'].includes(pv.status);
    div.classList.toggle('dependent', !!why);
    div.classList.toggle('locked', !!locked);
    div.querySelectorAll('input,select').forEach(e => { if (!e.closest('.comp') || !why) e.disabled = !!why || !!locked || VIEW_MODE; });
    div.querySelector('.chipbox').innerHTML = chip(pid);
    const ovr = div.querySelector('.ovr'); if (ovr) ovr.hidden = provOf(pid).status !== 'override';
    const vs = div.querySelector('.vsrc'); if (vs) vs.textContent = valueSource(pid);
    const rst = div.querySelector('.rst');
    if (rst) { const dest = resetDestination(pid); rst.textContent = `↺ ${dest === 'imported value' ? 'imported' : 'template'}`; rst.title = `Reset to ${dest}`; }
    let note = div.querySelector('.depnote'); if (!note) { note = document.createElement('div'); note.className = 'depnote desc'; div.insertBefore(note, div.querySelector('.msgbox')); }
    const dv = derivedValue(pid);
    note.textContent = (locked ? 'Locked: patient-derived value (unlock Patient inputs to edit). ' : '') +
      (why ? (`${/^calculated/.test(why) ? '' : 'Calculated — '}${why}` + (dv != null && typeof dv === 'number' ? ` = ${Number(dv).toFixed(3)}` : '')) : '');
    const tag = div.querySelector('.vtag');
    if (tag) {
      const calc = why && /calculated|computed|derived|current length|solved/.test(why);
      const user = provOf(pid).status === 'override';
      tag.className = 'vtag ' + (calc ? 'derivedtag' : user ? 'usertag' : '');
      tag.textContent = calc ? 'calculated' : user ? 'user-set' : '';
      tag.title = calc ? 'Automatically derived value: switch the mode to make it user-set' : user ? 'Set by you' : '';
    }
    const bs = div.querySelector('.basis'); if (bs) { const b = basisOf(pid); bs.className = 'basis ' + b; bs.textContent = b; }
  });
}
function refreshDependents() { updateControlState(); }

function collectMessages() {
  S.msgs = {};
  (S.scene.checks || []).forEach(c => {
    if (!['fail', 'warn', 'info'].includes(c.status) || c.category === 'provenance' || c.category === 'scope') return;
    (c.params || []).slice(0, 3).forEach(pid => (S.msgs[pid] = S.msgs[pid] || []).push({ level: c.status, text: c.label + ': ' + (c.message || c.status) }));
  });
  (S.scene.warnings || []).forEach(w => (S.msgs[w.param] = S.msgs[w.param] || []).push({ level: 'warn', text: w.message }));
}
function refreshMessages() {
  document.querySelectorAll('#tab-design .ctl').forEach(div => {
    const box = div.querySelector('.msgbox'); const ms = S.msgs[div.dataset.pid] || [];
    box.innerHTML = ms.map(m => `<div class="msg ${m.level === 'fail' ? '' : m.level}">${m.level === 'fail' ? '✖' : m.level === 'warn' ? '⚠' : 'ℹ'} ${m.text}</div>`).join('');
  });
}

// ------------------------------------------------------------------ inspect panel
function renderObjList() {
  const box = $('objList'); box.innerHTML = '';
  const order = ['bone', 'cartilage', 'landmark', 'footprint', 'reference', 'bundle', 'retained', 'retained_structure', 'injury', 'observed', 'scaffold', 'scaffold_region', 'guide',
    'layer_sheath', 'layer_interface', 'layer_braid', 'layer_channels', 'layer_coating', 'layer_grooves', 'layer_end', 'layer_footprint', 'layer_plug', 'layer_pores', 'layer_anchors'];
  const objs = S.scene.objects.slice().sort((a, b) => order.indexOf(a.category) - order.indexOf(b.category));
  objs.forEach(meta => {
    const row = document.createElement('div'); row.className = 'objrow'; row.dataset.obj = meta.id;
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = S.vis[meta.id]; cb.setAttribute('aria-label', 'show ' + meta.label);
    cb.onchange = () => { S.vis[meta.id] = cb.checked; const o = S.objects.get(meta.id); if (o) o.visible = cb.checked; invalidate(); saveView(); };
    const lab = document.createElement('span'); lab.innerHTML = `${meta.label} <span class="chip ${meta.status.replace(' ', '_')}" title="${(meta.note || '').replace(/"/g, '&quot;')}">${meta.status}</span>`;
    const op = document.createElement('input'); op.type = 'range'; op.min = 0.05; op.max = 1; op.step = 0.05; op.value = S.opac[meta.id]; op.title = 'opacity';
    op.oninput = () => { S.opac[meta.id] = Number(op.value); const o = S.objects.get(meta.id); if (o) matOptsFor(o); invalidate(); saveView(); };
    const nb = document.createElement('span'); nb.className = 'muted small'; nb.textContent = meta.kind === 'mesh' ? '' : meta.n_lines;
    cb.dataset.ctl = 'obj.' + meta.id + '.visible'; op.dataset.ctl = 'obj.' + meta.id + '.opacity';
    row.append(cb, lab, op, nb); box.appendChild(row);
  });
}
function saveView() { try { localStorage.setItem('acl.v2.view', JSON.stringify({ vis: S.vis, opac: S.opac })); } catch (e) { /* ignore */ } }

function renderDisplayControls() {
  const d = S.state.display; const box = $('displayCtl'); box.innerHTML = '';
  const add = (label, el, key) => { if (key) el.dataset.ctl = key; const r = document.createElement('div'); r.className = 'row'; const l = document.createElement('label'); l.textContent = label; l.style.flex = '1'; r.append(l, el); box.appendChild(r); };
  const sel = (opts, val, fn) => { const s = document.createElement('select'); opts.forEach(o => { const op = document.createElement('option'); const pair = Array.isArray(o); op.value = pair ? o[0] : o; op.textContent = pair ? o[1] : o; s.appendChild(op); }); s.value = val; s.onchange = () => fn(s.value); return s; };
  const num = (min, max, step, val, fn) => { const n = document.createElement('input'); n.type = 'range'; n.min = min; n.max = max; n.step = step; n.value = val; n.onchange = () => fn(Number(n.value)); return n; };
  add('Colour by', sel([['anatomy', 'structural'], ['material', 'material'], ['family', 'family'], ['status', 'status']], S.colorMode, v => { S.colorMode = v; clearMatCache(); rebuildSceneObjects(); renderLegend(); }), 'view.color_mode');
  add('Fiber rendering', sel(['auto', 'lines', 'tubes', 'hidden'], d.fiber_mode, v => { setDisplay('fiber_mode', v); rebuildSceneObjects(); }), 'display.fiber_mode');
  add('Native fibers shown per fascicle', num(1, 19, 1, d.native_fibers_per_fascicle, v => setDisplay('native_fibers_per_fascicle', v, true)), 'display.native_fibers_per_fascicle');
  add('Scaffold fibers shown per yarn', num(1, 19, 1, d.scaffold_fibers_per_yarn, v => setDisplay('scaffold_fibers_per_yarn', v, true)), 'display.scaffold_fibers_per_yarn');
  add('Crimp drawn at × (display)', num(1, 60, 1, d.crimp_exaggeration, v => setDisplay('crimp_exaggeration', v, true)), 'display.crimp_exaggeration');
  add('Exploded view (mm)', num(0, 25, 1, d.exploded || 0, v => { setDisplay('exploded', v); applyExploded(); }), 'view.explode');
  const n = document.createElement('p'); n.className = 'note'; n.textContent = 'Representative density and crimp magnification change only what is drawn, never the design or exports (the design hash stays the same).'; box.appendChild(n);
  if (VIEW_MODE) { const f = document.createElement('label'); f.innerHTML = '<input type="checkbox" id="follow" checked> Follow changes from the designer tab'; box.appendChild(f); }
}

function renderClipControls() {
  const box = $('clipCtl'); box.innerHTML = '';
  S.clip = S.clip || { on: false, axis: 'sagittal', pos: 0, flip: false };
  S.cut = S.cut || { on: true, pos: 14 };
  S.slab = S.slab || { on: false, width: 1.2 };
  const row = (html) => { const r = document.createElement('div'); r.className = 'row'; r.innerHTML = html; box.appendChild(r); return r; };
  const r1 = row(`<label><input type="checkbox" id="cutOn"> Bone cutaway</label> <input type="range" id="cutPos" min="-20" max="30" step="0.5" style="flex:1" title="cut position (mm, lateral of the plateau centre)">`);
  r1.querySelector('#cutOn').checked = S.cut.on; r1.querySelector('#cutPos').value = S.cut.pos;
  r1.querySelector('#cutOn').onchange = e => { S.cut.on = e.target.checked; $('cutQuick').checked = S.cut.on; updateClip(); };
  r1.querySelector('#cutPos').oninput = e => { S.cut.pos = Number(e.target.value); updateClip(); };
  const r2 = row(`<label><input type="checkbox" id="clipOn"> Clip all</label> <select id="clipAx"><option>sagittal</option><option>coronal</option><option>axial</option></select> <input type="range" id="clipPos" min="-40" max="60" step="0.5" style="flex:1"> <label><input type="checkbox" id="clipFlip"> flip</label>`);
  r2.querySelector('#clipOn').checked = S.clip.on; r2.querySelector('#clipAx').value = S.clip.axis; r2.querySelector('#clipPos').value = S.clip.pos;
  r2.querySelector('#clipOn').onchange = e => { S.clip.on = e.target.checked; updateClip(); };
  r2.querySelector('#clipAx').onchange = e => { S.clip.axis = e.target.value; updateClip(); };
  r2.querySelector('#clipPos').oninput = e => { S.clip.pos = Number(e.target.value); updateClip(); };
  r2.querySelector('#clipFlip').onchange = e => { S.clip.flip = e.target.checked; updateClip(); };
  const r4 = row(`<label><input type="checkbox" id="slabOn"> Show only a thin slab at the section (3D)</label>`);
  r4.querySelector('#slabOn').checked = S.slab.on; r4.querySelector('#slabOn').onchange = e => { S.slab.on = e.target.checked; rebuildSceneObjects(); };
  const lb = $('btnLongCut'); if (lb) lb.classList.toggle('on', !!(S.clip && S.clip.on && S.clip.long));
  renderSectionControls();
}
const SEC_KINDS = { ligament: 'Perpendicular to the ACL', am: 'Perpendicular to the AM bundle', pl: 'Perpendicular to the PL bundle', axial: 'Axial (anatomical)', sagittal: 'Sagittal (anatomical)', coronal: 'Coronal (anatomical)' };
function renderSectionControls() {
  const box = $('secCtl'); box.innerHTML = '';
  const d = S.state.display; const kind = d.section_kind || 'ligament'; const local = ['ligament', 'am', 'pl'].includes(kind);
  const r = document.createElement('div'); r.className = 'row';
  r.innerHTML = `<select id="secKind" aria-label="section plane">${Object.entries(SEC_KINDS).map(([k, v]) => `<option value="${k}">${v}</option>`).join('')}</select>
    <label style="flex:1">${local ? 's' : 'offset'} = <b id="secVal"></b>${local ? '' : ' mm'}</label>
    <input type="range" id="secPos" style="flex:2" min="${local ? 0.02 : -25}" max="${local ? 0.98 : 25}" step="${local ? 0.01 : 0.25}">
    <input type="number" id="secNum" style="width:70px" step="${local ? 0.01 : 0.25}" aria-label="section position">`;
  box.appendChild(r);
  const v = local ? Number(d.section_s) : Number(d.section_offset_mm || 0);
  r.querySelector('#secKind').value = kind; r.querySelector('#secPos').value = v; r.querySelector('#secNum').value = v; r.querySelector('#secVal').textContent = v.toFixed(2);
  const set = (x, commit) => {
    if (local) d.section_s = x; else d.section_offset_mm = x;
    r.querySelector('#secVal').textContent = Number(x).toFixed(2); r.querySelector('#secNum').value = x;
    updateSlab(); if (commit) { persist(); requestSection(); }
  };
  r.querySelector('#secKind').onchange = e => { d.section_kind = e.target.value; persist(); renderSectionControls(); requestSection(); };
  r.querySelector('#secPos').oninput = e => set(Number(e.target.value), false);
  r.querySelector('#secPos').onchange = e => set(Number(e.target.value), true);
  r.querySelector('#secNum').onchange = e => set(Number(e.target.value), true);
}
async function requestSection() {
  const d = S.state.display; const kind = d.section_kind || 'ligament';
  const pos = ['ligament', 'am', 'pl'].includes(kind) ? Number(d.section_s) : Number(d.section_offset_mm || 0);
  const my = (S.secSeq = (S.secSeq || 0) + 1);
  $('sectionTitle').textContent = 'cutting…';
  try {
    const j = await api('/api/v2/section', { state: S.state, kind, pos, quality: S.scene ? S.scene.quality : 'full' });
    if (my !== S.secSeq) return;
    S.scene.section = j.section; renderSection(); updateSlab();
  } catch (e) { $('sectionTitle').textContent = 'section failed: ' + e.message; }
}
function updateClip() { setClipPlanes(); rebuildSceneObjects(); }
function setClipPlanes() {
  if (!S.cut || !S.clip) return;
  const side = S.scene && S.scene.side === 'left' ? -1 : 1;
  // bone cutaway removes bone lateral of the cut (opens the notch from the lateral side)
  clipPlanes.bone.normal.set(-side, 0, 0); clipPlanes.bone.constant = S.cut.pos;   // keeps |x| < pos on the lateral side (both knees)
  const ax = { sagittal: [1, 0, 0], coronal: [0, 0, -1], axial: [0, 1, 0] }[S.clip.axis];
  const n = new THREE.Vector3(...ax).multiplyScalar(S.clip.flip ? -1 : 1);
  clipPlanes.global.normal.copy(n.clone().negate()); clipPlanes.global.constant = (S.clip.flip ? -1 : 1) * S.clip.pos;
}
function updateSlab() {
  const sec = S.scene && S.scene.section; if (!sec || !sec.origin_mm) return;
  // the slab follows the section plane (same plane as the 2D section)
  let O = sec.origin_mm.slice(), n = sec.normal.slice();
  const d = S.state.display, kind = d.section_kind || 'ligament';
  if (kind === 'ligament') {            // live slider preview along the centreline
    const cl = S.objects.get('centerline');
    if (cl) { const P = cl.userData.lines.P, m = P.length / 3, i = Math.round(d.section_s * (m - 1)), a = Math.max(0, i - 2), b2 = Math.min(m - 1, i + 2);
      O = [P[3 * i], P[3 * i + 1], P[3 * i + 2]]; n = [P[3 * b2] - P[3 * a], P[3 * b2 + 1] - P[3 * a + 1], P[3 * b2 + 2] - P[3 * a + 2]]; }
  } else if (sec.mirrored) { O[0] = -O[0]; n[0] = -n[0]; }
  const p = new THREE.Vector3(...O), t = new THREE.Vector3(...n).normalize();
  root.updateMatrixWorld(true);
  const pw = p.clone().applyMatrix4(root.matrixWorld), tw = t.clone().transformDirection(root.matrixWorld);
  const w = S.slab ? S.slab.width / 2 : 0.6;
  clipPlanes.slabA = clipPlanes.slabA || new THREE.Plane(); clipPlanes.slabB = clipPlanes.slabB || new THREE.Plane();
  clipPlanes.slabA.setFromNormalAndCoplanarPoint(tw, pw.clone().addScaledVector(tw, -w));
  clipPlanes.slabB.setFromNormalAndCoplanarPoint(tw.clone().negate(), pw.clone().addScaledVector(tw, w));
  invalidate();
}

const SEC_STYLE = {
  bone: ['#d9cfbd', '#8c8270', 1.0], cartilage: ['#a9c3ea', '#5b7fb8', 1.0], landmark: ['#c9c9dd', '#8a8aa6', 1.0], footprint: [null, '#c06a2a', 1.4],
  reference: [null, '#222', 1.6], bundle: ['', '', 1.0], retained: ['#d9776d40', '#b55045', 1.0], injury: ['#e03b2f2e', '#e03b2f', 1.0],
  repair_space: ['#6fcfff26', '#1f8fcf', 1.2], observed: [null, '#35c4d8', 1.2], scaffold_region: [null, null, 0],
};
function renderSection() {
  const sec = S.scene && S.scene.section;
  drawSection($('sectionCanvas'), sec, null, true);
  if (!$('lvSecWrap').hidden) { drawSection($('lvSecCanvas'), sec, S.levelSel, false); $('lvSecTitle').textContent = sec ? sec.title : ''; }
}
function drawSection(cv, sec, emph, main) {
  const g = cv.getContext('2d');
  const A = (lv) => (!emph || !lv || lv === emph) ? 1 : 0.18;     // non-selected levels are dimmed
  g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height); g.fillStyle = '#fff'; g.fillRect(0, 0, cv.width, cv.height);
  if (!sec || !sec.layers) return;
  // extent: the ligament/repair layers (and strands); bones only if nothing else is cut
  const pts = []; const addSegs = (L) => { for (let i = 0; i < L.segs.length; i += 2) pts.push([L.segs[i], L.segs[i + 1]]); };
  sec.layers.filter(L => !['bone', 'cartilage', 'landmark'].includes(L.category)).forEach(addSegs);
  sec.yarns.forEach(e => pts.push([e.x, e.y])); sec.fascicles.forEach(e => pts.push([e.x, e.y]));
  (sec.channels || []).forEach(e => pts.push([e.x, e.y])); (sec.braid || []).forEach(e => pts.push([e.x, e.y]));
  const local = ['ligament', 'am', 'pl'].includes(sec.kind);
  if (!pts.length) sec.layers.forEach(addSegs);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  pts.forEach(p => { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); });
  if (!isFinite(x0)) { x0 = -10; x1 = 10; y0 = -10; y1 = 10; }
  const padmm = local ? 1.5 : 4; x0 -= padmm; x1 += padmm; y0 -= padmm; y1 += padmm;
  const W = cv.width, H = cv.height, M = 44;
  const sc = Math.min((W - 2 * M) / (x1 - x0), (H - 2 * M - 30) / (y1 - y0));
  const cx = W / 2 - sc * (x0 + x1) / 2, cy = (H - 30) / 2 + sc * (y0 + y1) / 2 + 10;
  const X = x => cx + x * sc, Y = y => cy - y * sc;
  const drawSegs = (L, stroke, width) => { g.strokeStyle = stroke; g.lineWidth = width; g.beginPath(); for (let i = 0; i < L.segs.length; i += 4) { g.moveTo(X(L.segs[i]), Y(L.segs[i + 1])); g.lineTo(X(L.segs[i + 2]), Y(L.segs[i + 3])); } g.stroke(); };
  const loopPath = (L) => {           // stitch segments into closed loops and add them to the current path
    const segs = []; for (let i = 0; i < L.segs.length; i += 4) segs.push([[L.segs[i], L.segs[i + 1]], [L.segs[i + 2], L.segs[i + 3]]]);
    const key = p => p[0].toFixed(3) + ',' + p[1].toFixed(3); const adj = new Map();
    segs.forEach((sg, k) => [0, 1].forEach(e => { const kk = key(sg[e]); if (!adj.has(kk)) adj.set(kk, []); adj.get(kk).push([k, e]); }));
    const used = new Array(segs.length).fill(false);
    for (let k = 0; k < segs.length; k++) { if (used[k]) continue; used[k] = true; let cur = segs[k][1]; g.moveTo(X(segs[k][0][0]), Y(segs[k][0][1])); g.lineTo(X(cur[0]), Y(cur[1]));
      for (let guard = 0; guard < segs.length; guard++) { const nb = (adj.get(key(cur)) || []).find(([j]) => !used[j]); if (!nb) break; used[nb[0]] = true; cur = segs[nb[0]][1 - nb[1]]; g.lineTo(X(cur[0]), Y(cur[1])); }
      g.closePath(); }
  };
  const fillSegs = (L, fill) => { g.fillStyle = fill; g.beginPath(); loopPath(L); g.fill('evenodd'); };
  // tissue regions (remnants, injury, repair space, bundles) belong to the ligament: draw them
  // clipped to the exact envelope cut, so the smoothed display meshes never appear to leak
  const env = sec.layers.find(L => L.id === 'intact_reference');
  const inLig = new Set(['bundle', 'retained', 'injury', 'repair_space']);
  const order = ['bone', 'cartilage', 'landmark', 'footprint', 'bundle', 'retained', 'injury', 'repair_space', 'reference', 'observed'];
  const layers = sec.layers.slice().sort((a, b) => order.indexOf(a.category) - order.indexOf(b.category));
  const clipEnv = (fn) => { if (!env) return fn(); g.save(); g.beginPath(); loopPath(env); g.clip('evenodd'); fn(); g.restore(); };
  layers.filter(L => !inLig.has(L.category)).forEach(L => { const st = SEC_STYLE[L.category]; if (st && st[0]) fillSegs(L, st[0]); });
  clipEnv(() => layers.filter(L => inLig.has(L.category)).forEach(L => {
    if (L.category === 'bundle') { fillSegs(L, L.id.includes('_am_') ? '#f0a05030' : '#5aa0d830'); return; }
    const st = SEC_STYLE[L.category]; if (st && st[0]) fillSegs(L, st[0]);
  }));
  // boundary uncertainty band and interface
  if (sec.uncertainty_band) { const b = sec.uncertainty_band; g.fillStyle = '#7a7a7a28'; g.beginPath(); for (let i = 0; i < b.length; i += 2) i ? g.lineTo(X(b[i]), Y(b[i + 1])) : g.moveTo(X(b[i]), Y(b[i + 1])); g.closePath(); g.fill(); }
  layers.filter(L => !inLig.has(L.category)).forEach(L => { const st = SEC_STYLE[L.category]; if (!st || !st[1]) return; drawSegs(L, st[1], st[2]); });
  clipEnv(() => layers.filter(L => inLig.has(L.category) && L.category !== 'bundle').forEach(L => { const st = SEC_STYLE[L.category]; if (!st || !st[1]) return; if (L.category === 'repair_space') g.setLineDash([5, 3]); drawSegs(L, st[1], st[2]); g.setLineDash([]); }));
  if (env) drawSegs(env, SEC_STYLE.reference[1], SEC_STYLE.reference[2]);
  const ell = (e) => { g.beginPath(); g.ellipse(X(e.x), Y(e.y), Math.max(0.8, e.a * sc), Math.max(0.8, e.b * sc), -e.ang, 0, 2 * Math.PI); };
  // layered construct: sheath (annulus), end domains, plugs, pores, coating; then braid and channels
  const LST = { layer_end: ['#c9b3d6aa', '#7d5a96'], layer_sheath: ['#cfe4eadd', '#4f7380'], layer_plug: ['#bfb39ecc', '#7a6c55'], layer_pores: ['#1d2629', '#1d2629'], layer_coating: ['#2ec4b688', '#1a8f85'], layer_interface: ['#8c73b855', '#6a5590'] };
  ['layer_end', 'layer_sheath', 'layer_plug', 'layer_pores', 'layer_coating', 'layer_interface'].forEach(cat => layers.filter(L => L.category === cat).forEach(L => {
    g.globalAlpha = A(L.level); fillSegs(L, LST[cat][0]); drawSegs(L, LST[cat][1], 1.1); g.globalAlpha = 1;
  }));
  (sec.braid || []).forEach(e => { g.globalAlpha = A(4); ell(e); g.fillStyle = e.group ? '#4f9bd6' : '#e8a33a'; g.fill(); g.lineWidth = 0.6; g.strokeStyle = '#333'; g.stroke(); g.globalAlpha = 1; });
  (sec.channels || []).forEach(e => {
    g.globalAlpha = A(3); ell(e); g.fillStyle = ['#e9cf9c', '#a9c8e6', '#cdb5e2'][e.group || 0]; g.fill(); g.lineWidth = 0.6; g.strokeStyle = '#555'; g.stroke();
    if (e.ai) { g.globalAlpha = A(3); g.beginPath(); g.ellipse(X(e.x), Y(e.y), Math.max(0.5, e.ai * sc), Math.max(0.5, e.bi * sc), -e.ang, 0, 2 * Math.PI);
      g.fillStyle = '#ffffff'; g.fill(); if (e.coat) { g.globalAlpha = A(1); g.lineWidth = Math.max(1, 1.5); g.strokeStyle = '#2ec4b6'; g.stroke(); } }
    g.globalAlpha = 1;
  });
  if (sec.interface) { const f = sec.interface; g.setLineDash([6, 4]); g.strokeStyle = '#555'; g.lineWidth = 1.3; g.beginPath(); for (let i = 0; i < f.length; i += 2) i ? g.lineTo(X(f[i]), Y(f[i + 1])) : g.moveTo(X(f[i]), Y(f[i + 1])); g.stroke(); g.setLineDash([]); }
  sec.fascicles.forEach(e => { ell(e); g.fillStyle = e.bundle ? '#4f8fcfcc' : '#e8894acc'; if (e.bridging) g.fillStyle = '#ffffffcc'; g.fill(); g.strokeStyle = '#0005'; g.lineWidth = 0.5; g.stroke(); });
  const famC = { '+': '#e8c14a', '-': '#3fb39b', core: '#c56fc0', axial: '#e8c14a' };
  sec.yarns.forEach(e => { ell(e); g.fillStyle = famC[e.family] || '#e8c14a'; g.fill(); g.lineWidth = e.viol ? 2 : 0.7; g.strokeStyle = e.viol ? (e.viol & 63 ? '#ff1a1a' : '#ff9a1a') : '#333'; g.stroke(); });
  sec.violations.forEach(v => { g.strokeStyle = '#ff1a1a'; g.lineWidth = 2; const x = X(v.x), y = Y(v.y); g.beginPath(); g.moveTo(x - 6, y - 6); g.lineTo(x + 6, y + 6); g.moveTo(x + 6, y - 6); g.lineTo(x - 6, y + 6); g.stroke(); });
  // axes words, scale bar
  g.fillStyle = '#333'; g.font = '13px system-ui';
  g.fillText(sec.axes.x[1] + ' →', W - M - 70, H - 36); g.fillText('← ' + sec.axes.x[0], M, H - 36);
  g.save(); g.translate(16, H / 2); g.rotate(-Math.PI / 2); g.fillText('← ' + sec.axes.y[0] + '      ' + sec.axes.y[1] + ' →', -80, 0); g.restore();
  let bar = 1; for (const c of [0.5, 1, 2, 5, 10, 20]) if (c * sc < 160) bar = c;
  g.fillStyle = '#111'; g.fillRect(W - M - bar * sc, H - 18, bar * sc, 4); g.fillText(bar + ' mm', W - M - bar * sc, H - 22);
  if (emph) { g.fillStyle = LV_COL[emph]; g.font = '600 15px system-ui'; g.fillText(`Level ${emph} emphasised`, M, 22); }
  if (!main) return;
  $('sectionTitle').textContent = sec.title + (sec.mirrored ? ' (left knee: anatomical labels shown)' : '');
  const ny = sec.yarns.length, nv = sec.yarns.filter(e => e.viol & 63).length;
  const nch = (sec.channels || []).length, nbr = (sec.braid || []).length;
  $('sectionInfo').innerHTML = `Cut from the same meshes and strands as the 3D view and the export (true scale; strands are their real elliptical cross-sections). ` +
    (sec.area_mm2 ? `Section ${sec.area_mm2.toFixed(1)} mm², AM ${(100 * sec.am_fraction).toFixed(0)}%. ` : '') +
    (nch || nbr ? `Layered construct: ${nch} channel cuts (white = open lumen, teal ring = coating), ${nbr} braid strand cuts (<span style="color:#d9772b">●</span>AM-like <span style="color:#3f8ac6">●</span>PL-like), sheath wall shaded blue-grey, violet band = inter-bundle zone. ` : '') +
    `Strands cut: ${ny}${nv ? `, <b style="color:#c00">${nv} violating</b>` : ''}. ` +
    `<span style="color:#222">━</span> ligament envelope · <span style="background:#f0a05050">AM</span>/<span style="background:#5aa0d850">PL</span> regions, dashed line = AM/PL boundary, grey band = boundary uncertainty · ` +
    `<span style="color:#b55045">▬</span> retained tissue · <span style="color:#e03b2f">▬</span> injury · <span style="color:#1f8fcf">┅</span> repair space · fascicles <span style="color:#e8894a">●</span>AM <span style="color:#4f8fcf">●</span>PL · yarns <span style="color:#e8c14a">●</span>+ <span style="color:#3fb39b">●</span>− <span style="color:#c56fc0">●</span>core, red outline = violation.`;
}

function renderLegend() {
  const items = [];
  if (S.colorMode === 'status') Object.entries(STATUS_COL).forEach(([k, c]) => items.push([c, k + (k === 'observed' ? ' (patient data)' : k === 'assumed' ? ' (template / synthetic)' : k === 'estimated' ? ' (reconstruction)' : k === 'proposed' ? ' (scaffold design)' : '')]));
  else {
    items.push([COL.bone, 'Bone (distal femur, proximal tibia)'], [COL.cartilage, 'Articular cartilage'], [COL.footprintAM, 'AM bundle: footprint, fascicles'], [COL.footprintPL, 'PL bundle: footprint, fascicles'],
      [COL.retained, 'Retained tissue surface (remnants)'], [COL.injury, 'Missing / damaged tissue (injury region)'], [COL.reference, 'Intact reference estimate'], [COL.repair, 'Repair space (permitted scaffold volume)'],
      [COL.violation, 'Fit violation: strand outside its repair space, in bone, or touching a strand / retained fascicle'], [COL.shortfall, 'Closer than the requested clearance (warning)']);
    if (S.scene.objects.some(o => isLayer(o))) items.push([LCOL.sheath, 'L5 outer sheath (translucent)'], [LCOL.iface, 'L5 inter-bundle zone (translucent band; no wall, not load-bearing)'],
      [LCOL.braidAM, 'L4 braided mantle, AM-like region'], [LCOL.braidPL, 'L4 braided mantle, PL-like region'],
      [LCOL.chAM, 'L3 hollow channels (AM-like; PL-like bluish, shared lilac); dark inside = lumen'], [LCOL.coat, 'L1 coated lumen (teal)'], [LCOL.groove, 'L2 groove paths (representative)'],
      [0xc58fd8, 'L6 flared ends: bone-facing → interface → ligament-facing composition'], [LCOL.fpBase, 'L6 flare base on the footprint (white = scaffold footprint outline)'],
      [LCOL.anchor, 'L6 bundle anchors (schematic markers at AM/PL attachment centroids)'], [LCOL.over, 'Channel that does not fit (requested beyond capacity)']);
    (S.scene.regions || []).forEach(r => { const mat = S.state.params['regions.' + r + '.material']; const m = S.reg.materials[mat]; if (m) items.push([new THREE.Color(...m.color).getHex(), 'Scaffold ' + r.replace('_', ' ') + ': ' + m.label]); });
  }
  $('legend').innerHTML = '<div class="leg">' + items.map(([c, t]) => `<i class="sw" style="background:#${new THREE.Color(c).getHexString()}"></i><span>${t}</span>`).join('') + '</div>' +
    '<p class="note">Darker native fascicles = higher regional damage estimate. Lighter = representative fibers. Status chips: observed (patient data), assumed (template or synthetic), estimated (reconstruction), proposed (design).</p>';
}

function renderChecks() {
  const ul = $('checks'); ul.innerHTML = '';
  let nf = 0, nw = 0;
  S.scene.checks.forEach(c => {
    if (c.status === 'fail') nf++; if (c.status === 'warn') nw++;
    const li = document.createElement('li'); li.className = c.status === 'not_evaluated' ? '' : c.status;
    const val = c.value == null ? '' : Array.isArray(c.value) ? ' [' + c.value.map(v => typeof v === 'number' ? v.toFixed(2) : v).join(', ') + ']' : typeof c.value === 'number' ? ' = ' + c.value.toFixed(3) + ' ' + (c.units || '') : '';
    li.innerHTML = `<b>${c.label}</b>: ${c.status.replace('_', ' ')}${val}<span class="m">${c.message || ''}</span>`;
    if (c.params && c.params.length && !VIEW_MODE) li.title = 'Show the related controls'; li.onclick = () => focusParam(c.params[0]);
    ul.appendChild(li);
  });
  $('checkCount').textContent = nf ? nf + ' fail' : nw ? nw + ' warn' : 'ok'; $('checkCount').className = 'pill tiny ' + (nf ? 'fail' : nw ? 'warn' : 'ok');
  const m = S.scene.metrics;
  const lines = [`ligament length ${m.ligament_length_mm.toFixed(1)} mm · sagittal ${m.inclinations.sagittal_deg.toFixed(1)}° (ref 47.4) · coronal ${m.inclinations.coronal_deg.toFixed(1)}° (ref 75.3)`,
    `footprints: tibial ${m.footprint_area_mm2.tibial.toFixed(0)} mm² · femoral ${m.footprint_area_mm2.femoral.toFixed(0)} mm² · midsubstance ${m.midsubstance_area_mm2.toFixed(1)} mm²`,
    `fascicle groups ${m.fascicles.count} (AM ${m.fascicles.per_bundle[0]}, PL ${m.fascicles.per_bundle[1]}) d=${m.fascicles.diameter_mm.toFixed(2)} mm packing ${m.fascicles.packing.toFixed(2)} · physical fibers ≈ ${Math.round(m.fibers.fibers_total_physical).toLocaleString()}`,
    `crimp angle (derived): native ${m.crimp_angle_deg.native.toFixed(1)}°, scaffold ${m.crimp_angle_deg.scaffold.toFixed(1)}°`];
  if (m.injury) lines.push(`injury: ${m.injury.type} (${m.injury.involvement}) · stumps ≥ ${m.injury.tibial_stump_min_mm.toFixed(1)} / ${m.injury.femoral_stump_min_mm.toFixed(1)} mm · nominal gap ${m.injury.nominal_gap_mm.toFixed(1)} mm${m.injury.measured_gap_mm != null ? ' · measured ' + m.injury.measured_gap_mm.toFixed(2) + ' mm' : ''} · boundary: ${m.injury.boundary_source}`);
  if (m.length) lines.push(`length ${m.length.length_mm.toFixed(1)} mm (${m.length.mode}${m.length.solved ? ', anterior bow solved = ' + m.length.bow_anterior_mm.toFixed(2) + ' mm' : ''}) · attachment distance ${m.length.attachment_distance_mm.toFixed(1)} mm · locked: attachments, bones`);
  if (m.section_profile) { const sp = m.section_profile, k = Math.floor(sp.s.length / 2); lines.push(`mid section (from the contour): ${sp.width_mm[k].toFixed(1)} × ${sp.thickness_mm[k].toFixed(1)} mm, area ${sp.area_mm2[k].toFixed(1)} mm², aspect ${sp.aspect[k].toFixed(2)}, AM ${(100 * sp.am_fraction[k]).toFixed(0)}% (${sp.am_area_mm2[k].toFixed(1)} / ${sp.pl_area_mm2[k].toFixed(1)} mm²)`); }
  (m.bundles || []).forEach(b => lines.push(`  ${b.name}: ${b.fascicles} fascicle groups d=${b.fascicle_diameter_mm.toFixed(2)} mm, packing ${b.packing.toFixed(2)}, within-bundle rotation ${b.within_bundle_rotation_deg.toFixed(1)}° (outer fascicle angle ${b.outermost_fascicle_angle_deg.toFixed(1)}°)`));
  if (m.scaffold && m.scaffold.units) {
    lines.push(`scaffold ${m.scaffold.architecture} / ${m.scaffold.engagement}: ${m.scaffold.yarn_count} yarns in ${m.scaffold.units.length} repair unit(s); clearances boundary ${m.scaffold.clearances_mm.boundary} · interface ${m.scaffold.clearances_mm.interface} · yarn ${m.scaffold.clearances_mm.yarn} mm`);
    m.scaffold.units.forEach(u => lines.push(`  ${u.label}: ${u.start_mm.toFixed(1)}–${u.end_mm.toFixed(1)} mm, ${u.yarns} yarns, repair area ${u.repair_area_mm2.min.toFixed(1)}–${u.repair_area_mm2.max.toFixed(1)} mm², bone: ${Object.entries(u.anchored_to_bone).filter(e => e[1]).map(e => e[0]).join(', ') || 'no'}`));
    (m.scaffold.braid || []).forEach(b => b.layers.forEach(l => lines.push(`  braid ${b.unit} layer ${l.layer}: angle ${l.angle_deg.toFixed(1)}° · pitch ${l.pitch_mm.toFixed(1)} mm · perimeter ${l.perimeter_mm.toFixed(1)} mm · level-pass spacing ${l.level_pass_spacing_mm.toFixed(2)} mm · bend radius ≥ ${l.min_bend_radius_mm.toFixed(2)} mm`)));
    (m.scaffold.regions || []).forEach(r => lines.push(`  ${r.id} (${r.kind}): ${r.length_mm.toFixed(1)} mm · ${r.material} · Vf ${r.fiber_volume_fraction.toFixed(2)} · pores ≈ ${r.inter_fiber_pore_um.toFixed(0)} µm · yarn ${r.yarn_diameter_mm.toFixed(2)} mm`));
  }
  lines.push('Not evaluated: strength, degradation, healing, re-tear risk.');
  renderFitPanel();
  lines.push(`design hash ${S.scene.design_hash} · geometry hash ${S.scene.geometry_hash}`);
  $('metrics').textContent = lines.join('\n');
}
function focusParam(pid) {
  if (!pid || VIEW_MODE) return;
  showTab('design');
  const spec = S.regById[pid];
  if (spec && spec.owner_level) { if (S.levelSel !== spec.owner_level) selectLevel(spec.owner_level); $('pnlLevels').open = true;
    if ([4, 6].includes(spec.owner_level) && spec.region && S.levelRegion[spec.owner_level] !== 'both' && S.levelRegion[spec.owner_level] !== spec.region) { S.levelRegion[spec.owner_level] = spec.region; renderControls(); } }
  else $('pnlAnat').open = true;
  const div = document.querySelector(`#tab-design .ctl[data-pid="${pid}"]`); if (!div) return;
  let det = div.closest('details'); while (det) { det.open = true; det = det.parentElement && det.parentElement.closest('details'); }
  div.scrollIntoView({ block: 'center' }); div.classList.add('flash'); setTimeout(() => div.classList.remove('flash'), 1500);
}

function renderPerf() {
  const r = perf.rebuilds.slice(-5);
  const mem = ren.info.memory;
  $('perf').textContent = [`startup: ${perf.steps.map(s => s[0] + ' ' + s[1] + ' ms').join(' · ')}`,
    `last rebuilds (server ms / 3D ms / reused+built): ${r.map(x => x.quality[0] + ' ' + x.server_ms + '/' + x.geometry_ms + (x.reuse ? ' ' + x.reuse.reused + '+' + x.reuse.built : '')).join(', ')}`,
    `tube worker: ${S.workerOK ? 'active (large tubes built off the main thread)' : 'fallback (in-page builder)'} · materials cached: ${S.matCache.size}`,
    `frame rate ≈ ${perf.fps.toFixed(0)} fps · GPU geometries ${mem.geometries} · triangles last frame ${ren.info.render.triangles.toLocaleString()}`].join('\n');
}

// ------------------------------------------------------------------ data & history tab
function renderData() {
  const st = S.state, c = st.case || {};
  $('caseLabel').textContent = c.label || c.id || 'design';
  $('synthBadge').hidden = !c.synthetic;
  const imp = st.imported;
  $('caseInfo').innerHTML = `<table class="t"><tr><th>Case</th><td>${c.id || ''}</td></tr><tr><th>Synthetic</th><td>${c.synthetic ? 'yes — not patient data' : 'no'}</td></tr>` +
    `<tr><th>Source</th><td>${imp && imp.meta.source ? (imp.meta.source.model_name + ' ' + imp.meta.source.model_version + ' run ' + imp.meta.source.run_id) : 'engine defaults / literature template'}</td></tr>` +
    `<tr><th>Saved link id</th><td class="mono">${S.stateId || '(saving…)'}</td></tr></table>`;
  const rows = [];
  for (const pid of Object.keys(S.regById)) {
    const cur = st.params[pid], base = imp ? imp.params[pid] : S.regById[pid].default;
    if (JSON.stringify(cur) !== JSON.stringify(base)) rows.push(`<tr><td>${S.regById[pid].label}<br><span class="muted mono">${pid}</span></td><td class="mono">${fmtV(base)}</td><td class="mono">${fmtV(cur)}</td><td><button data-reset="${pid}">reset</button></td></tr>`);
  }
  $('diff').innerHTML = rows.length ? `<table class="t"><tr><th>Parameter</th><th>${imp ? 'Imported' : 'Default'}</th><th>Current</th><th></th></tr>${rows.join('')}</table>` : '<p class="note">No differences.</p>';
  $('diff').querySelectorAll('button[data-reset]').forEach(b => b.onclick = () => { resetParam(b.dataset.reset); renderData(); });
  const rep = imp && imp.meta.report;
  $('importReport').innerHTML = rep ? `mapped ${rep.mapped.length} values · missing: ${rep.missing.join(', ') || 'none'} · unsupported: ${rep.unsupported.join(', ') || 'none'}<br>${(rep.warnings || []).map(w => '⚠ ' + w).join('<br>')}` : 'No MRI-analysis import in this design.';
  const counts = {}; Object.values(st.provenance).forEach(p => { counts[p.status] = (counts[p.status] || 0) + 1; });
  $('importReport').innerHTML += '<p class="note">Parameter status: ' + Object.entries(counts).map(([k, v]) => chipText(k) + ' ' + v).join(' ') + '</p>';
  $('history').innerHTML = st.history.slice(-60).reverse().map(h => `${(h.t || '').slice(11, 19)} ${h.action} ${h.param || h.case || ''} ${h.to !== undefined ? '→ ' + fmtV(h.to) : ''}`).join('<br>') || 'no edits yet';
}
const chipText = k => `<span class="chip ${k}">${k}</span>`;
function fmtV(v) { if (v == null) return '—'; if (typeof v === 'number') return +v.toFixed(3); const s = JSON.stringify(v); return s.length > 60 ? s.slice(0, 57) + '…' : s; }
function download(name, text, type) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: type || 'application/json' })); a.download = name; a.click(); }
$('btnHistory').onclick = () => download('history.json', JSON.stringify({ case: S.state.case, imported: S.state.imported && S.state.imported.meta, history: S.state.history }, null, 1));
$('btnDesign').onclick = () => download('design.json', JSON.stringify(S.state, null, 1));

// ------------------------------------------------------------------ cases & import
async function openCase(id) {
  progress('Opening case', [id]);
  try {
    const kind = (S.cases || []).find(c => c.id === id);
    const j = kind && kind.kind === 'patient' ? await api('/api/v2/import', { case: id, mode: 'replace' }) : await api('/api/v2/new', { case: id });
    S.state = j.state; S.firstFramed = false; S.selection = null;
    if (j.report) S.lastReport = j.report;
    afterLoad(); persist(); $('caseDlg').close && $('caseDlg').open && $('caseDlg').close();
  } catch (e) { showErr('Could not open the case', e.message, [['Open the demo case', () => openCase('demo_complete')]]); }
}
async function renderCases() {
  const j = await api('/api/v2/cases'); S.cases = j.cases;
  const box = $('caseList'); box.innerHTML = '';
  j.cases.forEach(c => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = c.label;
    b.onclick = async () => {
      const mode = document.querySelector('input[name=mode]:checked').value;
      if (c.kind === 'patient' && mode === 'merge') await importBody({ case: c.id }); else openCase(c.id);
    };
    box.appendChild(b);
  });
}
async function importBody(body) {
  const mode = document.querySelector('input[name=mode]:checked').value;
  body.mode = mode; body.current = mode === 'merge' ? S.state : null;
  $('importMsg').textContent = 'importing…';
  try {
    const j = await api('/api/v2/import', body);
    $('importMsg').textContent = `imported: ${j.report.mapped.length} values mapped, ${j.report.missing.length} missing, ${j.report.unsupported.length} unsupported`;
    if (j.conflicts && j.conflicts.length) await resolveConflicts(j.state, j.conflicts);
    S.state = j.state; S.firstFramed = false; afterLoad(); persist(); $('caseDlg').close();
  } catch (e) {
    const rep = e.body && e.body.report; $('importMsg').innerHTML = '✖ ' + e.message + (rep && rep.errors ? '<br>' + rep.errors.map(x => '• ' + x).join('<br>') : '');
  }
}
function resolveConflicts(st, conflicts) {
  return new Promise(res => {
    const box = $('conflicts');
    box.innerHTML = '<p class="note">Your manual values are kept unless you choose the new upstream value.</p>' + conflicts.map((c, i) =>
      `<div class="row"><b style="flex:1">${(S.regById[c.param] || {}).label || c.param}</b><label><input type="radio" name="c${i}" value="mine" checked> keep mine (${fmtV(c.yours)})</label><label><input type="radio" name="c${i}" value="up"> use new (${fmtV(c.new_upstream)})</label></div>`).join('');
    const dlg = $('conflictDlg'); dlg.showModal();
    dlg.onclose = () => { conflicts.forEach((c, i) => { const v = box.querySelector(`input[name=c${i}]:checked`).value; if (v === 'up') { st.params[c.param] = c.new_upstream; st.provenance[c.param] = st.imported.provenance[c.param]; st.history.push({ action: 'conflict_take_upstream', param: c.param, to: c.new_upstream }); } }); res(); };
  });
}
$('btnDoImport').onclick = async (ev) => {
  ev.preventDefault();
  const files = [...$('fileInput').files];
  const dfile = $('designInput').files[0];
  if (dfile) {
    try { const st = JSON.parse(await dfile.text()); if (!st.params) throw new Error('not a design.json'); S.state = st; S.firstFramed = false; afterLoad(); persist(); $('caseDlg').close(); }
    catch (e) { $('importMsg').textContent = '✖ ' + e.message; }
    return;
  }
  const doc = files.find(f => f.name.endsWith('.json'));
  if (!doc) { $('importMsg').textContent = 'Select the input JSON (and its referenced files).'; return; }
  const assets = {};
  for (const f of files) if (f !== doc) { const buf = new Uint8Array(await f.arrayBuffer()); let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000)); assets[f.name] = btoa(s); }
  let document_;
  try { document_ = JSON.parse(await doc.text()); } catch (e) { $('importMsg').textContent = '✖ ' + doc.name + ' is not valid JSON'; return; }
  await importBody({ document: document_, assets });
};

// ------------------------------------------------------------------ level detail (magnified) and section inset
const DETAIL_CRUMB = {
  1: ['Scaffold', '{bundle} bundle', 'Core channel', 'Inner wall', 'Coating + nanotexture'],
  2: ['Scaffold', '{bundle} bundle', 'Core channel', 'Inner wall', 'Microgrooves'],
  3: ['Scaffold', '{bundle} bundle', 'Core channel', 'Lumen + wall'],
  4: ['Scaffold', '{bundle} bundle', 'Braided mantle', 'Strand'],
  5: ['Scaffold', 'Sheath', 'Wall + pores'],
  6: ['Scaffold', '{end} insertion', 'Flare', 'Graded interface'],
};
async function openLevelDetail(n, quiet) {
  if (!n) { $('lvlHint').textContent = 'Open a level first; the magnified detail shows that level at true scale.'; return; }
  $('detailWrap').hidden = false; if (!detail.ren) detail.init();
  S.detailLevel = n;
  const reg = n === 4 ? (S.levelRegion[4] === 'pl' ? 'pl' : 'am') : n === 6 ? (S.levelRegion[6] === 'fem' ? 'fem' : 'tib') : null;
  const bundle = n === 4 ? (reg === 'pl' ? 'PL' : 'AM') : 'AM/PL';
  const end = reg === 'fem' ? 'Femoral' : 'Tibial';
  $('detailCrumb').textContent = (DETAIL_CRUMB[n] || []).join(' → ').replace('{bundle}', bundle).replace('{end}', end);
  const L = S.reg.levels.find(x => x.n === n);
  $('detailNote').textContent = `Level ${n} · ${L.name}${reg ? ' · ' + REG_LABEL[reg] : ''}` + (quiet ? '' : ' — loading…');
  const my = (S.detSeq = (S.detSeq || 0) + 1);
  try {
    const j = await api('/api/v2/level_detail', { state: S.state, level: n, region: reg, relief_x: Number(S.reliefX || 1) });
    if (my !== S.detSeq) return;
    if (detail.obj) { detail.scene.remove(detail.obj); disposeObj(detail.obj); }
    const grp = new THREE.Group();
    j.parts.forEach(pt => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(b64(pt.v, Float32Array), 3)); g.setIndex(new THREE.BufferAttribute(b64(pt.f, Uint32Array), 1));
      let vc = false;
      if (pt.vcol) { const c8 = b64(pt.vcol, Uint8Array), c = new Float32Array(c8.length); for (let i = 0; i < c8.length; i++) c[i] = c8[i] / 255; g.setAttribute('color', new THREE.BufferAttribute(c, 3)); vc = true; }
      g.computeVertexNormals();
      const m = new THREE.MeshStandardMaterial({ color: vc ? 0xffffff : new THREE.Color(...pt.color), vertexColors: vc, roughness: 0.75, side: THREE.DoubleSide,
        transparent: pt.opacity < 1, opacity: pt.opacity, depthWrite: pt.opacity >= 1 });
      const mesh = new THREE.Mesh(g, m); mesh.name = pt.name; grp.add(mesh);
    });
    if (j.up !== 'y') grp.rotation.x = -Math.PI / 2;      // engine patches are z-up; the end strip is already y-up
    detail.obj = grp; detail.scene.add(grp); grp.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(grp), ce = box.getCenter(new THREE.Vector3()), rr = box.getSize(new THREE.Vector3()).length() / 2;
    if (!quiet || !S.detailFramed || S.detailFramed !== n) {
      // look into the open lumen (L2/L3), down onto patches (L1/L5), obliquely otherwise
      const dir = { 1: [0.5, 1.3, 0.7], 2: [-1.4, 0.7, 0.9], 3: [0.45, 1.5, 0.85], 5: [0.6, 1.3, 0.9], 6: [1.8, 0.75, 1.5] }[n] || [1.1, 1.0, 1.5];
      detail.cam.position.copy(ce).add(new THREE.Vector3(...dir).multiplyScalar(rr)); detail.cam.near = rr / 200; detail.cam.far = rr * 60; detail.cam.updateProjectionMatrix();
      detail.ctl.target.copy(ce); detail.ctl.update(); S.detailFramed = n;
    }
    S.detailInfo = j; $('detailNote').textContent = `Level ${n} · ${L.name}${reg ? ' · ' + REG_LABEL[reg] : ''}`;
    updateMag(); invalidate();
  } catch (e) { $('detailNote').textContent = 'Detail failed: ' + e.message; }
}
function updateMag() {
  // enlargement of the inset relative to the main view (pixels per mm at each orbit target)
  const j = S.detailInfo; if (!j || !detail.cam) return;
  const pm = ren.domElement.clientHeight / (2 * cam.position.distanceTo(ctl.target) * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)));
  const host = $('detailView');
  const pd = host.clientHeight / (2 * detail.cam.position.distanceTo(detail.ctl.target) * Math.tan(THREE.MathUtils.degToRad(detail.cam.fov / 2)));
  const mag = pd / Math.max(pm, 1e-9);
  const w = j.window_mm < 0.1 ? (j.window_mm * 1000).toFixed(2) + ' µm' : j.window_mm < 1 ? (j.window_mm * 1000).toFixed(0) + ' µm' : j.window_mm.toFixed(1) + ' mm';
  const txt = j.relief_x > 1
    ? `Illustrative magnification — not to scale · relief drawn ×${j.relief_x} · window ≈ ${w}`
    : `Magnified ×${mag >= 100 ? Math.round(mag).toLocaleString() : mag.toFixed(1)} vs the main view · window ≈ ${w} · true proportions inside the inset`;
  const el = $('detailMag'); if (el.dataset.mag !== txt) { el.dataset.mag = txt; el.innerHTML = `<b>${txt}</b><br><span style="color:#b9c8c8">${j.note}</span>`; }
  // scale bar for the inset (actual feature dimensions at the inset's orbit target)
  let bar = 0.001;
  for (const c of [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]) if (c * pd <= 110) bar = c;
  $('detailScaleLine').style.width = Math.max(8, bar * pd) + 'px';
  $('detailScaleTxt').textContent = (bar < 1 ? (bar * 1000).toFixed(bar < 0.01 ? 1 : 0) + ' µm' : bar + ' mm') + ' (at inset centre)';
}
function openLvSection(force) {
  const w = $('lvSecWrap');
  w.hidden = force ? false : !w.hidden;
  $('btnLvSec').classList.toggle('on', !w.hidden);
  if (w.hidden) return;
  const d = S.state.display;
  if ((d.section_kind || 'ligament') !== 'ligament') { d.section_kind = 'ligament'; renderSectionControls(); }
  $('lvSecPos').value = d.section_s; $('lvSecVal').textContent = Number(d.section_s).toFixed(2);
  requestSection();
}
$('lvSecClose').onclick = () => { $('lvSecWrap').hidden = true; $('btnLvSec').classList.remove('on'); };
$('lvSecPos').oninput = e => { S.state.display.section_s = Number(e.target.value); $('lvSecVal').textContent = Number(e.target.value).toFixed(2); updateSlab(); };
$('lvSecPos').onchange = e => { S.state.display.section_s = Number(e.target.value); persist(); requestSection(); renderSectionControls(); };
$('lvSecSlab').onchange = e => { S.slab = S.slab || { on: false, width: 1.2 }; S.slab.on = e.target.checked; const c = $('slabOn'); if (c) c.checked = S.slab.on; rebuildSceneObjects(); };
$('btnLvSec').onclick = () => openLvSection(false);
$('btnLvDetail').onclick = () => openLevelDetail(S.levelSel);
function setLevelMode(m) {
  S.levelMode = m;
  document.querySelectorAll('#lvlbar [data-lvmode]').forEach(x => x.classList.toggle('on', x.dataset.lvmode === m));
}
document.querySelectorAll('#lvlbar [data-lvmode]').forEach(b => b.onclick = () => {
  setLevelMode(b.dataset.lvmode);
  selectLevel(S.levelSel);
});

// ------------------------------------------------------------------ tabs, buttons, boot
function showTab(t) { document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === t)); document.querySelectorAll('.tab').forEach(d => d.classList.toggle('on', d.id === 'tab-' + t)); if (t === 'data') renderData(); }
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => showTab(b.dataset.tab));
document.querySelectorAll('#viewbar button[data-cam]').forEach(b => b.onclick = () => setCam(b.dataset.cam === 'fit' ? 'q34' : b.dataset.cam));
$('cutQuick').onchange = e => { S.cut.on = e.target.checked; const c = $('cutOn'); if (c) c.checked = S.cut.on; updateClip(); };
$('paramSearch').oninput = () => renderControls();
$('btnResetAll').onclick = () => { if (confirm('Reset every parameter to its imported value (or default)?')) resetAll(); };
$('btnUndo').onclick = doUndo;
$('btnRedo').onclick = doRedo;
$('rgAll').onclick = () => gotoRegion('all');
$('rgAM').onclick = () => gotoRegion('am');
$('rgPL').onclick = () => gotoRegion('pl');
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'z' && !/^(TEXT|SEARCH)/.test((e.target.type || '').toUpperCase())) { e.preventDefault(); doUndo(); }
  else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'y' || (e.shiftKey && e.key.toLowerCase() === 'z'))) { e.preventDefault(); doRedo(); }
});
$('btnViewAnat').onclick = () => setBuilder(false);
$('btnViewBuilder').onclick = () => setBuilder(true);
$('ovACL').onchange = applyAclOverlay;
$('ovACLOp').oninput = () => { if ($('ovACL').checked) applyAclOverlay(); };
$('ovBone').onchange = applyBoneOverlay;
$('btnLongCut').onclick = toggleLongCut;
$('explodeRange').oninput = e => { setDisplay('exploded', Number(e.target.value)); applyExploded(); };
$('btnFitScaf').onclick = fitScaffold;
$('btnBuilderReset').onclick = restoreBuilderView;
$('tglLabels').onchange = e => { S.labelsOn = e.target.checked; refreshLabelAnchors(); invalidate(); };
$('btnCompFocus').onclick = () => {
  const cs = S.compSel; if (!cs) return;
  if (cs.marker) { const mk = markersOf().find(m => m.id === cs.marker); if (mk) focusMarker(mk); }
  else if (cs.def) focusComp(compDef(cs.def));
};
$('btnCompRestore').onclick = restoreBuilderView;
$('reliefSel').onchange = e => { S.reliefX = Number(e.target.value); if (!$('detailWrap').hidden && S.detailLevel) openLevelDetail(S.detailLevel, true); };
$('btnL0Lock').onclick = () => { S.level0Locked = !S.level0Locked; refreshL0Lock(); updateControlState(); };
$('btnCases').onclick = async () => { await renderCases(); $('importMsg').textContent = ''; $('caseDlg').showModal(); };
$('btnLink').onclick = async () => { const id = await saveRemote(); if (id) { const u = location.origin + '/?state=' + id; try { await navigator.clipboard.writeText(u); $('timing').textContent = 'link copied: ' + u; } catch (e) { prompt('Link to this model:', u); } } };
$('btnNewTab').onclick = async () => { const id = await saveRemote(); if (id) window.open('/view?state=' + id, '_blank'); };
function waitForSettled(afterSeq, timeoutMs) {
  // resolves when the build scheduled after `afterSeq` has started and the
  // latest result applied (or the timeout elapsed)
  return new Promise((resolve) => {
    const t0 = performance.now();
    const tick = () => {
      if ((S.seq > afterSeq && !S.building && S.applied === S.seq) || performance.now() - t0 > (timeoutMs || 45000)) resolve();
      else setTimeout(tick, 120);
    };
    tick();
  });
}
$('btnExport').onclick = async () => {
  // Exports always use the latest completed, validated full design — never a
  // simplified preview or a stale model.
  $('btnExport').disabled = true; $('btnExport').textContent = 'Exporting…';
  try {
    if (S.dirty || S.building || !S.scene || S.scene.quality !== 'full') {
      $('timing').textContent = 'finalizing the model before export…';
      const s0 = S.seq;
      scheduleBuild('full', 0);
      await waitForSettled(s0);
    }
    const v = S.scene && S.scene.validity;
    if (!v || !v.valid) {
      $('timing').textContent = `export blocked: the design is not valid (${v ? v.fail : '?'} failed checks) — see Checks, or restore the last valid design`;
      showTab('checks');
      return;
    }
    if (S.scene.design_hash !== S.lastFullHash) {
      $('timing').textContent = 'export blocked: the model changed during finalizing — try Export again';
      return;
    }
    const r = await fetch('/api/v2/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ state: S.state }) });
    if (!r.ok) throw new Error((await r.json()).error);
    const b = await r.blob(); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'acl_scaffold_' + r.headers.get('X-Design-Hash') + '.zip'; a.click();
    $('timing').textContent = 'exported validated design ' + r.headers.get('X-Design-Hash');
  } catch (e) { $('timing').textContent = 'export failed: ' + e.message; }
  $('btnExport').disabled = false; $('btnExport').textContent = 'Export';
};
if (VIEW_MODE) { $('btnNewTab').textContent = 'Open designer'; $('btnNewTab').onclick = () => window.open('/?state=' + (S.stateId || ''), '_blank'); $('btnCases').hidden = true; document.querySelector('#tabs button[data-tab=design]').hidden = true; showTab('inspect'); }

function afterLoad() {
  S.undoStack.length = 0; S.redoStack.length = 0; S.gesture = null; refreshUndoButtons();
  const st = S.state;
  st.display = Object.assign({}, S.reg.display_defaults, st.display || {});
  st.history = st.history || []; st.provenance = st.provenance || {};
  if (st.params['scaffold.engagement'] === 'sleeve') { st.params['scaffold.engagement'] = 'abut'; st.history.push({ action: 'migrate', param: 'scaffold.engagement', from: 'sleeve', to: 'abut', note: 'sleeve (scaffold outside the ligament) was retired', t: new Date().toISOString() }); }
  for (const p of S.reg.params) if (!(p.id in st.params)) { st.params[p.id] = JSON.parse(JSON.stringify(p.default)); st.provenance[p.id] = st.provenance[p.id] || { status: 'default', source: 'engine default (added in a newer version)' }; }
  renderControls(); renderDisplayControls(); renderClipControls(); renderData();
  build('full');
}

async function boot() {
  progress('Starting', ['3D library loaded', 'loading parameter registry…']);
  mark('3D library loaded');
  init3();
  try { const v = JSON.parse(localStorage.getItem('acl.v2.view') || 'null'); if (v) { S.vis = v.vis || {}; S.opac = v.opac || {}; } } catch (e) { /* ignore */ }
  S.reg = await api('/api/v2/registry'); S.reg.params.forEach(p => (S.regById[p.id] = p)); mark('registry');
  S.cases = (await api('/api/v2/cases')).cases;
  const id = new URL(location.href).searchParams.get('state');
  if (id) {
    progress('Restoring the saved model', ['link id ' + id]);
    try { const j = await api('/api/v2/state?id=' + encodeURIComponent(id)); S.state = j.state; S.stateId = id; mark('state restored'); }
    catch (e) {
      let local = null; try { local = JSON.parse(localStorage.getItem('acl.v2.current') || 'null'); } catch (x) { /* ignore */ }
      showErr('This saved model is not available', `The link refers to model "${id}", which this server does not have (${e.message}). ` +
        'It may come from another computer, or the state store was cleared.', [
        ['Open the synthetic demo', () => { history.replaceState(null, '', location.pathname); openCase('demo_complete'); }],
        ...(local ? [['Open my last design in this browser', () => { history.replaceState(null, '', location.pathname); S.state = local; afterLoad(); persist(); }]] : []),
        ['Choose a case…', async () => { await renderCases(); $('caseDlg').showModal(); }]]);
      return;
    }
  } else {
    let local = null; try { local = JSON.parse(localStorage.getItem('acl.v2.current') || 'null'); } catch (e) { /* ignore */ }
    if (local && local.params) S.state = local;
    else { const j = await api('/api/v2/new', { case: 'demo_complete' }); S.state = j.state; }
  }
  afterLoad(); persist();
}
function setSceneBackground(hex) { if (scene) { scene.background = new THREE.Color(hex); invalidate(); } }
window.__app = { gestureStart, applySnap, afterEdit, renderControls, DEPENDENT, derivedValue, setDisplay, setLevelMode, selectLevel, isolateLevel, focusLevel,
  applyAclOverlay, applyBoneOverlay, saveView, refreshLevelView, renderObjList, setSceneBackground, invalidate, COMP_DEFS, toggleLongCut, updateClip,
  refreshUndoButtons, resetParam, provOf, renderClipControls, renderSectionControls, updateSlab, matOptsFor, setClipPlanes, renderDisplayControls, rebuildSceneObjects, applyExploded, clearMatCache, renderLegend, renderData,
  cam: () => cam, ctl: () => ctl, root: () => root, objects: () => S.objects, renderSection, requestSection, S, setParam, setCam, build, perf, openCase, openDetail, showTab, scene: () => scene, renderer: () => ren, saveRemote, showGuide, focusGuide, highlightForParam, doUndo, doRedo, waitForSettled, gestureCommit,
  setBuilder, selectComponent, fitScaffold, frameBox, toggleComp, isolateComp, focusComp, restoreBuilderView, renderCompTree, refreshObject, compDef };
boot().catch(e => showErr('Could not start the designer', String(e && e.stack || e), [['Reload', () => location.reload()]]));
