// Imaging state and MPR views. One controller (IM) owns loaded volumes, masks, display settings, crosshairs,
// annotations and the active tool; every MPR pane is a view of it. Display settings never modify voxel data.
import * as M from './volmath.js';

const DTYPES = { uint8: Uint8Array, int16: Int16Array, uint16: Uint16Array, float32: Float32Array };
export const TOOLS = [
  ['crosshair', 'Crosshair', 'C', 'Click or drag to move the linked crosshair'],
  ['pan', 'Pan', 'P', 'Drag to pan'],
  ['zoom', 'Zoom', 'Z', 'Drag up/down to zoom'],
  ['window', 'Window/level', 'W', 'Drag: horizontal = width (contrast), vertical = level (brightness)'],
  ['brush', 'Brush', 'B', 'Paint the active label on this slice'],
  ['erase', 'Erase', 'E', 'Erase labels on this slice'],
  ['polygon', 'Polygon', 'G', 'Click vertices, double-click or Enter to fill the active label'],
  ['landmark', 'Landmark', 'L', 'Click to place an anatomical landmark'],
  ['distance', 'Distance', 'D', 'Click two points to measure a distance'],
];
const EDIT_TOOLS = new Set(['brush', 'erase', 'polygon']);
const MEASURE_TOOLS = new Set(['landmark', 'distance']);
export const NEW_LABELS = ['ACL', 'tear region', 'femur', 'tibia', 'fibula', 'PCL', 'meniscus', 'effusion'];

async function j(path, body, signal) {
  const opt = body === undefined ? { signal } : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal };
  const r = await fetch(path, opt);
  const out = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(out.error || 'HTTP ' + r.status); e.body = out; throw e; }
  return out;
}
export const api = j;

class Emitter {
  constructor() { this.h = new Map(); }
  on(t, f) { (this.h.get(t) || this.h.set(t, []).get(t)).push(f); }
  emit(t, d) { (this.h.get(t) || []).forEach(f => { try { f(d); } catch (e) { console.error(e); } }); }
}

export class Imaging extends Emitter {
  constructor() {
    super();
    this.caseId = null; this.manifest = null; this.vols = new Map(); this.masks = new Map();
    this.disp = new Map(); this.viewDisp = new Map(); this.cross = new Map(); this.crossVisible = true; this.linkOff = new Set();
    this.activeMask = new Map(); this.overlay = { visible: true, opacity: 0.5, labels: new Map() }; this.annotationsVisible = true;
    this.edit = null; this.activeLabel = 1; this.brush = 1.5; this.tool = 'crosshair';
    this.selectedAnn = null; this.measureKind = 'acl_length'; this.landmarkKind = 'femoral_attachment';
    this.views = new Map(); this.activeVol = null; this.vocab = null; this.jobs = new Map();
  }
  // ---------------------------------------------------------------- case
  async openCase(id) {
    for (const v of this.vols.values()) if (v.abort) v.abort.abort();
    this.caseId = id; this.vols.clear(); this.masks.clear(); this.disp.clear(); this.cross.clear(); this.activeMask.clear(); this.edit = null;
    await this.refresh();
    try { localStorage.setItem('ws.case', id); } catch (e) { /* ignore */ }
    const first = this.imageVolumes()[0];
    this.activeVol = first ? first.id : null;
    this.emit('case');
  }
  async refresh() {
    this.manifest = await j('/api/img/case?case=' + encodeURIComponent(this.caseId));
    for (const [vid, meta] of Object.entries(this.manifest.volumes)) {
      const v = this.vols.get(vid);
      if (v) Object.assign(v, meta, { ainv: null }); else this.vols.set(vid, Object.assign({ loaded: false, data: null }, meta));
    }
    for (const vid of [...this.vols.keys()]) if (!this.manifest.volumes[vid]) this.vols.delete(vid);
    for (const mid of [...this.masks.keys()]) if (!this.manifest.masks[mid]) this.masks.delete(mid);
    for (const [vid] of this.vols) if (!this.activeMask.has(vid)) {
      const ms = this.masksFor(vid);
      if (ms.length) this.activeMask.set(vid, ms[ms.length - 1].id);
    }
    this.emit('manifest');
  }
  imageVolumes() { return [...this.vols.values()].filter(v => v.role !== 'mask'); }
  masksFor(vid) { return Object.values((this.manifest && this.manifest.masks) || {}).filter(m => m.volume === vid).sort((a, b) => (a.created < b.created ? -1 : 1)); }
  annotationsFor(vid) { return ((this.manifest && this.manifest.annotations) || []).filter(a => !vid || a.volume === vid); }
  vol(vid) { return this.vols.get(vid); }
  // ---------------------------------------------------------------- loading (progressive, cancellable, cached)
  async load(vid) {
    const v = this.vols.get(vid);
    if (!v || v.loaded) return v;
    if (v.loading) return v.loading;
    v.abort = new AbortController(); v.progress = 0;
    v.loading = (async () => {
      const t0 = performance.now();
      const r = await fetch(`/api/img/volume?case=${this.caseId}&id=${vid}`, { signal: v.abort.signal });
      if (!r.ok) throw new Error('volume download failed: HTTP ' + r.status);
      const total = Number(r.headers.get('Content-Length')) || 0, rd = r.body.getReader(), parts = []; let got = 0;
      for (;;) {
        const { done, value } = await rd.read();
        if (done) break;
        parts.push(value); got += value.length; v.progress = total ? got / total : 0; this.emit('progress', vid);
      }
      const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
      const T = DTYPES[v.dtype];
      if (!T) throw new Error('unsupported dtype ' + v.dtype);
      v.data = new T(buf.buffer);
      if (v.data.length !== v.shape[0] * v.shape[1] * v.shape[2]) throw new Error('volume size mismatch');
      v.loaded = true; v.loading = null; v.loadMs = Math.round(performance.now() - t0);
      v.checksum0 = M.checksum(v.data);
      v.missingSet = new Set((v.missing_slices || []).map(m => m.axis + ':' + m.index));
      if (!this.disp.has(vid)) this.disp.set(vid, this.initialDisp(v));
      if (!this.cross.has(this.groupKey(vid))) this.cross.set(this.groupKey(vid), this.centerWorld(v));
      const mid = this.activeMask.get(vid); if (mid) await this.loadMask(mid).catch(() => {});
      this.emit('loaded', vid);
      return v;
    })().catch(e => { v.loading = null; v.error = e.name === 'AbortError' ? 'cancelled' : e.message; this.emit('progress', vid); throw e; });
    return v.loading;
  }
  cancelLoad(vid) { const v = this.vols.get(vid); if (v && v.abort) v.abort.abort(); }
  async loadMask(mid) {
    if (this.masks.has(mid)) return this.masks.get(mid);
    const meta = this.manifest.masks[mid];
    const r = await fetch(`/api/img/mask?case=${this.caseId}&id=${mid}`);
    if (!r.ok) throw new Error('mask download failed');
    const m = { meta, data: new Uint8Array(await r.arrayBuffer()) };
    this.masks.set(mid, m); this.emit('mask', mid);
    return m;
  }
  // ---------------------------------------------------------------- display (window / level / colour map)
  initialDisp(v) {
    const w = v.initial_window || v.auto_window || { center: (v.range[0] + v.range[1]) / 2, width: v.range[1] - v.range[0] || 1 };
    return { center: w.center, width: Math.max(1e-6, w.width), invert: false, cmap: v.role === 'derived' ? 'viridis' : 'gray', linked: true,
      source: w.method || 'data range' };
  }
  dispFor(viewId, vid) {
    const d = this.disp.get(vid);
    if (d && !d.linked && this.viewDisp.has(viewId)) return this.viewDisp.get(viewId);
    return d;
  }
  setDisp(viewId, vid, patch) {
    const d = this.disp.get(vid); if (!d) return;
    if (!d.linked && viewId) { const own = Object.assign({}, this.viewDisp.get(viewId) || d, patch); this.viewDisp.set(viewId, own); }
    else Object.assign(d, patch);
    if ('linked' in patch && patch.linked) for (const [k, vw] of this.views) if (vw.vid === vid) this.viewDisp.delete(k);
    if ('linked' in patch && !patch.linked) for (const [k, vw] of this.views) if (vw.vid === vid) this.viewDisp.set(k, Object.assign({}, d));
    this.emit('display', vid);
  }
  resetWindow(viewId, vid, mode) {
    const v = this.vols.get(vid); const init = this.initialDisp(v);
    const w = mode === 'auto' ? (v.auto_window || init) : (v.initial_window || v.auto_window || init);
    this.setDisp(viewId, vid, { center: w.center, width: w.width, invert: false, ...(mode === 'auto' ? {} : { cmap: init.cmap }) });
  }
  // ---------------------------------------------------------------- crosshair (world RAS mm; linked by frame of reference)
  groupKey(vid) { const v = this.vols.get(vid); return v && v.frame_id && !this.linkOff.has(vid) ? 'f:' + v.frame_id : 'v:' + vid; }
  linkedWith(vid) { const k = this.groupKey(vid); return [...this.vols.values()].filter(v => v.id !== vid && this.groupKey(v.id) === k); }
  centerWorld(v) { return M.voxelToWorld(v, v.shape.map(n => (n - 1) / 2).map(Math.round)); }
  crossWorld(vid) { return this.cross.get(this.groupKey(vid)); }
  crossVoxel(vid) { const v = this.vols.get(vid); return M.worldToVoxel(v, this.crossWorld(vid)); }
  setCross(vid, world, why) {
    const v = this.vols.get(vid);
    // keep the point inside this volume's grid
    const ijk = M.worldToVoxel(v, world).map((x, a) => Math.max(0, Math.min(v.shape[a] - 1, x)));
    this.cross.set(this.groupKey(vid), M.voxelToWorld(v, ijk));
    this.emit('cross', { vid, why });
  }
  moveCrossVoxels(vid, d) { const ijk = this.crossVoxel(vid).map((x, a) => Math.round(x) + (d[a] || 0)); this.setCross(vid, M.voxelToWorld(this.vols.get(vid), ijk), 'keys'); }
  centerCross(vid) { const v = this.vols.get(vid); if (v && v.loaded) this.setCross(vid, this.centerWorld(v), 'center'); }
  gotoWorld(world, vid) {
    vid = vid || this.activeVol; const v = this.vols.get(vid); if (!v) return false;
    if (v.loaded) this.setCross(vid, world, 'goto'); else this.load(vid).then(() => this.setCross(vid, world, 'goto'));
    return true;
  }
  gotoAnnotation(id) {
    const a = this.annotationsFor().find(x => x.id === (id || this.selectedAnn));
    if (!a) return { ok: false, why: 'No annotation is selected.' };
    const p = a.points_world.length === 2 ? a.points_world[0].map((x, i) => (x + a.points_world[1][i]) / 2) : a.points_world[0];
    this.selectedAnn = a.id; this.activeVol = a.volume;
    this.gotoWorld(p, a.volume); this.emit('selection');
    return { ok: true, annotation: a, world: p };
  }
  labelCentroid(vid, value) {
    const v = this.vols.get(vid), m = this.editData(vid);
    if (!m) return null;
    const [nx, ny] = v.shape; let n = 0, si = 0, sj = 0, sk = 0;
    for (let q = 0; q < m.length; q++) if (m[q] === value) { n++; si += q % nx; sj += Math.floor(q / nx) % ny; sk += Math.floor(q / (nx * ny)); }
    return n ? M.voxelToWorld(v, [si / n, sj / n, sk / n]) : null;
  }
  // ---------------------------------------------------------------- masks and editing
  maskMeta(vid) { const mid = this.activeMask.get(vid); return mid ? this.manifest.masks[mid] : null; }
  editData(vid) {
    if (this.edit && this.edit.vid === vid) return this.edit.data;
    const mid = this.activeMask.get(vid); const m = mid && this.masks.get(mid);
    return m ? m.data : null;
  }
  labelsFor(vid) {
    if (this.edit && this.edit.vid === vid) return this.edit.labels;
    const mm = this.maskMeta(vid); return mm ? mm.labels : [];
  }
  styleMap(vid) {
    const st = new Map();
    for (const l of this.labelsFor(vid)) {
      const o = this.overlay.labels.get(l.name) || { visible: true, opacity: 1 };
      st.set(Number(l.value), { rgb: M.hexRgb(l.color || '#f59e0b'), opacity: o.opacity, visible: o.visible });
    }
    return st;
  }
  async selectMask(vid, mid) {
    if (this.edit && this.edit.vid === vid && this.edit.dirty && !confirm('Discard unsaved edits?')) return;
    if (this.edit && this.edit.vid === vid) this.edit = null;
    if (mid) { await this.loadMask(mid); this.activeMask.set(vid, mid); } else this.activeMask.delete(vid);
    this.emit('overlay');
  }
  startEdit(vid) {
    if (this.edit && this.edit.vid === vid) return this.edit;
    const v = this.vols.get(vid); if (!v || !v.loaded) return null;
    const mid = this.activeMask.get(vid), base = mid && this.masks.get(mid);
    const labels = base ? JSON.parse(JSON.stringify(base.meta.labels)) : [{ value: 1, name: 'ACL', color: (this.vocab && this.vocab.label_colors.ACL) || '#f59e0b', source: 'manual' }];
    this.edit = { vid, parent: mid || null, data: base ? new Uint8Array(base.data) : new Uint8Array(v.shape[0] * v.shape[1] * v.shape[2]),
      labels, undo: [], redo: [], dirty: false, origin: base ? 'edited' : 'manual', stroke: null };
    if (!labels.some(l => Number(l.value) === this.activeLabel)) this.activeLabel = Number(labels[0].value);
    this.emit('edit');
    return this.edit;
  }
  addLabel(name) {
    const e = this.edit; if (!e) return null;
    const ex = e.labels.find(l => l.name === name); if (ex) { this.activeLabel = Number(ex.value); this.emit('edit'); return ex; }
    const value = Math.max(0, ...e.labels.map(l => Number(l.value))) + 1;
    const color = (this.vocab && this.vocab.label_colors[name]) || '#eab308';
    const l = { value, name, color, source: 'manual' }; e.labels.push(l); this.activeLabel = value; this.emit('edit');
    return l;
  }
  beginStroke() { if (this.edit) this.edit.stroke = new Map(); }
  paint(vid, idxs, value) {
    const e = this.edit; if (!e || e.vid !== vid) return;
    const s = e.stroke || (e.stroke = new Map());
    for (const q of idxs) { if (e.data[q] === value) continue; if (!s.has(q)) s.set(q, e.data[q]); e.data[q] = value; }
  }
  endStroke(value, label) {
    const e = this.edit; if (!e || !e.stroke || !e.stroke.size) { if (e) e.stroke = null; return; }
    const idx = Int32Array.from(e.stroke.keys()), old = Uint8Array.from(e.stroke.values());
    e.undo.push({ idx, old, neu: value, label }); if (e.undo.length > 100) e.undo.shift();
    e.redo.length = 0; e.stroke = null; e.dirty = true; this.emit('edit');
  }
  undoEdit() { const e = this.edit; const s = e && e.undo.pop(); if (!s) return false; for (let n = 0; n < s.idx.length; n++) e.data[s.idx[n]] = s.old[n]; e.redo.push(s); e.dirty = true; this.emit('edit'); return true; }
  redoEdit() { const e = this.edit; const s = e && e.redo.pop(); if (!s) return false; for (let n = 0; n < s.idx.length; n++) e.data[s.idx[n]] = s.neu; e.undo.push(s); e.dirty = true; this.emit('edit'); return true; }
  async saveEdit(note) {
    const e = this.edit; if (!e) return null;
    const q = new URLSearchParams({ case: this.caseId, volume: e.vid, origin: e.origin });
    if (e.parent) q.set('parent', e.parent);
    const r = await fetch('/api/img/mask_save?' + q, { method: 'POST', body: e.data,
      headers: { 'Content-Type': 'application/octet-stream', 'X-Labels': JSON.stringify(e.labels.map(l => ({ value: l.value, name: l.name, source: l.source || 'manual' }))), 'X-Note': note || '' } });
    const out = await r.json(); if (!r.ok) throw new Error(out.error);
    const vid = e.vid; this.edit = null;
    await this.refresh(); this.masks.set(out.mask.id, { meta: out.mask, data: e.data }); this.activeMask.set(vid, out.mask.id);
    this.emit('overlay');
    return out.mask;
  }
  discardEdit() { this.edit = null; this.emit('overlay'); }
  async review(mid, status) { await j('/api/img/review', { case: this.caseId, mask: mid, status }); await this.refresh(); this.emit('overlay'); }
  // ---------------------------------------------------------------- annotations
  async saveAnnotation(item) {
    const out = await j('/api/img/annotation', { case: this.caseId, item });
    await this.refresh(); this.selectedAnn = out.annotation.id; this.emit('annotations');
    return out.annotation;
  }
  async removeData(kind, id, label) {
    if (!confirm(`Remove ${label || kind} from this case? This deletes the stored data${kind === 'volume' ? ', its masks and annotations' : ''}.`)) return false;
    await j('/api/img/remove', { case: this.caseId, kind, id });
    if (kind === 'volume') { this.vols.delete(id); for (const [k, vw] of this.views) if (vw.vid === id) vw.setVolume(null); }
    if (kind === 'mask') { this.masks.delete(id); for (const [vid, mid] of this.activeMask) if (mid === id) this.activeMask.delete(vid); }
    await this.refresh(); this.emit('case');
    return true;
  }
  setTool(t) { if (!TOOLS.some(x => x[0] === t)) return false; this.tool = t; this.emit('tool'); return true; }
}

// ===================================================================== MPR view
export class MprView {
  constructor(IM, viewId, vid, planeKey) {
    this.IM = IM; this.id = viewId; this.vid = vid; this.planeKey = planeKey;
    this.zoom = null; this.pan = [0, 0]; this.poly = []; this.pending = null; this.hover = null;
    const el = this.el = document.createElement('div'); el.className = 'mpr'; el.tabIndex = 0;
    el.innerHTML = `<canvas class="mprc"></canvas>
      <div class="ori t"></div><div class="ori b"></div><div class="ori l"></div><div class="ori r"></div>
      <div class="hud tl"><b class="pl"></b> <span class="sl mono"></span><div class="toolchip"></div></div>
      <div class="hud tr"><span class="ovl"></span></div>
      <div class="hud bl mono"><span class="pos"></span></div>
      <div class="hud br"><div class="legend" hidden></div><span class="scale mono"></span></div>
      <div class="notice" hidden></div>
      <div class="loadbox" hidden><div class="lbt"></div><div class="bar"><i></i></div><button class="cancel">Cancel</button></div>
      <div class="empty" hidden>No volume in this pane. Choose a series in the imaging browser.</div>`;
    this.cv = el.querySelector('canvas'); this.ctx = this.cv.getContext('2d');
    this.img = document.createElement('canvas'); this.ov = document.createElement('canvas');
    new ResizeObserver(() => this.resize()).observe(el);
    el.querySelector('.cancel').onclick = () => IM.cancelLoad(this.vid);
    this.bind();
    for (const t of ['cross', 'display', 'loaded', 'overlay', 'mask', 'edit', 'annotations', 'tool', 'selection', 'manifest'])
      IM.on(t, () => this.draw());
    IM.on('progress', (vid) => { if (vid === this.vid) this.drawLoading(); });
    IM.views.set(viewId, this);
    this.setVolume(vid);
  }
  get v() { return this.IM.vols.get(this.vid); }
  get plane() { return M.planeOf(this.v, this.planeKey); }
  title() {
    const v = this.v; const pl = v ? M.planeOf(v, this.planeKey).name : this.planeKey;
    return v ? `${pl} · ${v.name.length > 22 ? v.name.slice(0, 21) + '…' : v.name}` : pl;
  }
  setVolume(vid) {
    this.vid = vid; this.zoom = null; this.pan = [0, 0];
    if (vid && this.v && !this.v.loaded) this.IM.load(vid).catch(() => {}).then(() => this.draw());
    this.draw();
  }
  resize() {
    const r = this.el.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    if (!r.width || !r.height) return;
    this.cv.width = Math.round(r.width * dpr); this.cv.height = Math.round(r.height * dpr);
    this.cv.style.width = r.width + 'px'; this.cv.style.height = r.height + 'px'; this.dpr = dpr; this.W = r.width; this.H = r.height;
    this.draw();
  }
  // ---------------------------------------------------------------- geometry: screen <-> plane image <-> voxel
  fit() {
    const { nu, nv, du, dv } = M.planeSize(this.v, this.plane);
    const su = this.v.spacing_known ? du : 1, sv = this.v.spacing_known ? dv : 1;
    this.zoom = 0.92 * Math.min(this.W / (nu * su), this.H / (nv * sv)); this.pan = [0, 0];
  }
  metrics() {
    const { nu, nv, du, dv } = M.planeSize(this.v, this.plane);
    const su = this.v.spacing_known ? du : 1, sv = this.v.spacing_known ? dv : 1;   // physical proportions: mm per pixel
    const w = nu * su * this.zoom, h = nv * sv * this.zoom;
    return { nu, nv, su, sv, w, h, ox: (this.W - w) / 2 + this.pan[0], oy: (this.H - h) / 2 + this.pan[1] };
  }
  screenToVoxel(sx, sy) {
    const m = this.metrics(); const X = (sx - m.ox) / (m.su * this.zoom), Y = (sy - m.oy) / (m.sv * this.zoom);
    return M.planeToVoxel(this.v, this.plane, X - 0.5, Y - 0.5, this.sliceIndex());
  }
  voxelToScreen(ijk) {
    const m = this.metrics(); const [x, y] = M.voxelToPlane(this.v, this.plane, ijk);
    return [m.ox + (x + 0.5) * m.su * this.zoom, m.oy + (y + 0.5) * m.sv * this.zoom];
  }
  sliceIndex() { const c = this.IM.crossVoxel(this.vid); const n = this.v.shape[this.plane.fixed]; return Math.max(0, Math.min(n - 1, Math.round(c[this.plane.fixed]))); }
  // ---------------------------------------------------------------- drawing
  drawLoading() {
    const v = this.v, lb = this.el.querySelector('.loadbox');
    if (!v || v.loaded) { lb.hidden = true; return; }
    lb.hidden = false;
    lb.querySelector('.lbt').textContent = v.error ? `Not loaded: ${v.error}` : `Loading ${v.name} — ${Math.round((v.progress || 0) * 100)}%`;
    lb.querySelector('.bar i').style.width = Math.round((v.progress || 0) * 100) + '%';
    lb.querySelector('.cancel').textContent = v.error ? 'Retry' : 'Cancel';
    lb.querySelector('.cancel').onclick = v.error ? () => { v.error = null; this.setVolume(this.vid); } : () => this.IM.cancelLoad(this.vid);
  }
  draw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this._draw(); });
  }
  _draw() {
    const el = this.el, v = this.v, ctx = this.ctx, IM = this.IM;
    el.querySelector('.empty').hidden = !!v;
    el.classList.toggle('tool-' + IM.tool, true); [...el.classList].filter(c => c.startsWith('tool-') && c !== 'tool-' + IM.tool).forEach(c => el.classList.remove(c));
    if (!this.W) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); ctx.fillStyle = '#0d1114'; ctx.fillRect(0, 0, this.W, this.H);
    this.drawLoading();
    if (!v || !v.loaded) { el.querySelectorAll('.ori,.hud span,.hud b').forEach(x => (x.textContent = '')); return; }
    if (!this.zoom) this.fit();
    const pl = this.plane, k = this.sliceIndex(), d = IM.dispFor(this.id, this.vid);
    const { nu, nv } = M.planeSize(v, pl);
    if (this.img.width !== nu || this.img.height !== nv) { this.img.width = this.ov.width = nu; this.img.height = this.ov.height = nv; this._id = null; }
    const ictx = this.img.getContext('2d'), id = ictx.createImageData(nu, nv);
    M.renderSlice(v, pl, k, d, id.data); ictx.putImageData(id, 0, 0);
    const m = this.metrics();
    ctx.imageSmoothingEnabled = false;                 // nearest neighbour: never imply detail the scan does not have
    ctx.drawImage(this.img, m.ox, m.oy, m.w, m.h);
    // overlay
    const mask = IM.overlay.visible ? IM.editData(this.vid) : null;
    if (mask) {
      const octx = this.ov.getContext('2d'), od = octx.createImageData(nu, nv);
      if (M.renderMaskSlice(v, pl, k, mask, IM.styleMap(this.vid), IM.overlay.opacity, od.data)) { octx.putImageData(od, 0, 0); ctx.drawImage(this.ov, m.ox, m.oy, m.w, m.h); }
    }
    this.drawAnnotations(ctx);
    if (this.poly.length) this.drawPoly(ctx);
    if (this.pending) this.drawPending(ctx);
    if (IM.crossVisible) this.drawCross(ctx);
    if (this.hover && EDIT_TOOLS.has(IM.tool) && IM.tool !== 'polygon') this.drawBrush(ctx);
    this.hud(k, d);
  }
  drawCross(ctx) {
    const [x, y] = this.voxelToScreen(this.IM.crossVoxel(this.vid));
    const linked = this.IM.linkedWith(this.vid).length > 0;
    ctx.save(); ctx.strokeStyle = linked ? 'rgba(80,200,255,.85)' : 'rgba(255,214,90,.9)'; ctx.lineWidth = 1;
    const g = 8;
    ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(x - g, y + 0.5); ctx.moveTo(x + g, y + 0.5); ctx.lineTo(this.W, y + 0.5);
    ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, y - g); ctx.moveTo(x + 0.5, y + g); ctx.lineTo(x + 0.5, this.H); ctx.stroke(); ctx.restore();
  }
  nearPlane(world) {
    const ijk = M.worldToVoxel(this.v, world), f = this.plane.fixed;
    return Math.abs(ijk[f] - this.sliceIndex()) <= 0.5;
  }
  drawAnnotations(ctx) {
    const IM = this.IM; if (!IM.annotationsVisible) return;
    for (const a of IM.annotationsFor(this.vid)) {
      const sel = a.id === IM.selectedAnn, col = a.status === 'reviewed' ? '#4ade80' : '#facc15';
      const pts = a.points_world.map(w => ({ s: this.voxelToScreen(M.worldToVoxel(this.v, w)), near: this.nearPlane(w) }));
      ctx.save(); ctx.strokeStyle = ctx.fillStyle = col; ctx.lineWidth = sel ? 2.5 : 1.5; ctx.font = '600 11px system-ui';
      if (a.type === 'distance') {
        ctx.setLineDash(pts.every(p => p.near) ? [] : [4, 4]);
        ctx.beginPath(); ctx.moveTo(...pts[0].s); ctx.lineTo(...pts[1].s); ctx.stroke(); ctx.setLineDash([]);
        pts.forEach(p => { ctx.beginPath(); ctx.arc(p.s[0], p.s[1], p.near ? 3.5 : 2, 0, 7); p.near ? ctx.fill() : ctx.stroke(); });
        const mx = (pts[0].s[0] + pts[1].s[0]) / 2, my = (pts[0].s[1] + pts[1].s[1]) / 2;
        const t = `${a.value} ${a.units === 'mm' ? 'mm' : 'vox'}${a.status === 'reviewed' ? ' ✓' : ''}`;
        ctx.fillStyle = 'rgba(0,0,0,.65)'; ctx.fillRect(mx + 6, my - 9, ctx.measureText(t).width + 8, 16); ctx.fillStyle = col; ctx.fillText(t, mx + 10, my + 3);
      } else if (a.type === 'landmark') {
        const p = pts[0]; if (!p.near && !sel) { ctx.restore(); continue; }
        ctx.beginPath(); ctx.arc(p.s[0], p.s[1], 5, 0, 7); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(p.s[0] - 8, p.s[1]); ctx.lineTo(p.s[0] + 8, p.s[1]); ctx.moveTo(p.s[0], p.s[1] - 8); ctx.lineTo(p.s[0], p.s[1] + 8); ctx.stroke();
        const t = (a.name || a.kind || 'landmark') + (a.status === 'reviewed' ? ' ✓' : '');
        ctx.fillStyle = 'rgba(0,0,0,.65)'; ctx.fillRect(p.s[0] + 9, p.s[1] - 18, ctx.measureText(t).width + 8, 16); ctx.fillStyle = col; ctx.fillText(t, p.s[0] + 13, p.s[1] - 6);
      }
      ctx.restore();
    }
  }
  drawPoly(ctx) {
    ctx.save(); ctx.strokeStyle = '#fff'; ctx.setLineDash([3, 3]); ctx.beginPath();
    this.poly.forEach((p, i) => { const s = this.voxelToScreen(p); i ? ctx.lineTo(...s) : ctx.moveTo(...s); });
    if (this.hover) ctx.lineTo(this.hover[0], this.hover[1]);
    ctx.stroke(); ctx.restore();
  }
  drawPending(ctx) {
    const s = this.voxelToScreen(this.pending); ctx.save(); ctx.strokeStyle = '#facc15'; ctx.beginPath(); ctx.arc(s[0], s[1], 4, 0, 7); ctx.stroke();
    if (this.hover) { ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(...s); ctx.lineTo(this.hover[0], this.hover[1]); ctx.stroke(); }
    ctx.restore();
  }
  drawBrush(ctx) {
    const m = this.metrics(), r = this.IM.brush * this.zoom * (this.v.spacing_known ? 1 : m.su);
    ctx.save(); ctx.strokeStyle = this.IM.tool === 'erase' ? '#f87171' : '#fff'; ctx.beginPath(); ctx.arc(this.hover[0], this.hover[1], Math.max(2, r), 0, 7); ctx.stroke(); ctx.restore();
  }
  hud(k, d) {
    const el = this.el, v = this.v, pl = this.plane, IM = this.IM, L = M.labelsFor(v, pl);
    el.querySelector('.ori.t').textContent = L ? L.top : ''; el.querySelector('.ori.b').textContent = L ? L.bottom : '';
    el.querySelector('.ori.l').textContent = L ? L.left : ''; el.querySelector('.ori.r').textContent = L ? L.right : '';
    el.querySelector('.pl').textContent = pl.name + (L && L.oblique ? ' (oblique grid)' : '');
    const { nfix, du, dv, dfix } = M.planeSize(v, pl);
    el.querySelector('.sl').textContent = `slice ${k + 1}/${nfix}` + (v.spacing_known ? ` · ${dfix.toFixed(2)} mm` : '');
    const t = (TOOLS.find(x => x[0] === IM.tool) || [])[1];
    el.querySelector('.toolchip').textContent = 'Tool: ' + t + (EDIT_TOOLS.has(IM.tool) && !(IM.edit && IM.edit.vid === this.vid) ? ' (starts an edit copy)' : '');
    el.querySelector('.toolchip').className = 'toolchip ' + (EDIT_TOOLS.has(IM.tool) ? 'edit' : MEASURE_TOOLS.has(IM.tool) ? 'measure' : 'nav');
    const w = IM.crossWorld(this.vid), c = IM.crossVoxel(this.vid).map(Math.round);
    const val = v.data[M.vidx(v.shape, ...c.map((x, a) => Math.max(0, Math.min(v.shape[a] - 1, x))))];
    el.querySelector('.pos').textContent = (v.orientation_known ? `RAS ${w.map(x => x.toFixed(1)).join(', ')} mm · ` : v.spacing_known ? '' : '') +
      `ijk ${c.join(', ')} · value ${Number.isInteger(val) ? val : val.toFixed(3)}`;
    // overlay status: where the mask came from and whether it was reviewed
    const mm = IM.maskMeta(this.vid), ed = IM.edit && IM.edit.vid === this.vid;
    el.querySelector('.ovl').textContent = !IM.overlay.visible ? 'overlays hidden' : ed ? `editing ${IM.edit.origin === 'edited' ? 'copy of prediction' : 'new mask'}${IM.edit.dirty ? ' (unsaved)' : ''}` :
      mm ? `${{ model: 'MODEL prediction', edited: 'edited', import: 'imported', manual: 'manual' }[mm.origin] || mm.origin} · ${mm.review.status}` : '';
    el.querySelector('.ovl').className = 'ovl ' + (mm && mm.origin === 'model' && !ed ? 'model' : mm && mm.review.status === 'reviewed' ? 'reviewed' : '');
    // notices: thick slices in reformatted planes, missing calibration/orientation
    const notes = [];
    const ratio = v.spacing_known ? Math.max(du, dv) / Math.min(du, dv) : 1;
    if (ratio >= 2.5) notes.push(`Reformatted plane: ${Math.max(du, dv).toFixed(1)} mm spacing limits detail ${du > dv ? 'left–right on screen' : 'up–down on screen'}; blocks are real voxels, not added detail.`);
    if (!v.orientation_known) notes.push('No orientation metadata: anatomical directions are not labelled.');
    if (!v.spacing_known) notes.push('Uncalibrated: no millimetre scale; proportions follow voxel counts.');
    if (v.missingSet && v.missingSet.size && v.missingSet.has(pl.fixed + ':' + k)) notes.push('This slice is missing from the series (blank).');
    const nb = el.querySelector('.notice'); nb.hidden = !notes.length; nb.textContent = notes.join(' ');
    // scale bar and legend
    const m = this.metrics();
    el.querySelector('.scale').textContent = v.spacing_known ? `${(this.W / (m.su * this.zoom) * m.su).toFixed(0)} mm across` : 'voxel units';
    const lg = el.querySelector('.legend');
    const showLegend = d.cmap !== 'gray' || v.role === 'derived';
    lg.hidden = !showLegend;
    if (showLegend) {
      const [lo, hi] = M.windowToRange(d.center, d.width), L2 = M.lut(d.cmap, d.invert);
      const stops = [0, 64, 128, 192, 255].map(i => `rgb(${L2[i * 3]},${L2[i * 3 + 1]},${L2[i * 3 + 2]}) ${i / 2.55}%`).join(',');
      const q = v.role === 'derived' ? `${v.quantity}${v.value_units ? ' (' + v.value_units + ')' : ''}` : 'MRI signal intensity (arbitrary units)';
      lg.innerHTML = `<div class="lgq">${q}</div><div class="lgbar" style="background:linear-gradient(90deg,${stops})"></div><div class="lgr mono"><span>${fmt(lo)}</span><span>${fmt(hi)}</span></div>` +
        (v.role === 'derived' ? '' : '<div class="lgn">Colour is a display choice, not a tissue or damage map.</div>');
    }
  }
  // ---------------------------------------------------------------- interaction
  bind() {
    const el = this.el, cv = this.cv;
    const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    let drag = null;
    cv.addEventListener('pointerdown', (e) => {
      if (!this.v || !this.v.loaded) return;
      el.focus(); cv.setPointerCapture(e.pointerId);
      const p = pos(e), IM = this.IM, t = e.button === 1 ? 'pan' : IM.tool;
      drag = { t, p0: p, pan0: this.pan.slice(), zoom0: this.zoom, disp0: Object.assign({}, IM.dispFor(this.id, this.vid)) };
      IM.activeVol = this.vid;
      if (t === 'crosshair') this.crossAt(p);
      else if (t === 'brush' || t === 'erase') { if (!IM.startEdit(this.vid)) { drag = null; return; } IM.beginStroke(); this.paintAt(p); }
      else if (t === 'polygon') { if (!IM.startEdit(this.vid)) return; if (e.detail >= 2 && this.poly.length >= 3) this.fillPoly(); else this.poly.push(this.screenToVoxel(...p)); this.draw(); }
      else if (t === 'landmark') this.placeLandmark(p);
      else if (t === 'distance') this.placeDistance(p);
    });
    cv.addEventListener('pointermove', (e) => {
      const p = pos(e); this.hover = p;
      if (drag) {
        const IM = this.IM, dx = p[0] - drag.p0[0], dy = p[1] - drag.p0[1];
        if (drag.t === 'crosshair') this.crossAt(p);
        else if (drag.t === 'pan') { this.pan = [drag.pan0[0] + dx, drag.pan0[1] + dy]; this.draw(); }
        else if (drag.t === 'zoom') { this.zoom = Math.max(0.05, drag.zoom0 * Math.exp(-dy / 200)); this.draw(); }
        else if (drag.t === 'window') {
          const v = this.v, span = (v.range[1] - v.range[0]) || 1;
          IM.setDisp(this.id, this.vid, { width: Math.max(span / 1000, drag.disp0.width * Math.exp(dx / 150)), center: drag.disp0.center - dy * span / 600 });
        } else if (drag.t === 'brush' || drag.t === 'erase') this.paintAt(p);
      } else if (EDIT_TOOLS.has(this.IM.tool) || this.poly.length || this.pending) this.draw();
    });
    const end = () => {
      if (drag && (drag.t === 'brush' || drag.t === 'erase')) this.IM.endStroke(drag.t === 'erase' ? 0 : this.IM.activeLabel, drag.t);
      drag = null;
    };
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', () => { this.hover = null; this.draw(); });
    cv.addEventListener('wheel', (e) => {
      if (!this.v || !this.v.loaded) return;
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) { this.zoom *= Math.exp(-e.deltaY / 300); this.draw(); return; }
      const d = [0, 0, 0]; d[this.plane.fixed] = e.deltaY > 0 ? -1 : 1; this.IM.moveCrossVoxels(this.vid, d);
    }, { passive: false });
    el.addEventListener('keydown', (e) => this.key(e));
    cv.addEventListener('dblclick', (e) => e.preventDefault());
  }
  key(e) {
    if (!this.v || !this.v.loaded) return;
    const IM = this.IM, pl = this.plane, step = e.shiftKey ? 5 : 1, d = [0, 0, 0];
    const ku = { ArrowRight: pl.uFlip ? -1 : 1, ArrowLeft: pl.uFlip ? 1 : -1 }, kv = { ArrowUp: pl.vFlip ? 1 : -1, ArrowDown: pl.vFlip ? -1 : 1 };
    if (e.key in ku) d[pl.u] = ku[e.key] * step;
    else if (e.key in kv) d[pl.v] = kv[e.key] * step;
    else if (e.key === 'PageUp' || e.key === '.') d[pl.fixed] = step;
    else if (e.key === 'PageDown' || e.key === ',') d[pl.fixed] = -step;
    else if (e.key === 'Enter' && this.poly.length >= 3) { this.fillPoly(); e.preventDefault(); return; }
    else if (e.key === 'Escape') { this.poly = []; this.pending = null; this.draw(); return; }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && IM.edit) { e.preventDefault(); e.stopPropagation(); e.shiftKey ? IM.redoEdit() : IM.undoEdit(); return; }
    else if (!e.ctrlKey && !e.metaKey && !e.altKey) {
      const t = TOOLS.find(x => x[2].toLowerCase() === e.key.toLowerCase()); if (t) { IM.setTool(t[0]); e.preventDefault(); }
      if (e.key === 'f') { this.fit(); this.draw(); }
      return;
    } else return;
    e.preventDefault(); e.stopPropagation(); IM.moveCrossVoxels(this.vid, d);
  }
  crossAt(p) { const ijk = this.screenToVoxel(...p); this.IM.setCross(this.vid, M.voxelToWorld(this.v, ijk), 'click'); }
  paintAt(p) {
    const IM = this.IM, v = this.v, pl = this.plane, c = this.screenToVoxel(...p).map(Math.round), out = [];
    for (const [a, b] of M.brushOffsets(v, pl, IM.brush)) {
      const q = c.slice(); q[pl.u] += a; q[pl.v] += b;
      if (q.every((x, ax) => x >= 0 && x < v.shape[ax])) out.push(M.vidx(v.shape, ...q));
    }
    IM.paint(this.vid, out, IM.tool === 'erase' ? 0 : IM.activeLabel); this.draw();
  }
  fillPoly() {
    const IM = this.IM, v = this.v, pl = this.plane, k = this.sliceIndex();
    const P = this.poly.map(p => [p[pl.u], p[pl.v]]), us = P.map(p => p[0]), vs = P.map(p => p[1]), idx = [];
    for (let b = Math.max(0, Math.floor(Math.min(...vs))); b <= Math.min(v.shape[pl.v] - 1, Math.ceil(Math.max(...vs))); b++)
      for (let a = Math.max(0, Math.floor(Math.min(...us))); a <= Math.min(v.shape[pl.u] - 1, Math.ceil(Math.max(...us))); a++)
        if (M.pointInPolygon(a, b, P)) { const q = [0, 0, 0]; q[pl.u] = a; q[pl.v] = b; q[pl.fixed] = k; idx.push(M.vidx(v.shape, ...q)); }
    IM.beginStroke(); IM.paint(this.vid, idx, IM.activeLabel); IM.endStroke(IM.activeLabel, 'polygon');
    this.poly = []; this.draw();
  }
  async placeLandmark(p) {
    const IM = this.IM, ijk = this.screenToVoxel(...p);
    const kind = IM.landmarkKind, name = (IM.vocab && IM.vocab.landmarks[kind]) || kind;
    try { await IM.saveAnnotation({ type: 'landmark', kind, name, volume: this.vid, points_voxel: [ijk] }); }
    catch (e) { alert('Landmark not saved: ' + e.message); }
  }
  async placeDistance(p) {
    const IM = this.IM, ijk = this.screenToVoxel(...p);
    if (!this.pending) { this.pending = ijk; this.draw(); return; }
    const a = this.pending; this.pending = null;
    const kind = IM.measureKind, name = (IM.vocab && IM.vocab.measurements[kind]) || kind;
    try { await IM.saveAnnotation({ type: 'distance', kind, name: name.split(':')[0], volume: this.vid, points_voxel: [a, ijk] }); }
    catch (e) { alert('Measurement not saved: ' + e.message); }
  }
}
const fmt = x => Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toFixed(1) : x.toFixed(3);
