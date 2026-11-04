// Workspace shell: tabbed split panes around the existing scaffold designer (loaded from /v2/app.js in this same
// page, so there is one authoritative design state), MRI panes, contextual inspector, imaging browser and Muse.
import { Layout } from './layout.js';
import { Imaging, MprView, api } from './imaging.js';
import { renderBrowser, renderImgInspector, qcPanel, evidencePanel, coveragePanel, proposalTable, sourcesList, h } from './panels.js';
import { buildRegistry, toolDefs, coverage } from './registry.js';
import { Txn } from './txn.js';
import { Muse } from './muse.js';
import { CustomParts } from './custom_parts.js';
import { MriUpload } from './upload.js';

const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));

class Workspace {
  constructor() {
    this.IM = new Imaging(); this.specs = new Map(); this.els = new Map(); this.n = 0; this.cases = [];
    this.inspector = 'scaffold'; this._reg = null;
    this.layout = new Layout($('panes'), {
      viewEl: id => this.viewEl(id), viewTitle: id => this.viewTitle(id), viewTip: id => this.viewTip(id),
      onActivate: id => this.onActivate(id), onChange: () => this.persistLayout(), afterRender: () => this.afterRender(),
      addMenu: (paneId, e) => this.addMenu(paneId, e), tabMenu: (v, e) => this.tabMenu(v, e),
    });
  }
  async start() {
    while (!(window.__app && window.__app.S.reg && (window.__app.S.state || window.__app.S.waiting))) await sleep(50);
    this.app = window.__app;
    this.app.setSceneBackground(0xe8ebee);
    // the designer's stage becomes the 'scaffold' view; its side panel becomes the scaffold inspector
    const sv = document.createElement('div'); sv.className = 'scafview';
    sv.append($('stage'), h('div', { class: 'unreg', id: 'unregBanner', hidden: true }, 'Side-by-side, UNREGISTERED: the scaffold is shown in the engine\'s knee frame. No spatial transform aligns it with the MRI, so positions in the two views do not correspond.'));
    this.els.set('scaffold', sv); this.specs.set('scaffold', { kind: 'scaffold' });
    $('insScaf').appendChild($('side'));
    this.txn = new Txn({ app: this.app, IM: this.IM, ws: this });
    this.custom = new CustomParts(this);
    this.muse = new Muse(this);
    this.IM.vocab = await api('/api/img/vocab').catch(() => null);
    for (const t of ['case', 'manifest', 'loaded', 'overlay', 'edit', 'annotations', 'selection', 'progress', 'tool']) this.IM.on(t, () => this.refreshSoon());
    this.IM.on('cross', () => this.refreshInspectorSoon()); this.IM.on('display', () => this.refreshInspectorSoon());
    window.addEventListener('acl:built', () => { this._reg = null; });
    window.addEventListener('acl:param', () => { this._reg = null; });
    this.bindTop();
    await this.loadCases();
    this.restoreLayout();
    const last = (() => { try { return localStorage.getItem('ws.case'); } catch (e) { return null; } })();
    if (last && this.cases.some(c => c.id === last)) await this.openCase(last, true);
    this.refresh();
    await this.checkReader();
    this.upload = new MriUpload(this);
    window.__wsUpload = () => this.upload.open();
    if (!this.app.S.state) this.upload.open();
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && this.layout.max) this.layout.restore(); if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); this.muse.toggle(true); } });
    window.__ws = this;
  }
  // ---------------------------------------------------------------- views
  viewEl(id) {
    if (this.els.has(id)) return this.els.get(id);
    const s = this.specs.get(id); if (!s) return null;
    let el;
    if (s.kind === 'mpr') { const v = new MprView(this.IM, id, s.vid, s.plane); s.view = v; el = v.el; v.el.addEventListener('focusin', () => this.layout.activate(id)); }
    else if (s.kind === 'scaffold_view') {
      el = h('div', { class: 'iframewrap' }, h('div', { class: 'note bar' }, 'Second view of the same design (follows every change).'), h('iframe', { src: '/view?state=' + (this.app.S.stateId || ''), title: 'Second scaffold view' }));
    } else if (s.kind === 'qc') { el = qcPanel(this); el.refresh(true); }
    else if (s.kind === 'evidence') { el = evidencePanel(this); el.refresh(); }
    else if (s.kind === 'coverage') { el = coveragePanel(this); el.refresh(); }
    this.els.set(id, el);
    return el;
  }
  viewTitle(id) {
    const s = this.specs.get(id) || {};
    if (s.kind === 'mpr') return s.view ? s.view.title() : s.plane;
    return { scaffold: 'Scaffold', scaffold_view: 'Scaffold (2nd view)', qc: 'Quality control', evidence: 'Evidence → scaffold', coverage: 'Muse coverage' }[s.kind] || id;
  }
  viewTip(id) { const s = this.specs.get(id) || {}; return s.kind === 'mpr' ? 'MRI pane — drag the tab to move it to another pane or a pane edge' : 'Drag the tab to move it'; }
  newMpr(vid, plane) { const id = 'mpr' + (++this.n); this.specs.set(id, { kind: 'mpr', vid, plane }); return id; }
  findMpr(vid, plane) { for (const [id, s] of this.specs) if (s.kind === 'mpr' && s.plane === plane && (s.view ? s.view.vid : s.vid) === vid) return id; return null; }
  mprFor(vid, plane) { return this.findMpr(vid, plane) || this.newMpr(vid, plane); }
  single(kind) { for (const [id, s] of this.specs) if (s.kind === kind) return id; const id = kind + (++this.n); this.specs.set(id, { kind }); return id; }
  activeMpr() {
    const s = this.specs.get(this.layout.activeView);
    if (s && s.kind === 'mpr') return s.view;
    const vis = this.layout.visibleViews().map(v => this.specs.get(v)).find(x => x && x.kind === 'mpr' && x.view);
    return vis ? vis.view : this.lastMpr && this.layout.views().includes(this.lastMpr.id) ? this.lastMpr : null;
  }
  openMpr(vid, plane, where) { const id = this.mprFor(vid, plane); this.layout.open(id, where || 'active'); return id; }
  openKind(kind, where) {
    const vid = this.IM.activeVol || (this.IM.imageVolumes()[0] || {}).id;
    if (['axial', 'sagittal', 'coronal'].includes(kind)) { if (!vid) throw new Error('No MRI series loaded.'); this.openMpr(vid, kind, where); return `opened ${kind} view`; }
    if (kind === 'scaffold') { this.layout.open('scaffold', where); return 'opened the scaffold view'; }
    if (kind === 'scaffold_view') { const id = 'scaffold_view' + (++this.n); this.specs.set(id, { kind: 'scaffold_view' }); this.layout.open(id, where || 'right'); return 'opened a second scaffold view'; }
    const id = this.single(kind); this.layout.open(id, where || 'right'); const el = this.els.get(id); if (el && el.refresh) el.refresh(kind === 'qc');
    return `opened ${this.viewTitle(id)}`;
  }
  preset(name) {
    const vid = this.IM.activeVol || (this.IM.imageVolumes()[0] || {}).id;
    if (name !== 'scaffold' && !vid) throw new Error('No MRI series is loaded: import a scan or load the demo case first.');
    const P = p => this.mprFor(vid, p);
    const grid = { mpr_scaffold: [[P('axial'), P('sagittal')], [P('coronal'), 'scaffold']], mpr: [[P('axial'), P('sagittal'), P('coronal')]], scaffold: [['scaffold']],
      review: [[P('sagittal'), this.single('qc')], [P('coronal'), this.single('evidence')]] }[name];
    if (!grid) throw new Error('unknown preset ' + name);
    this.layout.setTree(Layout.grid(grid));
    const first = grid[0][0]; this.layout.activate(first);
    for (const id of this.layout.views()) { const el = this.els.get(id); if (el && el.refresh) el.refresh(); }
    return { mpr_scaffold: 'MPR + Scaffold: axial, sagittal and coronal planes with the scaffold', mpr: 'three MRI planes', scaffold: 'scaffold only', review: 'review layout' }[name];
  }
  ensureMpr(vid) { if (!this.layout.visibleViews().some(v => { const s = this.specs.get(v); return s && s.kind === 'mpr' && s.view && s.view.vid === vid; })) this.openMpr(vid, 'sagittal', 'right'); }
  showSeries(vid) {
    const a = this.activeMpr();
    const mprOpen = this.layout.views().some(v => (this.specs.get(v) || {}).kind === 'mpr');
    if (!mprOpen) this.preset('mpr_scaffold');
    else if (a && a.vid !== vid) { a.setVolume(vid); this.layout.render(); }
    this.refresh();
  }
  addMenu(paneId, e) {
    const items = [['Axial', () => this.openKind('axial', paneId)], ['Sagittal', () => this.openKind('sagittal', paneId)], ['Coronal', () => this.openKind('coronal', paneId)],
      ['Scaffold', () => this.layout.open('scaffold', paneId)], ['Scaffold (2nd view)', () => this.openKind('scaffold_view', paneId)],
      ['Quality control', () => this.openKind('qc', paneId)], ['Evidence → scaffold', () => this.openKind('evidence', paneId)], ['Muse coverage', () => this.openKind('coverage', paneId)]];
    this.menu(e, items);
  }
  tabMenu(v, e) {
    this.menu(e, [['Maximize', () => this.layout.maximize(v)], ['Split right', () => this.layout.moveView(v, this.layout.paneOf(v).id, 'right')], ['Split down', () => this.layout.moveView(v, this.layout.paneOf(v).id, 'down')],
      ...this.layout.panes().filter(p => !p.tabs.includes(v)).map((p, i) => [`Move to pane ${i + 1} (${p.tabs.map(t => this.viewTitle(t)).join(', ').slice(0, 30)})`, () => this.layout.moveView(v, p.id, 'center')]),
      ['Close (data stays loaded)', () => this.layout.close(v)]]);
  }
  menu(e, items) {
    const m = $('menu');
    const anchor = e.currentTarget || e.target;
    if (!m.hidden && this.menuAnchor === anchor) {
      m.hidden = true;
      this.menuAnchor = null;
      return;
    }
    this.menuAnchor = anchor;
    m.innerHTML = ''; items.forEach(([t, f]) => m.appendChild(h('button', { onclick: () => { m.hidden = true; this.menuAnchor = null; try { f(); } catch (x) { this.toast(x.message); } } }, t)));
    const rect = anchor.getBoundingClientRect();
    m.hidden = false;
    m.style.left = Math.max(6, Math.min(rect.left, innerWidth - 260)) + 'px';
    m.style.top = Math.max(6, Math.min(rect.bottom + 4, innerHeight - 30 * items.length - 10)) + 'px';
    setTimeout(() => document.addEventListener('pointerdown', ev => {
      if (!m.contains(ev.target) && !(this.menuAnchor && this.menuAnchor.contains(ev.target))) {
        m.hidden = true;
        this.menuAnchor = null;
      }
    }, { once: true }), 0);
  }
  onActivate(id) {
    const s = this.specs.get(id) || {};
    if (s.kind === 'mpr') { this.lastMpr = s.view; if (s.view) this.IM.activeVol = s.view.vid; this.setInspector('imaging'); }
    else if (s.kind === 'scaffold' || s.kind === 'scaffold_view') this.setInspector('scaffold');
    this.refreshSoon();
  }
  afterRender() {
    const mpr = this.layout.visibleViews().some(v => (this.specs.get(v) || {}).kind === 'mpr');
    const banner = $('unregBanner');
    if (banner) banner.hidden = !(mpr && this.layout.visibleViews().includes('scaffold'));
    this.app.invalidate();
  }
  setInspector(which) {
    this.inspector = which;
    $('insImg').hidden = which !== 'imaging'; $('insScaf').hidden = which !== 'scaffold';
    document.querySelectorAll('#insSwitch button').forEach(b => b.classList.toggle('on', b.dataset.ins === which));
    if (which === 'imaging') renderImgInspector(this);
  }
  persistLayout() {
    try {
      const specs = {}; for (const id of this.layout.views()) { const s = this.specs.get(id); if (s) specs[id] = { kind: s.kind, plane: s.plane, vid: s.view ? s.view.vid : s.vid }; }
      localStorage.setItem('ws.layout', JSON.stringify({ tree: this.layout.tree, specs, n: this.n }));
    } catch (e) { /* ignore */ }
  }
  restoreLayout() {
    let saved = null; try { saved = JSON.parse(localStorage.getItem('ws.layout') || 'null'); } catch (e) { /* ignore */ }
    if (saved && saved.tree) {
      for (const [id, s] of Object.entries(saved.specs || {})) if (id !== 'scaffold') this.specs.set(id, { kind: s.kind, plane: s.plane, vid: s.vid });
      this.n = saved.n || 0;
      const known = new Set(this.specs.keys());
      const prune = n => { if (n.t === 'pane') { n.tabs = n.tabs.filter(t => known.has(t)); n.active = n.tabs.includes(n.active) ? n.active : n.tabs[0]; } else n.kids.forEach(prune); };
      prune(saved.tree); this.layout.setTree(saved.tree);
      if (!this.layout.views().includes('scaffold') && !this.layout.views().length) this.layout.setTree(Layout.grid([['scaffold']]));
    } else this.layout.setTree(Layout.grid([['scaffold']]));
    this.layout.activate(this.layout.views()[0]);
  }
  // ---------------------------------------------------------------- refresh
  refreshSoon() { if (this._rt) return; this._rt = requestAnimationFrame(() => { this._rt = null; this.refresh(); }); }
  refreshInspectorSoon() { clearTimeout(this._it); this._it = setTimeout(() => this.refreshInspector(), 120); }
  refresh() {
    renderBrowser(this);
    this.refreshInspector();
    document.querySelectorAll('.wtab').forEach(t => { const l = t.querySelector('.wtabl'); if (l) l.textContent = this.viewTitle(t.dataset.view); });
  }
  refreshInspector() {
    if (this.inspector !== 'imaging') return;
    const a = document.activeElement;
    if (a && $('insImg').contains(a) && (a.type === 'number' || a.type === 'range' || a.tagName === 'TEXTAREA')) return;  // do not steal focus while typing
    renderImgInspector(this);
  }
  toast(t, kind) { const el = $('toast'); el.textContent = t; el.className = 'toast ' + (kind || ''); el.hidden = false; clearTimeout(this._tt); this._tt = setTimeout(() => (el.hidden = true), 4200); }
  // ---------------------------------------------------------------- registry access for UI and Muse
  registry() { if (!this._reg || this._regRev !== this.IM.manifest) { this._reg = buildRegistry({ app: this.app, IM: this.IM, ws: this }); this._regRev = this.IM.manifest; } return this._reg; }
  tools() { return toolDefs(this.registry()); }
  coverage() { this._reg = null; return coverage(this.registry()); }
  inspectDesign(args = {}) {
    const S = this.app.S, reg = this.registry(), query = String(args.query || 'summary').toLowerCase();
    const params = [...reg.entries.values()].filter(e => e.param && (!args.level || e.level === Number(args.level))).map(e => {
      let value = null; try { value = e.get(); } catch (x) { /* unavailable */ }
      const pv = (S.state.provenance || {})[e.key] || {};
      return { key: e.key, label: e.label, level: e.level, units: e.units, value, editable: !e.readOnly, derived: e.readOnly || null,
        target: e.target, provenance: { status: pv.status || 'default', source: pv.source || 'template/default', evidence: pv.evidence || null } };
    });
    const visible = (S.scene && S.scene.objects || []).map(o => ({ id: o.id, label: o.label, level: o.level, visible: S.vis[o.id] !== false, opacity: S.opac[o.id] ?? 1 }));
    const cam = this.app.cam(), ctl = this.app.ctl(), validity = S.scene && S.scene.validity;
    const anns = this.IM.annotationsFor().map(a => ({ id: a.id, name: a.name || a.kind, type: a.type, value: a.value, units: a.units, status: a.status, calibrated: a.calibrated }));
    const checks = validity ? { evaluated: true, valid: validity.valid, failed: validity.fail, warnings: validity.warn, items: (S.scene.checks || []).map(c => ({ id: c.id, label: c.label, status: c.status, message: c.message, value: c.value, threshold: c.threshold, units: c.units, params: c.params })) } : { evaluated: false };
    const minimumWall = S.scene && S.scene.metrics && (S.scene.metrics.minimum_wall_mm ?? S.scene.metrics.min_wall_mm);
    return { revision: this.txn.revision, request: query, case: { design: S.state.case && S.state.case.name || S.state.case || 'current design', state_id: S.stateId || null,
        imaging: this.IM.manifest ? { id: this.IM.caseId, label: this.IM.manifest.label, synthetic: this.IM.manifest.synthetic } : null },
      selection: { level: S.levelSel || null, component: S.compSel || null }, layout: { active_view: this.layout.activeView, views: this.layout.visibleViews(), maximized: this.layout.max || null },
      camera: cam ? { position: cam.position.toArray().map(x => +x.toFixed(2)), target: ctl.target.toArray().map(x => +x.toFixed(2)) } : null,
      components: this.app.COMP_DEFS.filter(c => c.id).map(c => ({ id: c.id, name: c.short, level: c.level, visible: (c.objs || [c.obj]).some(id => S.vis[id] !== false) })),
      custom_components: this.custom.list().map(p => this.custom.describe(p)), visible_layers: visible, parameters: params,
      measurements: anns, validation: checks,
      calculated: { minimum_wall_thickness_mm: minimumWall == null ? { status: 'not_evaluated', next_step: 'run an implemented wall-thickness check' } : minimumWall },
      imaging_evidence: this.IM.manifest ? { findings: this.IM.manifest.findings || [], masks: Object.values(this.IM.manifest.masks || {}).map(m => ({ id: m.id, review: m.review.status, origin: m.origin, labels: m.labels.map(l => l.name) })) } : null,
      recent_changes: this.txn.cards.slice(-8).map(c => ({ revision: c.revision, label: c.label, status: c.status, changes: c.changes.map(x => ({ key: x.key, from: x.from, to: x.to })) })) };
  }
  async listVerifiedFixes() {
    const S = this.app.S, revision = this.txn.revision, hash = S.scene && S.scene.design_hash;
    if (!hash) throw new Error('No completed design is available for fix verification.');
    const out = await api('/api/v2/layer_fixes', { state: S.state });
    if (out.design_hash !== (S.scene && S.scene.design_hash) || revision !== this.txn.revision) throw new Error('The design changed while fixes were being verified; list fixes again.');
    const map = new Map(), fixes = (out.fixes || []).filter(f => f.verified).map((f, i) => {
      const id = `${out.design_hash}:${i}`; const item = { id, conflict: f.conflict, label: f.label, changes: f.set, new_conflicts: f.new_conflicts || [] }; map.set(id, item); return item;
    });
    this.verifiedFixes = { revision, design_hash: out.design_hash, map };
    return { revision, design_hash: out.design_hash, fixes, note: fixes.length ? 'Each fix was rebuilt by the deterministic engine; applying one still triggers a full rebuild and validation.' : 'No verified single correction is available for the current design.' };
  }
  async resolveVerifiedFix(id) {
    if (!this.verifiedFixes || this.verifiedFixes.revision !== this.txn.revision || this.verifiedFixes.design_hash !== (this.app.S.scene && this.app.S.scene.design_hash)) await this.listVerifiedFixes();
    const all = [...this.verifiedFixes.map.values()];
    const fix = id ? this.verifiedFixes.map.get(id) : all.length === 1 ? all[0] : null;
    if (!fix) throw new Error(id ? 'That verified fix is stale or unknown; list fixes again.' : all.length ? 'Several verified fixes are available; choose one by its fix id.' : 'No verified fix is available for the current design.');
    return fix;
  }
  set(key, value) { const e = this.registry().entries.get(key); if (!e) return this.toast('unknown control ' + key, 'err'); try { e.set(value); } catch (x) { this.toast(x.message, 'err'); } this.refreshInspectorSoon(); }
  async act(name, args) {
    const a = this.registry().actions.get(name);
    try { const r = await a.run(args || {}); if (typeof r === 'string') this.toast(r); return r; } catch (e) { this.toast(e.message, 'err'); return null; }
  }
  // ---------------------------------------------------------------- cases and import
  async loadCases() { this.cases = (await api('/api/img/cases')).cases; }
  async openCase(id, quiet) {
    await this.IM.openCase(id);
    for (const [, s] of this.specs) if (s.kind === 'mpr') { const vid = s.view ? s.view.vid : s.vid; if (!this.IM.vol(vid)) { const f = this.IM.imageVolumes()[0]; if (s.view) s.view.setVolume(f ? f.id : null); else s.vid = f && f.id; } else if (s.view) s.view.setVolume(vid); }
    this.layout.render(); this.refresh();
    if (!quiet) this.toast('Opened ' + this.IM.manifest.label);
  }
  async demoCase() {
    this.toast('Creating the synthetic phantom case…');
    const c = await api('/api/img/demo', {});
    await this.loadCases(); await this.openCase(c.id);
    const sag = this.IM.imageVolumes().find(v => /sagittal/i.test(v.name)); if (sag) this.IM.activeVol = sag.id;
    this.preset('mpr_scaffold');
  }
  async resetDemo() {
    this.toast('Resetting the hackathon demo…');
    const seq = this.app.S.seq;
    await this.app.openCase('demo_complete');
    await this.app.waitForSettled(seq, 90000);
    await this.demoCase();
    this.txn.reset(); this.verifiedFixes = null; this._reg = null;
    this.custom.selected = null; this.custom.undo.length = 0; this.custom.renderGeometry(); this.custom.renderInspector();
    this.muse.reset(); this.app.setCam('q34'); this.preset('mpr_scaffold'); this.refresh();
    this.toast('Hackathon demo restored: synthetic evidence, valid scaffold, clean Muse history.');
  }
  async importFiles(files, opt) {
    const IM = this.IM; if (!IM.caseId) return this.toast('Open or create a case first', 'err');
    if (opt.role === 'mask' && !IM.activeVol) return this.toast('Select the image series the mask belongs to', 'err');
    const batch = Math.random().toString(36).slice(2, 12);
    this.importState = { status: 'uploading', message: `uploading ${files.length} file(s)…`, progress: 0 }; this.refresh();
    const ac = this.importAbort = new AbortController();
    try {
      let done = 0;
      for (const f of files) {
        const r = await fetch(`/api/img/upload?case=${IM.caseId}&batch=${batch}&name=${encodeURIComponent(String(done).padStart(5, '0') + '_' + (f.webkitRelativePath || f.name).replace(/[\\/]/g, '_'))}`, { method: 'POST', body: f, signal: ac.signal });
        if (!r.ok) throw new Error((await r.json()).error);
        done++; this.importState.progress = done / files.length * 0.4; this.importState.message = `uploaded ${done}/${files.length}`; if (done % 10 === 0 || done === files.length) this.refresh();
      }
      const job = await api('/api/img/import', { case: IM.caseId, batch, role: opt.role, for_volume: IM.activeVol, quantity: opt.quantity, value_units: opt.value_units, resample: opt.resample });
      this.importJob = job.id;
      const res = await this.poll(job.id, j => { this.importState = { status: 'running', message: j.message, progress: 0.4 + 0.6 * j.progress }; this.refresh(); }, ac.signal);
      this.importState = res.status === 'done' ? { status: 'done', result: res.result } : res.status === 'cancelled' ? { status: 'cancelled' } : { status: 'error', error: res.error.message, checks: res.error.checks };
      await IM.refresh();
      if (res.status === 'done' && res.result.volumes.length) { IM.activeVol = res.result.volumes[0].id; this.showSeries(IM.activeVol); }
      if (res.status === 'done' && res.result.masks.length) await IM.selectMask(res.result.masks[0].volume, res.result.masks[0].id);
    } catch (e) { this.importState = e.name === 'AbortError' ? { status: 'cancelled' } : { status: 'error', error: e.message, checks: e.body && e.body.checks }; }
    this.refresh();
  }
  cancelImport() { if (this.importAbort) this.importAbort.abort(); if (this.importJob) api('/api/img/job/cancel', { id: this.importJob }).catch(() => {}); }
  async poll(jid, onStep, signal) {
    for (;;) {
      if (signal && signal.aborted) { await api('/api/img/job/cancel', { id: jid }).catch(() => {}); throw Object.assign(new Error('cancelled'), { name: 'AbortError' }); }
      const j = await api('/api/img/job?id=' + jid);
      if (j.status !== 'running') return j;
      onStep && onStep(j); await sleep(400);
    }
  }
  // ---------------------------------------------------------------- segmentation service
  async checkReader() {
    try { this.reader = await api('/api/img/reader' + (this.readerUrl ? '?url=' + encodeURIComponent(this.readerUrl) : '')); if (!this.readerUrl) this.readerUrl = this.reader.default_url; }
    catch (e) { this.reader = { ok: false, error: e.message }; }
    this.refreshInspectorSoon();
    if (this.upload) this.upload.renderReader();
    clearTimeout(this._readerPoll);
    if (!this.reader.ok && (this.reader.launcher || {}).state === 'starting') this._readerPoll = setTimeout(() => this.checkReader(), 3000);
  }
  async runSegmentation(args) {
    const IM = this.IM, vid = IM.activeVol;
    if (!vid) throw new Error('No MRI series selected.');
    const job = await api('/api/img/segment', { case: IM.caseId, volume: vid, reader_url: this.readerUrl, do_bones: !!(args && args.do_bones) });
    this.segJob = job.id; this.segState = { status: 'running', message: 'starting…', progress: 0 }; this.refreshInspectorSoon();
    const res = await this.poll(job.id, j => { this.segState = { status: 'running', message: j.message, progress: j.progress }; this.refreshInspectorSoon(); });
    if (res.status !== 'done') { this.segState = { status: res.status === 'cancelled' ? 'cancelled' : 'error', message: res.error ? res.error.message : 'cancelled' }; this.refreshInspectorSoon();
      throw new Error(res.error ? res.error.message : 'segmentation cancelled'); }
    await IM.refresh(); await IM.selectMask(vid, res.result.mask.id);
    this.segState = { status: 'done', message: `KneeScan result stored as an editable overlay (${res.result.mask.labels.map(l => l.name).join(', ')}; 2D outline). Review it before use.` };
    if (!this.app.S.state && res.result.kneescan) {
      this.segState.message += ' ' + await this.designFromKneeScan(res.result.kneescan, args && args.side, this.volName(vid));
    }
    this.refreshInspectorSoon();
    return { text: this.segState.message };
  }
  cancelSeg() { if (this.segJob) api('/api/img/job/cancel', { id: this.segJob }).catch(() => {}); }
  async importModelResult(vid, result, filename, side) {
    const out = await api('/api/img/segment_result', { case: this.IM.caseId, volume: vid, result, filename });
    await this.IM.refresh(); await this.IM.selectMask(vid, out.mask.id); this.toast('KneeScan result imported as an editable 2D overlay');
    if (!this.app.S.state) this.toast(await this.designFromKneeScan(result, side, filename));
  }
  volName(vid) { const v = this.IM.vols && this.IM.vols.get(vid); return (v && v.name) || 'scan'; }
  kneeSide(vid, side) {
    if (side === 'right' || side === 'left') return side;
    const v = this.IM.vols && this.IM.vols.get(vid), m = (v && v.meta) || {};
    const lat = String(m.Laterality || m.ImageLaterality || '').toUpperCase();
    return lat === 'R' ? 'right' : lat === 'L' ? 'left' : null;
  }
  async designFromKneeScan(result, side, filename) {
    const resolvedSide = this.kneeSide(this.IM.activeVol, side);
    if (!resolvedSide) return 'No scaffold yet: choose the knee side and run KneeScan again.';
    const out = await api('/api/mri/from_result', { result, side: resolvedSide, filename });
    this.app.loadState(out.state, out.state_id);
    const applied = out.mapping.filter(r => r.result === 'applied').map(r => r.reading.toLowerCase());
    return `Scaffold built from verified readings (${resolvedSide} knee; applied: ${[...new Set(applied)].join(', ') || 'none'}).`;
  }
  // ---------------------------------------------------------------- evidence -> scaffold (explicit mappings, one transaction)
  async applyEvidence(a) {
    const IM = this.IM, S = this.app.S;
    if (!IM.caseId) throw new Error('No imaging case is open.');
    if (!S.state) throw new Error('No scaffold design yet: run KneeScan on this series first.');
    const p = await api('/api/img/propose', { case: IM.caseId, params: S.state.params, provenance: S.state.provenance, scope: a.scope || null, include_assumptions: !!a.include_assumptions });
    const keep = a.keep || [], kept = k => keep.some(x => x.endsWith('*') ? k.startsWith(x.slice(0, -1)) : k === x);
    const changes = [], assumptions = [];
    for (const r of p.rows.filter(r => r.status === 'apply')) {
      for (const [k, v] of Object.entries(r.changes)) {
        if (kept(k)) continue;
        const src = r.source, label = `${src.label}${src.volume_name ? ' · ' + src.volume_name : ''}`;
        changes.push({ key: k, value: v, evidence: `${label} [${src.certainty}] via mapping ${r.mapping}`,
          prov: { status: src.certainty === 'measured' ? 'measured' : src.certainty === 'reported' ? 'reported' : 'assumed', source: (src.source_type === 'report' ? 'Radiology report' : src.source_type === 'model' ? 'KneeScan model' : 'MRI measurement') + ': ' + label, evidence: src.ref, mapping: r.mapping, note: r.assumptions.join(' ') } });
      }
      assumptions.push(...r.assumptions);
    }
    const extra = [h('details', { class: 'small' }, h('summary', {}, `Mappings: ${Object.entries(p.summary).filter(([, n]) => n).map(([k, n]) => `${n} ${k.replace('_', ' ')}`).join(', ')}`), proposalTable(p, this))];
    const unresolved = p.rows.filter(r => ['conflict', 'missing', 'blocked', 'needs_review'].includes(r.status));
    for (const r of unresolved) extra.push(h('div', { class: 'small ' + (r.status === 'conflict' ? 'err' : 'note') }, `${r.mapping.replace(/_/g, ' ')}: ${r.status.replace('_', ' ')} — ${r.reason}`));
    if (keep.length) extra.push(h('div', { class: 'small' }, 'Kept unchanged as requested: ' + keep.join(', ')));
    let card;
    if (changes.length) card = await this.txn.apply({ id: 'ev' + Date.now().toString(36), base: a.base, label: a.label || 'Apply reviewed evidence', source: 'evidence', changes, keep, authorizePatient: true, assumptions: [...new Set(assumptions)], model_ms: a.model_ms }, this.registry());
    else card = { label: a.label || 'Apply reviewed evidence', status: unresolved.some(r => r.status === 'conflict') ? 'rejected' : 'applied', changes: [], actions: [], errors: [], warnings: [],
      validation: 'no supported, unambiguous mapping to apply; current values kept', timings: { model_ms: a.model_ms } };
    card.extra = extra;
    const ev = this.els.get(this.single('evidence')); if (ev && ev.isConnected) ev.refresh();
    return card;
  }
  showSources() {
    this.openKind('evidence', 'right');
    const S = this.app.S, rows = Object.entries(S.state.provenance || {}).filter(([, pv]) => /MRI|report|KneeScan|measurement|evidence/i.test(pv.source || ''));
    return rows.length ? `${rows.length} setting(s) came from imaging or reports: ` + rows.map(([k, pv]) => `${(S.regById[k] || {}).label || k} (${pv.source})`).join('; ') : 'No current setting came from imaging or a report.';
  }
  // ---------------------------------------------------------------- top bar
  bindTop() {
    $('btnPresetMpr').onclick = () => { try { this.preset('mpr_scaffold'); } catch (e) { this.toast(e.message, 'err'); } };
    $('btnPresetScaf').onclick = () => this.preset('scaffold');
    $('btnPresetReview').onclick = () => { try { this.preset('review'); } catch (e) { this.toast(e.message, 'err'); } };
    $('btnAddView').onclick = (e) => this.addMenu(this.layout.paneOf(this.layout.activeView) ? this.layout.paneOf(this.layout.activeView).id : 'active', e);
    $('btnMuse').onclick = () => this.muse.toggle();
    $('btnDemoReset').onclick = () => this.resetDemo().catch(e => this.toast('Demo reset failed: ' + e.message, 'err'));
    document.querySelectorAll('#insSwitch button').forEach(b => (b.onclick = () => this.setInspector(b.dataset.ins)));
  }
}

const ws = new Workspace();
ws.start().catch(e => { console.error(e); const t = document.getElementById('toast'); t.hidden = false; t.textContent = 'Workspace failed to start: ' + e.message; });
