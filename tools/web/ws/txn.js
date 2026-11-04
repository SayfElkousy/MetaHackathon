// Transactions: Muse (and evidence transfers) change the *same* state as the manual controls, through one
// validated batch per request: one undo step, one rebuild, previous view kept for "restore the previous view".
// Revisions protect newer work: a response computed against an older revision is rejected when it would touch a
// control that changed since.

export function validateValue(e, value, S) {
  if (!e) return { error: 'unknown control' };
  if (e.readOnly) return { error: `read-only (${e.readOnly}); change the inputs that determine it` };
  const k = e.kind;
  if (k === 'scalar' || k === 'int') {
    const x = typeof value === 'string' && value.trim() !== '' ? Number(value) : value;
    if (typeof x !== 'number' || !Number.isFinite(x)) return { error: 'must be a number' };
    if (k === 'int' && Math.round(x) !== x) return { error: 'must be a whole number' };
    if (e.min != null && x < e.min) return { error: `${x} is below the minimum ${e.min}${e.units ? ' ' + e.units : ''}` };
    if (e.max != null && x > e.max) return { error: `${x} is above the maximum ${e.max}${e.units ? ' ' + e.units : ''}` };
    return { value: x };
  }
  if (k === 'bool') {
    if (typeof value === 'boolean') return { value };
    if (value === 'true' || value === 'false') return { value: value === 'true' };
    return { error: 'must be true or false' };
  }
  if (k === 'enum') return e.options && e.options.includes(value) ? { value } : { error: `must be one of ${(e.options || []).join(', ')}` };
  if (k === 'profile') {
    const n = (S.reg && S.reg.s_knots ? S.reg.s_knots.length : 5);
    if (!Array.isArray(value) || value.length !== n || value.some(x => typeof x !== 'number' || !Number.isFinite(x))) return { error: `must be ${n} numbers along the ligament` };
    const bad = value.find(x => (e.min != null && x < e.min) || (e.max != null && x > e.max));
    return bad != null ? { error: `${bad} is outside ${e.min}–${e.max} ${e.units}` } : { value };
  }
  return value === undefined ? { error: 'missing value' } : { value };       // structured kinds: checked by the engine on rebuild
}

const clone = x => JSON.parse(JSON.stringify(x));
const DERIVED = /computed|derived|solved|current length/i;
const fmtV = v => v == null ? '—' : typeof v === 'number' ? String(+v.toFixed(4)) : typeof v === 'object' ? JSON.stringify(v).slice(0, 48) : String(v);

export class Txn {
  constructor(ctx) {
    this.ctx = ctx; this.revision = 0; this.keyRev = new Map(); this.seen = new Set(); this.cards = []; this.last = null; this.lastActionAt = 0;
    window.addEventListener('acl:param', (ev) => {
      if (ev.detail.actor === 'muse') return;                   // our own writes are counted when the transaction commits
      this.revision++; for (const k of ev.detail.keys || []) this.keyRev.set(k, this.revision);
    });
  }
  touch(keys) { this.revision++; for (const k of keys) this.keyRev.set(k, this.revision); }
  reset() { this.revision = 0; this.keyRev.clear(); this.seen.clear(); this.cards.length = 0; this.last = null; this.lastView = null; this.lastGeom = null; this.lastActionAt = 0; }
  viewSnapshot() {
    const { app, IM, ws } = this.ctx, S = app.S, cam = app.cam(), ctl = app.ctl();
    return { cam: cam && { p: cam.position.toArray(), t: ctl.target.toArray() }, vis: clone(S.vis), opac: clone(S.opac), builder: !!S.builder,
      layout: ws.layout.snapshot(), overlay: { visible: IM.overlay.visible, opacity: IM.overlay.opacity, labels: [...IM.overlay.labels] }, ann: IM.annotationsVisible,
      disp: [...IM.disp].map(([k, v]) => [k, clone(v)]) };
  }
  restoreView() {
    const snap = this.lastView && this.lastView.view;
    if (!snap) return 'No earlier view to restore.';
    const { app, IM, ws } = this.ctx, S = app.S;
    if (S.builder !== snap.builder) app.setBuilder(snap.builder);
    S.vis = clone(snap.vis); S.opac = clone(snap.opac); app.refreshLevelView(); app.renderObjList();
    for (const [, o] of S.objects) { const m = o.userData && o.userData.meta; if (m) { o.visible = S.vis[m.id] !== false; app.matOptsFor(o); } }
    if (snap.cam) { app.cam().position.fromArray(snap.cam.p); app.ctl().target.fromArray(snap.cam.t); app.ctl().update(); }
    ws.layout.restoreSnapshot(snap.layout);
    IM.overlay.visible = snap.overlay.visible; IM.overlay.opacity = snap.overlay.opacity; IM.overlay.labels = new Map(snap.overlay.labels); IM.annotationsVisible = snap.ann;
    for (const [k, v] of snap.disp) IM.disp.set(k, v);
    IM.emit('overlay'); IM.emit('display'); app.invalidate();
    return 'Previous view restored; the geometry is unchanged.';
  }
  undoLast() {
    const { app } = this.ctx, S = app.S, L = this.lastGeom;
    if (!L || !L.undoEntry) return 'There is no Muse geometry change to undo.';
    if (L.undone) return 'That change was already undone.';
    if (S.undoStack[S.undoStack.length - 1] !== L.undoEntry) return 'Newer edits were made after that change; undo them first (Undo in the design panel).';
    app.doUndo(); L.undone = true; this.touch(L.card.changes.filter(c => c.param).map(c => c.key));
    if (L.card) L.card.status = 'undone';
    return `Undone: ${L.card.label}.`;
  }
  // tx: { id, base, label, source, changes: [{key, value, evidence?, prov?}], keep: [prefix], actions: [{name, args}], authorizePatient, assumptions, model_ms }
  async apply(tx, reg) {
    const { app, ws } = this.ctx, S = app.S, t0 = performance.now();
    const card = { id: tx.id, label: tx.label || 'Muse change', source: tx.source || 'muse', changes: [], actions: [], assumptions: tx.assumptions || [], errors: [], warnings: [],
      timings: { model_ms: tx.model_ms ?? null }, status: 'pending', revision: this.revision };
    if (this.seen.has(tx.id)) return Object.assign(card, { status: 'duplicate', errors: ['This request was already applied.'] });
    this.seen.add(tx.id);
    const keep = (tx.keep || []).filter(Boolean);
    const kept = k => keep.some(p => k === p || (p.endsWith('*') ? k.startsWith(p.slice(0, -1)) : k.startsWith(p + '.')));
    const staged = [];
    // read-only status depends on modes: evaluate it on the state as it would be after this transaction
    const trial = Object.assign({}, S.state.params);
    for (const c of tx.changes || []) { const e = reg.entries.get(c.key); if (e && e.param) trial[c.key] = c.value; }
    const saved = S.state.params; S.state.params = trial; let depAfter = {};
    try { depAfter = this.ctx.app.DEPENDENT(); } finally { S.state.params = saved; }
    for (const c of tx.changes || []) {
      let e = reg.entries.get(c.key);
      if (e && e.param) { const why = depAfter[c.key]; e = Object.assign({}, e, { readOnly: why && DERIVED.test(why) ? why : null, inactive: why && !DERIVED.test(why) ? why : null }); }
      if (kept(c.key)) { card.errors.push(`${c.key}: you asked to keep it unchanged`); continue; }
      if (tx.base != null && (this.keyRev.get(c.key) || 0) > tx.base) { card.errors.push(`${c.key}: changed since this request was made (stale response)`); continue; }
      const r = validateValue(e, c.value, S);
      if (r.error) { card.errors.push(`${e ? e.label : c.key}: ${r.error}`); continue; }
      if (e && e.param && e.level === 0 && S.level0Locked && !tx.authorizePatient) {
        const st = (app.provOf(c.key) || {}).status;
        if (['measured', 'imported', 'observed', 'partly observed'].includes(st)) { card.errors.push(`${e.label}: patient-derived input is locked (unlock Level 0, or transfer reviewed evidence)`); continue; }
      }
      if (e && e.inactive) card.warnings.push(`${e.label}: ${e.inactive}`);
      staged.push({ e, value: r.value, c });
    }
    card.timings.validation_ms = Math.round(performance.now() - t0);
    if (card.errors.length) { card.status = 'rejected'; this.push(card); return card; }   // atomic: nothing applied
    const meta = (tx.actions || []).some(a => a.name === 'design.undo_last' || a.name === 'design.restore_view');
    const rec = { view: this.viewSnapshot(), card };
    if (!meta) this.lastView = rec;
    const params = staged.filter(s => s.e.param), others = staged.filter(s => !s.e.param);
    // ---- parameter batch: one undo entry, one rebuild
    let seq0 = S.seq, before = null;
    if (params.length) {
      before = { params: clone(S.state.params), provenance: clone(S.state.provenance) };
      app.gestureStart();
      for (const s of params) {
        const from = clone(S.state.params[s.e.key]);
        const prov = s.c.prov || { status: 'override', source: tx.source === 'evidence' ? 'MRI/report evidence via Muse' : 'Muse (user instruction)' };
        app.setParam(s.e.key, s.value, { noBuild: true, noPair: true, actor: 'muse', prov, txid: tx.id });
        card.changes.push({ key: s.e.key, label: s.e.label, units: s.e.units, level: s.e.level, from, to: clone(s.value), param: true, evidence: s.c.evidence || null });
      }
      app.gestureCommit('Muse: ' + card.label);
      rec.undoEntry = S.undoStack[S.undoStack.length - 1]; this.lastGeom = rec; this.lastActionAt = Date.now();
      app.renderControls(); app.refreshUndoButtons();
      seq0 = S.seq;
      app.afterEdit(true);                                   // preview build, then the final build (the designer chains them)
      try { app.highlightForParam(params[0].e.key, false); } catch (e) { /* no guide for this parameter */ }
      flash(params.map(s => s.e.key));
    }
    for (const s of others) {
      const from = (() => { try { return clone(s.e.get()); } catch (e) { return null; } })();
      try { s.e.set(s.value); card.changes.push({ key: s.e.key, label: s.e.label, units: s.e.units, from, to: s.value, param: false }); }
      catch (err) { card.errors.push(`${s.e.label}: ${err.message}`); }
    }
    for (const a of tx.actions || []) {
      const def = reg.actions.get(a.name);
      if (!def) { card.errors.push(`unknown action ${a.name}`); continue; }
      try { const r = await def.run(a.args || {}); if (r && r.ask) card.ask = r.ask; else card.actions.push({ name: a.name, result: r && r.text ? r.text : r }); }
      catch (err) { card.errors.push(`${a.name}: ${err.message}`); }
    }
    if (params.length) {
      const res = await waitBuilds(seq0, t0);
      Object.assign(card.timings, res.timings);
      if (res.error) {
        app.applySnap(before); S.undoStack.pop(); app.refreshUndoButtons(); app.renderControls(); app.afterEdit(false);
        card.status = 'failed'; card.errors.push('Rebuild failed: ' + res.error + '. The previous design was restored.');
        rec.undoEntry = null; this.lastGeom = null; this.push(card); return card;
      }
      const v = S.scene && S.scene.validity;
      card.validation = v ? (v.valid ? 'all fit checks pass' : `${v.fail} fit check(s) fail — see Checks (undo is available)`) : 'not evaluated';
      this.touch(params.map(s => s.e.key));
    } else card.validation = 'no geometry change (display/workspace only)';
    card.levels = [...new Set(card.changes.filter(c => c.param).map(c => c.level))].sort();
    card.status = card.errors.length ? 'partial' : 'applied';
    card.timings.total_ms = Math.round(performance.now() - t0);
    this.push(card);
    return card;
  }
  push(card) { this.cards.push(card); if (this.cards.length > 50) this.cards.shift(); }
}

function flash(keys) {
  for (const k of keys) document.querySelectorAll(`.ctl[data-pid="${CSS.escape(k)}"]`).forEach(el => { el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); });
}

function waitBuilds(seq0, t0) {
  // resolves after the final build that follows seq0, recording preview and final times separately
  return new Promise((resolve) => {
    const timings = {};
    const done = (x) => { window.removeEventListener('acl:built', ok); window.removeEventListener('acl:build-error', bad); clearTimeout(to); resolve(x); };
    const ok = (ev) => {
      if (ev.detail.seq <= seq0) return;
      const ms = Math.round(performance.now() - t0);
      if (ev.detail.quality === 'preview' && timings.preview_ms == null) timings.preview_ms = ms;
      if (ev.detail.quality === 'full') { timings.final_ms = ms; timings.engine_s = ev.detail.engine_s; done({ timings }); }
    };
    const bad = (ev) => { if (ev.detail.seq > seq0) done({ timings, error: ev.detail.errors ? ev.detail.errors.map(e => e.param + ': ' + e.message).join('; ') : ev.detail.message }); };
    const to = setTimeout(() => done({ timings, error: 'timed out waiting for the rebuild' }), 90000);
    window.addEventListener('acl:built', ok); window.addEventListener('acl:build-error', bad);
  });
}
export { fmtV };
