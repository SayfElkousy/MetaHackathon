// Muse assistant panel: interprets a request (configured language model, or the built-in deterministic parser),
// then executes the selected actions through the transaction engine. The model never computes geometry, units,
// coordinates or measurements; application code does.
import { parse } from './intent.js';
import { rank, describe } from './registry.js';
import { h } from './panels.js';
import { fmtV } from './txn.js';

const EXAMPLES = ['Open axial, sagittal, and coronal views next to the scaffold', 'Show the ACL segmentation', 'Reduce overlay opacity to 30 percent',
  'Move the crosshair to the selected annotation', 'Reset the windowing', 'Show the original MRI without annotations', 'Open the quality-control panel',
  'Update the scaffold using this report and the reviewed MRI measurements', 'Show which measurements were used for the scaffold', 'Set the sheath wall to 0.3 mm', 'Undo that change'];

export class Muse {
  constructor(ws) {
    this.ws = ws; this.history = []; this.inflight = null; this.seq = 0; this.status = { configured: false };
    const el = this.el = document.getElementById('muse');
    el.innerHTML = `<div class="mhead"><b>Muse Spark</b> <span class="mstat small"></span><span class="grow"></span><button class="mmin" title="Collapse">–</button></div>
      <div class="mlog" aria-live="polite"></div>
      <details class="mpatient"><summary>Patient details <span>optional context</span></summary>
        <textarea rows="3" maxlength="6000" placeholder="Enter user-provided measurements, tear description, review status, or design constraints…" aria-label="Patient details"></textarea>
        <div class="mprivacy">Sent to the configured Meta model with your instruction. Use de-identified details for demos. Explicit values can adjust supported controls; Muse will not invent scan findings.</div>
      </details>
      <div class="mex"></div>
      <form class="mform"><textarea rows="2" placeholder="Ask Muse to change the scaffold or the imaging workspace…" aria-label="Message to Muse"></textarea>
      <div class="mbtns"><button type="button" class="mcancel" hidden>Cancel</button><button class="primary" type="submit">Send</button></div></form>`;
    this.log = el.querySelector('.mlog'); this.ta = el.querySelector('.mform textarea'); this.patientTa = el.querySelector('.mpatient textarea');
    el.querySelector('.mmin').onclick = () => this.toggle(false);
    el.querySelector('form').onsubmit = (e) => { e.preventDefault(); const t = this.ta.value.trim(); if (t) { this.ta.value = ''; this.run(t); } };
    this.ta.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); el.querySelector('form').requestSubmit(); } };
    el.querySelector('.mcancel').onclick = () => this.cancel();
    const ex = el.querySelector('.mex');
    EXAMPLES.forEach(t => ex.appendChild(h('button', { class: 'exb', onclick: () => this.run(t) }, t)));
    fetch('/api/muse/status').then(r => r.json()).then(s => { this.status = s; this.renderStatus(); }).catch(() => this.renderStatus());
    this.say('muse', 'Tell me what to adjust, and optionally add de-identified patient details. I use the same validated controls as the inspector, with one undoable transaction per request. I do not claim to inspect scan pixels.');
  }
  toggle(on) { const el = this.el; const show = on == null ? el.classList.contains('min') : on; el.classList.toggle('min', !show); if (show) this.ta.focus(); }
  renderStatus() {
    const s = this.el.querySelector('.mstat');
    s.textContent = this.status.configured ? `${this.status.model} · connected` : `${this.status.model || 'muse-spark-1.3'} · API key required`;
    s.classList.toggle('connected', !!this.status.configured); s.classList.toggle('needskey', !this.status.configured);
  }
  say(who, content) { const m = h('div', { class: 'msg ' + who }, content); this.log.appendChild(m); this.log.scrollTop = this.log.scrollHeight; return m; }
  cancel() { if (this.inflight) { this.inflight.abort(); this.inflight = null; } this.el.querySelector('.mcancel').hidden = true; }
  context(text, reg) {
    const app = window.__app, S = app.S, IM = this.ws.IM, av = this.ws.activeMpr();
    const patientDetails = (this.patientTa && this.patientTa.value || '').trim().slice(0, 6000);
    const anns = IM.annotationsFor().map(a => ({ id: a.id, type: a.type, kind: a.kind, name: a.name, value: a.value, units: a.units, status: a.status, volume: a.volume }));
    return {
      model_has_seen_image_pixels: false,
      active_view: this.ws.layout.activeView, active_volume: av ? { id: av.vid, name: av.v && av.v.name, plane: av.planeKey } : null,
      selected_level: S.levelSel || null, selected_component: S.compSel || null, builder_mode: !!S.builder,
      patient_details: patientDetails || null, patient_details_source: patientDetails ? 'entered by the user; not independently verified' : null,
      imaging: IM.manifest ? { case: IM.manifest.label, synthetic: IM.manifest.synthetic,
        volumes: IM.imageVolumes().map(v => ({ id: v.id, name: v.name, shape: v.shape, zooms_mm: v.spacing_known ? v.zooms : null, oriented: v.orientation_known, frame: v.frame_id })),
        masks: Object.values(IM.manifest.masks).map(m => ({ id: m.id, volume: m.volume, origin: m.origin, review: m.review.status, labels: m.labels.map(l => l.name), model: m.model && m.model.name })),
        annotations: anns, report_findings: IM.manifest.findings } : null,
      controls: rank(reg, text + ' ' + patientDetails, 40).map(describe),
      recent_changes: this.ws.txn.cards.slice(-3).map(c => ({ label: c.label, status: c.status, changes: c.changes.map(x => `${x.key}: ${fmtV(x.from)} -> ${fmtV(x.to)}`) })),
      revision: this.ws.txn.revision,
    };
  }
  async run(text) {
    this.cancel();                                   // obsolete requests are cancelled
    const my = ++this.seq, ws = this.ws;
    this.say('user', text);
    const ack = this.say('muse pending', 'Working on it…');
    const ac = this.inflight = new AbortController(); this.el.querySelector('.mcancel').hidden = false;
    const reg = ws.registry(), base = ws.txn.revision, t0 = performance.now();
    let plan, model_ms, via;
    try {
      if (this.status.configured) {
        via = this.status.model;
        const r = await fetch('/api/muse/plan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ac.signal,
          body: JSON.stringify({ message: text, context: this.context(text, reg), tools: ws.tools(), history: this.history.slice(-6) }) });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || 'model error');
        plan = { calls: j.calls.map(c => ({ name: c.name === 'set_controls' ? c.name : c.name.replace(/__/g, '.'), args: c.args })), notes: j.text ? [j.text] : [] };
        model_ms = j.latency_ms;
      } else { via = 'built-in parser'; plan = parse(text, reg); model_ms = Math.round(performance.now() - t0); }
    } catch (e) {
      ack.remove();
      if (e.name === 'AbortError') { this.say('muse', 'Cancelled.'); return; }
      this.say('muse err', 'Could not interpret the request: ' + e.message + (this.status.configured ? ' (the built-in parser is used next time if the model stays unavailable)' : ''));
      return;
    }
    if (my !== this.seq) return;                      // superseded
    this.history.push({ role: 'user', content: text });
    ack.remove();
    if (!plan.calls.length) { this.say('muse', (plan.notes.join(' ') || 'I did not find a supported action for that. Try naming a control and a value, or a view.') ); return; }
    const ask = plan.calls.find(c => c.name === 'ask_user');
    if (ask) { this.say('muse ask', ask.args.question); this.history.push({ role: 'assistant', content: ask.args.question }); this.el.querySelector('.mcancel').hidden = true; return; }
    // one transaction for all control changes; actions in order
    const changes = [], actions = [], keep = new Set(plan.keep || []);
    for (const c of plan.calls) {
      if (c.name === 'set_controls') { (c.args.changes || []).forEach(x => changes.push(x)); (c.args.keep || []).forEach(k => keep.add(k)); }
      else actions.push(c);
    }
    const ev = actions.filter(a => a.name === 'evidence.apply'), rest = actions.filter(a => a.name !== 'evidence.apply');
    const cards = [];
    if (ev.length) cards.push(await ws.applyEvidence(Object.assign({}, ev[0].args, { keep: [...keep, ...((ev[0].args && ev[0].args.keep) || [])], model_ms, base, label: text })));
    if (changes.length || rest.length) {
      const card = await ws.txn.apply({ id: 'tx' + Date.now().toString(36) + my, base, label: text.length > 70 ? text.slice(0, 67) + '…' : text, source: 'muse', changes, actions: rest, keep: [...keep], model_ms }, reg);
      cards.push(card);
    }
    this.el.querySelector('.mcancel').hidden = true; this.inflight = null;
    for (const c of cards) if (c) this.say('muse', this.card(c, via, plan.notes));
    this.history.push({ role: 'assistant', content: cards.map(c => c && c.status).join(', ') });
  }
  card(c, via, notes) {
    const st = { applied: 'pass', partial: 'warn', rejected: 'fail', failed: 'fail', duplicate: 'warn', undone: 'warn' }[c.status] || 'warn';
    const el = h('div', { class: 'ccard' },
      h('div', { class: 'chead' }, h('span', { class: 'st ' + st }, c.status), ' ', h('b', {}, c.label)),
      c.levels && c.levels.length ? h('div', { class: 'small' }, 'Levels: ' + c.levels.map(l => l === 0 ? 'patient geometry (0)' : 'L' + l).join(', ')) : null,
      c.changes.length ? h('table', { class: 'tbl small' }, h('tr', {}, h('th', {}, 'Control'), h('th', {}, 'Before'), h('th', {}, 'After'), h('th', {}, 'Source')),
        ...c.changes.map(x => h('tr', {}, h('td', {}, x.label, h('div', { class: 'muted mono' }, x.key)), h('td', { class: 'mono' }, fmtV(x.from) + (x.units && typeof x.from === 'number' ? ' ' + x.units : '')),
          h('td', { class: 'mono' }, fmtV(x.to) + (x.units && typeof x.to === 'number' ? ' ' + x.units : '')), h('td', { class: 'small' }, x.evidence || (c.source === 'evidence' ? 'evidence' : 'your instruction'))))) : null,
      ...(c.actions || []).map(a => h('div', { class: 'small' }, '• ' + (typeof a.result === 'string' ? a.result : a.name))),
      ...(c.extra || []),
      c.assumptions && c.assumptions.length ? h('div', { class: 'small' }, h('b', {}, 'Assumptions: '), c.assumptions.join(' ')) : null,
      ...(notes || []).map(n => h('div', { class: 'small note' }, n)),
      ...(c.warnings || []).map(w => h('div', { class: 'small warn' }, '⚠ ' + w)),
      ...(c.errors || []).map(e => h('div', { class: 'small err' }, '✖ ' + e)),
      c.ask ? h('div', { class: 'ask' }, c.ask) : null,
      h('div', { class: 'small muted' }, `Validation: ${c.validation || '—'}`),
      h('div', { class: 'small muted mono' }, timingText(c.timings, via)));
    const btns = h('div', { class: 'irow' });
    if (c.status === 'applied' || c.status === 'partial') {
      if (c.changes.some(x => x.param)) btns.append(h('button', { onclick: () => this.run('Undo that change') }, 'Undo'));
      btns.append(h('button', { onclick: () => this.run('Keep the geometry but restore the previous view') }, 'Restore previous view'));
    }
    el.append(btns);
    return el;
  }
}
function timingText(t, via) {
  if (!t) return '';
  const p = [`interpretation ${t.model_ms ?? '—'} ms (${via})`, `validation ${t.validation_ms ?? '—'} ms`];
  if (t.preview_ms != null) p.push(`preview ${t.preview_ms} ms`);
  if (t.final_ms != null) p.push(`final rebuild ${t.final_ms} ms` + (t.engine_s != null ? ` (engine ${t.engine_s.toFixed(2)} s)` : ''));
  return p.join(' · ');
}
