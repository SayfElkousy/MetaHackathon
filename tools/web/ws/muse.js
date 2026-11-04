// Muse assistant panel: interprets a request (configured language model, or the built-in deterministic parser),
// then executes the selected actions through the transaction engine. The model never computes geometry, units,
// coordinates or measurements; application code does.
import { parse } from './intent.js';
import { rank, describe } from './registry.js';
import { h } from './panels.js';
import { fmtV } from './txn.js';

const EXAMPLES = ['Inspect the current design and explain any failed fit checks', 'Set the sheath wall to 0.3 mm and keep patient geometry unchanged',
  'Use the reviewed ACL length, width, and tear gap, but preserve microscopic scaffold settings', 'Show which settings came from the MRI',
  'Apply the selected verified fit correction', 'Undo that change'];

export class Muse {
  constructor(ws) {
    this.ws = ws; this.history = []; this.inflight = null; this.seq = 0; this.status = { configured: false };
    const el = this.el = document.getElementById('muse');
    el.innerHTML = `<div class="mhead"><b>Muse Spark</b> <span class="mstat small"></span><button class="mtest" title="Run one small server-side connectivity test">Test connection</button><span class="grow"></span><button class="mmin" title="Collapse">–</button></div>
      <div class="mprogress" role="status" aria-live="polite">Ready</div><div class="mlog" aria-live="polite"></div>
      <details class="mpatient"><summary>Patient details <span>optional context</span></summary>
        <textarea rows="3" maxlength="6000" placeholder="Enter user-provided measurements, tear description, review status, or design constraints…" aria-label="Patient details"></textarea>
        <div class="mprivacy">Sent to the configured Meta model with your instruction. Use de-identified details for demos. Explicit values can adjust supported controls; Muse will not invent scan findings.</div>
      </details>
      <div class="mex"></div>
      <form class="mform"><textarea rows="2" placeholder="Ask Muse to change the scaffold or the imaging workspace…" aria-label="Message to Muse"></textarea>
      <div class="mbtns"><button type="button" class="mcancel" hidden>Cancel</button><button class="primary" type="submit">Send</button></div></form>`;
    this.log = el.querySelector('.mlog'); this.ta = el.querySelector('.mform textarea'); this.patientTa = el.querySelector('.mpatient textarea'); this.send = el.querySelector('button[type=submit]');
    el.querySelector('.mmin').onclick = () => this.toggle(false);
    el.querySelector('form').onsubmit = (e) => { e.preventDefault(); const t = this.ta.value.trim(); if (t) { this.ta.value = ''; this.run(t); } };
    this.ta.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); el.querySelector('form').requestSubmit(); } };
    el.querySelector('.mcancel').onclick = () => this.cancel();
    el.querySelector('.mtest').onclick = () => this.testConnection();
    const ex = el.querySelector('.mex');
    EXAMPLES.forEach(t => ex.appendChild(h('button', { class: 'exb', onclick: () => this.run(t) }, t)));
    fetch('/api/muse/status').then(r => r.json()).then(s => { this.status = s; this.renderStatus(); }).catch(() => this.renderStatus());
    fetch('/api/research/status').then(r => r.json()).then(s => { this.research = s; }).catch(() => { this.research = { available: false }; });
    this.say('muse', 'Tell me what to adjust, and optionally add de-identified patient details. I use the same validated controls as the inspector, with one undoable transaction per request. I do not claim to inspect scan pixels.');
  }
  toggle(on) { const el = this.el; const show = on == null ? el.classList.contains('min') : on; el.classList.toggle('min', !show); if (show) this.ta.focus(); }
  reset() { this.cancel(); this.history.length = 0; this.log.innerHTML = ''; if (this.patientTa) this.patientTa.value = '';
    this.say('muse', 'Demo reset. Muse translates engineering intent into registered operations; the deterministic geometry engine calculates and verifies physical fit.'); this.setProgress('Ready', false); }
  renderStatus() {
    const s = this.el.querySelector('.mstat');
    const labels = { connected_verified: 'Muse Spark · connected', configured_untested: 'Muse Spark · not yet tested', missing_key: 'Muse Spark · API key required',
      authentication_failed: 'Muse Spark · authentication failed', model_unavailable: 'Muse Spark · model unavailable', rate_limited: 'Muse Spark · rate limited',
      network_unavailable: 'Muse Spark · network unavailable', request_timed_out: 'Muse Spark · timed out', temporarily_degraded: 'Muse Spark · temporarily degraded' };
    s.textContent = labels[this.status.state] || (this.status.configured ? 'Muse Spark · not yet tested' : 'Muse Spark · API key required');
    s.title = this.status.message || ''; s.classList.toggle('connected', this.status.state === 'connected_verified'); s.classList.toggle('needskey', this.status.state !== 'connected_verified');
  }
  async testConnection() {
    this.setProgress('Checking Meta connection…', true);
    try { const r = await fetch('/api/muse/test', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }); const j = await r.json();
      this.status = Object.assign({}, this.status, j, { configured: true });
      if (!r.ok) this.status.state = ({ authorization_failed: 'authentication_failed', timeout: 'request_timed_out', provider_unavailable: 'temporarily_degraded' }[j.category] || j.category || 'temporarily_degraded');
      this.renderStatus();
      this.say(r.ok ? 'muse' : 'muse err', r.ok ? `Connection verified in ${j.latency_ms} ms.` : `${j.error}${j.suggested_action ? ' ' + j.suggested_action : ''}`);
    } catch (e) { this.status.state = 'network_unavailable'; this.status.message = 'The local server could not complete the test.'; this.renderStatus(); this.say('muse err', this.status.message); }
    this.setProgress('Ready', false);
  }
  setProgress(text, busy) { const p = this.el.querySelector('.mprogress'); p.textContent = text; p.classList.toggle('busy', !!busy); this.send.disabled = !!busy; }
  say(who, content) { const m = h('div', { class: 'msg ' + who }, content); this.log.appendChild(m); this.log.scrollTop = this.log.scrollHeight; return m; }
  cancel() { if (this.inflight) { this.inflight.abort(); this.inflight = null; this.seq++; } this.el.querySelector('.mcancel').hidden = true; this.setProgress('Cancelled', false); }
  context(text, reg) {
    const app = window.__app, S = app.S, IM = this.ws.IM, av = this.ws.activeMpr();
    const patientDetails = (this.patientTa && this.patientTa.value || '').trim().slice(0, 6000);
    const anns = IM.annotationsFor().map(a => ({ id: a.id, type: a.type, kind: a.kind, name: a.name, value: a.value, units: a.units, status: a.status, volume: a.volume }));
    const cam = app.cam(), ctl = app.ctl(), validity = S.scene && S.scene.validity;
    return {
      model_has_seen_image_pixels: false,
      active_view: this.ws.layout.activeView, active_volume: av ? { id: av.vid, name: av.v && av.v.name, plane: av.planeKey } : null,
      selected_level: S.levelSel || null, selected_component: S.compSel || null, builder_mode: !!S.builder,
      patient_details: patientDetails || null, patient_details_source: patientDetails ? 'entered by the user; not independently verified' : null,
      imaging: IM.manifest ? { case: IM.manifest.label, synthetic: IM.manifest.synthetic,
        volumes: IM.imageVolumes().map(v => ({ id: v.id, name: v.name, shape: v.shape, zooms_mm: v.spacing_known ? v.zooms : null, oriented: v.orientation_known, frame: v.frame_id })),
        masks: Object.values(IM.manifest.masks).map(m => ({ id: m.id, volume: m.volume, origin: m.origin, review: m.review.status, labels: m.labels.map(l => l.name), model: m.model && m.model.name })),
        annotations: anns, report_findings: IM.manifest.findings } : null,
      controls: rank(reg, text + ' ' + patientDetails, 48).map(e => Object.assign(describe(e), { provenance: e.param ? (S.state.provenance || {})[e.key] || { status: 'default' } : null,
        locked: !!(e.param && e.level === 0 && S.level0Locked) })),
      components: app.COMP_DEFS.filter(c => c.id).map(c => ({ id: c.id, name: c.short, level: c.level, visible: (c.objs || [c.obj]).some(id => S.vis[id] !== false) })),
      custom_components: this.ws.custom.list().map(p => this.ws.custom.describe(p)),
      layout: { active: this.ws.layout.activeView, visible_views: this.ws.layout.visibleViews(), maximized: this.ws.layout.max || null },
      camera: cam ? { position: cam.position.toArray(), target: ctl.target.toArray() } : null,
      validation: validity ? { evaluated: true, valid: validity.valid, failed: validity.fail, warnings: validity.warn,
        failed_checks: (S.scene.checks || []).filter(c => c.status === 'fail').map(c => ({ id: c.id, label: c.label, message: c.message, value: c.value, threshold: c.threshold, units: c.units, params: c.params })) } : { evaluated: false },
      verified_fixes: this.ws.verifiedFixes && this.ws.verifiedFixes.revision === this.ws.txn.revision && this.ws.verifiedFixes.design_hash === (S.scene && S.scene.design_hash) ? [...this.ws.verifiedFixes.map.values()] : [],
      parameter_coverage: this.ws.coverage(), custom_capabilities: this.ws.custom.supported(), research: this.research || { available: false },
      recent_changes: this.ws.txn.cards.slice(-3).map(c => ({ label: c.label, status: c.status, changes: c.changes.map(x => `${x.key}: ${fmtV(x.from)} -> ${fmtV(x.to)}`) })),
      revision: this.ws.txn.revision,
    };
  }
  async run(text) {
    this.cancel();                                   // obsolete requests are cancelled
    const my = ++this.seq, ws = this.ws;
    this.say('user', text);
    const ack = this.say('muse pending', 'Interpreting the request…'); this.setProgress('Interpreting with Muse Spark…', true);
    const ac = this.inflight = new AbortController(); this.el.querySelector('.mcancel').hidden = false;
    const reg = ws.registry(), base = ws.txn.revision, t0 = performance.now();
    let plan, model_ms, via;
    try {
      if (this.status.configured) {
        via = this.status.model;
        const r = await fetch('/api/muse/plan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ac.signal,
          body: JSON.stringify({ message: text, context: this.context(text, reg), tools: ws.tools(), history: this.history.slice(-6) }) });
        const j = await r.json();
        if (!r.ok) { const e = new Error(j.error || 'model error'); Object.assign(e, j); throw e; }
        plan = { calls: j.calls.map(c => ({ name: c.name === 'set_controls' ? c.name : c.name.replace(/__/g, '.'), args: c.args })), notes: j.text ? [j.text] : [] };
        model_ms = j.latency_ms;
      } else { via = 'Local deterministic fallback'; plan = parse(text, reg); model_ms = Math.round(performance.now() - t0);
        this.say('muse fallback', 'Local deterministic fallback — Meta’s model did not process this request.'); }
    } catch (e) {
      if (e.name === 'AbortError') { this.say('muse', 'Cancelled. No action was applied.'); this.setProgress('Cancelled', false); return; }
      if (this.status.configured) {
        via = 'Local deterministic fallback'; plan = parse(text, reg); model_ms = Math.round(performance.now() - t0);
        this.status.state = ({ authorization_failed: 'authentication_failed', timeout: 'request_timed_out', provider_unavailable: 'temporarily_degraded',
          request_rejected: 'temporarily_degraded', invalid_response: 'temporarily_degraded', malformed_tool_call: 'temporarily_degraded', invalid_tool_call: 'temporarily_degraded' }[e.category] || e.category || 'temporarily_degraded');
        this.status.message = e.message; this.renderStatus();
        this.say('muse fallback', `Local deterministic fallback — Meta’s model did not process this request. ${e.message}${e.suggested_action ? ' ' + e.suggested_action : ''}`);
      } else { ack.remove(); this.inflight = null; this.el.querySelector('.mcancel').hidden = true; this.say('muse err', 'Could not interpret the request: ' + e.message); return; }
    }
    if (my !== this.seq) return;                      // superseded
    this.history.push({ role: 'user', content: text });
    ack.remove();
    if (!plan.calls.length) { const answer = plan.notes.join(' ') || 'I did not find a supported action for that. Try naming a control and a value, a component, a check, or a view.'; this.say('muse', answer); this.history.push({ role: 'assistant', content: answer }); this.inflight = null; this.el.querySelector('.mcancel').hidden = true; this.setProgress('Ready', false); return; }
    const ask = plan.calls.find(c => c.name === 'ask_user');
    if (ask) { this.say('muse ask', ask.args.question); this.history.push({ role: 'assistant', content: ask.args.question }); this.inflight = null; this.el.querySelector('.mcancel').hidden = true; this.setProgress('Waiting for clarification', false); return; }
    // one transaction for all control changes; actions in order
    const changes = [], actions = [], keep = new Set(plan.keep || []);
    for (const c of plan.calls) {
      if (c.name === 'set_controls') { (c.args.changes || []).forEach(x => changes.push(x)); (c.args.keep || []).forEach(k => keep.add(k)); }
      else actions.push(c);
    }
    const fixActions = actions.filter(a => a.name === 'design.apply_verified_fix');
    try {
      for (const a of fixActions) {
        const fix = await ws.resolveVerifiedFix(a.args && a.args.fix_id);
        for (const [key, value] of Object.entries(fix.changes)) changes.push({ key, value, evidence: `deterministic verified fix ${fix.id}: ${fix.label}` });
        plan.notes.push(`Applied engine-verified correction “${fix.label}”. A new full build determines whether the complete design now passes.`);
      }
    } catch (e) { ack.remove(); this.say('muse ask', e.message); this.inflight = null; this.el.querySelector('.mcancel').hidden = true; this.setProgress('Waiting for a verified correction', false); return; }
    for (let i = actions.length - 1; i >= 0; i--) if (actions[i].name === 'design.apply_verified_fix') actions.splice(i, 1);
    const ev = actions.filter(a => a.name === 'evidence.apply'), rest = actions.filter(a => a.name !== 'evidence.apply');
    const cards = []; this.setProgress('Validating registered actions…', true);
    if (ev.length) cards.push(await ws.applyEvidence(Object.assign({}, ev[0].args, { keep: [...keep, ...((ev[0].args && ev[0].args.keep) || [])], model_ms, base, label: text })));
    if (changes.length || rest.length) {
      this.setProgress(changes.some(x => (reg.entries.get(x.key) || {}).param) ? 'Building preview and final geometry…' : 'Applying workspace actions…', true);
      const card = await ws.txn.apply({ id: 'tx' + Date.now().toString(36) + my, base, label: text.length > 70 ? text.slice(0, 67) + '…' : text, source: via === 'Local deterministic fallback' ? 'fallback' : 'muse', changes, actions: rest, keep: [...keep], model_ms }, reg);
      cards.push(card);
    }
    for (const c of cards) if (c) this.say('muse', this.card(c, via, plan.notes));
    let grounded = '';
    if (this.status.configured && via !== 'Local deterministic fallback' && cards.length) {
      try {
        this.setProgress('Summarizing verified results…', true);
        const fresh = ws.registry();
        const rr = await fetch('/api/muse/respond', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ac.signal,
          body: JSON.stringify({ message: text, context: this.context(text, fresh), results: cards.map(c => ({ status: c.status, revision: c.revision, validation: c.validation, changes: c.changes, actions: c.actions, errors: c.errors, warnings: c.warnings })) }) });
        const rj = await rr.json(); if (rr.ok && rj.text) { grounded = rj.text; this.say('muse grounded', h('span', {}, h('b', {}, `Grounded Muse response · ${rj.latency_ms} ms · `), rj.text)); }
        else this.say('muse err', `Actions completed, but the grounded Muse response failed: ${rj.error || 'empty response'}. The geometry change was not repeated or rolled back.`);
      } catch (e) { if (e.name !== 'AbortError') this.say('muse err', 'Actions completed, but Muse could not summarize the results: ' + e.message + ' The geometry change remains applied.'); }
    }
    if (this.inflight === ac) this.inflight = null;
    this.el.querySelector('.mcancel').hidden = true;
    this.setProgress('Ready', false);
    this.history.push({ role: 'assistant', content: grounded || cards.map(c => c && c.status).join(', ') });
  }
  card(c, via, notes) {
    const st = { applied: 'pass', partial: 'warn', rejected: 'fail', failed: 'fail', duplicate: 'warn', undone: 'warn' }[c.status] || 'warn';
    const el = h('div', { class: 'ccard' },
      h('div', { class: 'chead' }, h('span', { class: 'st ' + st }, c.status), ' ', h('b', {}, c.label)),
      c.levels && c.levels.length ? h('div', { class: 'small' }, 'Levels: ' + c.levels.map(l => l === 0 ? 'patient geometry (0)' : 'L' + l).join(', ')) : null,
      c.changes.length ? h('table', { class: 'tbl small' }, h('tr', {}, h('th', {}, 'Control'), h('th', {}, 'Before'), h('th', {}, 'After'), h('th', {}, 'Source')),
        ...c.changes.map(x => h('tr', {}, h('td', {}, x.label, h('div', { class: 'muted mono' }, x.key)), h('td', { class: 'mono' }, fmtV(x.from) + (x.units && typeof x.from === 'number' ? ' ' + x.units : '')),
          h('td', { class: 'mono' }, fmtV(x.to) + (x.units && typeof x.to === 'number' ? ' ' + x.units : '')), h('td', { class: 'small' }, x.evidence || (c.source === 'evidence' ? 'evidence' : 'your instruction'))))) : null,
      ...(c.actions || []).map(a => actionResult(a)),
      ...(c.extra || []),
      c.assumptions && c.assumptions.length ? h('div', { class: 'small' }, h('b', {}, 'Assumptions: '), c.assumptions.join(' ')) : null,
      ...(notes || []).map(n => h('div', { class: 'small note' }, n)),
      ...(c.warnings || []).map(w => h('div', { class: 'small warn' }, '⚠ ' + w)),
      ...(c.errors || []).map(e => h('div', { class: 'small err' }, '✖ ' + e)),
      c.ask ? h('div', { class: 'ask' }, c.ask) : null,
      h('div', { class: 'small muted' }, `Validation: ${c.validation || '—'}`),
      h('div', { class: 'execsource ' + (via === 'Local deterministic fallback' ? 'fallback' : 'model') }, via === 'Local deterministic fallback' ? 'Local deterministic fallback · Meta did not process this request' : `Muse Spark interpretation · ${via}`),
      h('div', { class: 'small muted mono' }, timingText(c.timings, via)));
    const btns = h('div', { class: 'irow' });
    if (c.status === 'applied' || c.status === 'partial') {
      if (c.changes.some(x => x.param) || (c.actions || []).some(x => /^custom\.(create|update|duplicate|delete)$/.test(x.name))) btns.append(h('button', { onclick: () => this.run('Undo that change') }, 'Undo'));
      btns.append(h('button', { onclick: () => this.run('Keep the geometry but restore the previous view') }, 'Restore previous view'));
    }
    el.append(btns);
    return el;
  }
}
function actionResult(a) {
  const r = a.result;
  if (r && r.sources) return h('details', { class: 'researchresult', open: true }, h('summary', {}, `${r.sources.length} PubMed source(s) for “${r.query}”`),
    ...r.sources.map(s => h('div', { class: 'sourcecard' }, h('a', { href: s.url, target: '_blank', rel: 'noopener' }, s.title),
      h('div', { class: 'small muted' }, `${(s.authors || []).slice(0, 3).join(', ')}${s.authors && s.authors.length > 3 ? ' et al.' : ''} · ${s.year || 'year unavailable'} · PMID ${s.pmid} · ${s.access}`),
      s.passage ? h('div', { class: 'small' }, s.passage.slice(0, 420) + (s.passage.length > 420 ? '…' : '')) : null)));
  if (r && r.revision != null) return h('details', { class: 'inspectresult' }, h('summary', {}, `Fresh design inspection · revision ${r.revision}`), h('pre', { class: 'small' }, JSON.stringify(r, null, 2)));
  if (r && typeof r === 'object') return h('details', { class: 'inspectresult' }, h('summary', {}, a.name), h('pre', { class: 'small' }, JSON.stringify(r, null, 2)));
  return h('div', { class: 'small' }, '• ' + (typeof r === 'string' ? r : a.name));
}
function timingText(t, via) {
  if (!t) return '';
  const p = [`interpretation ${t.model_ms ?? '—'} ms (${via})`, `validation ${t.validation_ms ?? '—'} ms`];
  if (t.preview_ms != null) p.push(`preview ${t.preview_ms} ms`);
  if (t.final_ms != null) p.push(`final rebuild ${t.final_ms} ms` + (t.engine_s != null ? ` (engine ${t.engine_s.toFixed(2)} s)` : ''));
  return p.join(' · ');
}
