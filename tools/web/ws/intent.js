// Deterministic command parser: used when no Muse language model is configured, and as a fast path for common
// commands. It only selects registered controls and actions; numbers come from the user's text, never invented.
// Output: { calls: [{ name, args }], notes: [] } using the same tool names as the language-model path.

const LEN = { nm: 1e-6, um: 1e-3, 'µm': 1e-3, micron: 1e-3, microns: 1e-3, mm: 1, millimeter: 1, millimeters: 1, cm: 10 };
const UNIT_RE = '(nm|µm|um|microns?|mm|millimet(?:er|re)s?|cm|°|deg(?:rees?)?|%|percent|x)?';

export function convert(value, unit, e) {
  unit = (unit || '').toLowerCase();
  const eu = (e.units || '').toLowerCase();
  if (unit === '%' || unit === 'percent') {
    if (eu === 'fraction' || (e.min != null && e.max != null && e.min >= 0 && e.max <= 1)) return { value: value / 100 };
    if (eu === '%') return { value };
    return { error: `${e.label} is in ${e.units || 'unitless values'}, not percent` };
  }
  const uLen = LEN[unit.replace(/s$/, '')] ?? LEN[unit], eLen = LEN[eu];
  if (uLen != null) {
    if (eLen == null) return { error: `${e.label} is not a length (${e.units || 'unitless'})` };
    return { value: +(value * uLen / eLen).toPrecision(10) };
  }
  if (/^(°|deg)/.test(unit) && eu && eu !== 'deg') return { error: `${e.label} is not an angle` };
  return { value };
}

const PREFIX_WORDS = {
  groove: 'gr.', grooves: 'gr.', microgroove: 'gr.', microgrooves: 'gr.', channel: 'ch.', channels: 'ch.', braid: 'scaffold.braid', sheath: 'sheath.',
  coating: 'sf.', coatings: 'sf.', surface: 'sf.', footprint: 'fp.', footprints: 'fp.', attachment: 'fp.', tear: 'injury.', injury: 'injury.', pose: 'pose.',
};
export function keepPrefixes(text, reg) {
  const out = new Set();
  const re = /\b(?:retain|keep|preserve|leave|don'?t change|do not change)\s+(?:the\s+)?(?:current\s+|existing\s+)?([a-z0-9 \-]+?)(?:\s+(?:settings?|values?|parameters?|unchanged|as is|alone))?(?=[,.;]|\s+and\b|$)/gi;
  let m;
  while ((m = re.exec(text))) {
    const words = m[1].toLowerCase().split(/\s+/);
    let hit = false;
    if (/\b(patient|anatomy|anatomical)\b/i.test(m[1])) { for (const e of reg.entries.values()) if (e.param && e.level === 0) out.add(e.key); hit = true; }
    if (/\b(microscopic|microstructure)\b/i.test(m[1])) { for (const e of reg.entries.values()) if (e.param && e.level >= 1 && e.level <= 4) out.add(e.key); hit = true; }
    if (/\bpl\b.*\b(mantle|braid|reinforcement)\b/i.test(m[1])) { for (const e of reg.entries.values()) if (e.param && (/\.pl\./.test(e.key) || /\bPL\b/.test(e.label))) out.add(e.key); hit = true; }
    for (const w of words) if (PREFIX_WORDS[w]) { out.add(PREFIX_WORDS[w] + '*'); hit = true; }
    const lv = m[1].match(/level\s*(\d)/i);
    if (lv) for (const e of reg.entries.values()) if (e.param && e.level === Number(lv[1])) { out.add(e.key); hit = true; }
    if (!hit) { const r = resolve(reg, m[1]); if (r.entry) out.add(r.entry.key); }
  }
  return [...out];
}

const tok = s => String(s || '').toLowerCase().replace(/µ/g, 'u').replace(/[^a-z0-9]+/g, ' ').split(' ').filter(w => w.length > 1 && !STOP.has(w));
const STOP = new Set(['the', 'of', 'to', 'by', 'and', 'set', 'change', 'make', 'scaffold', 'value', 'please', 'it', 'a', 'an', 'in', 'on', 'for', 'level']);
const ALIAS = { gr: ['groove', 'grooves', 'microgroove'], ch: ['channel', 'channels'], sf: ['coating', 'surface'], fp: ['footprint'], lig: ['ligament', 'acl'],
  um: ['micron', 'microns'], tib: ['tibial'], fem: ['femoral'], deg: ['angle'] };
function words(e) {
  const base = tok(e.key.replace(/[._]/g, ' ') + ' ' + e.label + ' ' + (e.subgroup || ''));
  const out = new Set(base);
  for (const w of base) if (ALIAS[w]) ALIAS[w].forEach(a => out.add(a));
  return out;
}
export function resolve(reg, phrase, filter) {
  const q = tok(phrase), lv = String(phrase).match(/level\s*(\d)/i);
  if (!q.length) return { none: true };
  const exact = reg.entries.get(phrase.trim());
  if (exact) return { entry: exact };
  const scored = [];
  for (const e of reg.entries.values()) {
    if (filter && !filter(e)) continue;
    if (lv && e.level !== Number(lv[1])) continue;
    const W = words(e); let s = 0;
    for (const w of q) if (W.has(w)) s += 1; else if ([...W].some(x => x.startsWith(w) && w.length >= 4)) s += 0.6;
    if (s <= 0) continue;
    s = s / q.length - 0.01 * W.size;
    // the user's phrase matching the control's own label is the strongest signal
    const lab = tok(e.label.replace(/\(.*?\)/g, '')).join(' '), ph = q.join(' ');
    if (lab === ph) s += 1; else if (lab.startsWith(ph)) s += 0.68; else if (lab.includes(ph)) s += 0.42; else if (ph.includes(lab) && lab.length > 3) s += 0.4;
    if (e.inactive) s -= 0.1;                      // prefer controls that currently affect the design
    scored.push({ e, s });
  }
  scored.sort((a, b) => b.s - a.s);
  if (!scored.length || scored[0].s < 0.5) return { none: true };
  const top = scored.filter(x => scored[0].s - x.s < 0.06 && !x.e.readOnly);
  if (top.length > 1 && !top.every(x => x.e.key === top[0].e.key)) return { ambiguous: top.slice(0, 5).map(x => x.e) };
  return { entry: scored[0].e };
}

const has = (t, re) => re.test(t);

export function parse(text, reg, ctx = {}) {
  const t = String(text || '').trim(), low = t.toLowerCase(), calls = [], notes = [];
  const keep = keepPrefixes(t, reg);
  const action = (name, args) => calls.push({ name, args: args || {} });
  // ---------------- goals without a defined objective
  if (has(low, /\bdiagnos(?:e|is|tic)\b/)) {
    action('ask_user', { question: 'I cannot diagnose MRI or infer clinical findings. I can inspect verified measurements, loaded annotations, segmentation provenance, quality-control flags, or the current scaffold design. Which should I show?' });
    return { calls, notes };
  }
  if (has(low, /\b(optimi[sz]e|maximi[sz]e|improve|best)\b.*\b(heal|healing|regenerat|integration|outcome|strength)/)) {
    action('ask_user', { question: 'Healing has no defined objective or validated model in this workspace. Which parameter should change, and to what value or target (for example a channel lumen in µm, or a sheath wall in mm)?' });
    return { calls, notes };
  }
  // ---------------- undo / restore view
  if (has(low, /\bkeep the geometry\b.*\b(restore|previous|earlier)\b.*\bview\b|\brestore the (previous|earlier) view\b/)) { action('design.restore_view'); return { calls, notes }; }
  if (has(low, /^\s*undo\b|\bundo (that|the last|my last|the previous)\b/)) { action('design.undo_last'); return { calls, notes }; }
  if (has(low, /\b(list|show)\b.*\bverified (?:fit )?(?:fix|fixes|corrections?)\b/)) { action('design.list_verified_fixes'); return { calls, notes }; }
  if (has(low, /\bapply\b.*\b(?:selected )?verified (?:fit )?(?:fix|correction)\b/)) { action('design.apply_verified_fix', {}); return { calls, notes }; }
  // ---------------- physical STL export (same validated pipeline as toolbar)
  if (has(low, /\b(export(?:ing)?|download|stl|cad import)\b/)) {
    const mode = /\b(each|individual|separate)\b.*\b(component|part|file)|\bseparate (?:files|parts)\b/.test(low) ? 'separate' : 'combined';
    let scope = 'entire', component_ids = [], levels = [];
    if (/\b(custom|user[- ]defined) (?:components?|parts?)\b/.test(low)) scope = 'custom';
    const lm = [...low.matchAll(/\blevel\s*([1-6])\b/g)].map(x => Number(x[1]));
    if (lm.length) { scope = 'levels'; levels = [...new Set(lm)]; }
    if (/\b(outer )?sheath\b/.test(low) && /\bonly\b/.test(low)) { scope = 'components'; component_ids = ['sheath']; }
    if (/\b(am|anteromedial)\b.*\b(mantle|braid)\b.*\bonly\b/.test(low)) { scope = 'components'; component_ids = ['mantle_am']; }
    if (/\b(pl|posterolateral)\b.*\b(mantle|braid)\b.*\bonly\b/.test(low)) { scope = 'components'; component_ids = ['mantle_pl']; }
    const args = { scope, mode, detail: /\bhigh detail\b/.test(low) ? 'high' : /\bdraft\b/.test(low) ? 'draft' : 'standard', component_ids, levels };
    if (/\b(explain|show)\b.*\b(export )?(warning|warnings)\b/.test(low)) action('export.explain_warnings', args);
    else if (/\b(check|inspect|validate|validation)\b/.test(low) && !/\b(download|generate)\b/.test(low)) action('export.inspect', args);
    else action('export.download', args);
    return { calls, notes };
  }
  // ---------------- fresh design inspection / grounded questions
  const designQuestion = /\?\s*$/.test(t) || /^(?:inspect|review|explain|why|what|which|how|are|is|tell me|show me (?:the )?current)\b/.test(low);
  if (designQuestion && has(low, /\b(what is|what are|which|how does|how do|are any|current|thinnest|minimum wall|not been performed|not performed|changed)\b/) &&
      has(low, /\b(channel|braid|reinforcement|sheath|wall|design|parameter|value|component|check|fit|invalid|failure|measurement|mri|attachment|material)\b/)) {
    action('design.inspect', { query: t });
    if (!has(low, /\b(evidence|literature|research|stud(?:y|ies)|published|pubmed)\b/)) return { calls, notes };
  }
  // ---------------- external evidence (generalized concepts only)
  if (has(low, /\b(evidence|literature|research|stud(?:y|ies)|published|pubmed)\b/)) {
    const topic = /channel|pore/.test(low) ? 'ACL scaffold channel diameter pore size tissue engineering' :
      /braid|reinforcement/.test(low) ? 'braided ligament scaffold reinforcement mechanical bench study' :
      /attachment|flare|insertion/.test(low) ? 'ACL scaffold bone attachment graded interface tissue engineering' :
      /sheath/.test(low) ? 'ligament tissue engineering scaffold sheath permeability' : 'ACL tissue engineering scaffold design';
    action('research.search', { query: topic, limit: 5 });
    return { calls, notes };
  }
  // ---------------- custom component requests require an explicit construction and dimensions
  if (has(low, /\b(add|create|make|insert)\b.*\b(sleeve|band|tube|channel array|strut|lattice|custom component)\b/)) {
    const op = /channel array/.test(low) ? 'channel_array' : /curved.*tube/.test(low) ? 'curved_tube' : (low.match(/\b(sleeve|band|tube|strut|lattice)\b/) || [])[1];
    const length = low.match(/(?:length|long)\s*(?:of|=|to)?\s*(\d+(?:\.\d+)?)\s*mm/);
    const diameter = low.match(/(?:diameter|wide|width)\s*(?:of|=|to)?\s*(\d+(?:\.\d+)?)\s*mm/);
    const wall = low.match(/wall(?: thickness)?\s*(?:of|=|to)?\s*(\d+(?:\.\d+)?)\s*mm/);
    const pos = low.match(/(?:at|position(?:ed)?(?: at)?)\s*\[?\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)\s*\]?\s*mm/);
    const independent = /\bindependent\b/.test(low);
    if (!op || !length || (!diameter && !/lattice/.test(op)) || !pos || !independent) {
      action('ask_user', { question: `I can create tube, curved_tube, band, sleeve, channel_array, strut, or lattice parts. What operation, length in mm, width/diameter in mm, and model-frame position or attachment target should I use?` });
      return { calls, notes };
    }
    const dimensions = { length_mm: Number(length[1]) };
    if (diameter) { dimensions.outer_diameter_mm = Number(diameter[1]); dimensions.width_mm = Number(diameter[1]); }
    if (wall) dimensions.wall_mm = Number(wall[1]);
    action('custom.create', { name: `Custom ${op.replace(/_/g, ' ')}`, purpose: 'User-requested scaffold component', operation: op,
      dimensions, position_mm: [Number(pos[1]), Number(pos[2]), Number(pos[3])], relationship: { mode: 'independent', target: null },
      assumptions: ['Position is interpreted in the scaffold model frame.'] });
    return { calls, notes };
  }
  // ---------------- sources / provenance
  if (has(low, /\bwhich (measurements?|settings?|values?|parameters?)\b.*\b(used|came from|come from|from the (mri|report|scan))\b|\bshow (the )?(sources|provenance)\b/)) { action('evidence.show_sources'); return { calls, notes }; }
  // ---------------- segmentation (must call the configured service)
  if (has(low, /\b(run|start)\b.*\b(segmentation|segment|model|kneescan)\b|\bsegment (the|this) (scan|volume|mri)\b/)) { action('seg.run', { do_bones: /\bbones?\b|fibula|femur/.test(low) }); return { calls, notes }; }
  // ---------------- evidence transfer (reports + reviewed measurements)
  const wantsEvidence = has(low, /\b(using|use|from|based on|apply)\b.*\b(report|reviewed|measurements?|mri measurements?|attachment positions?|measured)\b/) || has(low, /\badapt\b.*\b(measured|measurements?)\b/);
  if (wantsEvidence) {
    const scope = [];
    if (/attachment/.test(low)) scope.push('acl_length', 'tib_fp_length', 'tib_fp_width', 'fem_fp_length', 'fem_fp_width');
    if (/envelope|dimension|width|size/.test(low)) scope.push('acl_width', 'acl_length');
    if (/report/.test(low) || /\btear\b/.test(low)) scope.push('tear_type', 'tear_bundle', 'tear_location', 'tear_gap');
    if (/\breviewed (mri )?measurements?\b|\ball\b|\bmeasurements\b/.test(low) && !/attachment|envelope/.test(low)) scope.push('acl_length', 'acl_width', 'tear_gap', 'tib_fp_length', 'tib_fp_width', 'fem_fp_length', 'fem_fp_width');
    action('evidence.apply', { scope: scope.length ? [...new Set(scope)] : null, keep, include_assumptions: /\binclude assumptions?\b|\buse assumptions?\b/.test(low) });
    if (/attachment positions?/.test(low)) notes.push('Attachment *positions* need a registration between the MRI and the scaffold frame, which is not available; the distance between reviewed attachment landmarks can still set the ligament length.');
  }
  // ---------------- workspace
  if (has(low, /\b(axial|sagittal|coronal)\b.*\b(axial|sagittal|coronal)\b/) && has(low, /\b(open|show|view|display|put|lay)\b/))
    action('ws.preset', { name: /scaffold|model|design/.test(low) ? 'mpr_scaffold' : 'mpr' });
  else { const m = low.match(/\b(open|show)\b(?: an?| the)? (axial|sagittal|coronal) (?:view|plane|pane)/); if (m) action('ws.open_view', { view: m[2], where: /right|next to|beside/.test(low) ? 'right' : 'active' }); }
  if (has(low, /\b(quality[- ]control|qc)\b/) && has(low, /\b(open|show|panel|run)\b/)) action('ws.open_view', { view: 'qc', where: 'right' });
  if (has(low, /\b(evidence|mapping) panel\b|\bopen (the )?evidence\b/)) action('ws.open_view', { view: 'evidence', where: 'right' });
  if (has(low, /\bmaximi[sz]e\b/)) action('ws.maximize');
  if (has(low, /\brestore (the )?layout\b|\bun-?maximi[sz]e\b/)) action('ws.restore');
  // ---------------- crosshair
  if (has(low, /\bcrosshair\b.*\b(selected )?annotation\b|\bgo to (the )?selected annotation\b/)) action('img.crosshair.goto_annotation');
  else if (has(low, /\bcent(er|re) (the )?crosshairs?\b/)) action('img.crosshair.center');
  else { const m = low.match(/\bcrosshairs? (?:to|at) (?:the )?(?:centre|center) of (?:the )?([a-z ]+?)(?: label| segmentation)?$/); if (m) action('img.crosshair.goto_label', { label: m[1].trim() }); }
  const xyz = low.match(/\bcrosshair\b.*?(-?\d+(?:\.\d+)?)[ ,]+(-?\d+(?:\.\d+)?)[ ,]+(-?\d+(?:\.\d+)?)/);
  if (xyz) action('img.crosshair.set_world', { x: +xyz[1], y: +xyz[2], z: +xyz[3] });
  if (has(low, /\b(hide|turn off)\b.*\bcrosshairs?\b/)) calls.push(setc('img.crosshair.visible', false));
  if (has(low, /\b(show|turn on)\b.*\bcrosshairs?\b/) && !has(low, /annotation/)) calls.push(setc('img.crosshair.visible', true));
  // ---------------- overlays and windowing
  if (has(low, /\b(original|raw|plain) (mri|scan|image)\b.*\bwithout\b|\bhide (all )?(the )?(annotations|overlays|segmentations?)\b/)) action('img.show_original');
  const seg = low.match(/\bshow (?:the |me the )?([a-z][a-z ]*?) (?:segmentation|label|mask|overlay)\b/);
  if (seg && !/original/.test(seg[1])) action('img.show_label', { label: seg[1].replace(/\b(acl)\b/, 'ACL') });
  const op = low.match(/\b(?:overlay|segmentation|label|mask)s? opacity\b.*?(\d+(?:\.\d+)?)\s*(%|percent)?|\b(?:reduce|set|lower|raise|increase|change)\b.*?\b(?:overlay|segmentation) opacity\b.*?(\d+(?:\.\d+)?)\s*(%|percent)?/);
  if (op) { const v = Number(op[1] || op[3]), pct = op[2] || op[4] || v > 1; calls.push(setc('img.overlay.opacity', pct ? v / 100 : v)); }
  if (has(low, /\breset\b.*\bwindow(ing)?\b|\breset (the )?(brightness|contrast)\b/)) action('img.window.reset', { mode: 'initial' });
  if (has(low, /\bauto(matic)? (window|display range|contrast)\b/)) action('img.window.reset', { mode: 'auto' });
  if (has(low, /\binvert\b/)) calls.push(setc('img.window.invert', !/\b(un-?invert|don'?t invert|no invert)\b/.test(low)));
  const ww = low.match(/\bwindow width\b.*?(\d+(?:\.\d+)?)/); if (ww) calls.push(setc('img.window.width', +ww[1]));
  const wl = low.match(/\bwindow (?:level|centre|center)\b.*?(-?\d+(?:\.\d+)?)/); if (wl) calls.push(setc('img.window.level', +wl[1]));
  const cm = low.match(/\b(gray|grey|hot|viridis|cool[_ ]warm)\b.*\b(colou?r ?map|colormap)\b|\b(colou?r ?map|colormap)\b.*\b(gray|grey|hot|viridis|cool[_ ]warm)\b/);
  if (cm) calls.push(setc('img.colormap', (cm[1] || cm[4]).replace('grey', 'gray').replace(' ', '_')));
  // ---------------- scaffold view
  if (has(low, /\b(inside|within|in) the (acl|ligament|knee)\b/) && has(low, /\bshow\b/)) { calls.push(setc('view.mode', 'anatomy')); calls.push(setc('view.acl_overlay', true)); }
  if (has(low, /\b(builder|scaffold alone|scaffold by itself)\b/) && has(low, /\b(show|open|switch)\b/)) calls.push(setc('view.mode', 'builder'));
  for (const [w, k] of [['bones', 'view.bones'], ['labels', 'view.labels']]) {
    if (new RegExp(`\\bhide (the )?${w}\\b`).test(low)) calls.push(setc(k, false));
    else if (new RegExp(`\\bshow (the )?${w}\\b`).test(low)) calls.push(setc(k, true));
  }
  const cv = low.match(/\b(anterior|posterior|medial|lateral|superior)\b (?:camera )?view\b/); if (cv) action('camera.view', { preset: cv[1] });
  if (has(low, /\b(?:show|open)\b.*\bmidpoint\b.*\bsagittal (?:section|cross[- ]section)\b/)) { calls.push(setc('display.section_kind', 'sagittal')); calls.push(setc('display.section_s', 0.5)); }
  const lvl = low.match(/\b(isolate|highlight|select|show)\b (?:scaffold )?level (\d)\b/);
  if (lvl) { calls.push(setc('view.level', +lvl[2])); calls.push(setc('view.level_mode', lvl[1] === 'isolate' ? 'isolate' : 'highlight')); }
  // ---------------- generic parameter edits: "set <control> to <number><unit>", "increase <control> by 10%"
  const clauses = t.split(/[;\n]|,?\s*\band\s+(?=(?:set|change|make|increase|decrease|reduce|raise|lower)\b)|,\s*(?=(?:set|change|make|increase|decrease|reduce|raise|lower)\b)|\band then\b|\.\s+/i);
  for (const cl of clauses) {
    const m = cl.match(new RegExp(`\\b(set|change|make|put|increase|decrease|reduce|raise|lower)\\s+(?:the\\s+)?(.+?)\\s+(to|by|=|at)\\s+(-?\\d+(?:\\.\\d+)?)\\s*${UNIT_RE}`, 'i'));
    const me = !m && cl.match(/\b(?:set|change|make|switch)\s+(?:the\s+)?(.+?)\s+to\s+([a-z_][a-z0-9_ -]*?)\s*$/i);
    if (m) {
      const [, verb, what, rel, num, unit] = m;
      if (/overlay|segmentation|window/.test(what.toLowerCase()) && calls.some(c => c.name === 'set_controls' && /^img\./.test(c.args.changes[0].key))) continue;
      const r = resolve(reg, what, e => !e.readOnly);
      if (r.ambiguous) { action('ask_user', { question: `"${what}" matches several controls: ${r.ambiguous.map(e => `${e.label} (${e.key}${e.units ? ', ' + e.units : ''})`).join('; ')}. Which one?` }); continue; }
      if (!r.entry) { const ro = resolve(reg, what); notes.push(ro.entry && ro.entry.readOnly ? `${ro.entry.label} is read-only (${ro.entry.readOnly}).` : `No control matches "${what}".`); continue; }
      const e = r.entry; let x = Number(num);
      const dec = /decrease|reduce|lower/i.test(verb) ? -1 : 1;
      if (rel.toLowerCase() === 'by') {
        const cur = Number(e.get());
        if (!Number.isFinite(cur)) { notes.push(`${e.label} has no numeric value to change by.`); continue; }
        if (unit === '%' || /percent/i.test(unit || '')) x = cur * (1 + dec * x / 100);
        else { const cv2 = convert(x, unit, e); if (cv2.error) { notes.push(cv2.error); continue; } x = cur + dec * cv2.value; }
      } else { const cv2 = convert(x, unit, e); if (cv2.error) { notes.push(cv2.error); continue; } x = cv2.value; }
      calls.push(setc(e.key, e.kind === 'int' ? Math.round(x) : +x.toPrecision(10)));
    } else if (me) {
      const r = resolve(reg, me[1], e => e.kind === 'enum' && !e.readOnly);
      if (r.entry) {
        const opt = (r.entry.options || []).find(o => o.toLowerCase() === me[2].trim().toLowerCase().replace(/ /g, '_'));
        if (opt) calls.push(setc(r.entry.key, opt)); else notes.push(`${r.entry.label} options: ${(r.entry.options || []).join(', ')}.`);
      } else if (r.ambiguous) action('ask_user', { question: `Which control: ${r.ambiguous.map(e => e.label + ' (' + e.key + ')').join('; ')}?` });
    }
  }
  // merge set_controls into one transaction
  const sets = calls.filter(c => c.name === 'set_controls'), rest = calls.filter(c => c.name !== 'set_controls');
  const merged = sets.length ? [{ name: 'set_controls', args: { changes: sets.flatMap(c => c.args.changes), keep } }] : [];
  if (!merged.length && keep.length && !rest.length) notes.push('Nothing to change; the settings you named stay as they are.');
  return { calls: [...merged, ...rest], notes, keep };
}
const setc = (key, value) => ({ name: 'set_controls', args: { changes: [{ key, value }] } });
