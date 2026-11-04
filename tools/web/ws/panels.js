// Workspace panels: imaging browser (left), imaging inspector (right, when an MRI pane is active), quality control,
// evidence & mappings, control coverage. All actions go through the imaging controller (IM) or the transaction engine.
import * as M from './volmath.js';
import { TOOLS, NEW_LABELS, api } from './imaging.js';

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const h = (tag, attrs, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') el.className = v; else if (k === 'html') el.innerHTML = v; else if (k.startsWith('on')) el[k] = v;
    else if (v === true) el.setAttribute(k, ''); else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(c));
  return el;
};
export { h, esc };
const chip = (status, text) => h('span', { class: 'st ' + status }, text || status.replace('_', ' '));
const worst = (checks) => checks.some(c => c.status === 'fail') ? 'fail' : checks.some(c => c.status === 'warn') ? 'warn' : 'pass';
const DEFAULT_OPEN_SECTIONS = new Set(['Series in this pane', 'Navigation', 'Image windowing']);
const sec = (title, ...kids) => h('details', { class: 'isec', open: DEFAULT_OPEN_SECTIONS.has(title) }, h('summary', {}, title), h('div', { class: 'isb' }, ...kids));
const row = (label, ...kids) => h('div', { class: 'irow' }, h('label', {}, label), ...kids);
const num = (val, onchange, attrs) => { const i = h('input', Object.assign({ type: 'number', step: 'any', value: val == null ? '' : +Number(val).toPrecision(6) }, attrs || {})); i.onchange = () => onchange(Number(i.value)); return i; };
const paintRange = input => {
  const min = Number(input.min), max = Number(input.max), value = Number(input.value);
  const pct = max === min ? 0 : Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  input.style.setProperty('--range-fill', pct + '%');
  return input;
};
const rangeInput = (attrs, oninput) => {
  const input = h('input', Object.assign({ type: 'range' }, attrs || {}));
  input.oninput = e => { paintRange(input); if (oninput) oninput(e); };
  return paintRange(input);
};
const rangeControl = (label, input, value, opts = {}) => h('div', { class: 'range-control' },
  h('div', { class: 'range-head' }, h('label', {}, label), h('div', { class: 'range-current' }, value, opts.unit ? h('span', { class: 'range-unit' }, opts.unit) : null)),
  input,
  h('div', { class: 'range-bounds' }, h('span', {}, opts.minLabel ?? fmt(Number(input.min))), h('span', {}, opts.maxLabel ?? fmt(Number(input.max)))));
const btn = (text, onclick, attrs) => h('button', Object.assign({ onclick }, attrs || {}), text);
const fmt = (v, d = 2) => v == null ? '—' : typeof v === 'number' ? (+v.toFixed(d)).toString() : String(v);

// ===================================================================== imaging browser (left sidebar)
export function renderBrowser(ws) {
  const IM = ws.IM, box = document.getElementById('browser');
  box.innerHTML = '';
  const cases = ws.cases || [];
  const sel = h('select', { 'aria-label': 'Imaging case', onchange: e => e.target.value && ws.openCase(e.target.value), 'data-ctl': '(case selector)' },
    h('option', { value: '' }, cases.length ? '— choose a case —' : 'no cases yet'), ...cases.map(c => h('option', { value: c.id, selected: c.id === IM.caseId }, c.label)));
  box.append(h('div', { class: 'bhead' }, h('b', {}, 'Imaging'), h('span', { class: 'grow' }),
    btn('New', async () => { const l = prompt('Case name', 'Knee MRI'); if (l) { const c = await api('/api/img/cases', { label: l }); await ws.loadCases(); ws.openCase(c.id); } }, { title: 'New empty case' }),
    btn('Demo', () => ws.demoCase(), { title: 'Load the synthetic knee phantom (never patient data)' })), sel);
  if (!IM.manifest) { box.append(h('p', { class: 'note' }, 'Open a case, create one and import a scan, or load the synthetic demo phantom.')); return; }
  if (IM.manifest.synthetic) box.append(h('div', { class: 'synth' }, 'SYNTHETIC phantom — not patient data'));
  // ---- import
  const role = h('select', { 'aria-label': 'Import as', 'data-ctl': '(import role)' }, h('option', { value: 'image' }, 'Image series / volume'), h('option', { value: 'mask' }, 'Mask for the selected series'), h('option', { value: 'derived' }, 'Derived / quantitative map'));
  const qty = h('input', { placeholder: 'quantity (e.g. T2 map)', class: 'small', hidden: true, 'data-ctl': '(import quantity)' });
  const qun = h('input', { placeholder: 'units (e.g. ms)', class: 'small', hidden: true, 'data-ctl': '(import units)' });
  const resample = h('label', { class: 'small', hidden: true }, h('input', { type: 'checkbox', 'data-ctl': '(import resample)' }), ' resample to image grid if needed');
  role.onchange = () => { qty.hidden = qun.hidden = role.value !== 'derived'; resample.hidden = role.value !== 'mask'; };
  const files = h('input', { type: 'file', multiple: true, 'data-ctl': '(import files)', accept: '.dcm,.zip,.nii,.gz,.npy,.png,.jpg,.jpeg,.tif,.tiff' });
  const folder = h('input', { type: 'file', multiple: true, webkitdirectory: true, 'data-ctl': '(import folder)' });
  const go = (list) => ws.importFiles([...list], { role: role.value, quantity: qty.value, value_units: qun.value, resample: resample.querySelector('input').checked });
  files.onchange = () => go(files.files); folder.onchange = () => go(folder.files);
  box.append(h('details', { class: 'imp', open: !Object.keys(IM.manifest.volumes).length }, h('summary', {}, 'Import'),
    role, qty, qun, resample, h('div', { class: 'small' }, 'Files: ', files), h('div', { class: 'small' }, 'DICOM folder: ', folder),
    h('p', { class: 'note' }, 'DICOM (.dcm, folder or .zip), NIfTI, .npy, PNG/JPEG/TIFF stacks. Series are grouped, ordered and validated; problems are listed, not hidden.'),
    h('div', { id: 'impStatus' })));
  if (ws.importState) box.querySelector('#impStatus').append(importStatus(ws));
  // ---- tree: study -> series -> masks
  const vols = IM.imageVolumes();
  const studies = new Map();
  for (const v of vols) { const k = (v.meta && v.meta.study_key) || ('file:' + v.fmt); (studies.get(k) || studies.set(k, []).get(k)).push(v); }
  for (const [k, list] of studies) {
    const st = list[0].meta && (list[0].meta.StudyDescription || list[0].meta.study_key);
    box.append(h('div', { class: 'study' }, k.startsWith('file:') ? `Files (${k.slice(5)})` : `Study ${esc(st)}`));
    for (const v of list) box.append(seriesRow(ws, v));
  }
}
function importStatus(ws) {
  const s = ws.importState, el = h('div', { class: 'job' });
  if (s.status === 'running' || s.status === 'uploading') {
    el.append(h('div', {}, s.message || 'working…'), h('div', { class: 'bar' }, h('i', { style: `width:${Math.round((s.progress || 0) * 100)}%` })), btn('Cancel', () => ws.cancelImport()));
  } else if (s.status === 'error') {
    el.classList.add('err');
    el.append(h('b', {}, 'Import failed: ' + (s.error || '')));
    for (const c of s.checks || []) el.append(h('div', { class: 'chk ' + c.status }, chip(c.status), ' ', c.message, c.hint ? h('div', { class: 'hint' }, '→ ' + c.hint) : null));
  } else if (s.status === 'done') {
    el.append(h('div', {}, `Imported ${s.result.volumes.length} series, ${s.result.masks.length} mask(s).`));
    for (const f of s.result.failed || []) { el.append(h('div', { class: 'err' }, h('b', {}, `Not imported: ${f.name}`))); for (const c of f.checks) el.append(h('div', { class: 'chk ' + c.status }, chip(c.status), ' ', c.message, c.hint ? h('div', { class: 'hint' }, '→ ' + c.hint) : null)); }
  } else if (s.status === 'cancelled') el.append(h('div', {}, 'Import cancelled.'));
  return el;
}
function seriesRow(ws, v) {
  const IM = ws.IM, act = v.id === IM.activeVol;
  const checks = v.checks || [], w = worst(checks);
  const m = v.meta || {};
  const seq = [m.SeriesDescription || m.description, m.MRAcquisitionType, m.RepetitionTime && `TR ${m.RepetitionTime}`, m.EchoTime && `TE ${m.EchoTime}`, m.MagneticFieldStrength && `${m.MagneticFieldStrength} T`].filter(Boolean).join(' · ');
  const st = v.loaded ? 'loaded' : v.loading ? `loading ${Math.round((v.progress || 0) * 100)}%` : v.error ? 'error: ' + v.error : 'not loaded';
  const el = h('div', { class: 'series' + (act ? ' on' : ''), onclick: () => { IM.activeVol = v.id; ws.showSeries(v.id); } },
    h('div', { class: 'sname' }, chip(w, w === 'pass' ? '✓' : w === 'warn' ? '!' : '✖'), ' ', h('b', {}, v.name), v.role === 'derived' ? h('span', { class: 'tag' }, 'derived: ' + v.quantity) : null),
    h('div', { class: 'smeta mono' }, `${v.shape.join(' × ')} · ${v.spacing_known ? v.zooms.map(z => (+z).toFixed(2)).join(' × ') + ' mm' : 'uncalibrated'} · ${v.fmt}`),
    seq ? h('div', { class: 'smeta' }, seq) : null,
    h('div', { class: 'smeta' }, `${v.orientation_known ? 'oriented (RAS)' : 'no orientation'} · ${v.frame_id ? 'frame ' + v.frame_id.slice(0, 6) : 'no shared frame'} · ${st}`),
    h('div', { class: 'sbtns' }, ...['axial', 'sagittal', 'coronal'].map(p => btn(p[0].toUpperCase() + p.slice(1, 3), (e) => { e.stopPropagation(); ws.openMpr(v.id, p, 'right'); }, { title: 'Open ' + p + ' view' })),
      btn('MPR+Scaffold', (e) => { e.stopPropagation(); IM.activeVol = v.id; ws.preset('mpr_scaffold'); }, { title: 'Three synchronized planes next to the scaffold' }),
      btn('Remove…', (e) => { e.stopPropagation(); IM.removeData('volume', v.id, v.name); }, { class: 'danger', title: 'Remove this series from the case (deletes its data)' })));
  const det = h('details', { class: 'schecks' }, h('summary', {}, `Validation (${checks.filter(c => c.status !== 'pass').length} flag${checks.filter(c => c.status !== 'pass').length === 1 ? '' : 's'})`),
    ...checks.map(c => h('div', { class: 'chk ' + c.status }, chip(c.status), ' ', c.message, c.hint ? h('div', { class: 'hint' }, '→ ' + c.hint) : null)));
  det.onclick = e => e.stopPropagation();
  el.append(det);
  for (const mk of IM.masksFor(v.id)) {
    const on = IM.activeMask.get(v.id) === mk.id;
    el.append(h('div', { class: 'mask' + (on ? ' on' : ''), onclick: (e) => { e.stopPropagation(); IM.selectMask(v.id, on ? null : mk.id); } },
      h('span', { class: 'swatches' }, ...mk.labels.map(l => h('i', { style: `background:${l.color}`, title: l.name }))),
      ` ${originText(mk)} `, chip(mk.review.status === 'reviewed' ? 'pass' : mk.review.status === 'rejected' ? 'fail' : 'warn', mk.review.status),
      h('span', { class: 'muted small' }, ' ' + mk.labels.map(l => l.name).join(', '))));
  }
  const n = IM.annotationsFor(v.id).length;
  if (n) el.append(h('div', { class: 'smeta' }, `${n} annotation${n > 1 ? 's' : ''} (${IM.annotationsFor(v.id).filter(a => a.status === 'reviewed').length} reviewed)`));
  return el;
}
export const originText = mk => ({ model: `Model: ${(mk.model && mk.model.name) || 'prediction'}`, edited: `Edited from ${mk.model ? mk.model.name : 'mask'}`, import: 'Imported mask', manual: 'Manual mask' }[mk.origin] || mk.origin);

// ===================================================================== imaging inspector
export function renderImgInspector(ws) {
  const IM = ws.IM, box = document.getElementById('insImg'), view = ws.activeMpr();
  const priorSections = [...box.querySelectorAll('details.isec')];
  const priorSectionTitles = new Set(priorSections.map(d => d.querySelector('summary').textContent));
  const openSections = new Set(priorSections.filter(d => d.open).map(d => d.querySelector('summary').textContent));
  box.innerHTML = '';
  if (!view) { box.append(h('p', { class: 'note' }, 'Activate an MRI pane to see imaging controls.')); return; }
  const v = view.v;
  // series & pane
  const volSel = h('select', { 'data-ctl': '(pane series)', onchange: e => { view.setVolume(e.target.value); ws.layout.render(); ws.refreshInspector(); } },
    ...IM.imageVolumes().map(x => h('option', { value: x.id, selected: x.id === view.vid }, x.name)));
  const plSel = h('select', { 'data-ctl': '(pane plane)', onchange: e => { view.planeKey = e.target.value; view.zoom = null; view.draw(); ws.layout.render(); } },
    ...['axial', 'sagittal', 'coronal'].map(p => h('option', { value: p, selected: p === view.planeKey }, v ? M.planeOf(v, p).name : p)));
  box.append(sec('Series in this pane', row('Series', volSel), row('Plane', plSel),
    v ? h('div', { class: 'note mono' }, `${v.shape.join('×')} · ${v.spacing_known ? v.zooms.map(z => (+z).toFixed(3)).join('×') + ' mm' : 'uncalibrated'} · ${v.orientation_known ? 'RAS+ (from ' + (v.fmt === 'dicom' ? 'DICOM' : 'NIfTI') + ')' : 'orientation unknown'}`) : null,
    v ? row('Linked series', h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: !IM.linkOff.has(v.id), disabled: !v.frame_id, 'data-ctl': 'img.series.linked',
      onchange: e => ws.set('img.series.linked', e.target.checked) }), v.frame_id ? ` ${IM.linkedWith(v.id).length} other series share this frame of reference` : ' none: no verified frame of reference')) : null));
  if (!v || !v.loaded) { box.append(h('p', { class: 'note' }, v ? 'Loading…' : 'Choose a series.')); return; }
  // navigation
  const tools = h('div', { class: 'tools', role: 'toolbar', 'data-ctl-group': 'img.tool' }, ...TOOLS.map(([id, name, key, tip]) =>
    btn(name, () => ws.set('img.tool', id), { class: 'tool' + (IM.tool === id ? ' on' : ''), title: `${tip} (${key})` })));
  const w = IM.crossWorld(v.id), c = IM.crossVoxel(v.id).map(Math.round);
  const wIn = [0, 1, 2].map(a => num(w[a], () => IM.setCross(v.id, wIn.map(i => Number(i.value)), 'numeric'), { 'aria-label': 'RAS ' + 'xyz'[a], 'data-ctl': 'img.crosshair.set_world', disabled: !v.orientation_known }));
  const vIn = [0, 1, 2].map(a => num(c[a], () => IM.setCross(v.id, M.voxelToWorld(v, vIn.map(i => Number(i.value))), 'numeric'), { 'aria-label': 'voxel ' + 'ijk'[a], step: 1, min: 0, max: v.shape[a] - 1, 'data-ctl': '(voxel navigation)' }));
  box.append(sec('Navigation', tools,
    h('div', { class: 'note' }, 'Active tool shown in every pane. Wheel: slice · Ctrl/⌘+wheel: zoom · middle-drag: pan · arrows: move crosshair (Shift ×5) · PageUp/PageDown: slice · F: fit.'),
    row('Crosshair', h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: IM.crossVisible, 'data-ctl': 'img.crosshair.visible', onchange: e => ws.set('img.crosshair.visible', e.target.checked) }), ' show'),
      btn('Center', () => ws.act('img.crosshair.center'), { 'data-ctl': 'img.crosshair.center' }), btn('Go to selected annotation', () => ws.act('img.crosshair.goto_annotation'), { disabled: !IM.selectedAnn, 'data-ctl': 'img.crosshair.goto_annotation' })),
    v.orientation_known ? row('RAS (mm)', ...wIn) : h('div', { class: 'note' }, 'No orientation: RAS coordinates unavailable.'), row('Voxel (i j k)', ...vIn)));
  // windowing
  const d = IM.dispFor(view.id, v.id), [lo, hi] = M.windowToRange(d.center, d.width), span = (v.range[1] - v.range[0]) || 1;
  const wl = rangeInput({ min: v.range[0], max: v.range[1], step: span / 500, value: d.center, 'data-ctl': 'img.window.level', 'aria-label': 'Window level' }, e => IM.setDisp(view.id, v.id, { center: Number(e.target.value) }));
  const ww = rangeInput({ min: span / 500, max: span * 2, step: span / 500, value: d.width, 'data-ctl': 'img.window.width', 'aria-label': 'Window width' }, e => IM.setDisp(view.id, v.id, { width: Number(e.target.value) }));
  const intact = v.checksum0 && M.checksum(v.data) === v.checksum0;
  box.append(sec('Image windowing',
    h('div', { class: 'control-intro' }, h('b', {}, 'Brightness and contrast'), h('span', {}, 'Drag in the image or use these exact controls.')),
    rangeControl('Level', wl, num(d.center, x => ws.set('img.window.level', x), { 'data-ctl': 'img.window.level', 'aria-label': 'Exact window level' }), { minLabel: fmt(v.range[0]), maxLabel: fmt(v.range[1]), unit: 'signal' }),
    rangeControl('Width', ww, num(d.width, x => ws.set('img.window.width', x), { 'data-ctl': 'img.window.width', 'aria-label': 'Exact window width' }), { minLabel: fmt(span / 500), maxLabel: fmt(span * 2), unit: 'range' }),
    h('div', { class: 'control-actions' }, btn('Reset display', () => ws.act('img.window.reset', { mode: 'initial' }), { title: 'Initial display: ' + ((v.initial_window || {}).method || (v.auto_window || {}).method || 'data range'), 'data-ctl': 'img.window.reset' }),
      btn('Auto range', () => ws.act('img.window.reset', { mode: 'auto' }), { title: (v.auto_window || {}).method, 'data-ctl': 'img.window.reset' }),
      h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: d.invert, 'data-ctl': 'img.window.invert', onchange: e => ws.set('img.window.invert', e.target.checked) }), ' invert'),
      h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: IM.disp.get(v.id).linked, 'data-ctl': 'img.window.linked', onchange: e => ws.set('img.window.linked', e.target.checked) }), ' link across views')),
    h('div', { class: 'note' }, `Display only: intensities are ${v.role === 'derived' ? v.quantity + (v.value_units ? ' (' + v.value_units + ')' : '') : 'MRI signal in arbitrary scanner units (not Hounsfield units)'}. Voxel data ${intact ? 'unchanged ✓ (checksum)' : 'CHANGED — report this'}.`)));
  // colour map
  const cm = h('select', { 'data-ctl': 'img.colormap', onchange: e => ws.set('img.colormap', e.target.value) }, ...M.COLORMAPS.map(n => h('option', { value: n, selected: d.cmap === n }, n)));
  box.append(sec('Colour map', row('Map', cm),
    row('Range', num(lo, x => IM.setDisp(view.id, v.id, { center: (x + hi) / 2, width: Math.max(1e-6, hi - x) }), { 'data-ctl': 'img.window.level' }), '–',
      num(hi, x => IM.setDisp(view.id, v.id, { center: (lo + x) / 2, width: Math.max(1e-6, x - lo) }), { 'data-ctl': 'img.window.width' }), btn('Reset', () => ws.act('img.window.reset', { mode: 'initial' }))),
    h('div', { class: 'note' }, v.role === 'derived' ? `Shows ${v.quantity}${v.value_units ? ' in ' + v.value_units : ''}, as imported.` :
      'Grayscale is the default. A colour map on MRI intensity is a display choice: it is not a tissue, damage or probability map.')));
  // overlay
  const masks = IM.masksFor(v.id), mm = IM.maskMeta(v.id), ed = IM.edit && IM.edit.vid === v.id;
  const mSel = h('select', { 'data-ctl': '(mask selector)', onchange: e => IM.selectMask(v.id, e.target.value || null) }, h('option', { value: '' }, '— none —'),
    ...masks.map(m => h('option', { value: m.id, selected: mm && mm.id === m.id }, `${originText(m)} · ${m.review.status} · ${m.created.slice(11, 19)}`)));
  const labs = IM.labelsFor(v.id);
  const overlayRange = rangeInput({ min: 0, max: 1, step: 0.05, value: IM.overlay.opacity, 'data-ctl': 'img.overlay.opacity', 'aria-label': 'Overlay opacity' }, e => { IM.overlay.opacity = Number(e.target.value); IM.emit('overlay'); });
  box.append(sec('Segmentation overlay',
    row('Mask', mSel),
    h('div', { class: 'visibility-row' }, h('label', {}, h('input', { type: 'checkbox', checked: IM.overlay.visible, 'data-ctl': 'img.overlay.visible', onchange: e => ws.set('img.overlay.visible', e.target.checked) }), ' Show overlays')),
    rangeControl('Overlay opacity', overlayRange, h('output', {}, Math.round(IM.overlay.opacity * 100)), { minLabel: '0%', maxLabel: '100%', unit: '%' }),
    ...labs.map(l => { const o = IM.overlay.labels.get(l.name) || { visible: true, opacity: 1 };
      const labelRange = rangeInput({ min: 0, max: 1, step: 0.05, value: o.opacity, 'data-ctl': `img.label.${l.name}.opacity`, 'aria-label': l.name + ' opacity' }, e => { IM.overlay.labels.set(l.name, Object.assign({}, o, { opacity: Number(e.target.value) })); IM.emit('overlay'); });
      return h('div', { class: 'irow labrow' }, h('i', { class: 'sw', style: `background:${l.color}` }),
        h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: o.visible, 'data-ctl': `img.label.${l.name}.visible`, onchange: e => ws.set(`img.label.${l.name}.visible`, e.target.checked) }), ' ' + l.name + (l.dimensionality ? ` (${l.dimensionality})` : '') + (l.experimental ? ' · experimental' : '')),
        labelRange,
        btn('Go to', () => ws.act('img.crosshair.goto_label', { label: l.name }), { title: 'Move the crosshair to this label', 'data-ctl': 'img.crosshair.goto_label' })); }),
    h('div', { class: 'note' }, 'Overlay opacity affects segmentation labels only — not image windowing and not 3D scaffold materials. Label colours are the same in every pane.')));
  // annotation editing
  const edBox = h('div');
  if (ed) {
    const e = IM.edit;
    const labSel = h('select', { 'data-ctl': '(active label)', onchange: ev => { IM.activeLabel = Number(ev.target.value); IM.emit('edit'); } }, ...e.labels.map(l => h('option', { value: l.value, selected: Number(l.value) === IM.activeLabel }, l.name)));
    const addSel = h('select', { 'data-ctl': '(add label)', onchange: ev => { if (ev.target.value) IM.addLabel(ev.target.value); ws.refreshInspector(); } }, h('option', { value: '' }, '+ label…'), ...NEW_LABELS.filter(n => !e.labels.some(l => l.name === n)).map(n => h('option', { value: n }, n)));
    edBox.append(h('div', { class: 'note' }, e.origin === 'edited' ? 'Editing a copy: the original prediction is kept for comparison.' : 'New manual mask.'),
      row('Label', labSel, addSel),
      rangeControl('Brush diameter', rangeInput({ min: 0.25, max: 10, step: 0.25, value: IM.brush, 'data-ctl': 'img.brush.size', 'aria-label': 'Brush diameter' }, ev => { IM.brush = Number(ev.target.value); ws.refreshInspector(); }), h('output', {}, IM.brush), { minLabel: '0.25', maxLabel: '10', unit: v.spacing_known ? 'mm' : 'vox' }),
      h('div', { class: 'irow' }, btn('Undo', () => IM.undoEdit(), { disabled: !e.undo.length, title: 'Ctrl/⌘+Z in the pane' }), btn('Redo', () => IM.redoEdit(), { disabled: !e.redo.length }),
        btn('Save version', async () => { try { await IM.saveEdit(); } catch (x) { alert(x.message); } }, { class: 'primary', disabled: !e.dirty }), btn('Discard', () => IM.discardEdit())),
      h('div', { class: 'note' }, `${e.undo.length} edit step(s)${e.dirty ? ', unsaved' : ''}. Use Brush (B), Erase (E) or Polygon (G).`));
  } else {
    edBox.append(h('div', { class: 'irow' }, btn(mm ? 'Edit a copy of this mask' : 'New manual mask', () => { IM.startEdit(v.id); if (!['brush', 'erase', 'polygon'].includes(IM.tool)) IM.setTool('brush'); }, { 'data-ctl': '(start edit)' })));
    if (mm) {
      const md = mm.model;
      edBox.append(h('div', { class: 'prov' },
        h('div', {}, h('b', {}, originText(mm)), ' · ', chip(mm.review.status === 'reviewed' ? 'pass' : mm.review.status === 'rejected' ? 'fail' : 'warn', mm.review.status)),
        md ? h('div', { class: 'small' }, `Model: ${md.name}. ${md.dimensionality || ''}`) : null,
        md && md.settings ? h('div', { class: 'small mono' }, 'Settings: ' + JSON.stringify(md.settings)) : null,
        md && md.confidence != null ? h('div', { class: 'small' }, `Model-reported confidence ${md.confidence} (${md.confidence_meaning}).`) : null,
        mm.edited ? h('div', { class: 'small' }, `Manually edited: ${mm.edit_voxels} voxel(s) changed from ${mm.parent}.`) : null,
        mm.note ? h('div', { class: 'small muted' }, mm.note) : null,
        h('div', { class: 'irow' }, btn('Mark reviewed', () => IM.review(mm.id, 'reviewed'), { disabled: mm.review.status === 'reviewed' }), btn('Reject', () => IM.review(mm.id, 'rejected')),
          btn('QC', () => ws.openKind('qc', 'right')), btn('Remove…', () => IM.removeData('mask', mm.id, 'this mask'), { class: 'danger' }))));
    }
  }
  box.append(sec('Annotation editing', edBox));
  // segmentation service
  box.append(sec('AI segmentation', segBox(ws, v)));
  // measurements
  const vocab = IM.vocab || { measurements: {}, landmarks: {} };
  const mk = h('select', { 'data-ctl': '(measurement kind)', onchange: e => { IM.measureKind = e.target.value; } }, ...Object.entries(vocab.measurements).map(([k, t]) => h('option', { value: k, selected: k === IM.measureKind, title: t }, t.split(':')[0])));
  const lk = h('select', { 'data-ctl': '(landmark kind)', onchange: e => { IM.landmarkKind = e.target.value; } }, ...Object.entries(vocab.landmarks).map(([k, t]) => h('option', { value: k, selected: k === IM.landmarkKind }, t)));
  const list = h('div', { class: 'anns' }, ...IM.annotationsFor(v.id).map(a => annRow(ws, a)));
  box.append(sec('Landmarks and measurements',
    v.spacing_known ? null : h('div', { class: 'warnbox' }, 'Uncalibrated volume: distances are in voxels and cannot be reviewed as millimetre measurements or transferred to the scaffold.'),
    row('Distance', mk, btn('Measure', () => ws.set('img.tool', 'distance'), { class: IM.tool === 'distance' ? 'on' : '' })),
    row('Landmark', lk, btn('Place', () => ws.set('img.tool', 'landmark'), { class: IM.tool === 'landmark' ? 'on' : '' })),
    list, h('div', { class: 'note' }, 'Positions and distances are computed by the server from the volume affine. Moving a reviewed point returns it to draft.')));
  if (priorSections.length) for (const d2 of box.querySelectorAll('details.isec')) {
    const title = d2.querySelector('summary').textContent;
    if (priorSectionTitles.has(title)) d2.open = openSections.has(title);
  }
}
export function annRow(ws, a) {
  const IM = ws.IM, sel = a.id === IM.selectedAnn;
  return h('div', { class: 'ann' + (sel ? ' on' : ''), onclick: () => { IM.selectedAnn = a.id; IM.emit('selection'); ws.refreshInspector(); } },
    h('div', {}, chip(a.status === 'reviewed' ? 'pass' : 'warn', a.status), ' ', h('b', {}, a.name || a.kind), a.type === 'distance' ? h('span', { class: 'mono' }, ` ${a.value} ${a.units}`) : null),
    h('div', { class: 'small mono muted' }, a.type === 'landmark' ? `RAS ${a.points_world[0].map(x => x.toFixed(1)).join(', ')}` : `${a.points_world.map(p => '(' + p.map(x => x.toFixed(1)).join(', ') + ')').join(' → ')}`),
    a.note ? h('div', { class: 'small muted' }, a.note) : null,
    h('div', { class: 'irow' }, btn('Go to', (e) => { e.stopPropagation(); IM.gotoAnnotation(a.id); }),
      a.status !== 'reviewed' ? btn('Mark reviewed', async (e) => { e.stopPropagation(); try { await IM.saveAnnotation(Object.assign({}, a, { status: 'reviewed' })); } catch (x) { alert(x.message); } }, { disabled: a.type === 'distance' && !a.calibrated })
        : btn('Back to draft', async (e) => { e.stopPropagation(); await IM.saveAnnotation(Object.assign({}, a, { status: 'draft' })); }),
      btn('Delete…', (e) => { e.stopPropagation(); IM.removeData('annotation', a.id, a.name || 'this annotation'); }, { class: 'danger' })));
}
function segBox(ws, v) {
  const IM = ws.IM, s = ws.segState || {};
  const first = !(window.__app && window.__app.S.state);
  const guessed = ws.kneeSide(v.id);
  ws.segSide = ws.segSide || guessed || '';
  const side = h('select', { onchange: e => { ws.segSide = e.target.value; ws.refreshInspectorSoon(); } },
    h('option', { value: '', selected: !ws.segSide }, 'Choose side…'),
    h('option', { value: 'right', selected: ws.segSide === 'right' }, 'Right knee'),
    h('option', { value: 'left', selected: ws.segSide === 'left' }, 'Left knee'));
  const url = h('input', { value: ws.readerUrl || '', placeholder: 'http://127.0.0.1:7860', 'data-ctl': '(segmentation service URL)', onchange: e => { ws.readerUrl = e.target.value; ws.checkReader(); } });
  const bones = h('input', { type: 'checkbox', 'data-ctl': '(segment bones)' });
  const file = h('input', { type: 'file', accept: '.json', 'data-ctl': '(import KneeScan results)' });
  file.onchange = async () => { const f = file.files[0]; if (!f) return; try { const r = JSON.parse(await f.text()); await ws.importModelResult(v.id, r, f.name, ws.segSide); } catch (e) { alert('Not imported: ' + e.message); } };
  const out = h('div', {},
    h('div', { class: 'note' }, 'Model: KneeScan (mri_analysis/) — MedSAM ACL outline on one sagittal slice (2D), plus experimental bone outlines. Results are shown as 2D outlines, never as a 3D segmentation.'),
    row('Service', url), h('div', { class: 'small ' + (ws.reader && ws.reader.ok ? 'ok' : 'bad') }, ws.reader ? (ws.reader.ok ? `reachable at ${ws.reader.url}` : `not reachable: ${ws.reader.error}`) : 'checking…'),
    first ? row('Knee side', side) : null,
    h('div', { class: 'irow' }, h('label', { class: 'small' }, bones, ' also outline femur/tibia/fibula (+1 min)'),
      btn('Run on this series', () => ws.act('seg.run', { do_bones: bones.checked, side: ws.segSide }), { class: 'primary', disabled: !(ws.reader && ws.reader.ok) || s.status === 'running' || (first && !ws.segSide), 'data-ctl': 'seg.run' })),
    s.status ? h('div', { class: 'job' + (s.status === 'error' ? ' err' : '') }, s.message || s.status, s.status === 'running' ? h('div', { class: 'bar' }, h('i', { style: `width:${Math.round((s.progress || 0) * 100)}%` })) : null,
      s.status === 'running' ? btn('Cancel', () => ws.cancelSeg()) : null) : null,
    h('div', { class: 'small' }, 'Or import a KneeScan results JSON (e.g. from the Colab notebook): ', file));
  return out;
}

// ===================================================================== quality control panel
export function qcPanel(ws) {
  const el = h('div', { class: 'wpanel qc' });
  el.refresh = async (run) => {
    const IM = ws.IM; el.innerHTML = '';
    const vid = IM.activeVol, v = IM.vol(vid);
    el.append(h('h3', {}, 'Quality control'), h('p', { class: 'note' }, 'Review flags, not proof that a segmentation is correct. No accuracy score is computed.'));
    if (!v) { el.append(h('p', { class: 'note' }, 'Select a series in the imaging browser.')); return; }
    const mm = IM.maskMeta(vid);
    el.append(h('div', { class: 'irow' }, h('b', {}, v.name), ' · ', mm ? originText(mm) : 'no mask selected', h('span', { class: 'grow' }), btn('Run checks', () => el.refresh(true), { class: 'primary' })));
    if (IM.edit && IM.edit.vid === vid && IM.edit.dirty) el.append(h('div', { class: 'warnbox' }, 'Unsaved edits are not checked: save a version first.'));
    if (!run && !el.last) return;
    let res;
    try { res = el.last = (run || !el.last) ? await api('/api/img/qc', { case: IM.caseId, volume: vid, mask: mm ? mm.id : null }) : el.last; }
    catch (e) { el.append(h('div', { class: 'err' }, e.message)); return; }
    for (const c of res.categories) {
      el.append(h('div', { class: 'qccat' }, chip(c.status), ' ', h('b', {}, c.title)));
      if (!c.checks.length) el.append(h('div', { class: 'chk not_evaluated small' }, chip('not_evaluated'), ' nothing to check'));
      for (const k of c.checks) {
        const r = h('div', { class: 'chk ' + k.status + (k.location ? ' nav' : '') }, chip(k.status), ' ', k.message, k.hint ? h('div', { class: 'hint' }, '→ ' + k.hint) : null);
        if (k.location) { r.title = 'Go to ' + k.location.world.join(', '); r.onclick = () => { ws.ensureMpr(vid); IM.gotoWorld(k.location.world, vid); }; r.append(h('span', { class: 'go' }, ' ⌖ go to')); }
        el.append(r);
      }
    }
  };
  ws.IM.on('overlay', () => { el.last = null; if (el.isConnected) el.refresh(false); });
  return el;
}

// ===================================================================== evidence & mapping panel
export function evidencePanel(ws) {
  const el = h('div', { class: 'wpanel evidence' });
  el.refresh = async () => {
    const IM = ws.IM; el.innerHTML = '';
    el.append(h('h3', {}, 'Evidence → scaffold'), h('p', { class: 'note' }, 'Reports and annotations are data, not instructions. Uploading never changes the design; only an explicit apply does. Reported facts, measured values, model inferences and assumptions are kept apart.'));
    if (!IM.manifest) { el.append(h('p', { class: 'note' }, 'Open an imaging case first.')); return; }
    const ta = h('textarea', { rows: 4, placeholder: 'Paste radiology report text…', 'data-ctl': '(report text)' }, IM.manifest.report_text || '');
    el.append(h('h4', {}, 'Report'), ta, h('div', { class: 'irow' }, btn('Extract findings', async () => { await api('/api/img/report', { case: IM.caseId, text: ta.value }); await IM.refresh(); el.refresh(); })));
    const f = IM.manifest.findings || [];
    if (f.length) el.append(h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, 'Finding'), h('th', {}, 'Value'), h('th', {}, 'Quote')),
      ...f.map(x => h('tr', {}, h('td', {}, x.kind.replace('_', ' ')), h('td', {}, `${x.value} ${x.units}`), h('td', { class: 'small' }, '“' + x.quote + '”')))));
    const anns = IM.annotationsFor();
    el.append(h('h4', {}, 'Measurements and landmarks'), anns.length ? h('div', { class: 'anns' }, ...anns.map(a => annRow(ws, a))) : h('p', { class: 'note' }, 'None yet: use the Distance and Landmark tools in an MRI pane.'));
    el.append(h('h4', {}, 'Proposed mappings'), h('div', { class: 'irow' }, btn('Propose', () => el.propose()), btn('Apply supported mappings', () => ws.muse.run('Update the scaffold using the report and the reviewed MRI measurements'), { class: 'primary' })));
    const box = h('div', { class: 'propbox' }); el.append(box);
    el.propose = async () => {
      const S = window.__app.S;
      const p = await api('/api/img/propose', { case: IM.caseId, params: S.state.params, provenance: S.state.provenance });
      box.innerHTML = ''; box.append(proposalTable(p, ws));
    };
    el.append(h('h4', {}, 'Settings that came from imaging or reports'), sourcesList());
    el.propose().catch(() => {});
  };
  ws.IM.on('annotations', () => el.isConnected && el.refresh());
  return el;
}
export function proposalTable(p, ws) {
  const wrap = h('div');
  wrap.append(h('table', { class: 'tbl' }, h('tr', {}, ...['Mapping', 'Source (series / annotation)', 'Value', 'Target', 'Relation', 'Status'].map(t => h('th', {}, t))),
    ...p.rows.map(r => h('tr', { class: 'r-' + r.status },
      h('td', {}, r.mapping.replace(/_/g, ' '), h('div', { class: 'small muted' }, r.transform)),
      h('td', { class: 'small' }, r.source ? `${r.source.label}${r.source.volume_name ? ' · ' + r.source.volume_name : ''} [${r.source.certainty}]` : (r.sources || []).map(s => `${s.label} [${s.certainty}]`).join('; ') || '—'),
      h('td', { class: 'mono' }, r.source ? `${r.source.value} ${r.source.units || ''}` : '—'),
      h('td', { class: 'mono small' }, r.changes ? Object.entries(r.changes).map(([k, v]) => `${k} = ${typeof v === 'object' ? JSON.stringify(v) : v}`).join('\n') : r.targets.join(', ')),
      h('td', { class: 'small' }, r.relation === 'direct' ? 'direct' : 'needs assumption', r.assumptions.length ? h('div', { class: 'muted' }, r.assumptions.join(' ')) : null),
      h('td', {}, chip({ apply: 'pass', unchanged: 'pass', needs_review: 'warn', conflict: 'fail', blocked: 'fail', missing: 'not_evaluated' }[r.status], r.status.replace('_', ' ')), h('div', { class: 'small' }, r.reason))))));
  const nr = p.rows.filter(r => r.status === 'needs_review');
  if (nr.length && ws) wrap.append(h('div', { class: 'irow' }, btn(`Apply ${nr.length} assumption mapping(s) as labelled assumptions`, () => ws.applyEvidence({ scope: nr.map(r => r.mapping), include_assumptions: true }))));
  wrap.append(h('details', {}, h('summary', {}, 'Not derived from MRI'), ...p.not_mapped.map(n => h('div', { class: 'small' }, h('b', {}, n.what), ` (${n.params}): ${n.why}`))));
  return wrap;
}
export function sourcesList() {
  const S = window.__app.S, out = h('div', { class: 'small' });
  const rows = Object.entries(S.state.provenance || {}).filter(([, pv]) => /MRI|report|KneeScan|measurement|evidence/i.test(pv.source || ''));
  if (!rows.length) out.append(h('p', { class: 'note' }, 'No current setting came from imaging or a report.'));
  for (const [k, pv] of rows) {
    const e = S.regById[k];
    out.append(h('div', { class: 'srcrow' }, h('b', {}, e ? e.label : k), ` = ${JSON.stringify(S.state.params[k])} ${e && e.units ? e.units : ''}`, h('div', { class: 'muted' }, `${pv.status}: ${pv.source}${pv.evidence ? ' · ' + pv.evidence : ''}`)));
  }
  return out;
}

// ===================================================================== registry coverage panel
export function coveragePanel(ws) {
  const el = h('div', { class: 'wpanel coverage' });
  el.refresh = () => {
    el.innerHTML = '';
    const reg = ws.registry(), cov = ws.coverage(), cats = {};
    for (const e of reg.entries.values()) cats[e.category] = (cats[e.category] || 0) + 1;
    const tools = ws.tools();
    el.append(h('h3', {}, 'Muse control coverage'),
      h('p', { class: 'note' }, 'The registry is built from the live application; Muse\'s tool definitions are generated from it.'),
      h('div', { class: 'mono small' }, `${reg.entries.size} controls · ${reg.actions.size} actions · ${tools.length} tool definitions (${(JSON.stringify(tools).length / 1024).toFixed(1)} kB)`),
      h('div', { class: 'mono small' }, Object.entries(cats).map(([k, n]) => `${k}: ${n}`).join(' · ')),
      h('div', { class: 'mono small' }, `Levels: ${[1, 2, 3, 4, 5, 6].map(l => `L${l} ${[...reg.entries.values()].filter(e => e.level === l).length}`).join(' · ')}`),
      h('h4', {}, `Coverage check: ${cov.missing.length ? cov.missing.length + ' editable control(s) missing' : 'every editable control is registered'}`),
      ...cov.missing.slice(0, 60).map(m => h('div', { class: 'chk fail small' }, m)),
      h('div', { class: 'small muted' }, `${cov.covered} rendered controls matched to registry entries.`), btn('Re-check', () => el.refresh()));
  };
  return el;
}
