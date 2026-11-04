// Control registry: every user-adjustable control in the workspace, built from the live application (the engine's
// parameter registry, the scene, the imaging controller and the layout). Muse's tool definitions are generated from
// it, so a control added to the engine registry is available to Muse without editing a command list.
//
// entry: { key, label, description, units, min, max, step, kind, options, target, level, category, depends,
//          readOnly (reason) | inactive (reason), geometry (does it rebuild?), get(), set(value) | param: true }

import { COLORMAPS } from './volmath.js';
import { TOOLS } from './imaging.js';

const DERIVED = /computed|derived|solved|current length/i;

export function buildRegistry(ctx) {
  const { app, IM, ws } = ctx, S = app.S, E = new Map(), A = new Map();
  const add = (e) => { E.set(e.key, Object.assign({ units: '', depends: [], category: 'workspace', geometry: false }, e)); };
  const act = (a) => A.set(a.name, Object.assign({ params: {}, required: [] }, a));
  const dep = S.state ? app.DEPENDENT() : {};
  // ---------------------------------------------------------------- engine parameters (all six scaffold levels + anatomy)
  for (const p of (S.reg ? S.reg.params : [])) {
    const why = dep[p.id];
    const lvl = p.owner_level;
    add({ key: p.id, param: true, label: p.label, description: (p.description || '').slice(0, 400), units: p.units || '', min: p.min, max: p.max, step: p.step,
      kind: p.kind, options: p.options || null, target: p.component || p.scope || '', level: lvl,
      category: lvl ? 'scaffold' : 'anatomy', group: p.group, subgroup: p.subgroup, depends: [p.depends, p.pair].filter(Boolean),
      readOnly: why && DERIVED.test(why) ? why : null, inactive: why && !DERIVED.test(why) ? why : null, geometry: (p.affects || 'geometry') === 'geometry',
      get: () => S.state.params[p.id] });
  }
  // ---------------------------------------------------------------- scaffold view: display, visibility, cutaways, camera
  const d = () => S.state.display;
  const dispKey = (k, label, kind, o) => add(Object.assign({ key: 'display.' + k, label, kind, category: 'display', description: 'Display only: changes what is drawn, never the design.',
    get: () => d()[k], set: v => { app.setDisplay(k, v, k !== 'fiber_mode'); if (k === 'fiber_mode') app.rebuildSceneObjects(); app.renderDisplayControls(); } }, o));
  dispKey('fiber_mode', 'Fiber rendering', 'enum', { options: ['auto', 'lines', 'tubes', 'hidden'] });
  dispKey('native_fibers_per_fascicle', 'Native fibers drawn per fascicle', 'int', { min: 1, max: 19, step: 1 });
  dispKey('scaffold_fibers_per_yarn', 'Scaffold fibers drawn per yarn', 'int', { min: 1, max: 19, step: 1 });
  dispKey('crimp_exaggeration', 'Crimp drawn at × (display)', 'scalar', { min: 1, max: 60, step: 1, units: '×' });
  add({ key: 'view.explode', label: 'Exploded view', kind: 'scalar', min: 0, max: 25, step: 1, units: 'mm', category: 'view', description: 'Inspection arrangement; positions are not true. Exports unchanged.',
    get: () => d().exploded || 0, set: v => { app.setDisplay('exploded', v); app.applyExploded(); const r = document.getElementById('explodeRange'); if (r) r.value = v; } });
  add({ key: 'view.color_mode', label: 'Colour scaffold by', kind: 'enum', options: ['anatomy', 'material', 'family', 'status'], category: 'view',
    get: () => S.colorMode, set: v => { S.colorMode = v; app.clearMatCache(); app.rebuildSceneObjects(); app.renderLegend(); app.renderDisplayControls(); } });
  add({ key: 'view.mode', label: 'Scaffold view mode', kind: 'enum', options: ['anatomy', 'builder'], category: 'view', description: 'anatomy = scaffold inside the knee; builder = scaffold alone with components',
    get: () => (S.builder ? 'builder' : 'anatomy'), set: v => app.setBuilder(v === 'builder') });
  const chk = (id, key, label, fn) => add({ key, label, kind: 'bool', category: 'view', get: () => !!(document.getElementById(id) || {}).checked,
    set: v => { const el = document.getElementById(id); if (el) { el.checked = !!v; el.dispatchEvent(new Event('change')); } if (fn) fn(v); } });
  chk('ovACL', 'view.acl_overlay', 'Show native ACL over the scaffold');
  chk('ovBone', 'view.bones', 'Show bones and surrounding anatomy');
  chk('cutQuick', 'view.lateral_cutaway', 'Lateral condyle cutaway');
  chk('tglLabels', 'view.labels', '3D labels');
  add({ key: 'view.acl_overlay_opacity', label: 'Native ACL overlay opacity', kind: 'scalar', min: 0.05, max: 1, step: 0.05, units: 'fraction', category: 'view',
    get: () => Number((document.getElementById('ovACLOp') || {}).value || 0.25),
    set: v => { const el = document.getElementById('ovACLOp'); el.value = v; el.dispatchEvent(new Event('input')); } });
  add({ key: 'view.longitudinal_cutaway', label: 'Longitudinal cutaway (builder)', kind: 'bool', category: 'view', get: () => !!(S.clip && S.clip.on && S.clip.long),
    set: v => { if (!!(S.clip && S.clip.on && S.clip.long) !== !!v) app.toggleLongCut(); } });
  const clip = (k, label, kind, o) => add(Object.assign({ key: 'view.clip.' + k, label, kind, category: 'view', get: () => (S.clip || {})[k],
    set: v => { S.clip = S.clip || {}; S.clip[k] = v; app.updateClip(); app.renderClipControls(); } }, o));
  clip('on', 'Clip all objects', 'bool'); clip('axis', 'Clip plane', 'enum', { options: ['sagittal', 'coronal', 'axial'] });
  clip('pos', 'Clip position', 'scalar', { min: -40, max: 60, step: 0.5, units: 'mm' }); clip('flip', 'Flip clip side', 'bool');
  add({ key: 'view.bone_cutaway_pos', label: 'Bone cutaway position', kind: 'scalar', min: -20, max: 30, step: 0.5, units: 'mm', category: 'view', get: () => (S.cut || {}).pos,
    set: v => { S.cut.pos = v; app.updateClip(); app.renderClipControls(); } });
  add({ key: 'view.slab', label: 'Show only a thin slab at the section (3D)', kind: 'bool', category: 'view', get: () => !!(S.slab || {}).on,
    set: v => { S.slab.on = !!v; app.rebuildSceneObjects(); app.renderClipControls(); } });
  add({ key: 'display.section_kind', label: 'Section plane', kind: 'enum', options: ['ligament', 'am', 'pl', 'axial', 'sagittal', 'coronal'], category: 'view',
    get: () => d().section_kind, set: v => { d().section_kind = v; app.renderSectionControls(); app.requestSection(); } });
  add({ key: 'display.section_s', label: 'Section position along the ligament', kind: 'scalar', min: 0.02, max: 0.98, step: 0.01, units: '0 tibia – 1 femur', category: 'view',
    get: () => d().section_s, set: v => { d().section_s = v; app.renderSectionControls(); app.requestSection(); } });
  add({ key: 'display.section_offset_mm', label: 'Anatomical section offset', kind: 'scalar', min: -25, max: 25, step: 0.25, units: 'mm', category: 'view',
    get: () => d().section_offset_mm, set: v => { d().section_offset_mm = v; app.renderSectionControls(); app.requestSection(); } });
  add({ key: 'view.level_mode', label: 'Level view', kind: 'enum', options: ['highlight', 'isolate', 'neighbors'], category: 'view', get: () => S.levelMode, set: v => app.setLevelMode(v) });
  add({ key: 'view.level', label: 'Selected scaffold level', kind: 'int', min: 0, max: 6, step: 1, category: 'view', description: '0 clears the selection',
    get: () => S.levelSel || 0, set: v => app.selectLevel(v || null) });
  for (const meta of (S.scene ? S.scene.objects : [])) {
    add({ key: `obj.${meta.id}.visible`, label: `Show ${meta.label}`, kind: 'bool', category: 'visibility', target: meta.id, get: () => S.vis[meta.id] !== false,
      set: v => { S.vis[meta.id] = !!v; const o = S.objects.get(meta.id); if (o) o.visible = !!v; app.refreshLevelView(); app.renderObjList(); app.invalidate(); app.saveView(); } });
    add({ key: `obj.${meta.id}.opacity`, label: `${meta.label} opacity`, kind: 'scalar', min: 0.05, max: 1, step: 0.05, units: 'fraction', category: 'visibility', target: meta.id,
      get: () => S.opac[meta.id], set: v => { S.opac[meta.id] = v; const o = S.objects.get(meta.id); if (o) app.matOptsFor(o); app.renderObjList(); app.invalidate(); app.saveView(); } });
  }
  for (const c of app.COMP_DEFS.filter(c => c.id)) {
    add({ key: `comp.${c.id}.visible`, label: `Show ${c.short}`, kind: 'bool', category: 'visibility', level: c.level, target: c.short,
      get: () => { const objs = c.objs || [c.obj]; return c.group != null ? S.groupVis[c.obj][c.group] !== false : objs.some(o => S.vis[o] !== false); },
      set: function (v) { if (this.get() !== !!v) app.toggleComp(c); } });
  }
  act({ name: 'camera.view', description: 'Point the 3D camera (no rebuild)', params: { preset: { type: 'string', enum: ['anterior', 'posterior', 'medial', 'lateral', 'superior', 'q34', 'along'] } }, required: ['preset'],
    run: a => { app.setCam(a.preset); return `camera: ${a.preset}`; } });
  act({ name: 'camera.focus', description: 'Focus the 3D camera on a scaffold component', params: { component: { type: 'string', enum: app.COMP_DEFS.filter(c => c.id).map(c => c.id) } }, required: ['component'],
    run: a => { const c = app.compDef(a.component); if (!c) throw new Error('unknown component'); app.setBuilder(true); app.focusComp(c); return `focused ${c.short}`; } });
  act({ name: 'component.isolate', description: 'Show only one scaffold component', params: { component: { type: 'string', enum: app.COMP_DEFS.filter(c => c.id).map(c => c.id) } }, required: ['component'],
    run: a => { const c = app.compDef(a.component); app.setBuilder(true); app.isolateComp(c); return `isolated ${c.short}`; } });
  act({ name: 'component.restore_all', description: 'Show every component and reset the camera', run: () => { app.restoreBuilderView(); return 'all components shown'; } });
  // ---------------------------------------------------------------- imaging (active MPR volume / view)
  const av = () => ws.activeMpr();
  const need = () => { const v = av(); if (!v || !v.v || !v.v.loaded) throw new Error('No MRI volume is open in an imaging pane.'); return v; };
  const di = () => { const v = need(); return IM.dispFor(v.id, v.vid); };
  add({ key: 'img.window.width', label: 'Window width (contrast)', kind: 'scalar', min: 1e-6, max: 1e9, units: 'image intensity units', category: 'imaging',
    description: 'Display range width. MRI intensities are arbitrary scanner units (not Hounsfield units). Display only: voxel values are unchanged.',
    get: () => { try { return di().width; } catch (e) { return null; } }, set: v => { const w = need(); IM.setDisp(w.id, w.vid, { width: v }); } });
  add({ key: 'img.window.level', label: 'Window level (brightness)', kind: 'scalar', min: -1e9, max: 1e9, units: 'image intensity units', category: 'imaging',
    get: () => { try { return di().center; } catch (e) { return null; } }, set: v => { const w = need(); IM.setDisp(w.id, w.vid, { center: v }); } });
  add({ key: 'img.window.invert', label: 'Invert greyscale', kind: 'bool', category: 'imaging', get: () => { try { return di().invert; } catch (e) { return null; } },
    set: v => { const w = need(); IM.setDisp(w.id, w.vid, { invert: !!v }); } });
  add({ key: 'img.window.linked', label: 'Link windowing across views of this series', kind: 'bool', category: 'imaging', get: () => { const w = av(); return w && IM.disp.get(w.vid) ? IM.disp.get(w.vid).linked : null; },
    set: v => { const w = need(); IM.setDisp(w.id, w.vid, { linked: !!v }); } });
  add({ key: 'img.colormap', label: 'Colour map', kind: 'enum', options: COLORMAPS, category: 'imaging', description: 'Display colour map. For MRI intensity it is a display choice, not a tissue or damage map.',
    get: () => { try { return di().cmap; } catch (e) { return null; } }, set: v => { const w = need(); IM.setDisp(w.id, w.vid, { cmap: v }); } });
  add({ key: 'img.overlay.visible', label: 'Show segmentation overlays', kind: 'bool', category: 'imaging', get: () => IM.overlay.visible, set: v => { IM.overlay.visible = !!v; IM.emit('overlay'); } });
  add({ key: 'img.overlay.opacity', label: 'Segmentation overlay opacity', kind: 'scalar', min: 0, max: 1, step: 0.05, units: 'fraction', category: 'imaging',
    description: 'Opacity of segmentation labels (independent of image windowing and of 3D scaffold materials).', get: () => IM.overlay.opacity, set: v => { IM.overlay.opacity = v; IM.emit('overlay'); } });
  add({ key: 'img.annotations.visible', label: 'Show landmarks and measurements', kind: 'bool', category: 'imaging', get: () => IM.annotationsVisible, set: v => { IM.annotationsVisible = !!v; IM.emit('annotations'); } });
  add({ key: 'img.crosshair.visible', label: 'Show crosshairs', kind: 'bool', category: 'imaging', get: () => IM.crossVisible, set: v => { IM.crossVisible = !!v; IM.emit('cross', {}); } });
  add({ key: 'img.tool', label: 'Active imaging tool', kind: 'enum', options: TOOLS.map(t => t[0]), category: 'imaging', get: () => IM.tool, set: v => IM.setTool(v) });
  add({ key: 'img.brush.size', label: 'Brush radius', kind: 'scalar', min: 0.25, max: 20, step: 0.25, units: 'mm (voxels when uncalibrated)', category: 'imaging', get: () => IM.brush, set: v => { IM.brush = v; IM.emit('tool'); } });
  add({ key: 'img.series.linked', label: 'Link this series to others in the same frame of reference', kind: 'bool', category: 'imaging',
    description: 'Series are linked only when they share a verified frame of reference (DICOM FrameOfReferenceUID). Matching slice numbers are never used.',
    get: () => { const w = av(); return w ? !IM.linkOff.has(w.vid) : null; }, set: v => { const w = need(); const k0 = IM.crossWorld(w.vid); v ? IM.linkOff.delete(w.vid) : IM.linkOff.add(w.vid); if (!IM.cross.has(IM.groupKey(w.vid))) IM.cross.set(IM.groupKey(w.vid), k0); IM.emit('cross', {}); } });
  const names = new Set(); for (const v of IM.vols.values()) for (const l of IM.labelsFor(v.id)) names.add(l.name);
  for (const n of names) {
    const o = () => IM.overlay.labels.get(n) || { visible: true, opacity: 1 };
    add({ key: `img.label.${n}.visible`, label: `Show label "${n}"`, kind: 'bool', category: 'imaging', target: n, get: () => o().visible,
      set: v => { IM.overlay.labels.set(n, Object.assign({}, o(), { visible: !!v })); IM.emit('overlay'); } });
    add({ key: `img.label.${n}.opacity`, label: `Opacity of label "${n}"`, kind: 'scalar', min: 0, max: 1, step: 0.05, units: 'fraction', category: 'imaging', target: n, get: () => o().opacity,
      set: v => { IM.overlay.labels.set(n, Object.assign({}, o(), { opacity: v })); IM.emit('overlay'); } });
  }
  act({ name: 'img.window.reset', description: 'Reset windowing to the initial display (DICOM window if present) or the automatic range', params: { mode: { type: 'string', enum: ['initial', 'auto'] } },
    run: a => { const w = need(); IM.resetWindow(w.id, w.vid, a.mode || 'initial'); return `windowing reset (${a.mode || 'initial'})`; } });
  act({ name: 'img.crosshair.center', description: 'Centre the crosshair in the active volume', run: () => { const w = need(); IM.centerCross(w.vid); return 'crosshair centred'; } });
  act({ name: 'img.crosshair.goto_annotation', description: 'Move the crosshair to the selected (or named) annotation', params: { id: { type: 'string' } },
    run: a => { const r = IM.gotoAnnotation(a.id); if (!r.ok) throw new Error(r.why); return `crosshair at ${r.annotation.name || r.annotation.kind}`; } });
  act({ name: 'img.crosshair.goto_label', description: 'Move the crosshair to the centre of a segmentation label', params: { label: { type: 'string' } }, required: ['label'],
    run: a => { const w = need(); const l = IM.labelsFor(w.vid).find(x => x.name.toLowerCase() === String(a.label).toLowerCase()); if (!l) throw new Error(`No "${a.label}" label in the current mask.`);
      const p = IM.labelCentroid(w.vid, Number(l.value)); if (!p) throw new Error(`Label "${a.label}" is empty.`); IM.gotoWorld(p, w.vid); return `crosshair at the centre of ${l.name}`; } });
  act({ name: 'img.crosshair.set_world', description: 'Set the crosshair to RAS millimetre coordinates THE USER TYPED (never invented)', params: { x: { type: 'number' }, y: { type: 'number' }, z: { type: 'number' } }, required: ['x', 'y', 'z'],
    run: a => { const w = need(); if (!w.v.orientation_known) throw new Error('This volume has no orientation: RAS coordinates are undefined.'); IM.setCross(w.vid, [a.x, a.y, a.z], 'numeric'); return 'crosshair moved'; } });
  act({ name: 'img.show_label', description: 'Show a segmentation label (and overlays) if it exists', params: { label: { type: 'string' } }, required: ['label'],
    run: a => { const all = [...IM.vols.values()].flatMap(v => IM.labelsFor(v.id)); const l = all.find(x => x.name.toLowerCase() === String(a.label).toLowerCase());
      if (!l) throw new Error(`There is no "${a.label}" segmentation loaded. Available: ${[...new Set(all.map(x => x.name))].join(', ') || 'none'}.`);
      IM.overlay.visible = true; IM.overlay.labels.set(l.name, Object.assign({ opacity: 1 }, IM.overlay.labels.get(l.name), { visible: true })); IM.emit('overlay'); return `showing ${l.name}`; } });
  act({ name: 'img.show_original', description: 'Show the original MRI without overlays or annotations (windowing unchanged)',
    run: () => { IM.overlay.visible = false; IM.annotationsVisible = false; IM.emit('overlay'); IM.emit('annotations'); return 'overlays and annotations hidden'; } });
  act({ name: 'seg.run', description: 'Run the configured segmentation service (KneeScan) on the active volume', params: { do_bones: { type: 'boolean' } },
    run: async a => ws.runSegmentation(a) });
  // ---------------------------------------------------------------- workspace
  act({ name: 'ws.preset', description: 'Apply a workspace layout', params: { name: { type: 'string', enum: ['mpr_scaffold', 'mpr', 'scaffold', 'review'] } }, required: ['name'],
    run: a => ws.preset(a.name) });
  act({ name: 'ws.open_view', description: 'Open a view as a tab', params: { view: { type: 'string', enum: ['axial', 'sagittal', 'coronal', 'scaffold', 'scaffold_view', 'qc', 'evidence', 'coverage'] }, where: { type: 'string', enum: ['active', 'right', 'down'] } }, required: ['view'],
    run: a => ws.openKind(a.view, a.where) });
  act({ name: 'ws.maximize', description: 'Maximize the active (or named) view', params: { view: { type: 'string' } }, run: a => { ws.layout.maximize(a.view); return 'maximized'; } });
  act({ name: 'ws.restore', description: 'Restore the layout after maximizing', run: () => { ws.layout.restore(); return 'layout restored'; } });
  act({ name: 'ws.close_view', description: 'Close a view tab (data stays loaded)', params: { view: { type: 'string' } }, required: ['view'], run: a => { ws.layout.close(a.view); return 'closed'; } });
  act({ name: 'evidence.apply', description: 'Map reviewed measurements and report findings to scaffold parameters (explicit mappings only)',
    params: { scope: { type: 'array', items: { type: 'string' } }, include_assumptions: { type: 'boolean' }, keep: { type: 'array', items: { type: 'string' } } },
    run: async a => ws.applyEvidence(a) });
  act({ name: 'evidence.show_sources', description: 'Show which settings came from MRI measurements, reports or models', run: () => ws.showSources() });
  act({ name: 'design.undo_last', description: 'Undo the last Muse change', run: () => ws.txn.undoLast() });
  act({ name: 'design.restore_view', description: 'Keep the geometry but restore the view from before the last Muse change', run: () => ws.txn.restoreView() });
  act({ name: 'ask_user', description: 'Ask the user one focused question instead of acting', params: { question: { type: 'string' } }, required: ['question'], run: a => ({ ask: a.question }) });
  return { entries: E, actions: A };
}

// ------------------------------------------------------------------ tool definitions (OpenAI-compatible)
export function toolDefs(reg) {
  const writable = [...reg.entries.values()].filter(e => !e.readOnly).map(e => e.key);
  const tools = [{ type: 'function', function: { name: 'set_controls', description: 'Change one or more controls in ONE transaction (single rebuild). Values in each control\'s units; see context.controls for bounds.',
    parameters: { type: 'object', properties: { changes: { type: 'array', items: { type: 'object', properties: { key: { type: 'string', enum: writable }, value: {} }, required: ['key', 'value'] } },
      keep: { type: 'array', items: { type: 'string' }, description: 'keys or key prefixes the user asked to keep unchanged' } }, required: ['changes'] } } }];
  for (const a of reg.actions.values())
    tools.push({ type: 'function', function: { name: a.name.replace(/\./g, '__'), description: a.description, parameters: { type: 'object', properties: a.params, required: a.required } } });
  return tools;
}

// ------------------------------------------------------------------ relevance (compact context for the model)
const tok = s => String(s || '').toLowerCase().replace(/[^a-z0-9µ]+/g, ' ').split(' ').filter(w => w.length > 1);
const SYN = { groove: ['gr', 'groove', 'grooves', 'microgroove'], channel: ['ch', 'channel', 'channels', 'lumen'], braid: ['braid', 'braided', 'mantle'],
  sheath: ['sheath', 'ribbon'], coating: ['sf', 'coating', 'coat'], footprint: ['fp', 'footprint', 'attachment'], tear: ['injury', 'tear', 'gap'],
  length: ['length'], width: ['width'], opacity: ['opacity', 'transparency'], window: ['window', 'windowing', 'contrast', 'brightness'] };
export function rank(reg, query, n = 25) {
  const q = new Set(tok(query)); for (const [k, ws] of Object.entries(SYN)) if (ws.some(w => q.has(w))) ws.forEach(w => q.add(w)), q.add(k);
  const scored = [];
  for (const e of reg.entries.values()) {
    const words = tok(e.key.replace(/\./g, ' ') + ' ' + e.label + ' ' + (e.target || '') + ' ' + (e.subgroup || ''));
    let s = 0; for (const w of words) if (q.has(w)) s += 1;
    if (s) scored.push([s / Math.sqrt(words.length + 1), e]);
  }
  return scored.sort((a, b) => b[0] - a[0]).slice(0, n).map(x => x[1]);
}
export const describe = (e) => ({ key: e.key, label: e.label, units: e.units, min: e.min, max: e.max, options: e.options, kind: e.kind, level: e.level,
  value: safe(e.get), ...(e.readOnly ? { read_only: e.readOnly } : {}), ...(e.inactive ? { inactive: e.inactive } : {}) });
function safe(f) { try { const v = f(); return typeof v === 'number' ? +v.toFixed(4) : v; } catch (e) { return null; } }

// ------------------------------------------------------------------ coverage: every editable control in the DOM must be registered
const STATIC = { ovACL: 'view.acl_overlay', ovACLOp: 'view.acl_overlay_opacity', ovBone: 'view.bones', explodeRange: 'view.explode', cutQuick: 'view.lateral_cutaway',
  tglLabels: 'view.labels', cutOn: 'view.lateral_cutaway', cutPos: 'view.bone_cutaway_pos', clipOn: 'view.clip.on', clipAx: 'view.clip.axis', clipPos: 'view.clip.pos',
  clipFlip: 'view.clip.flip', slabOn: 'view.slab', secKind: 'display.section_kind', secPos: 'display.section_s', secNum: 'display.section_s',
  btnViewAnat: 'view.mode', btnViewBuilder: 'view.mode', btnLongCut: 'view.longitudinal_cutaway', paramSearch: '(search box)', reliefSel: '(detail display)', lvSecPos: '(cross-section preview)', lvSecSlab: '(cross-section preview)' };
export function coverage(reg, root = document) {
  const missing = [], covered = [];
  root.querySelectorAll('.ctl[data-pid]').forEach(el => (reg.entries.has(el.dataset.pid) ? covered : missing).push(el.dataset.pid));
  root.querySelectorAll('[data-ctl]').forEach(el => { const k = el.dataset.ctl; (reg.entries.has(k) || reg.actions.has(k) || k.startsWith('(') ? covered : missing).push(k); });
  root.querySelectorAll('#side input, #side select, #stage input, #stage select, #insImg input, #insImg select').forEach(el => {
    if (el.closest('.ctl[data-pid]') || el.dataset.ctl || el.closest('[data-ctl-group]')) return;
    const k = STATIC[el.id];
    if (k) (k.startsWith('(') || reg.entries.has(k) ? covered : missing).push(k);
    else missing.push('#' + (el.id || el.name || el.type) + ' in ' + ((el.closest('[id]') || {}).id || '?'));
  });
  if (reg.entries.size) for (const p of (window.__app.S.reg.params || [])) if (!reg.entries.has(p.id)) missing.push(p.id + ' (engine registry)');
  return { covered: new Set(covered).size, missing: [...new Set(missing)] };
}
