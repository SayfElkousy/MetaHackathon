// Parametric custom scaffold components. Definitions are data, never executable code.
// Geometry is deliberately limited to audited construction operations and remains editable.
import { h } from './panels.js';

const clone = x => JSON.parse(JSON.stringify(x));
const OPS = ['tube', 'curved_tube', 'band', 'sleeve', 'channel_array', 'strut', 'lattice'];
const REL = ['independent', 'attached', 'following_path', 'conforming', 'fused_visual', 'clearance'];
const MATERIALS = ['PLLA', 'PCL', 'PLGA', 'silk', 'collagen_I', 'PET', 'beta_TCP', 'HA', 'custom'];
const clamp = (x, a, b) => Math.max(a, Math.min(b, Number(x)));

export class CustomParts {
  constructor(ws) {
    this.ws = ws; this.app = ws.app; this.group = new this.app.THREE.Group(); this.group.name = 'custom-components';
    this.app.root().add(this.group); this.undo = []; this.selected = null; this.lastActionAt = 0;
    const st = this.app.S.state;
    if (st) st.custom_parts = Array.isArray(st.custom_parts) ? st.custom_parts : [];
    this.renderGeometry(); this.renderInspector();
    window.addEventListener('acl:built', () => { if (this.group.parent !== this.app.root()) this.app.root().add(this.group); this.renderGeometry(); });
    window.addEventListener('acl:loaded', () => { this.renderGeometry(); this.renderInspector(); });
  }
  list() {
    // Case loading replaces the whole design state before the build-complete
    // event fires. Older/demo cases may not contain the workspace extension,
    // so keep that transient state safe for event-driven renders.
    const state = this.app.S.state;
    if (!state) return [];
    if (!Array.isArray(state.custom_parts)) state.custom_parts = [];
    return state.custom_parts;
  }
  supported() { return { operations: OPS, relationships: REL, materials: MATERIALS,
    limits: 'Model-frame parametric primitives. Referenced parts track a target origin; path conformance, Boolean fusion, contact, clearance solving and mechanical bonding are not modeled.' }; }
  persist() { localStorage.setItem('acl.v2.current', JSON.stringify(this.app.S.state)); this.app.invalidate(); this.ws._reg = null; }
  checkpoint(label) { this.undo.push({ label, parts: clone(this.list()) }); if (this.undo.length > 30) this.undo.shift(); this.lastActionAt = Date.now(); }
  assertAcyclic(candidate) {
    const parts = this.list().filter(x => x.id !== candidate.id).concat([candidate]);
    const targets = new Map(parts.map(x => [x.id, x.relationship && x.relationship.target]));
    for (const start of targets.keys()) {
      const seen = new Set(); let id = start;
      while (id && targets.has(id)) { if (seen.has(id)) throw new Error(`Dependency cycle detected at "${id}".`); seen.add(id); id = targets.get(id); }
    }
  }
  normalize(d) {
    if (!OPS.includes(d.operation)) throw new Error(`Unsupported construction operation. Use: ${OPS.join(', ')}`);
    const q = clone(d), id = String(q.id || ('custom_' + Date.now().toString(36))).replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 60);
    if (this.list().some(x => x.id === id)) throw new Error(`Custom component id "${id}" already exists.`);
    q.id = id; q.name = String(q.name || q.operation.replace(/_/g, ' ')).slice(0, 80); q.purpose = String(q.purpose || 'User-defined scaffold component').slice(0, 240);
    q.level = q.level == null ? null : clamp(q.level, 1, 6); q.position_mm = (q.position_mm || [0, 0, 0]).slice(0, 3).map(Number);
    q.rotation_deg = (q.rotation_deg || [0, 0, 0]).slice(0, 3).map(Number); q.material = MATERIALS.includes(q.material) ? q.material : 'custom';
    q.visible = q.visible !== false; q.dimensions = Object.assign({ length_mm: 12, width_mm: 3, thickness_mm: 0.6, outer_diameter_mm: 2, wall_mm: 0.3, count: 3, spacing_mm: 2 }, q.dimensions || {});
    for (const [k, v] of Object.entries(q.dimensions)) q.dimensions[k] = clamp(v, k === 'count' ? 1 : 0.05, k === 'count' ? 64 : 100);
    q.relationship = Object.assign({ mode: 'independent', target: null, clearance_mm: 0 }, q.relationship || {});
    if (!REL.includes(q.relationship.mode)) throw new Error(`Unsupported relationship. Use: ${REL.join(', ')}`);
    if (q.relationship.mode !== 'independent' && !q.relationship.target) throw new Error(`${q.relationship.mode} requires an explicit target component id.`);
    if (q.relationship.target === q.id) throw new Error('A component cannot depend on itself.');
    q.assumptions = (q.assumptions || []).map(String).slice(0, 8); q.created = q.created || new Date().toISOString(); q.edited = new Date().toISOString();
    if (q.relationship.target && !this.list().some(x => x.id === q.relationship.target) && !this.app.compDef(q.relationship.target)) throw new Error(`Unknown relationship target "${q.relationship.target}".`);
    q.mechanical_model = q.relationship.mode === 'fused_visual' ? 'visual fusion only; bond strength is not modeled' : 'not modeled';
    q.relationship_status = q.relationship.mode === 'independent' ? 'independent model-frame placement' :
      q.relationship.mode === 'attached' ? 'tracks target origin; attachment mechanics are not modeled' :
      `tracks target origin only; ${q.relationship.mode.replace(/_/g, ' ')} geometry is not constraint-solved`;
    this.assertAcyclic(q);
    return q;
  }
  create(def) { if (!this.app.S.state) throw new Error('Load an MRI design before creating scaffold components.'); this.checkpoint('create custom component'); const q = this.normalize(def); this.list().push(q); this.selected = q.id; this.persist(); this.renderGeometry(); this.renderInspector(); return this.describe(q); }
  update(id, patch) {
    const i = this.list().findIndex(x => x.id === id); if (i < 0) throw new Error(`Unknown custom component "${id}".`);
    this.checkpoint('update ' + id); const base = Object.assign({}, clone(this.list()[i]), clone(patch || {}), { id });
    if (patch.dimensions) base.dimensions = Object.assign({}, this.list()[i].dimensions, patch.dimensions);
    if (patch.relationship) base.relationship = Object.assign({}, this.list()[i].relationship, patch.relationship);
    this.list()[i] = this.normalizeExisting(base); this.persist(); this.renderGeometry(); this.renderInspector(); return this.describe(this.list()[i]);
  }
  normalizeExisting(q) { const old = this.list().find(x => x.id === q.id); const tmp = this.list(); this.app.S.state.custom_parts = tmp.filter(x => x.id !== q.id); try { return this.normalize(q); } finally { this.app.S.state.custom_parts = tmp; } }
  remove(id) { const i = this.list().findIndex(x => x.id === id); if (i < 0) throw new Error(`Unknown custom component "${id}".`); const broken = this.list().filter(x => x.relationship && x.relationship.target === id); if (broken.length) throw new Error(`Cannot delete: referenced by ${broken.map(x => x.name).join(', ')}.`); this.checkpoint('delete ' + id); const gone = this.list().splice(i, 1)[0]; this.selected = null; this.persist(); this.renderGeometry(); this.renderInspector(); return `deleted ${gone.name}`; }
  duplicate(id) { const p = this.list().find(x => x.id === id); if (!p) throw new Error(`Unknown custom component "${id}".`); const q = clone(p); delete q.id; q.name += ' copy'; q.position_mm[0] += Number(q.dimensions.spacing_mm || 2); return this.create(q); }
  undoLast() { const u = this.undo.pop(); if (!u) return null; this.app.S.state.custom_parts = u.parts; this.lastActionAt = Date.now(); this.persist(); this.renderGeometry(); this.renderInspector(); return `Undone custom-component action: ${u.label}.`; }
  describe(p) { return { id: p.id, name: p.name, purpose: p.purpose, operation: p.operation, dimensions: p.dimensions, units: 'mm', position_mm: p.position_mm, rotation_deg: p.rotation_deg, material: p.material, level: p.level, relationship: p.relationship, relationship_status: p.relationship_status, assumptions: p.assumptions, mechanical_model: p.mechanical_model, editable: true }; }
  clearGroup() { while (this.group.children.length) { const x = this.group.children.pop(); x.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); }); } }
  material(p) { return new this.app.THREE.MeshStandardMaterial({ color: p.material === 'PCL' ? 0x74b7b2 : p.material === 'collagen_I' ? 0xd4a4a4 : 0x7667c7, roughness: .48, metalness: .02, transparent: true, opacity: p.visible ? .88 : 0 }); }
  cylinder(radius, length, mat) { const T = this.app.THREE, m = new T.Mesh(new T.CylinderGeometry(radius, radius, length, 20, 1, false), mat); return m; }
  geometry(p) {
    const T = this.app.THREE, d = p.dimensions, mat = this.material(p), g = new T.Group();
    const len = d.length_mm, r = d.outer_diameter_mm / 2;
    if (p.operation === 'band') g.add(new T.Mesh(new T.BoxGeometry(d.width_mm, len, d.thickness_mm), mat));
    else if (p.operation === 'sleeve') {
      const outer = new T.Mesh(new T.CylinderGeometry(r, r, len, 28, 1, true), mat); g.add(outer);
      const innerMat = mat.clone(); innerMat.side = T.BackSide; const ri = Math.max(.03, r - d.wall_mm); g.add(new T.Mesh(new T.CylinderGeometry(ri, ri, len, 28, 1, true), innerMat));
    } else if (p.operation === 'curved_tube') {
      const pts = (p.path_points_mm || [[0, -len / 2, 0], [d.width_mm, 0, 0], [0, len / 2, 0]]).map(x => new T.Vector3(...x));
      g.add(new T.Mesh(new T.TubeGeometry(new T.CatmullRomCurve3(pts), 48, r, 12, false), mat));
    } else if (p.operation === 'channel_array') {
      const n = Math.round(d.count); for (let i = 0; i < n; i++) { const m = this.cylinder(r, len, mat.clone()); m.position.x = (i - (n - 1) / 2) * d.spacing_mm; g.add(m); }
    } else if (p.operation === 'lattice') {
      const n = Math.round(Math.min(20, d.count)); for (let i = 0; i < n; i++) { const a = this.cylinder(d.thickness_mm / 2, len, mat.clone()); a.rotation.z = (i % 2 ? 1 : -1) * Math.PI / 6; a.position.x = (i - (n - 1) / 2) * d.spacing_mm / 2; g.add(a); }
    } else g.add(this.cylinder(r, len, mat));
    g.name = p.id; g.userData.custom = p.id; g.position.set(...p.position_mm); g.rotation.set(...p.rotation_deg.map(x => x * Math.PI / 180)); g.visible = p.visible; return g;
  }
  targetCenter(id) {
    const T = this.app.THREE, box = new T.Box3();
    const custom = this.group.getObjectByName(id);
    if (custom) box.setFromObject(custom);
    else {
      const def = this.app.compDef(id); if (!def) return null;
      for (const oid of def.objs || [def.obj]) { const o = this.app.S.objects.get(oid); if (o) box.expandByObject(o); }
    }
    if (box.isEmpty()) return null;
    const c = box.getCenter(new T.Vector3()); this.app.root().worldToLocal(c); return c;
  }
  renderGeometry() {
    this.clearGroup(); for (const p of this.list()) this.group.add(this.geometry(p));
    // Definitions keep a model-frame offset. Referenced modes follow only the target origin; no contact/path solver is implied.
    for (const p of this.list()) if (p.relationship.mode !== 'independent') {
      const o = this.group.getObjectByName(p.id), c = this.targetCenter(p.relationship.target); if (o && c) o.position.set(c.x + p.position_mm[0], c.y + p.position_mm[1], c.z + p.position_mm[2]);
    }
    this.app.invalidate();
  }
  focus(id) { const o = this.group.getObjectByName(id); if (!o) throw new Error(`Custom component "${id}" is not visible.`); const box = new this.app.THREE.Box3().setFromObject(o); this.app.frameBox(box); this.selected = id; this.renderInspector(); return `focused ${id}`; }
  isolate(id) { for (const o of this.group.children) o.visible = o.name === id; this.app.setBuilder(true); return `isolated ${id}; standard scaffold context is unchanged`; }
  renderInspector() {
    const host = document.getElementById('tab-design'); if (!host) return; let root = document.getElementById('customParts');
    if (!root) { root = h('details', { id: 'customParts', class: 'panel', open: true }, h('summary', {}, 'Custom components'), h('div', { class: 'custombody' })); host.append(root); }
    const body = root.querySelector('.custombody'); body.innerHTML = '';
    body.append(h('p', { class: 'note' }, 'Parametric model-frame parts. Visual contact does not imply a mechanical bond.'));
    if (!this.app.S.state) { body.append(h('p', { class: 'note' }, 'Waiting for an MRI-derived scaffold design.')); return; }
    if (!this.list().length) { body.append(h('p', { class: 'note' }, 'No custom components. Ask Muse to create one using a supported operation.')); return; }
    for (const p of this.list()) {
      const card = h('div', { class: 'custompart' + (p.id === this.selected ? ' on' : ''), onclick: () => { this.selected = p.id; this.renderInspector(); } },
        h('div', { class: 'customhead' }, h('b', {}, p.name), h('span', { class: 'mono muted' }, p.operation)),
        h('div', { class: 'small muted' }, `${p.id} · ${p.material} · ${p.relationship.mode}`));
      if (p.id === this.selected) {
        const field = (label, key, value, unit) => h('label', { class: 'customfield' }, h('span', {}, label), h('input', { type: 'number', step: .1, value, onchange: e => this.update(p.id, { dimensions: { [key]: Number(e.target.value) } }) }), h('small', {}, unit));
        const axis = (label, i) => h('label', { class: 'customfield' }, h('span', {}, label), h('input', { type: 'number', step: .1, value: p.position_mm[i], onchange: e => { const v = p.position_mm.slice(); v[i] = Number(e.target.value); this.update(p.id, { position_mm: v }); } }), h('small', {}, 'mm'));
        card.append(h('label', { class: 'customwide' }, h('span', {}, 'Name'), h('input', { value: p.name, maxlength: 80, onchange: e => this.update(p.id, { name: e.target.value }) })),
          h('label', { class: 'customwide' }, h('span', {}, 'Material'), h('select', { onchange: e => this.update(p.id, { material: e.target.value }) }, ...MATERIALS.map(x => h('option', { value: x, selected: x === p.material }, x)))),
          h('div', { class: 'customgrid' }, field('Length', 'length_mm', p.dimensions.length_mm, 'mm'), field('Width', 'width_mm', p.dimensions.width_mm, 'mm'), field('Diameter', 'outer_diameter_mm', p.dimensions.outer_diameter_mm, 'mm'), field('Wall', 'wall_mm', p.dimensions.wall_mm, 'mm'), axis('Position X', 0), axis('Position Y', 1), axis('Position Z', 2)),
          h('div', { class: 'irow' }, h('button', { onclick: e => { e.stopPropagation(); this.focus(p.id); } }, 'Focus'), h('button', { onclick: e => { e.stopPropagation(); this.isolate(p.id); } }, 'Isolate'), h('button', { onclick: e => { e.stopPropagation(); this.duplicate(p.id); } }, 'Duplicate'), h('button', { class: 'danger', onclick: e => { e.stopPropagation(); this.remove(p.id); } }, 'Delete')),
          h('div', { class: 'note' }, `Relationship: ${p.relationship.mode}${p.relationship.target ? ' → ' + p.relationship.target : ''}. ${p.relationship_status}. Mechanical behavior: ${p.mechanical_model}.`));
      }
      body.append(card);
    }
  }
}

export { OPS as CUSTOM_OPERATIONS, REL as CUSTOM_RELATIONSHIPS, MATERIALS as CUSTOM_MATERIALS };
