// N3 design page. Written by Muse (N3). Fixes by Claude are marked "FIX (Claude)".
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let request = null, sliders = { sliders: [] }, hints = {}, buildSeq = 0, inFlight = false, pending = false, framed = false;
const $ = id => document.getElementById(id);
const badge = $('badge'), errBox = $('err');
function showErr(m) { errBox.textContent = m; }
async function get(p) { const r = await fetch(p); return r.json(); }

// ---- request editing -------------------------------------------------------
function getPath(obj, path, create) {
  if (path.startsWith('zones.')) {
    const [, zid, field] = path.split('.');
    const z = (obj.zones || []).find(z => z.id === zid);
    return z ? { o: z, k: field } : null;
  }
  const ks = path.split('.'); let o = obj;
  for (let i = 0; i < ks.length - 1; i++) {
    if (o[ks[i]] == null) { if (!create) return null; o[ks[i]] = {}; }  // FIX (Claude): crashed on missing parents
    o = o[ks[i]];
  }
  return { o, k: ks[ks.length - 1] };
}
function readVal(v) { return (v && typeof v === 'object' && 'value' in v) ? v.value : v; }
function writeVal(slot, val) {
  const o = slot.o, k = slot.k;
  if (o[k] && typeof o[k] === 'object' && 'value' in o[k]) { o[k].value = val; o[k].origin = 'user_design'; } else o[k] = val;
}
function currentVal(s) { const slot = getPath(request, s.path, false); return slot ? readVal(slot.o[slot.k]) : undefined; }

// ---- 3D ---------------------------------------------------------------------
let scene, cam, ren, ctl, loader, modelRoot = null;
function init3() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1b2322);
  cam = new THREE.PerspectiveCamera(35, 1, 0.5, 2000);
  ren = new THREE.WebGLRenderer({ antialias: true });
  ren.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  $('view').appendChild(ren.domElement);
  ctl = new OrbitControls(cam, ren.domElement); ctl.enableDamping = true;
  loader = new GLTFLoader();
  // FIX (Claude): Muse's scene had no lights, so every (PBR) mesh rendered black.
  scene.add(new THREE.HemisphereLight(0xffffff, 0x667788, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(60, 100, 80); scene.add(key);
  const rim = new THREE.DirectionalLight(0xffffff, 0.6); rim.position.set(-80, 40, -60); scene.add(rim);
  const size = () => { const w = $('view').clientWidth || 640, h = $('view').clientHeight || 480; ren.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix(); };
  new ResizeObserver(size).observe($('view')); size();
  document.querySelectorAll('#cams button').forEach(b => b.onclick = () => setCam(b.dataset.cam));
  setCam('reset');
  const tick = () => { requestAnimationFrame(tick); ctl.update(); ren.render(scene, cam); }; tick();
}
// Views use the scaffold (not the bones) as the target so it fills the screen.
function setCam(which) {
  const s = (request && request.anatomy && request.anatomy.side === 'left') ? -1 : 1;
  const target = new THREE.Box3();
  if (modelRoot) modelRoot.traverse(o => { if (o.isMesh && o.name.startsWith('scaffold')) target.expandByObject(o); });
  const c = target.isEmpty() ? new THREE.Vector3(0, 15, 0) : target.getCenter(new THREE.Vector3());
  const r = target.isEmpty() ? 60 : target.getSize(new THREE.Vector3()).length();
  const d = r * 1.8;
  // three.js is Y-up; the engine is Z-up with +Y anterior, so engine front (-Y) is three +Z.
  // modelRoot is rotated -90 deg about X: three +X = lateral, +Y = proximal, +Z = posterior.
  if (which === 'front') setC([c.x, c.y, c.z + d]);
  else if (which === 'side') setC([c.x + s * d, c.y, c.z]);
  else if (which === 'anterior') setC([c.x, c.y, c.z - d]);
  else if (which === 'posterior') setC([c.x, c.y, c.z + d]);
  else if (which === 'lateral') setC([c.x + s * d, c.y, c.z]);
  else if (which === 'medial') setC([c.x - s * d, c.y, c.z]);
  else if (which === 'superior') setC([c.x, c.y + d, c.z]);
  else setC([c.x + s * d * 0.6, c.y + d * 0.45, c.z + d * 0.7]);
  function setC(p) { cam.position.set(p[0], p[1], p[2]); ctl.target.copy(c); ctl.update(); }
}
function applyTogs() {
  if (!modelRoot) return;
  modelRoot.traverse(o => {
    if (!o.isMesh) return; const n = o.name || '';
    if (n.startsWith('scaffold_strand_')) o.visible = $('t_scaf').checked;
    else if (n.startsWith('acl_strands_')) { o.visible = $('t_strand').checked; o.material.transparent = false; o.material.opacity = 1.0; o.material.depthWrite = true; o.renderOrder = 3; }
    else if (n.startsWith('acl_')) { o.visible = $('t_ghost').checked; o.material.transparent = true; o.material.opacity = n.startsWith('acl_edge_') ? 1.0 : 0.3; o.material.depthWrite = false; o.renderOrder = 2; }
    else if (n.startsWith('stump_')) { o.visible = $('t_tear').checked; o.material.transparent = true; o.material.opacity = 0.85; o.material.depthWrite = false; o.renderOrder = 2; }
    else if (n.startsWith('footprint_')) { o.visible = $('t_fp').checked; o.material.transparent = true; o.material.opacity = 0.9; o.material.depthWrite = false; o.renderOrder = 2; }
    else if (n.startsWith('bone_')) { o.visible = $('t_bone').checked; o.material.transparent = true; o.material.opacity = 0.25; o.material.depthWrite = false; o.renderOrder = 1; }
    else if (n.startsWith('tunnel_')) { o.visible = $('t_tun').checked; o.material.transparent = true; o.material.opacity = 0.25; o.material.depthWrite = false; o.renderOrder = 1; }
  });
}
['t_scaf', 't_ghost', 't_strand', 't_tear', 't_fp', 't_bone', 't_tun'].forEach(id => $(id).addEventListener('change', applyTogs));

// ---- builds -------------------------------------------------------------------
let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(() => { if (inFlight) { pending = true; return; } doBuild(); }, 120); }
async function doBuild() {
  inFlight = true; pending = false; buildSeq++; const my = buildSeq;
  badge.textContent = 'building…'; badge.className = 'badge busy';
  try {
    const r = await fetch('/api/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ request, quality: 'preview', build_id: my }) });
    const j = await r.json();
    if (!r.ok) throw new Error((j.error || 'build failed') + (j.hint ? ' (' + j.hint + ')' : ''));
    const bin = Uint8Array.from(atob(j.glb_base64), c => c.charCodeAt(0));
    const gltf = await loader.parseAsync(bin.buffer, '');
    if (modelRoot) scene.remove(modelRoot);
    modelRoot = gltf.scene;
    // FIX (Claude): the GLB is in metres; scale to mm so the camera distances make sense.
    modelRoot.scale.setScalar(1000);
    modelRoot.rotation.x = -Math.PI / 2;
    scene.add(modelRoot); applyTogs();
    if (!framed) { setCam('q34'); framed = true; }  // FIX (Claude): camera was never pointed at the first model
    renderChecks(j.checks, j.summary);
    const s = j.summary;
    $('sum').textContent = `crossings ${s.nevents} · strand gap ${fmt(s.gap_mm, 2)} mm · sticks out ${fmt(Math.max(0, s.penetration_mm), 2)} mm · tightest bend R ${fmt(1 / s.kmax, 2)} mm`;
    $('timing').textContent = 'rebuilt in ' + (j.timings_s.total || 0).toFixed(2) + ' s';
    showErr('');
  } catch (e) { showErr(String(e && e.message || e)); }
  badge.textContent = 'ready'; badge.className = 'badge'; inFlight = false;
  if (pending) doBuild();
}
function fmt(v, n) { return (typeof v === 'number' && isFinite(v)) ? v.toFixed(n) : '–'; }

// FIX (Claude): Muse wrote hints.json but the page never used it; checks showed raw JSON.
function hintFor(c, summary) {
  let tpl = hints[c.check_id];
  if (!tpl && c.check_id.startsWith('fiber_length')) tpl = hints.fiber_length;
  if (!tpl) tpl = hints.fallback || '{check_id}: {status}';
  const vals = Object.assign({}, c, { list: (summary.assumptions || []).join(', ') });
  return tpl.replace(/\{(\w+)(?::\.(\d)f)?\}/g, (_, k, d) => {
    const v = vals[k];
    if (typeof v === 'number') return d ? v.toFixed(Number(d)) : String(v);
    return v == null ? '?' : String(v);
  });
}
const NICE = { native_containment: 'Fits inside natural ACL', strand_gap: "Strands don't touch", block_wall: 'Mold wall ≥ 0.8 mm',
  tube_mold_feasible: 'Tube mold ≥ 1.7 mm', inclination: 'ACL angles match', flare: 'Insertion flare', bone_clearance_schematic: 'Clears tibia',
  bundle_inside_acl: 'Bundles inside ACL', stump_closure: 'Stump faces closed',
  anatomy_regen: 'Deterministic rebuild', attachment_on_bone: 'Attachments on bone', frame_continuity: 'Frame continuity',
  mesh_valid_anatomy: 'Anatomy meshes valid', bone_penetration_midbody: 'Mid-body clears bone',
  anatomy_self_intersection: 'No self-intersection', export_scale: 'Export scale (mm)',
  process_compat: 'Manufacturing', mechanical_validation: 'Mechanics', biological_validation: 'Biology' };
function renderChecks(checks, summary) {
  const box = $('checks'); box.innerHTML = '';
  (checks || []).forEach(c => {
    const li = document.createElement('li'); li.className = c.status;
    const name = NICE[c.check_id] || c.check_id.replace(/_/g, ' ');
    li.innerHTML = '<b></b> ';
    li.querySelector('b').textContent = name + ': ' + c.status;
    if (c.status !== 'pass') li.append(' ' + hintFor(c, summary));
    box.appendChild(li);
  });
  if (summary.uses_assumptions) {
    const li = document.createElement('li'); li.className = 'warn';
    li.textContent = hintFor({ check_id: 'uses_assumptions' }, summary); box.appendChild(li);
  }
}

// ---- sliders -------------------------------------------------------------------
function zoneControl(path) {
  if (!path.startsWith('zones.')) return null;
  const z = (request.zones || []).find(z => z.id === path.split('.')[1]);
  return z ? (z.control || 'braid_angle') : null;
}
function buildSliders() {
  const box = $('sliders'); box.innerHTML = '';
  sliders.sliders.forEach(s => {
    const div = document.createElement('div'); div.className = 'sl'; div.dataset.id = s.id;
    const lab = document.createElement('label'); lab.htmlFor = 'sl_' + s.id;
    lab.textContent = s.label + (s.unit ? ' (' + s.unit + ')' : '');
    const val = document.createElement('span'); val.className = 'val';
    const inp = document.createElement('input'); inp.type = 'range'; inp.id = 'sl_' + s.id; inp.min = s.min; inp.max = s.max; inp.step = s.step;
    const cur = currentVal(s);
    if (s.kind === 'toggle') { const ix = s.options.indexOf(cur); inp.value = ix >= 0 ? ix : s.default; val.textContent = s.options[Number(inp.value)]; }  // Milestone A: general toggle (was side-only)
    else { inp.value = (typeof cur === 'number') ? cur : s.default; val.textContent = inp.value; }
    inp.oninput = () => {
      val.textContent = s.kind === 'toggle' ? s.options[Number(inp.value)] : inp.value;
      applySlider(s, inp.value); markActive(); schedule();
    };
    div.append(lab, val, inp);
    if (s.studied_range) { const b = document.createElement('div'); b.className = 'band'; b.textContent = 'studied range ' + s.studied_range[0] + '–' + s.studied_range[1] + ' (' + (s.source || '') + ')'; div.appendChild(b); }
    box.appendChild(div);
  });
  markActive();
}
// A zone is driven either by braid angle or by fiber length. Dim the one not in use.
function markActive() {
  sliders.sliders.forEach(s => {
    const ctl = zoneControl(s.path); const el = document.querySelector('.sl[data-id="' + s.id + '"]');
    if (!el || !ctl || !s.sets_control) return;
    el.classList.toggle('inactive', ctl !== s.sets_control);
    el.title = ctl !== s.sets_control ? 'Not in use: this zone is set by ' + ctl.replace('_', ' ') + '. Move this slider to switch.' : '';
  });
}
function applySlider(s, v) {
  const slot = getPath(request, s.path, true); if (!slot) return;
  if (s.kind === 'toggle') {
    const opt = s.options[Number(v)];  // Milestone A: general toggle (was side-only)
    writeVal(slot, opt);
    if (s.path === 'anatomy.side' && request.measurement_conditions) request.measurement_conditions.side = opt;  // FIX (Claude): keep both sides in step
    return;
  }
  if (s.sets_control && s.path.startsWith('zones.')) slot.o.control = s.sets_control;
  writeVal(slot, Number(v));
}

// ---- boot ----------------------------------------------------------------------------
async function boot() {
  showErr('');
  init3();
  [sliders, hints] = await Promise.all([get('/api/sliders'), get('/web/hints.json')]);
  const pr = await get('/api/presets');
  const sel = $('presets'); sel.innerHTML = '';
  pr.presets.forEach(p => { const o = document.createElement('option'); o.value = p.name; o.textContent = p.name.replace(/_/g, ' '); sel.appendChild(o); });
  const load = async name => { const j = await get('/api/design?name=' + encodeURIComponent(name)); request = j.request; buildSliders(); schedule(); };
  sel.onchange = () => load(sel.value);
  // FIX (Claude): default to the ACL-in-knee design, whose anatomy sliders are all live.
  const first = pr.presets.find(p => p.name === 'acl_context_request') || pr.presets[0];
  sel.value = first.name; await load(first.name);
  $('save').onclick = async () => {
    const n = prompt('Name for this design (letters, digits, - and _):'); if (!n) return;
    const r = await fetch('/api/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: n, request }) });
    const j = await r.json(); alert(r.ok ? 'Saved as fixtures/' + j.saved : 'Not saved: ' + j.error + ' (' + j.hint + ')');
    if (r.ok) { const o = document.createElement('option'); o.value = n; o.textContent = n; sel.appendChild(o); sel.value = n; }
  };
  $('stl').onclick = async () => {
    $('stl').disabled = true; $('stl').textContent = 'Exporting…';
    try {
      const r = await fetch('/api/export_stl', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ request }) });
      if (!r.ok) { const j = await r.json(); throw new Error(j.error); }
      const b = await r.blob(); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'scaffold.stl'; a.click();
    } catch (e) { showErr('Export failed: ' + e.message); }
    $('stl').disabled = false; $('stl').textContent = 'Export STL';
  };
  $('anatomy').onclick = async () => {
    $('anatomy').disabled = true; $('anatomy').textContent = 'Exporting…';
    try {
      const r = await fetch('/api/export_anatomy', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ request }) });
      if (!r.ok) { const j = await r.json(); throw new Error(j.error); }
      const b = await r.blob(); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'anatomy.stl'; a.click();
    } catch (e) { showErr('Export failed: ' + e.message); }
    $('anatomy').disabled = false; $('anatomy').textContent = 'Export anatomy';
  };
}
boot().catch(e => showErr('Could not start: ' + e.message));
