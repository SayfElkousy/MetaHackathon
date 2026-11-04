// Node tests for the workspace's pure modules (no browser): node --test tests/js/
import test from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../../tools/web/ws/volmath.js';
import { parse, convert, resolve, keepPrefixes } from '../../tools/web/ws/intent.js';
import { validateValue } from '../../tools/web/ws/txn.js';

// canonical RAS+ volume, anisotropic: 3 mm along i (L-R), 0.5 mm in-plane — like a sagittal knee acquisition
const vol = (over) => Object.assign({ shape: [21, 160, 180], zooms: [3, 0.5, 0.5], orientation_known: true, spacing_known: true,
  affine: [[3, 0, 0, -30], [0, 0.5, 0, -39.5], [0, 0, 0.5, -44.5], [0, 0, 0, 1]] }, over);

test('orientation labels come from the affine (radiological display)', () => {
  const v = vol();
  assert.deepEqual(pick(M.labelsFor(v, M.PLANES.axial)), { left: 'R', right: 'L', top: 'A', bottom: 'P' });
  assert.deepEqual(pick(M.labelsFor(v, M.PLANES.coronal)), { left: 'R', right: 'L', top: 'S', bottom: 'I' });
  assert.deepEqual(pick(M.labelsFor(v, M.PLANES.sagittal)), { left: 'A', right: 'P', top: 'S', bottom: 'I' });
  // a volume whose first axis points LEFT is labelled from its affine, not from screen position
  const flipped = vol({ affine: [[-3, 0, 0, 30], [0, 0.5, 0, -39.5], [0, 0, 0.5, -44.5], [0, 0, 0, 1]] });
  assert.equal(M.labelsFor(flipped, M.PLANES.axial).left, 'L');
  assert.equal(M.labelsFor(vol({ orientation_known: false }), M.PLANES_UNORIENTED.axial), null);
});
const pick = L => ({ left: L.left, right: L.right, top: L.top, bottom: L.bottom });

test('world <-> voxel round trip and crosshair agreement across planes', () => {
  const v = vol(), w = [6, 1.25, -3.5];
  const ijk = M.worldToVoxel(v, w);
  assert.deepEqual(M.voxelToWorld(v, ijk).map(x => +x.toFixed(9)), w);
  for (const p of Object.values(M.PLANES)) {
    const [x, y] = M.voxelToPlane(v, p, ijk);
    const back = M.planeToVoxel(v, p, x, y, ijk[p.fixed]);
    assert.deepEqual(back.map(a => +a.toFixed(9)), ijk.map(a => +a.toFixed(9)), p.name);
  }
});

test('physical proportions: plane size in mm uses voxel spacing', () => {
  const v = vol(), s = M.planeSize(v, M.PLANES.coronal);
  assert.equal(s.nu * s.du, 63);      // 21 slices x 3 mm across
  assert.equal(s.nv * s.dv, 90);      // 180 rows x 0.5 mm
});

test('windowing and colour maps never modify voxel data', () => {
  const v = vol({ shape: [4, 5, 6] }), n = 4 * 5 * 6;
  v.data = Int16Array.from({ length: n }, (_, i) => (i * 37) % 1000 - 200);
  const c0 = M.checksum(v.data), copy = Int16Array.from(v.data), out = new Uint8ClampedArray(5 * 6 * 4);
  for (const cmap of M.COLORMAPS) for (const invert of [false, true]) M.renderSlice(v, M.PLANES.sagittal, 2, { center: 100, width: 300, cmap, invert }, out);
  assert.equal(M.checksum(v.data), c0);
  assert.deepEqual(v.data, copy);
  // window maps lo -> black, hi -> white in grayscale
  const w = vol({ shape: [2, 1, 1] }); w.data = Int16Array.from([0, 100]);
  const o2 = new Uint8ClampedArray(8); M.renderSlice(w, M.PLANES.axial, 0, { center: 50, width: 100, cmap: 'gray' }, o2);
  assert.deepEqual([o2[0], o2[4]], [255, 0]);   // axial is mirrored (patient right on screen left): i=1 drawn first
});

test('brush footprint respects anisotropic spacing', () => {
  const v = vol(), off = M.brushOffsets(v, M.PLANES.coronal, 3);  // u = i (3 mm), v = k (0.5 mm)
  const us = off.map(o => o[0]), vs = off.map(o => o[1]);
  assert.equal(Math.max(...us), 1); assert.equal(Math.max(...vs), 6);
});

// ---------------- parser and validation against a small fake registry
const entry = (key, label, o) => Object.assign({ key, label, units: '', kind: 'scalar', get: () => 1 }, o);
const reg = { entries: new Map([
  entry('sheath.wall_mm', 'Sheath wall thickness', { units: 'mm', min: 0.05, max: 2, param: true, level: 5, get: () => 0.4 }),
  entry('ch.wall_um', 'Channel wall thickness', { units: 'um', min: 20, max: 400, param: true, level: 3 }),
  entry('ch.lumen_um', 'Channel lumen diameter', { units: 'um', min: 50, max: 800, param: true, level: 3, get: () => 200 }),
  entry('gr.width_um', 'Groove width', { units: 'um', min: 2, max: 100, param: true, level: 2 }),
  entry('img.overlay.opacity', 'Segmentation overlay opacity', { units: 'fraction', min: 0, max: 1 }),
  entry('scaffold.braid_angle_deg', 'Braid angle', { units: 'deg', min: 10, max: 60, readOnly: 'computed: tan(angle) = 2πR / pitch' }),
  entry('scaffold.braid_pitch_mm', 'Braid pitch', { units: 'mm', min: 1, max: 30 }),
].map(e => [e.key, e])), actions: new Map() };

test('unit conversion', () => {
  assert.equal(convert(0.3, 'mm', reg.entries.get('ch.lumen_um')).value, 300);
  assert.equal(convert(30, '%', reg.entries.get('img.overlay.opacity')).value, 0.3);
  assert.ok(convert(30, '%', reg.entries.get('sheath.wall_mm')).error);
  assert.ok(convert(2, 'mm', reg.entries.get('scaffold.braid_pitch_mm')).value === 2);
});

test('brief examples map to the right actions', () => {
  const p = t => parse(t, reg).calls;
  assert.deepEqual(p('Open axial, sagittal, and coronal views next to the scaffold'), [{ name: 'ws.preset', args: { name: 'mpr_scaffold' } }]);
  assert.equal(p('Move the crosshair to the selected annotation')[0].name, 'img.crosshair.goto_annotation');
  assert.deepEqual(p('Show the ACL segmentation')[0], { name: 'img.show_label', args: { label: 'ACL' } });
  assert.deepEqual(p('Reduce overlay opacity to 30 percent')[0].args.changes, [{ key: 'img.overlay.opacity', value: 0.3 }]);
  assert.equal(p('Reset the windowing')[0].name, 'img.window.reset');
  assert.equal(p('Show the original MRI without annotations')[0].name, 'img.show_original');
  assert.equal(p('Open the quality-control panel')[0].args.view, 'qc');
  assert.equal(p('Show which measurements were used for the scaffold')[0].name, 'evidence.show_sources');
  assert.equal(p('Undo that change')[0].name, 'design.undo_last');
  assert.equal(p('Keep the geometry but restore the previous view')[0].name, 'design.restore_view');
  assert.equal(p('Run the segmentation')[0].name, 'seg.run');
});

test('generic edits, units, relative changes, several edits in one transaction', () => {
  const c = parse('Set the sheath wall thickness to 0.3 mm and set the channel lumen to 0.25 mm', reg).calls;
  assert.equal(c.length, 1);
  assert.deepEqual(c[0].args.changes, [{ key: 'sheath.wall_mm', value: 0.3 }, { key: 'ch.lumen_um', value: 250 }]);
  assert.deepEqual(parse('increase the channel lumen by 10%', reg).calls[0].args.changes, [{ key: 'ch.lumen_um', value: 220 }]);
});

test('ambiguous targets and undefined goals ask a question', () => {
  const a = parse('set the wall thickness to 0.3', reg).calls[0];
  assert.equal(a.name, 'ask_user'); assert.match(a.args.question, /Sheath wall/); assert.match(a.args.question, /Channel wall/);
  assert.equal(parse('optimize healing', reg).calls[0].name, 'ask_user');
});

test('composite evidence request keeps groove settings and shows the result inside the ACL', () => {
  const r = parse('Use the reviewed attachment positions, adapt the scaffold envelope to the measured dimensions, retain the current groove settings, and show the result inside the ACL.', reg);
  const ev = r.calls.find(c => c.name === 'evidence.apply');
  assert.ok(ev && ev.args.scope.includes('acl_length') && ev.args.scope.includes('acl_width'));
  assert.ok(ev.args.keep.includes('gr.*'));
  assert.ok(r.notes.some(n => /registration/.test(n)));
  assert.deepEqual(r.calls.find(c => c.name === 'set_controls').args.changes, [{ key: 'view.mode', value: 'anatomy' }, { key: 'view.acl_overlay', value: true }]);
  assert.deepEqual(keepPrefixes('retain the current groove settings', reg), ['gr.*']);
});

test('validation: bounds, read-only derived values, kinds', () => {
  const S = { reg: { s_knots: [0, .25, .5, .75, 1] } };
  assert.ok(validateValue(reg.entries.get('sheath.wall_mm'), 3, S).error.includes('maximum'));
  assert.ok(validateValue(reg.entries.get('scaffold.braid_angle_deg'), 30, S).error.includes('read-only'));
  assert.equal(validateValue(reg.entries.get('sheath.wall_mm'), '0.5', S).value, 0.5);
  assert.ok(validateValue({ kind: 'enum', options: ['a', 'b'] }, 'c', S).error);
  assert.ok(validateValue({ kind: 'profile', min: 3, max: 22 }, [1, 2, 3], S).error);
  assert.equal(resolve(reg, 'braid angle', e => !e.readOnly).none, true);
});
