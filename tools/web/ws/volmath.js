// Pure imaging math for the MPR viewer (no DOM): plane geometry on the stored canonical grid, orientation labels
// read from the volume's affine, display transforms (window/level, colour maps). Display functions never write to
// the voxel array; they map values to colours.
//
// Volumes arrive from the server on the nearest canonical RAS+ grid (nibabel): +i Right, +j Anterior, +k Superior
// when orientation is known. World coordinates are RAS+ millimetres: world = A · [i, j, k, 1].

export function inv4(m) {
  // 4x4 inverse (row-major nested arrays)
  const a = m.flat(), r = new Array(16);
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = a;
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12, b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) throw new Error('singular affine');
  const d = 1 / det;
  r[0] = (a11 * b11 - a12 * b10 + a13 * b09) * d; r[1] = (a02 * b10 - a01 * b11 - a03 * b09) * d;
  r[2] = (a31 * b05 - a32 * b04 + a33 * b03) * d; r[3] = (a22 * b04 - a21 * b05 - a23 * b03) * d;
  r[4] = (a12 * b08 - a10 * b11 - a13 * b07) * d; r[5] = (a00 * b11 - a02 * b08 + a03 * b07) * d;
  r[6] = (a32 * b02 - a30 * b05 - a33 * b01) * d; r[7] = (a20 * b05 - a22 * b02 + a23 * b01) * d;
  r[8] = (a10 * b10 - a11 * b08 + a13 * b06) * d; r[9] = (a01 * b08 - a00 * b10 - a03 * b06) * d;
  r[10] = (a30 * b04 - a31 * b02 + a33 * b00) * d; r[11] = (a21 * b02 - a20 * b04 - a23 * b00) * d;
  r[12] = (a11 * b07 - a10 * b09 - a12 * b06) * d; r[13] = (a00 * b09 - a01 * b07 + a02 * b06) * d;
  r[14] = (a31 * b01 - a30 * b03 - a32 * b00) * d; r[15] = (a20 * b03 - a21 * b01 + a22 * b00) * d;
  return [r.slice(0, 4), r.slice(4, 8), r.slice(8, 12), r.slice(12, 16)];
}
export const apply4 = (m, p) => [0, 1, 2].map(r => m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3]);
export const voxelToWorld = (vol, ijk) => apply4(vol.affine, ijk);
export const worldToVoxel = (vol, w) => apply4(vol.ainv || (vol.ainv = inv4(vol.affine)), w);
export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// Plane = which stored axis is fixed, which axes run across (u) and down (v) the screen, and whether they are
// mirrored. With known orientation the radiological convention is used: patient right on screen left for
// axial/coronal, anterior on screen left for sagittal, superior/anterior up. The *labels* are not assumed from
// that convention: they are computed from the affine (labelsFor), so an unusual orientation is labelled correctly.
export const PLANES = {
  axial: { fixed: 2, u: 0, v: 1, uFlip: true, vFlip: true, name: 'Axial' },
  coronal: { fixed: 1, u: 0, v: 2, uFlip: true, vFlip: true, name: 'Coronal' },
  sagittal: { fixed: 0, u: 1, v: 2, uFlip: true, vFlip: true, name: 'Sagittal' },
};
// Without orientation metadata the views are named after voxel axes and drawn without mirroring.
export const PLANES_UNORIENTED = {
  axial: { fixed: 2, u: 0, v: 1, uFlip: false, vFlip: true, name: 'Slice plane (k)' },
  coronal: { fixed: 1, u: 0, v: 2, uFlip: false, vFlip: true, name: 'Reformat (j)' },
  sagittal: { fixed: 0, u: 1, v: 2, uFlip: false, vFlip: true, name: 'Reformat (i)' },
};
export function planeOf(vol, key) { return (vol.orientation_known ? PLANES : PLANES_UNORIENTED)[key]; }

const LETTERS = [['L', 'R'], ['P', 'A'], ['I', 'S']];   // [negative, positive] per RAS axis
export function directionLetter(vol, voxelDir) {
  // world direction of a voxel-space direction; return the dominant anatomical letter (+ obliquity hint)
  const A = vol.affine;
  const w = [0, 1, 2].map(r => A[r][0] * voxelDir[0] + A[r][1] * voxelDir[1] + A[r][2] * voxelDir[2]);
  const n = Math.hypot(...w) || 1;
  let ax = 0;
  for (let a = 1; a < 3; a++) if (Math.abs(w[a]) > Math.abs(w[ax])) ax = a;
  return { letter: LETTERS[ax][w[ax] > 0 ? 1 : 0], purity: Math.abs(w[ax]) / n };
}
export function labelsFor(vol, plane) {
  if (!vol.orientation_known) return null;
  const e = a => [0, 1, 2].map(i => (i === a ? 1 : 0));
  const scale = (v, s) => v.map(x => x * s);
  const right = scale(e(plane.u), plane.uFlip ? -1 : 1);       // screen +x in voxel space
  const up = scale(e(plane.v), plane.vFlip ? 1 : -1);          // screen up in voxel space
  const R = directionLetter(vol, right), U = directionLetter(vol, up);
  const L = directionLetter(vol, scale(right, -1)), D = directionLetter(vol, scale(up, -1));
  return { right: R.letter, left: L.letter, top: U.letter, bottom: D.letter, oblique: Math.min(R.purity, U.purity) < 0.985 };
}

// slice <-> screen (in voxel units of the plane, before the physical scale)
export function planeSize(vol, plane) {
  const s = vol.shape, z = vol.zooms;
  return { nu: s[plane.u], nv: s[plane.v], du: z[plane.u], dv: z[plane.v], nfix: s[plane.fixed], dfix: z[plane.fixed] };
}
// voxel (continuous) -> plane image coordinates (pixel units, origin top-left of the slice image)
export function voxelToPlane(vol, plane, ijk) {
  const { nu, nv } = planeSize(vol, plane);
  const u = ijk[plane.u], v = ijk[plane.v];
  return [plane.uFlip ? nu - 1 - u : u, plane.vFlip ? nv - 1 - v : v];
}
export function planeToVoxel(vol, plane, x, y, fixedIndex) {
  const { nu, nv } = planeSize(vol, plane);
  const out = [0, 0, 0];
  out[plane.u] = plane.uFlip ? nu - 1 - x : x;
  out[plane.v] = plane.vFlip ? nv - 1 - y : y;
  out[plane.fixed] = fixedIndex;
  return out;
}
export const vidx = (shape, i, j, k) => i + shape[0] * (j + shape[1] * k);

// ------------------------------------------------------------------ display transforms (read-only on voxels)
export function windowToRange(center, width) { return [center - width / 2, center + width / 2]; }
const CMAPS = {
  gray: t => [t, t, t],
  hot: t => [Math.min(1, 3 * t), Math.min(1, Math.max(0, 3 * t - 1)), Math.min(1, Math.max(0, 3 * t - 2))],
  viridis: t => {            // polynomial fit of matplotlib's viridis (accurate to ~1/255)
    const c0 = [0.2777, 0.0054, 0.3341], c1 = [0.1051, 1.4046, 1.3846], c2 = [-0.3309, 0.2148, 0.0951],
      c3 = [-4.6342, -5.7991, -19.3324], c4 = [6.2283, 14.1799, 56.6906], c5 = [4.7764, -13.7451, -65.353], c6 = [-5.4355, 4.6459, 26.3124];
    return [0, 1, 2].map(i => Math.max(0, Math.min(1, c0[i] + t * (c1[i] + t * (c2[i] + t * (c3[i] + t * (c4[i] + t * (c5[i] + t * c6[i]))))))));
  },
  cool_warm: t => [Math.min(1, 0.23 + 1.3 * t), 0.3 + 0.6 * (1 - Math.abs(2 * t - 1)), Math.min(1, 0.75 + 0.2 * (1 - t)) * (1 - 0.7 * t)],
};
export const COLORMAPS = Object.keys(CMAPS);
const lutCache = new Map();
export function lut(name, invert) {
  const key = name + (invert ? ':i' : '');
  if (lutCache.has(key)) return lutCache.get(key);
  const f = CMAPS[name] || CMAPS.gray, L = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) { const t = invert ? 1 - i / 255 : i / 255; const c = f(t); L[i * 3] = c[0] * 255; L[i * 3 + 1] = c[1] * 255; L[i * 3 + 2] = c[2] * 255; }
  lutCache.set(key, L);
  return L;
}
// Write the slice into an RGBA buffer. data is never modified.
export function renderSlice(vol, plane, fixedIndex, disp, out) {
  const { nu, nv } = planeSize(vol, plane);
  const [lo, hi] = windowToRange(disp.center, disp.width);
  const span = hi - lo || 1e-9, L = lut(disp.cmap || 'gray', !!disp.invert), d = vol.data, sh = vol.shape;
  const ijk = [0, 0, 0]; ijk[plane.fixed] = fixedIndex;
  const missing = vol.missingSet && vol.missingSet.has(plane.fixed + ':' + fixedIndex);
  for (let y = 0; y < nv; y++) {
    ijk[plane.v] = plane.vFlip ? nv - 1 - y : y;
    for (let x = 0; x < nu; x++) {
      ijk[plane.u] = plane.uFlip ? nu - 1 - x : x;
      const o = (y * nu + x) * 4;
      if (missing) { out[o] = 40; out[o + 1] = 20; out[o + 2] = 20; out[o + 3] = 255; continue; }
      const val = d[ijk[0] + sh[0] * (ijk[1] + sh[1] * ijk[2])];
      let t = (val - lo) / span; t = t < 0 ? 0 : t > 1 ? 1 : t;
      const li = (t * 255 + 0.5) | 0;
      out[o] = L[li * 3]; out[o + 1] = L[li * 3 + 1]; out[o + 2] = L[li * 3 + 2]; out[o + 3] = 255;
    }
  }
  return { nu, nv };
}
export function renderMaskSlice(vol, plane, fixedIndex, mask, styles, globalOpacity, out) {
  // styles: Map(labelValue -> {rgb:[r,g,b], opacity, visible}); writes premultiplied-free RGBA
  const { nu, nv } = planeSize(vol, plane), sh = vol.shape;
  const ijk = [0, 0, 0]; ijk[plane.fixed] = fixedIndex;
  let any = false;
  for (let y = 0; y < nv; y++) {
    ijk[plane.v] = plane.vFlip ? nv - 1 - y : y;
    for (let x = 0; x < nu; x++) {
      ijk[plane.u] = plane.uFlip ? nu - 1 - x : x;
      const o = (y * nu + x) * 4, lab = mask[ijk[0] + sh[0] * (ijk[1] + sh[1] * ijk[2])];
      const st = lab && styles.get(lab);
      if (st && st.visible) { out[o] = st.rgb[0]; out[o + 1] = st.rgb[1]; out[o + 2] = st.rgb[2]; out[o + 3] = 255 * st.opacity * globalOpacity; any = true; }
      else out[o + 3] = 0;
    }
  }
  return any;
}
export function hexRgb(h) { const n = parseInt(h.replace('#', ''), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

// brush footprint in the plane: voxel offsets within radius (mm when calibrated, voxels otherwise)
export function brushOffsets(vol, plane, radius) {
  const { du, dv } = planeSize(vol, plane);
  const su = vol.spacing_known ? du : 1, sv = vol.spacing_known ? dv : 1;
  const ru = Math.floor(radius / su), rv = Math.floor(radius / sv), out = [];
  for (let b = -rv; b <= rv; b++) for (let a = -ru; a <= ru; a++) if ((a * su) ** 2 + (b * sv) ** 2 <= radius * radius + 1e-9) out.push([a, b]);
  return out.length ? out : [[0, 0]];
}
export function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (((yi > y) !== (yj > y)) && x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi) inside = !inside;
  }
  return inside;
}
// a stable, cheap checksum used to prove display operations leave voxel data untouched
export function checksum(arr) {
  let h = 2166136261 >>> 0;
  const b = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  for (let i = 0; i < b.length; i += 7) { h ^= b[i]; h = Math.imul(h, 16777619) >>> 0; }
  return h.toString(16) + ':' + b.length;
}
