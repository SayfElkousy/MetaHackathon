// Tube-geometry worker: builds merged-tube vertex/index buffers off the main
// thread. Same parallel-transport frames as the in-page fallback.
// Message in: {id, P, offs, R, nsec, col, lidBase} (transferable buffers).
// Message out: {id, pos, nor, idx, lid, nv} (transferable buffers).
self.onmessage = (ev) => {
  const { id, P, offs, R, nsec, col, lidBase } = ev.data;
  try {
    const P32 = new Float32Array(P), O32 = new Uint32Array(offs),
      R32 = new Float32Array(R), C32 = new Float32Array(col);
    let nv = 0, ni = 0;
    for (let l = 0; l < O32.length - 1; l++) {
      const n = O32[l + 1] - O32[l];
      if (n < 2) continue;
      nv += n * nsec; ni += (n - 1) * nsec * 6;
    }
    const pos = new Float32Array(nv * 3), nor = new Float32Array(nv * 3), lid = new Float32Array(nv);
    const idx = new Uint32Array(ni);
    const cs = [], sn = [];
    for (let k = 0; k < nsec; k++) { cs.push(Math.cos(2 * Math.PI * k / nsec)); sn.push(Math.sin(2 * Math.PI * k / nsec)); }
    let vo = 0, io = 0;
    let Tx = 0, Ty = 0, Tz = 0, Nx = 0, Ny = 0, Nz = 0, Bx = 0, By = 0, Bz = 0;
    for (let l = 0; l < O32.length - 1; l++) {
      const a = O32[l], n = O32[l + 1] - a;
      if (n < 2) continue;
      const base = vo;
      for (let i = 0; i < n; i++) {
        const j = a + i, jp = a + Math.min(i + 1, n - 1), jm = a + Math.max(i - 1, 0);
        Tx = P32[3 * jp] - P32[3 * jm]; Ty = P32[3 * jp + 1] - P32[3 * jm + 1]; Tz = P32[3 * jp + 2] - P32[3 * jm + 2];
        const tl = Math.hypot(Tx, Ty, Tz) || 1;
        Tx /= tl; Ty /= tl; Tz /= tl;
        if (i === 0) {
          let tx = 0, ty = 0, tz = 1;
          if (Math.abs(Tz) > 0.9) { tx = 1; ty = 0; tz = 0; }
          Nx = Ty * tz - Tz * ty; Ny = Tz * tx - Tx * tz; Nz = Tx * ty - Ty * tx;
          const nl = Math.hypot(Nx, Ny, Nz) || 1;
          Nx /= nl; Ny /= nl; Nz /= nl;
        } else {
          const d = Nx * Tx + Ny * Ty + Nz * Tz;
          Nx -= d * Tx; Ny -= d * Ty; Nz -= d * Tz;
          const nl = Math.hypot(Nx, Ny, Nz) || 1;
          Nx /= nl; Ny /= nl; Nz /= nl;
        }
        Bx = Ty * Nz - Tz * Ny; By = Tz * Nx - Tx * Nz; Bz = Tx * Ny - Ty * Nx;
        const r = R32[j];
        for (let k = 0; k < nsec; k++) {
          const nx = cs[k] * Nx + sn[k] * Bx, ny = cs[k] * Ny + sn[k] * By, nz = cs[k] * Nz + sn[k] * Bz;
          const q = 3 * vo;
          pos[q] = P32[3 * j] + r * nx; pos[q + 1] = P32[3 * j + 1] + r * ny; pos[q + 2] = P32[3 * j + 2] + r * nz;
          nor[q] = nx; nor[q + 1] = ny; nor[q + 2] = nz; lid[vo] = lidBase + l; vo++;
        }
      }
      for (let i = 0; i < n - 1; i++) for (let k = 0; k < nsec; k++) {
        const a0 = base + i * nsec + k, a1 = base + i * nsec + (k + 1) % nsec, b0 = a0 + nsec, b1 = a1 + nsec;
        idx[io++] = a0; idx[io++] = b0; idx[io++] = a1; idx[io++] = a1; idx[io++] = b0; idx[io++] = b1;
      }
    }
    // expand per-point colors to per-vertex colors
    const vcol = new Float32Array(nv * 3);
    let w = 0;
    for (let l = 0; l < O32.length - 1; l++) {
      const a = O32[l], n = O32[l + 1] - a;
      if (n < 2) continue;
      for (let i = 0; i < n; i++) {
        const j = a + i;
        for (let k = 0; k < nsec; k++) {
          vcol[3 * w] = C32[3 * j]; vcol[3 * w + 1] = C32[3 * j + 1]; vcol[3 * w + 2] = C32[3 * j + 2]; w++;
        }
      }
    }
    self.postMessage({ id, pos: pos.buffer, nor: nor.buffer, idx: idx.buffer, lid: lid.buffer, vcol: vcol.buffer, nv },
      [pos.buffer, nor.buffer, idx.buffer, lid.buffer, vcol.buffer]);
  } catch (e) {
    self.postMessage({ id, error: String(e && e.message || e) });
  }
};
