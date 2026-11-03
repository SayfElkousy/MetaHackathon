"""Vectorized meshing utilities.

- surface_nets: closed triangle mesh of the zero level set of a sampled scalar
  field (negative = inside). The field is padded with "outside" so every surface
  is closed. No per-cell Python loops.
- splat_field: turns a cloud of interior sample points into a smooth field,
  used for tissue remnants, the injury region and scaffold regions.
- audit: open/non-manifold edge counts, volume, orientation, degenerate faces.
- smooth_taubin: shrink-free smoothing of surface-nets output.
"""
import numpy as np
from scipy import ndimage


def surface_nets(F, origin, h):
    """Mesh the zero level set of F (shape nx,ny,nz) sampled at origin + h*idx.

    Returns V (n,3) float64 in world units and Fc (m,3) int64, outward-wound.
    """
    F = np.asarray(F, dtype=np.float32)
    F = np.pad(F, 1, mode="constant", constant_values=1.0)
    origin = np.asarray(origin, dtype=float) - h
    s = F < 0.0
    nx, ny, nz = F.shape
    csum = np.zeros((nx - 1, ny - 1, nz - 1, 3), dtype=np.float64)
    ccnt = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.int32)

    def edge_points(axis):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, -1)
        sl1[axis] = slice(1, None)
        f0 = F[tuple(sl0)]
        f1 = F[tuple(sl1)]
        cross = s[tuple(sl0)] != s[tuple(sl1)]
        den = np.where(cross, f0 - f1, 1.0)
        t = np.where(cross, f0 / den, 0.0)
        idx = np.indices(f0.shape).astype(np.float64)
        idx[axis] += t
        P = np.moveaxis(idx, 0, -1) * cross[..., None]
        return P, cross

    for axis in range(3):
        P, cross = edge_points(axis)
        c = cross.astype(np.int32)
        others = [a for a in range(3) if a != axis]
        for d0 in (0, 1):
            for d1 in (0, 1):
                sl = [slice(None)] * 3
                sl[others[0]] = slice(d0, d0 + P.shape[others[0]] - 1)
                sl[others[1]] = slice(d1, d1 + P.shape[others[1]] - 1)
                csum += P[tuple(sl)]
                ccnt += c[tuple(sl)]
    active = ccnt > 0
    vid = -np.ones(active.shape, dtype=np.int64)
    nv = int(active.sum())
    vid[active] = np.arange(nv)
    V = csum[active] / ccnt[active][:, None]
    V = origin + V * h

    faces = []
    # x-edges: cells (i, j-1..j, k-1..k)
    cx = s[:-1, :, :] != s[1:, :, :]
    cx[:, 0, :] = False
    cx[:, -1, :] = False
    cx[:, :, 0] = False
    cx[:, :, -1] = False
    i, j, k = np.nonzero(cx)
    inside_lo = s[i, j, k]
    q = np.stack([vid[i, j - 1, k - 1], vid[i, j, k - 1], vid[i, j, k], vid[i, j - 1, k]], 1)
    faces.append((q, inside_lo))
    cy = s[:, :-1, :] != s[:, 1:, :]
    cy[0, :, :] = False
    cy[-1, :, :] = False
    cy[:, :, 0] = False
    cy[:, :, -1] = False
    i, j, k = np.nonzero(cy)
    inside_lo = s[i, j, k]
    q = np.stack([vid[i - 1, j, k - 1], vid[i - 1, j, k], vid[i, j, k], vid[i, j, k - 1]], 1)
    faces.append((q, inside_lo))
    cz = s[:, :, :-1] != s[:, :, 1:]
    cz[0, :, :] = False
    cz[-1, :, :] = False
    cz[:, 0, :] = False
    cz[:, -1, :] = False
    i, j, k = np.nonzero(cz)
    inside_lo = s[i, j, k]
    q = np.stack([vid[i - 1, j - 1, k], vid[i, j - 1, k], vid[i, j, k], vid[i - 1, j, k]], 1)
    faces.append((q, inside_lo))
    tris = []
    for q, ins in faces:
        q = np.where(ins[:, None], q, q[:, ::-1])
        # split along the shorter diagonal for better triangles
        d02 = np.linalg.norm(V[q[:, 0]] - V[q[:, 2]], axis=1)
        d13 = np.linalg.norm(V[q[:, 1]] - V[q[:, 3]], axis=1)
        a = d02 <= d13
        t1 = np.where(a[:, None], q[:, [0, 1, 2]], q[:, [0, 1, 3]])
        t2 = np.where(a[:, None], q[:, [0, 2, 3]], q[:, [1, 2, 3]])
        tris.append(t1)
        tris.append(t2)
    Fc = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    return V, Fc.astype(np.int64)


def vertex_neighbors(F, nv):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    e = np.unique(e, axis=0)
    return e


def smooth_taubin(V, F, iters=6, lam=0.5, mu=-0.53):
    if V.shape[0] == 0:
        return V
    e = vertex_neighbors(F, V.shape[0])
    deg = np.bincount(e[:, 0], minlength=V.shape[0]).astype(float)
    deg[deg == 0] = 1.0
    V = V.copy()
    for _ in range(iters):
        for f in (lam, mu):
            acc = np.zeros_like(V)
            np.add.at(acc, e[:, 0], V[e[:, 1]])
            V += f * (acc / deg[:, None] - V)
    return V


def splat_field(P, lo, hi, h, sigma_vox=0.9, weights=None):
    """Occupancy splat of interior sample points -> smooth field (neg inside).

    P: (n,3) points that lie inside the solid, dense enough that neighbours are
    closer than about one voxel. Returns (field, origin).
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    shape = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
    G = np.zeros(shape, dtype=np.float32)
    if P.shape[0]:
        idx = np.rint((P - lo) / h).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < shape), axis=1)
        idx = idx[ok]
        w = 1.0 if weights is None else weights[ok]
        np.add.at(G, (idx[:, 0], idx[:, 1], idx[:, 2]), w)
    G = np.minimum(G, 1.0)
    G = ndimage.binary_closing(G > 0, iterations=1).astype(np.float32)
    G = ndimage.gaussian_filter(G, sigma_vox)
    return 0.5 - G, lo


def mesh_from_points(P, h=0.35, pad=2.0, sigma_vox=0.9, smooth=4):
    """Closed surface around interior points. Surface nets can leave a few
    non-manifold edges at pinch points; retry at nearby resolutions (deterministic)
    and keep the first clean result (or the cleanest)."""
    if P.shape[0] < 8:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    best = None
    for hh, sg in ((h, sigma_vox), (h * 1.07, sigma_vox * 1.15), (h * 0.93, sigma_vox * 1.3), (h * 1.15, sigma_vox * 1.4), (h * 0.97, sigma_vox * 1.6), (h * 1.25, sigma_vox * 1.25)):
        lo = P.min(0) - pad
        hi = P.max(0) + pad
        G, o = splat_field(P, lo, hi, hh, sg)
        V, F = surface_nets(G, o, hh)
        if smooth:
            V = smooth_taubin(V, F, iters=smooth)
        bad = audit(V, F)["nonmanifold_edges"]
        if best is None or bad < best[0]:
            best = (bad, V, F)
        if bad == 0:
            break
    return best[1], best[2]


def audit(V, F):
    """Edge-manifold audit. Closed = every edge shared by exactly two faces."""
    out = {"vertices": int(V.shape[0]), "faces": int(F.shape[0])}
    if F.shape[0] == 0:
        out.update(open_edges=0, nonmanifold_edges=0, closed=False, volume_mm3=0.0,
                   degenerate_faces=0, orientation_consistent=False)
        return out
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
    es = np.sort(e, axis=1)
    nv = np.int64(max(int(V.shape[0]), 1))
    _, cnt = np.unique(es[:, 0] * nv + es[:, 1], return_counts=True)
    # directed edges: a consistently oriented closed mesh uses each directed edge once
    _, dcnt = np.unique(e[:, 0] * nv + e[:, 1], return_counts=True)
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    out.update(open_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
               closed=bool((cnt == 2).all()), volume_mm3=vol,
               degenerate_faces=int((area < 1e-10).sum()),
               orientation_consistent=bool((dcnt == 1).all()))
    return out


def components(F, nv):
    """Connected components of a triangle mesh -> label per vertex."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if F.shape[0] == 0:
        return 0, np.zeros(nv, int)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]])
    A = coo_matrix((np.ones(e.shape[0]), (e[:, 0], e[:, 1])), shape=(nv, nv))
    return connected_components(A, directed=False)


def split_components(V, F, min_faces=20):
    n, lab = components(F, V.shape[0])
    out = []
    flab = lab[F[:, 0]]
    for c in range(n):
        m = flab == c
        if m.sum() < min_faces:
            continue
        Fc = F[m]
        used = np.unique(Fc)
        remap = -np.ones(V.shape[0], np.int64)
        remap[used] = np.arange(used.size)
        out.append((V[used], remap[Fc]))
    out.sort(key=lambda vf: -vf[1].shape[0])
    return out


def tube_mesh(P, radius, nsec=8, cap=True):
    """Closed tube along polyline P (n,3); radius scalar or (n,). Vectorized."""
    P = np.asarray(P, float)
    n = P.shape[0]
    r = np.broadcast_to(np.asarray(radius, float), (n,))
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    N = np.zeros_like(P)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N[0] = np.cross(T[0], ref)
    N[0] /= np.linalg.norm(N[0])
    for k in range(1, n):  # parallel transport (n is small: one tube)
        v = N[k - 1] - np.dot(N[k - 1], T[k]) * T[k]
        nv = np.linalg.norm(v)
        N[k] = v / nv if nv > 1e-12 else N[k - 1]
    B = np.cross(T, N)
    a = np.linspace(0, 2 * np.pi, nsec, endpoint=False)
    ring = (np.cos(a)[None, :, None] * N[:, None, :] + np.sin(a)[None, :, None] * B[:, None, :])
    V = (P[:, None, :] + r[:, None, None] * ring).reshape(-1, 3)
    i = np.arange(n - 1)[:, None]
    j = np.arange(nsec)[None, :]
    a0 = i * nsec + j
    a1 = i * nsec + (j + 1) % nsec
    b0 = a0 + nsec
    b1 = a1 + nsec
    F = np.concatenate([np.stack([a0, a1, b1], -1).reshape(-1, 3), np.stack([a0, b1, b0], -1).reshape(-1, 3)])
    if cap:
        c0 = V.shape[0]
        V = np.vstack([V, P[0], P[-1]])
        jj = np.arange(nsec)
        F = np.vstack([F, np.stack([np.full(nsec, c0), (jj + 1) % nsec, jj], 1),
                       np.stack([np.full(nsec, c0 + 1), (n - 1) * nsec + jj, (n - 1) * nsec + (jj + 1) % nsec], 1)])
    return V, F.astype(np.int64)


def concat(parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += V.shape[0]
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.vstack(Vs), np.vstack(Fs)
