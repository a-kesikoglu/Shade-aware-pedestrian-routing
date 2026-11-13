"""Raster sampling along network edges."""
import numpy as np


def sample_points(geoms, along, offsets, transform, shape):
    """Raster indices of points every `along` m along each line at perpendicular `offsets` (m).
    Returns edge index, row and column arrays for points inside the raster."""
    H, W = shape
    E, R, C = [], [], []
    for i, ls in enumerate(geoms):
        L = ls.length
        d = np.arange(along / 2, L, along) if L > along else np.array([L / 2])
        pts = np.array([ls.interpolate(x).coords[0] for x in d])
        pts2 = np.array([ls.interpolate(min(L, x + 0.5)).coords[0] for x in d])
        t = pts2 - pts
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        nrm = np.c_[-t[:, 1], t[:, 0]]
        for o in offsets:
            q = pts + o * nrm
            col = ((q[:, 0] - transform.c) / transform.a).astype(int)
            row = ((q[:, 1] - transform.f) / transform.e).astype(int)
            E.append(np.full(len(q), i)); R.append(row); C.append(col)
    E, R, C = map(np.concatenate, (E, R, C))
    ok = (R >= 0) & (R < H) & (C >= 0) & (C < W)
    return E[ok], R[ok], C[ok]


def corridor_points(geoms, bmask, transform):
    """Walkable corridor of +/-3 m around each centreline with building pixels removed;
    edges lying entirely inside footprints (passages) keep all their points."""
    n = len(geoms)
    E, R, C = sample_points(geoms, 1.0, [-3, -2, -1, 0, 1, 2, 3], transform, bmask.shape)
    free = ~bmask[R, C]
    has_free = np.bincount(E[free], minlength=n) > 0
    use = free | ~has_free[E]
    return E[use], R[use], C[use]


def edge_sunlit(tr, idx, covered):
    """Edge sunlit fraction from a transmission raster; covered edges (pedestrian underpasses) receive 0."""
    E, R, C = idx
    s = edge_mean(E, tr[R, C], len(covered))
    s[np.asarray(covered, bool)] = 0.0
    return s


def edge_mean(E, vals, n):
    s = np.bincount(E, weights=vals, minlength=n)
    k = np.bincount(E, minlength=n)
    return np.where(k > 0, s / np.maximum(k, 1), np.nan)
