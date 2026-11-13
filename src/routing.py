"""Exact bi-objective label-setting search (Martins, 1984) with lower-bound pruning
and a maximum-detour constraint, plus route evaluation utilities."""
import heapq
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class Net:
    def __init__(self, u, v, length):
        nodes = np.unique(np.r_[u, v])
        self.nodes = nodes
        self.idx = {n: i for i, n in enumerate(nodes)}
        a = np.array([self.idx[x] for x in u]); b = np.array([self.idx[x] for x in v])
        self.a, self.b = a, b
        self.N = len(nodes)
        self.length = np.asarray(length, float)
        # adjacency: list of (nbr, edge id)
        self.adj = [[] for _ in range(self.N)]
        for e, (i, j) in enumerate(zip(a, b)):
            self.adj[i].append((j, e)); self.adj[j].append((i, e))

    def csr(self, w):
        return csr_matrix((np.r_[w, w], (np.r_[self.a, self.b], np.r_[self.b, self.a])), shape=(self.N, self.N))

    def sssp(self, w, src, pred=False):
        return dijkstra(self.csr(w), directed=False, indices=src, return_predecessors=pred)

    def path_edges(self, pred, s, t):
        """Edge ids of the predecessor-tree path s -> t."""
        out = []
        j = t
        while j != s:
            i = pred[j]
            if i < 0:
                return None
            out.append(self._edge(i, j)); j = i
        return out[::-1]

    def _edge(self, i, j):
        best, be = np.inf, None
        for k, e in self.adj[i]:
            if k == j and self.length[e] < best:
                best, be = self.length[e], e
        return be


def pareto_paths(net, o, d, c1, c2, detour=1.5, max_labels=400000):
    """All non-dominated (c1, c2) paths o -> d with c1 <= detour * shortest c1.
    Label-setting in lexicographic order of (f1 + h1, f2 + h2), where h are exact
    single-objective distances to d (admissible lower bounds)."""
    h1 = net.sssp(c1, d); h2 = net.sssp(c2, d)
    L0 = h1[o]
    cap = detour * L0 + 1e-6
    labels = [(0.0, 0.0, o, -1, -1)]          # f1, f2, node, parent label, edge
    best2 = np.full(net.N, np.inf)            # min f2 among permanent labels per node
    heap = [(h1[o], h2[o], 0)]
    sols = []
    while heap:
        k1, k2, li = heapq.heappop(heap)
        f1, f2, i, _, _ = labels[li]
        # dominated by an existing permanent label at this node (f1 ascending -> check min f2)
        # labels leave the heap in increasing f1 at every node (consistent bounds), so a label
        # is dominated iff an earlier permanent label at the node has f2 <= its f2
        if best2[i] <= f2 + 1e-9:
            continue
        if sols and sols[-1][1] <= k2 + 1e-9:   # dominated by a solution already found
            continue
        best2[i] = f2
        if i == d:
            sols.append((f1, f2, li))
            continue
        for j, e in net.adj[i]:
            g1 = f1 + c1[e]; g2 = f2 + c2[e]
            if g1 + h1[j] > cap:
                continue
            if best2[j] <= g2 + 1e-9:
                continue
            labels.append((g1, g2, j, li, e))
            heapq.heappush(heap, (g1 + h1[j], g2 + h2[j], len(labels) - 1))
        if len(labels) > max_labels:
            break
    pareto_paths.n_labels = len(labels)
    pareto_paths.truncated = len(labels) > max_labels
    out = []
    for f1, f2, li in sols:
        es = []
        while labels[li][3] >= 0:
            es.append(labels[li][4]); li = labels[li][3]
        out.append((f1, f2, es[::-1]))
    return out, L0


def knee(sols):
    """Solution with maximum perpendicular distance to the line joining the extremes
    in min-max normalised objective space."""
    if len(sols) <= 2:
        return 0
    f = np.array([[s[0], s[1]] for s in sols])
    lo, hi = f.min(0), f.max(0)
    g = (f - lo) / np.where(hi > lo, hi - lo, 1)
    a, b = g[0], g[-1]
    dvec = b - a
    dist = np.abs(dvec[0] * (a[1] - g[:, 1]) - dvec[1] * (a[0] - g[:, 0])) / np.linalg.norm(dvec)
    return int(np.argmax(dist))


def budget(sols, L0, eps):
    ok = [k for k, s in enumerate(sols) if s[0] <= (1 + eps) * L0 + 1e-6]
    return min(ok, key=lambda k: sols[k][1])


def run_metrics(edges_seq, length, sunlit, thr=0.5):
    """Longest continuous distance on edges with sunlit fraction > thr."""
    best = cur = 0.0
    for e in edges_seq:
        if sunlit[e] > thr:
            cur += length[e]; best = max(best, cur)
        else:
            cur = 0.0
    return best

