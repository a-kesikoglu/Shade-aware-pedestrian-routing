"""Deterministic edge-case tests of the bounded biobjective search (routing.pareto_paths, budget, knee).

1 two disjoint paths with identical (length, cost) vectors: one representative is returned, always the same
2 zero secondary cost on every edge: the front collapses to the shortest path
3 a path whose length equals the 10% budget exactly is admissible for B10; one 1 cm longer is not
4 a path exactly at the 50% detour cap is retained; one 1 cm longer is pruned
5 collinear front of three vectors: the middle vector is supported and the knee falls back to SP
6 a concave front (middle vector above the chord) has no knee; a convex one returns its middle vector
Results are written to data/processed/rev_search_tests.json.
"""
import json
import numpy as np
from config import PROC
from routing import Net, pareto_paths, budget, knee


def diamond(paths):
    """Graph with o=0, d=1 and one two-edge path per entry (length, cost) via its own middle node."""
    u, v, L, C = [], [], [], []
    for k, (l, c) in enumerate(paths):
        m = 2 + k
        u += [0, m]; v += [m, 1]; L += [l / 2, l / 2]; C += [c / 2, c / 2]
    return Net(np.array(u), np.array(v), np.array(L)), np.array(L, float), np.array(C, float)


out = {}
# 1 identical vectors
reps = []
for _ in range(5):
    net, L, C = diamond([(100, 50), (110, 40), (110, 40)])
    sols, L0 = pareto_paths(net, 0, 1, L, C)
    reps.append([tuple(s[2]) for s in sols])
out["identical_vectors"] = {"n_solutions": len(reps[0]), "deterministic": all(r == reps[0] for r in reps),
                            "pass": len(reps[0]) == 2 and all(r == reps[0] for r in reps)}
# 2 zero secondary cost
net, L, C = diamond([(100, 0), (120, 0), (140, 0)])
sols, L0 = pareto_paths(net, 0, 1, L, C)
out["zero_cost"] = {"n_solutions": len(sols), "pass": len(sols) == 1 and abs(sols[0][0] - 100) < 1e-9}
# 3 budget boundary
net, L, C = diamond([(100, 50), (110, 30), (110.01, 20)])
sols, L0 = pareto_paths(net, 0, 1, L, C)
k = budget(sols, L0, 0.10)
out["budget_boundary"] = {"selected_length": sols[k][0], "pass": abs(sols[k][0] - 110) < 1e-6}
# 4 detour cap boundary
net, L, C = diamond([(100, 50), (150, 30), (150.01, 10)])
sols, L0 = pareto_paths(net, 0, 1, L, C)
lens = sorted(s[0] for s in sols)
out["detour_cap"] = {"lengths": lens, "pass": lens == [100.0, 150.0]}
# 5 collinear front
net, L, C = diamond([(100, 60), (110, 50), (120, 40)])
sols, L0 = pareto_paths(net, 0, 1, L, C)
out["collinear"] = {"n_solutions": len(sols), "knee": knee(sols), "pass": len(sols) == 3 and knee(sols) == 0}
# 6 concave and convex fronts
net, L, C = diamond([(100, 60), (110, 58), (120, 40)])
s_cc, _ = pareto_paths(net, 0, 1, L, C)
net, L, C = diamond([(100, 60), (102, 42), (120, 40)])
s_cv, _ = pareto_paths(net, 0, 1, L, C)
out["knee_shape"] = {"concave_knee": knee(s_cc), "convex_knee": knee(s_cv),
                     "pass": knee(s_cc) == 0 and knee(s_cv) == 1}
out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
json.dump(out, open(PROC / "rev_search_tests.json", "w"), indent=1, default=lambda o: o.item())
print(json.dumps(out, indent=1, default=lambda o: o.item()))
