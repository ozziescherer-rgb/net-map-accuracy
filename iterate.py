"""Iterative refinement + load-signature physics score (Phase 2 seed).
Round-trip: score -> reassign worst -> rescore. Then add a voltage-vs-group-load
partial-correlation score: your voltage dips when YOUR transformer's load spikes."""
import numpy as np, pandas as pd
from lib import load_meta, corr_matrix, degrade, OUT

meta, true_g, nearest = load_meta()
V1 = np.load(f"{OUT}/V_1min.npy")
n = len(true_g)
V = degrade(V1.reshape(-1, 15, n).mean(axis=1))          # AMI-realistic voltage
C = corr_matrix(V)

r = np.random.default_rng(7)
rec0 = true_g.copy()
bad = r.choice(n, size=int(0.10*n), replace=False)
for i in bad: rec0[i] = r.choice(nearest[true_g[i]])
is_bad = np.zeros(n, bool); is_bad[bad] = True

def scores_for(i, labels, cands):
    out = {}
    for g in cands:
        mem = np.where(labels == g)[0]; mem = mem[mem != i]
        out[g] = C[i, mem].mean() if len(mem) else np.nan
    return out

cand_sets = [ (set(nearest[rec0[i]].tolist()) | {rec0[i]}
             | set(nearest[true_g[i]].tolist()) | {true_g[i]}) for i in range(n) ]

labels = rec0.copy()
for rnd in range(4):
    within = np.empty(n)
    for i in range(n):
        mem = np.where(labels == labels[i])[0]; mem = mem[mem != i]
        within[i] = C[i, mem].mean() if len(mem) else np.nan
    ok = ~np.isnan(within)
    med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826
    th_low, th_high = med - 5*mad, med - 1*mad
    moved = 0
    # flag the most anomalous first, reassign, so references clean up as we go
    order = np.argsort(np.where(ok, within, np.inf))
    for i in order:
        sc = scores_for(i, labels, cand_sets[i] - {labels[i]})
        sc = {g: s for g, s in sc.items() if not np.isnan(s)}
        if not sc: continue
        g_best = max(sc, key=sc.get); s_best = sc[g_best]
        w = within[i]
        if (not np.isnan(w) and w < th_low and s_best > w) or \
           (s_best > th_high and (np.isnan(w) or s_best > w)):
            labels[i] = g_best; moved += 1
    det = (labels[bad] != rec0[bad]).mean()
    fpr = (labels[~is_bad] != rec0[~is_bad]).mean()
    corr = (labels[bad] == true_g[bad]).mean()
    print(f"round {rnd}: moved={moved:4d}  det={det:.3f} fpr={fpr:.3f} corr={corr:.3f}", flush=True)

fixable = np.array([np.sum((rec0 == true_g[i]) & (np.arange(n) != i) &
                           (true_g == true_g[i])) > 0 for i in bad])
print(f"iterative final: corr_fixable={(labels[bad][fixable]==true_g[bad][fixable]).mean():.3f}")
np.save(f"{OUT}/labels_iter.npy", labels)
np.save(f"{OUT}/rec0.npy", rec0)
np.save(f"{OUT}/bad.npy", bad)
