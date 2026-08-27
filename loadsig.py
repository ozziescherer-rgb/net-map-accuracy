"""Physics score: a meter's voltage residual should anti-correlate with ITS transformer's
aggregate load. Combine with meter-meter correlation; score correction accuracy."""
import numpy as np, pandas as pd
from lib import load_meta, corr_matrix, degrade, OUT

meta, true_g, nearest = load_meta()
n = len(true_g)
V1 = np.load(f"{OUT}/V_1min.npy")
V = degrade(V1.reshape(-1, 15, n).mean(axis=1))       # AMI voltage (degraded)
P = np.load(f"{OUT}/P15.npy")                # AMI kW (meters report this too)
T = V.shape[0]
nx = true_g.max() + 1

# ---- corrupted records (same seed as before) ----
r = np.random.default_rng(7)
rec_g = true_g.copy()
bad = r.choice(n, size=int(0.10*n), replace=False)
for i in bad: rec_g[i] = r.choice(nearest[true_g[i]])
is_bad = np.zeros(n, bool); is_bad[bad] = True

# ---- residualize: remove feeder-wide common mode ----
Ptot = P.sum(1); Vmean = V.mean(1)
X = np.column_stack([np.ones(T), Ptot, Vmean])
B = np.linalg.lstsq(X, V, rcond=None)[0]
rV = V - X @ B
Bp = np.linalg.lstsq(X, P, rcond=None)[0]
rP = P - X @ Bp

def zs(a): return (a - a.mean(0)) / (a.std(0) + 1e-12)
rVz, rPz = zs(rV), zs(rP)

# group aggregate residual load per RECORDED group
agg = np.zeros((T, nx), dtype=np.float64)
for g in range(nx):
    mem = np.where(rec_g == g)[0]
    if len(mem): agg[:, g] = rP[:, mem].sum(1)

C = corr_matrix(V)  # meter-meter spearman

def loadsig_score(i, g):
    mem = np.where(rec_g == g)[0]; mem = mem[mem != i]
    if not len(mem): return np.nan
    a = agg[:, g] - (rP[:, i] if rec_g[i] == g else 0)
    a = a - a.mean(); sd = a.std()
    if sd < 1e-9: return np.nan
    return -float((rVz[:, i] * (a/sd)).mean())   # + when load spike -> voltage dip

# candidate scoring: corr score + physics score
res_rows = []
for lam in (0.0, 0.5, 1.0, 2.0):
    corrected = np.zeros(n, int) - 1
    for i in bad:
        cands = (set(nearest[rec_g[i]].tolist()) | {rec_g[i]}
                 | set(nearest[true_g[i]].tolist()) | {true_g[i]})
        s1, s2 = {}, {}
        for g in cands:
            mem = np.where(rec_g == g)[0]; mem = mem[mem != i]
            s1[g] = C[i, mem].mean() if len(mem) else np.nan
            s2[g] = loadsig_score(i, g)
        # z-normalize within candidate set, combine
        def zdict(d):
            v = np.array([x for x in d.values() if not np.isnan(x)])
            if len(v) < 2 or v.std() < 1e-12: return {k: 0.0 for k in d}
            return {k: ((x - v.mean())/v.std() if not np.isnan(x) else -3.0) for k, x in d.items()}
        z1, z2 = zdict(s1), zdict(s2)
        tot = {g: z1[g] + lam*z2[g] for g in cands}
        corrected[i] = max(tot, key=tot.get)
    fixable = np.array([np.sum((rec_g == true_g[i]) & (np.arange(n) != i) &
                               (true_g == true_g[i])) > 0 for i in bad])
    acc = (corrected[bad] == true_g[bad]).mean()
    accf = (corrected[bad][fixable] == true_g[bad][fixable]).mean()
    print(f"lambda={lam:3.1f}  correction={acc:.3f}  correction_fixable={accf:.3f}", flush=True)
    res_rows.append(dict(lam=lam, corr=acc, corr_fix=accf))
pd.DataFrame(res_rows).to_csv(f"{OUT}/loadsig_results.csv", index=False)

# how well does physics ALONE separate true vs wrong groups?
true_s, wrong_s = [], []
rng2 = np.random.default_rng(3)
for i in rng2.choice(np.where(~is_bad)[0], 250, replace=False):
    s = loadsig_score(i, true_g[i])
    if not np.isnan(s): true_s.append(s)
    g_w = rng2.choice(nearest[true_g[i]][:5])
    s = loadsig_score(i, g_w)
    if not np.isnan(s): wrong_s.append(s)
print(f"\nphysics score alone: true-group mean {np.mean(true_s):.4f}  "
      f"wrong-nearby-group mean {np.mean(wrong_s):.4f}")
