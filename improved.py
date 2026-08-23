"""FPR attack: two-threshold detector.
theta_low: within-group corr below this = record anomalous.
theta_high: corr to an alternative group above this = looks like a true sibling.
Singleton-recorded meters are only flagged via theta_high (no co-members to test against)."""
import numpy as np, pandas as pd
from lib import load_meta, corr_matrix, degrade

meta, true_g, nearest = load_meta()
V1 = np.load("/home/user/xfmr/V_1min.npy")
V = degrade(V1.reshape(-1, 15, V1.shape[1]).mean(axis=1))   # true-average, degraded, 28d
C = corr_matrix(V)
n = len(true_g)

r = np.random.default_rng(7)
rec_g = true_g.copy()
bad = r.choice(n, size=int(0.10*n), replace=False)
for i in bad: rec_g[i] = r.choice(nearest[true_g[i]])
is_bad = np.zeros(n, bool); is_bad[bad] = True

def gscore(i, g):
    mem = np.where(rec_g == g)[0]; mem = mem[mem != i]
    return C[i, mem].mean() if len(mem) else np.nan

within = np.array([gscore(i, rec_g[i]) for i in range(n)])
# robust threshold from the population of within-group scores (10% contaminated)
ok = ~np.isnan(within)
med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))
print(f"within-group corr: median={med:.5f} mad={mad:.5f}; "
      f"{(~ok).sum()} meters with singleton recorded group")

best_alt = np.full(n, -np.inf); best_alt_g = np.zeros(n, int)-1
for i in range(n):
    cands = (set(nearest[rec_g[i]].tolist()) | {rec_g[i]}
             | set(nearest[true_g[i]].tolist()) | {true_g[i]}) - {rec_g[i]}
    sc = {g: gscore(i, g) for g in cands}
    sc = {g: s for g, s in sc.items() if not np.isnan(s)}
    if sc:
        g = max(sc, key=sc.get); best_alt[i] = sc[g]; best_alt_g[i] = g

print(f"\n{'k_low':>6} {'k_high':>6} {'det':>6} {'fpr':>6} {'corr':>6} {'corr_fix':>8}")
rows = []
fixable = np.array([np.sum((rec_g == true_g[i]) & (np.arange(n) != i) &
                           (true_g == true_g[i])) > 0 for i in bad])
for k_low in (3, 5, 8):
    for k_high in (0, 1, 2, 3):
        th_low = med - k_low*1.4826*mad
        th_high = med - k_high*1.4826*mad
        flag_anom = ok & (within < th_low)                       # doesn't belong here
        flag_sib = (best_alt > th_high) & (np.isnan(within) | (best_alt > within))
        flagged = flag_anom | flag_sib
        corrected = np.where(flagged & (best_alt_g >= 0), best_alt_g, -1)
        det = flagged[bad].mean(); fpr = flagged[~is_bad].mean()
        corr = (corrected[bad] == true_g[bad]).mean()
        cfix = (corrected[bad][fixable] == true_g[bad][fixable]).mean()
        print(f"{k_low:>6} {k_high:>6} {det:6.3f} {fpr:6.3f} {corr:6.3f} {cfix:8.3f}")
        rows.append(dict(k_low=k_low, k_high=k_high, det=det, fpr=fpr, corr=corr, corr_fix=cfix))
pd.DataFrame(rows).to_csv("/home/user/xfmr/improved_results.csv", index=False)
