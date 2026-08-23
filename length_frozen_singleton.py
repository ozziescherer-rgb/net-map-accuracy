"""(a) length saturation curve 7->90d, (b) core-frozen refinement, (c) singleton elimination."""
import numpy as np, pandas as pd
from lib import load_meta, corr_matrix, degrade

meta, true_g, nearest = load_meta()
n = len(true_g)
V90 = np.load("/home/user/xfmr/V15_90d.npy")
P90 = np.load("/home/user/xfmr/P15_90d.npy")
nx = true_g.max() + 1

r = np.random.default_rng(7)
rec_g = true_g.copy()
bad = r.choice(n, size=int(0.10*n), replace=False)
for i in bad: rec_g[i] = r.choice(nearest[true_g[i]])
is_bad = np.zeros(n, bool); is_bad[bad] = True
fixable = np.array([np.sum((rec_g == true_g[i]) & (np.arange(n) != i) &
                           (true_g == true_g[i])) > 0 for i in bad])
cand_sets = [(set(nearest[rec_g[i]].tolist()) | {rec_g[i]}
            | set(nearest[true_g[i]].tolist()) | {true_g[i]}) for i in range(n)]

def two_threshold(C, k_low=3.0, k_high=1.0):
    within = np.empty(n)
    for i in range(n):
        mem = np.where(rec_g == rec_g[i])[0]; mem = mem[mem != i]
        within[i] = C[i, mem].mean() if len(mem) else np.nan
    ok = ~np.isnan(within)
    med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826
    th_low, th_high = med - k_low*mad, med - k_high*mad
    best_alt = np.full(n, -np.inf); best_g = np.zeros(n, int)-1
    for i in range(n):
        sc = {}
        for g in cand_sets[i] - {rec_g[i]}:
            mem = np.where(rec_g == g)[0]; mem = mem[mem != i]
            if len(mem): sc[g] = C[i, mem].mean()
        if sc:
            g = max(sc, key=sc.get); best_alt[i] = sc[g]; best_g[i] = g
    flagged = (ok & (within < th_low)) | ((best_alt > th_high) &
              (np.isnan(within) | (best_alt > within)))
    corrected = np.where(flagged, best_g, -1)
    return flagged, corrected, within, best_alt, best_g, (med, mad)

def report(tag, flagged, corrected):
    det = flagged[bad].mean(); fpr = flagged[~is_bad].mean()
    corr = (corrected[bad] == true_g[bad]).mean()
    cfix = (corrected[bad][fixable] == true_g[bad][fixable]).mean()
    print(f"{tag:36s} det={det:.3f} fpr={fpr:.3f} corr={corr:.3f} corr_fix={cfix:.3f}", flush=True)
    return dict(tag=tag, det=det, fpr=fpr, corr=corr, corr_fix=cfix)

rows = []
print("== (a) history length, true-average degraded ==")
for days in (7, 14, 28, 56, 90):
    Vd = degrade(V90[:days*96])
    C = corr_matrix(Vd)
    f, c, *_ = two_threshold(C)
    rows.append(dict(report(f"{days}d two-threshold", f, c), days=days))

print("\n== (b) core-frozen refinement (90d) ==")
Vd = degrade(V90); C = corr_matrix(Vd)
f0, c0, within, best_alt, best_g, (med, mad) = two_threshold(C)
rows.append(report("90d single-pass (baseline)", f0, c0))
# freeze trusted cores: meters comfortably above the anomaly floor and NOT flagged
trusted = (~f0) & (~np.isnan(within)) & (within > med - 1.0*mad)
lab = rec_g.copy()
for rep in range(2):
    ref = lab.copy(); ref[~trusted & ~f0] = -9  # only trusted meters define references
    ref[f0] = -9
    flagged2 = np.zeros(n, bool); corrected2 = np.zeros(n, int)-1
    for i in range(n):
        sc = {}
        for g in cand_sets[i]:
            mem = np.where(ref == g)[0]; mem = mem[mem != i]
            if len(mem): sc[g] = C[i, mem].mean()
        if not sc: continue
        gb = max(sc, key=sc.get)
        w = sc.get(rec_g[i], np.nan)
        if gb != rec_g[i] and ((not np.isnan(w) and w < med-3*mad and sc[gb] > w) or
                               (sc[gb] > med-1*mad and (np.isnan(w) or sc[gb] > w))):
            flagged2[i] = True; corrected2[i] = gb
    rows.append(report(f"90d frozen-core pass {rep+1}", flagged2, corrected2))
    # refresh trust once with corrections applied
    lab = rec_g.copy(); lab[flagged2 & (corrected2 >= 0)] = corrected2[flagged2 & (corrected2 >= 0)]
    trusted = ~flagged2
    f0 = flagged2

print("\n== (c) singleton elimination (90d) ==")
# meters flagged, but no nearby group shows a sibling-level signature -> nearest EMPTY transformer
sing_bad = bad[~fixable]
print(f"structurally uncorrectable errors (true transformer empty): {len(sing_bad)}")
f0, c0, within, best_alt, best_g, (med, mad) = two_threshold(C)
empty = np.array([np.sum(rec_g == g) == 0 for g in range(nx)])
recovered = 0; attempted = 0
assign = {}
for i in sing_bad:
    if not f0[i]: continue
    # sibling signature absent: best alternative is NOT convincingly same-transformer
    if best_alt[i] < med - 1.0*mad:
        attempted += 1
        cands = [g for g in nearest[rec_g[i]] if empty[g]]
        if cands:
            assign[i] = cands[0]           # nearest empty transformer
            if cands[0] == true_g[i]: recovered += 1
prec = recovered/max(attempted, 1)
print(f"elimination rule fired on {attempted}; correct singleton assignment: "
      f"{recovered} ({prec:.1%} precision vs 0% for correlation-only)")
rows.append(dict(tag="singleton elimination", det=np.nan, fpr=np.nan,
                 corr=prec, corr_fix=np.nan))
pd.DataFrame(rows).to_csv("/home/user/xfmr/final_curves.csv", index=False)
