"""Premise-GPS prior: utilities know each meter's service address. Model as true-transformer
location + jitter; use it to (a) rescue singleton assignment, (b) re-rank all corrections."""
import re
import numpy as np, pandas as pd
from lib import load_meta, corr_matrix, degrade, CKT

meta, true_g, nearest = load_meta()
n = len(true_g); nx = true_g.max()+1
V90 = np.load("/home/user/xfmr/V15_90d.npy")

# transformer coordinates
coords = {}
for ln in open(f"{CKT}/Buscoords_ckt5.dss"):
    p = ln.replace(",", " ").split()
    if len(p) >= 3:
        try: coords[p[0].lower()] = (float(p[1]), float(p[2]))
        except ValueError: pass
xtxt = open(f"{CKT}/XFR_Loads_ckt5.dss").read()
xfmrs = sorted(meta.xfmr.unique())
xf_xy = {}
for m in re.finditer(r"New Transformer\.(\S+).*?wdg=1 bus=(\S+)", xtxt):
    b = m.group(2).split(".")[0].lower()
    if b in coords: xf_xy[m.group(1)] = coords[b]
XY = np.array([xf_xy.get(x, (np.nan, np.nan)) for x in xfmrs])
spacing = np.nanmedian(np.sort(np.sqrt(((XY[:,None,:]-XY[None,:,:])**2).sum(-1)), axis=1)[:, 1])
print(f"median nearest-transformer spacing: {spacing:.0f} units")

rng = np.random.default_rng(5)
meter_xy = XY[true_g] + rng.normal(0, 0.35*spacing, (n, 2))   # premise GPS, service-drop jitter

r = np.random.default_rng(7)
rec_g = true_g.copy()
bad = r.choice(n, size=int(0.10*n), replace=False)
for i in bad: rec_g[i] = r.choice(nearest[true_g[i]])
is_bad = np.zeros(n, bool); is_bad[bad] = True
fixable = np.array([np.sum((rec_g == true_g[i]) & (np.arange(n) != i) &
                           (true_g == true_g[i])) > 0 for i in bad])

Vd = degrade(V90); C = corr_matrix(Vd)
# distance of every meter to every transformer
Dm = np.sqrt(((meter_xy[:, None, :] - XY[None, :, :])**2).sum(-1))
Dm = np.nan_to_num(Dm, nan=np.inf)
gps_rank = np.argsort(Dm, axis=1)

within = np.empty(n)
for i in range(n):
    mem = np.where(rec_g == rec_g[i])[0]; mem = mem[mem != i]
    within[i] = C[i, mem].mean() if len(mem) else np.nan
ok = ~np.isnan(within)
med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826

def run(tag, K_GPS, use_empty):
    flagged = np.zeros(n, bool); corrected = np.zeros(n, int)-1
    for i in range(n):
        cands = set(gps_rank[i, :K_GPS].tolist()) | {rec_g[i]}
        sc = {}
        for g in cands:
            mem = np.where(rec_g == g)[0]; mem = mem[mem != i]
            if len(mem): sc[g] = C[i, mem].mean()
        alt = {g: s for g, s in sc.items() if g != rec_g[i]}
        w = sc.get(rec_g[i], np.nan)
        gb = max(alt, key=alt.get) if alt else None
        sb = alt[gb] if alt else -np.inf
        if (not np.isnan(w) and w < med-3*mad and sb > w) or \
           (sb > med-1*mad and (np.isnan(w) or sb > w)):
            flagged[i] = True; corrected[i] = gb
        elif use_empty and (np.isnan(w) or w < med-3*mad) and \
             (sb < med-1*mad):
            # no sibling signature anywhere near premise -> nearest EMPTY transformer to premise
            for g in gps_rank[i]:
                if np.sum(rec_g == g) == 0:
                    if Dm[i, g] < 2.0*spacing:
                        flagged[i] = True; corrected[i] = g
                    break
    det = flagged[bad].mean(); fpr = flagged[~is_bad].mean()
    corr = (corrected[bad] == true_g[bad]).mean()
    cfix = (corrected[bad][fixable] == true_g[bad][fixable]).mean()
    sing = (corrected[bad][~fixable] == true_g[bad][~fixable]).mean()
    print(f"{tag:44s} det={det:.3f} fpr={fpr:.3f} corr={corr:.3f} "
          f"corr_fix={cfix:.3f} corr_singleton={sing:.3f}", flush=True)

print("baseline (network-nearest candidates, no GPS): corr_fix=0.916, corr_singleton=0.0")
for K in (5, 10, 20):
    run(f"GPS candidates K={K}", K, use_empty=False)
run("GPS K=10 + empty-transformer fallback", 10, use_empty=True)
run("GPS K=20 + empty-transformer fallback", 20, use_empty=True)
