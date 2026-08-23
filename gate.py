"""Confidence-gated correction: apply fixes in descending confidence, track NET map accuracy.
The question: how many corrections can you auto-apply before FPs eat the gains?"""
import numpy as np, pandas as pd
from lib import load_meta, corr_matrix, degrade

meta, true_g, nearest = load_meta()
n = len(true_g)
V90 = np.load("/home/user/xfmr/V15_90d.npy")
import re
from lib import CKT
coords = {}
for ln in open(f"{CKT}/Buscoords_ckt5.dss"):
    p = ln.replace(",", " ").split()
    if len(p) >= 3:
        try: coords[p[0].lower()] = (float(p[1]), float(p[2]))
        except ValueError: pass
xtxt = open(f"{CKT}/XFR_Loads_ckt5.dss").read()
xfmrs = sorted(meta.xfmr.unique())
xy = {}
for m in re.finditer(r"New Transformer\.(\S+).*?wdg=1 bus=(\S+)", xtxt):
    b = m.group(2).split(".")[0].lower()
    if b in coords: xy[m.group(1)] = coords[b]
XY = np.array([xy.get(x, (np.nan, np.nan)) for x in xfmrs])

r = np.random.default_rng(7)
rec_g = true_g.copy()
bad = r.choice(n, size=int(0.10*n), replace=False)
for i in bad: rec_g[i] = r.choice(nearest[true_g[i]])

rng2 = np.random.default_rng(5)
meter_xy = XY[true_g] + rng2.normal(0, 0.35*178.0, (n, 2))
Dm = np.nan_to_num(np.sqrt(((meter_xy[:, None, :]-XY[None, :, :])**2).sum(-1)), nan=np.inf)
gps_rank = np.argsort(Dm, axis=1)
C = corr_matrix(degrade(V90))

within = np.empty(n)
for i in range(n):
    mem = np.where(rec_g == rec_g[i])[0]; mem = mem[mem != i]
    within[i] = C[i, mem].mean() if len(mem) else np.nan
ok = ~np.isnan(within)
med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826

# candidate corrections with confidence = best-alt minus within (or floor)
props = []
for i in range(n):
    cands = set(gps_rank[i, :5].tolist()) | {rec_g[i]}
    sc = {}
    for g in cands:
        mem = np.where(rec_g == g)[0]; mem = mem[mem != i]
        if len(mem): sc[g] = C[i, mem].mean()
    alt = {g: s for g, s in sc.items() if g != rec_g[i]}
    if not alt: continue
    gb = max(alt, key=alt.get); sb = alt[gb]
    w = sc.get(rec_g[i], np.nan)
    flag = (not np.isnan(w) and w < med-3*mad and sb > w) or \
           (sb > med-1*mad and (np.isnan(w) or sb > w))
    if flag:
        conf = sb - (w if not np.isnan(w) else (med - 3*mad))
        props.append((conf, i, gb))
props.sort(reverse=True)
print(f"{len(props)} proposed corrections; base map accuracy {np.mean(rec_g==true_g):.3f}")

lab = rec_g.copy()
best = (np.mean(rec_g == true_g), 0)
print(f"{'applied':>8} {'net acc':>8} {'TP':>5} {'FP':>5} {'precision':>9}")
tp = fp = 0
rows = []
for k, (conf, i, gb) in enumerate(props, 1):
    was_right = rec_g[i] == true_g[i]
    lab[i] = gb
    if not was_right and gb == true_g[i]: tp += 1
    elif was_right: fp += 1
    acc = np.mean(lab == true_g)
    rows.append(dict(applied=k, conf=conf, acc=acc, tp=tp, fp=fp))
    if acc > best[0]: best = (acc, k)
    if k in (25, 50, 75, 100, 125, 150, 175, 200) or k == len(props):
        print(f"{k:>8} {acc:>8.3f} {tp:>5} {fp:>5} {tp/(tp+fp+1e-9):>9.2f}")
print(f"\nbest: apply top-{best[1]} corrections -> map accuracy {best[0]:.3f} "
      f"(from 0.901; ceiling with perfect corrections would be ~0.99)")
pd.DataFrame(rows).to_csv("/home/user/xfmr/gate_curve.csv", index=False)
