"""Score ckt24 with the frozen recipe: degrade -> Spearman -> GPS-K5 two-threshold."""
import re
import numpy as np, pandas as pd
from lib import corr_matrix, degrade

CKT = "/home/user/electricdss-tst/Version8/Distrib/EPRITestCircuits/ckt24"
V = degrade(np.load("/home/user/xfmr/V24.npy"))
meta = pd.read_csv("/home/user/xfmr/meta24.csv")
n = len(meta)
xfmrs = sorted(meta.xf.unique())
gid = {x: i for i, x in enumerate(xfmrs)}
true_g = meta.xf.map(gid).values
sizes = np.bincount(true_g, minlength=len(xfmrs))
print(f"{n} meters, {len(xfmrs)} transformers; sizes 1:{(sizes==1).sum()} "
      f"2-3:{((sizes>=2)&(sizes<=3)).sum()} 4+:{(sizes>=4).sum()}")

coords = {}
for ln in open(f"{CKT}/buscoords_ckt24.dss"):
    p = ln.replace(",", " ").split()
    if len(p) >= 3:
        try: coords[p[0].lower()] = (float(p[1]), float(p[2]))
        except ValueError: pass
xtxt = open(f"{CKT}/transformers_ckt24.dss").read()
xf_xy = {}
for m in re.finditer(r"New\s+Transformer\.(\S+).*?wdg=1\s+bus=(\S+)", xtxt):
    b = m.group(2).split(".")[0].lower()
    if b in coords: xf_xy[m.group(1)] = coords[b]
XY = np.array([xf_xy.get(x, (np.nan, np.nan)) for x in xfmrs])
Dx = np.sqrt(((XY[:, None, :]-XY[None, :, :])**2).sum(-1))
np.fill_diagonal(Dx, np.inf); Dx = np.nan_to_num(Dx, nan=np.inf)
nearest = np.argsort(Dx, axis=1)[:, :15]
spacing = np.median(np.sort(Dx, axis=1)[:, 0][np.isfinite(np.sort(Dx, axis=1)[:, 0])])
print(f"median spacing {spacing:.0f}")

rng2 = np.random.default_rng(5)
meter_xy = XY[true_g] + rng2.normal(0, 0.35*spacing, (n, 2))
Dm = np.nan_to_num(np.sqrt(((meter_xy[:, None, :]-XY[None, :, :])**2).sum(-1)), nan=np.inf)
gps_rank = np.argsort(Dm, axis=1)

C = corr_matrix(V)
r = np.random.default_rng(7)
rec_g = true_g.copy()
bad = r.choice(n, size=int(0.10*n), replace=False)
for i in bad: rec_g[i] = r.choice(nearest[true_g[i]])
is_bad = np.zeros(n, bool); is_bad[bad] = True
fixable = np.array([np.sum((rec_g == true_g[i]) & (np.arange(n) != i) &
                           (true_g == true_g[i])) > 0 for i in bad])

within = np.empty(n)
for i in range(n):
    mem = np.where(rec_g == rec_g[i])[0]; mem = mem[mem != i]
    within[i] = C[i, mem].mean() if len(mem) else np.nan
ok = ~np.isnan(within)
med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826
print(f"within median {med:.4f} mad {mad:.5f}; singleton-recorded: {(~ok).sum()}")

props = []
flagged = np.zeros(n, bool); corrected = np.zeros(n, int)-1
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
    if (not np.isnan(w) and w < med-3*mad and sb > w) or \
       (sb > med-1*mad and (np.isnan(w) or sb > w)):
        flagged[i] = True; corrected[i] = gb
        props.append((sb - (w if not np.isnan(w) else med-3*mad), i, gb))

det = flagged[bad].mean(); fpr = flagged[~is_bad].mean()
corr = (corrected[bad] == true_g[bad]).mean()
cfix = (corrected[bad][fixable] == true_g[bad][fixable]).mean()
print(f"\nckt24: det={det:.3f} fpr={fpr:.3f} corr={corr:.3f} corr_fixable={cfix:.3f}")

props.sort(reverse=True)
lab = rec_g.copy(); accs = [np.mean(lab == true_g)]
best = (accs[0], 0)
for k, (cf, i, gb) in enumerate(props, 1):
    lab[i] = gb
    a = np.mean(lab == true_g); accs.append(a)
    if a > best[0]: best = (a, k)
print(f"gate: base {accs[0]:.3f} -> best {best[0]:.3f} at {best[1]} applied "
      f"-> all-applied {accs[-1]:.3f}")
