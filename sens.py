"""Corruption-rate sensitivity on ckt5 (90d): does the story hold at 5% / 10% / 20%?"""
import re
import numpy as np
from lib import load_meta, corr_matrix, degrade, CKT, OUT

meta, true_g, nearest = load_meta()
n = len(true_g)
C = corr_matrix(degrade(np.load(f"{OUT}/V15_90d.npy")))
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
rng2 = np.random.default_rng(5)
meter_xy = XY[true_g] + rng2.normal(0, 0.35*178.0, (n, 2))
Dm = np.nan_to_num(np.sqrt(((meter_xy[:, None, :]-XY[None, :, :])**2).sum(-1)), nan=np.inf)
gps_rank = np.argsort(Dm, axis=1)

print(f"{'rate':>5} {'det':>6} {'fpr':>6} {'corr':>6} {'base':>6} {'gated':>6} {'@N':>4} {'blind':>6}")
for frac in (0.05, 0.10, 0.20):
    for seed in (7,):
        r = np.random.default_rng(seed)
        rec_g = true_g.copy()
        bad = r.choice(n, size=int(frac*n), replace=False)
        for i in bad: rec_g[i] = r.choice(nearest[true_g[i]])
        is_bad = np.zeros(n, bool); is_bad[bad] = True
        within = np.empty(n)
        for i in range(n):
            mem = np.where(rec_g == rec_g[i])[0]; mem = mem[mem != i]
            within[i] = C[i, mem].mean() if len(mem) else np.nan
        ok = ~np.isnan(within)
        med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826
        props = []; flagged = np.zeros(n, bool); corrected = np.zeros(n, int)-1
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
                props.append((sb-(w if not np.isnan(w) else med-3*mad), i, gb))
        det = flagged[bad].mean(); fpr = flagged[~is_bad].mean()
        corr = (corrected[bad] == true_g[bad]).mean()
        props.sort(reverse=True)
        lab = rec_g.copy(); base = np.mean(lab == true_g); best = (base, 0); last = base
        for k, (cf, i, gb) in enumerate(props, 1):
            lab[i] = gb; a = np.mean(lab == true_g)
            if a > best[0]: best = (a, k)
            last = a
        print(f"{frac:5.0%} {det:6.3f} {fpr:6.3f} {corr:6.3f} {base:6.3f} "
              f"{best[0]:6.3f} {best[1]:4d} {last:6.3f}")
