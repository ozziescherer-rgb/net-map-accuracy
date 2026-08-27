"""(a) multi-seed 5% and 20% corruption sweeps; (b) observable-gate stopping variance
across verification draws. Standalone (no audit.py import side effects)."""
import re
import numpy as np, pandas as pd
from lib import corr_matrix, degrade, CKT, OUT

meta = pd.read_csv(f"{OUT}/meta.csv")
xfmrs = sorted(meta.xfmr.unique())
gid = {x: i for i, x in enumerate(xfmrs)}
true_g = meta.xfmr.map(gid).values
n = len(true_g)
coords = {}
for ln in open(f"{CKT}/Buscoords_ckt5.dss"):
    p = ln.replace(",", " ").split()
    if len(p) >= 3:
        try: coords[p[0].lower()] = (float(p[1]), float(p[2]))
        except ValueError: pass
xy = {}
for m in re.finditer(r"New Transformer\.(\S+).*?wdg=1 bus=(\S+)", open(f"{CKT}/XFR_Loads_ckt5.dss").read()):
    b = m.group(2).split(".")[0].lower()
    if b in coords: xy[m.group(1)] = coords[b]
XY = np.array([xy.get(x, (np.nan, np.nan)) for x in xfmrs])
D = np.sqrt(((XY[:, None, :]-XY[None, :, :])**2).sum(-1))
np.fill_diagonal(D, np.inf); D = np.nan_to_num(D, nan=np.inf)
nearest = np.argsort(D, axis=1)[:, :15]
sp = np.median(np.sort(D, axis=1)[:, 0][np.isfinite(np.sort(D, axis=1)[:, 0])])
C = corr_matrix(degrade(np.load(f"{OUT}/V15_90d.npy")))

def run(frac, cseed, jseed, vseed):
    rj = np.random.default_rng(jseed)
    mxy = XY[true_g] + rj.normal(0, 0.35*sp, (n, 2))
    Dm = np.nan_to_num(np.sqrt(((mxy[:, None, :]-XY[None, :, :])**2).sum(-1)), nan=np.inf)
    gps = np.argsort(Dm, axis=1)[:, :5]
    rc = np.random.default_rng(cseed)
    rec = true_g.copy()
    bad = rc.choice(n, size=int(frac*n), replace=False)
    for i in bad: rec[i] = rc.choice(nearest[true_g[i]])
    within = np.empty(n)
    for i in range(n):
        mem = np.where(rec == rec[i])[0]; mem = mem[mem != i]
        within[i] = C[i, mem].mean() if len(mem) else np.nan
    ok = ~np.isnan(within)
    med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826
    props = []
    for i in range(n):
        cands = set(gps[i].tolist()) | {rec[i]}
        sc = {}
        for g in cands:
            mem = np.where(rec == g)[0]; mem = mem[mem != i]
            if len(mem): sc[g] = C[i, mem].mean()
        alt = {g: s for g, s in sc.items() if g != rec[i]}
        if not alt: continue
        gb = max(alt, key=alt.get); sb = alt[gb]
        w = sc.get(rec[i], np.nan)
        if (not np.isnan(w) and w < med-3*mad and sb > w) or \
           (sb > med-1*mad and (np.isnan(w) or sb > w)):
            props.append((sb-(w if not np.isnan(w) else med-3*mad), i, gb))
    props.sort(reverse=True)
    lab = rec.copy(); base = np.mean(lab == true_g)
    accs = []
    for (cf, i, gb) in props:
        lab[i] = gb; accs.append(np.mean(lab == true_g))
    oracle = max(accs) if accs else base
    blind = accs[-1] if accs else base
    rv = np.random.default_rng(vseed)
    ver = set(rv.choice(len(props), size=min(40, len(props)), replace=False).tolist())
    cfl = [(true_g[i] != rec[i]) and (gb == true_g[i]) for (cf, i, gb) in props]
    stop = len(props); seen = []
    for k in range(len(props)):
        if k in ver: seen.append((k, cfl[k]))
        recent = [c for (kk, c) in seen if kk > k-60]
        if len(recent) >= 8 and np.mean(recent) < 0.5: stop = k; break
    obs = accs[stop-1] if stop > 0 and accs else base
    return dict(base=base, oracle=oracle, blind=blind, obs=obs, stop=stop, nprops=len(props))

print("== (a) multi-seed corruption sweep ==")
for frac in (0.05, 0.20):
    rows = [run(frac, cs, js, cs+1000) for cs, js in
            [(7, 5), (17, 15), (27, 25), (37, 35), (47, 45)]]
    df = pd.DataFrame(rows); m, s = df.mean(), df.std()
    print(f"{frac:.0%}: base {m.base:.3f}  oracle {m.oracle:.3f}±{s.oracle:.3f}  "
          f"obs {m.obs:.3f}±{s.obs:.3f}  blind {m.blind:.3f}±{s.blind:.3f}")
    df.to_csv(f"{OUT}/sweep_{int(frac*100)}pct.csv", index=False)

print("\n== (b) gate variance across 20 verification draws (10%, cseed=7) ==")
rows = [run(0.10, 7, 5, v) for v in range(2000, 2020)]
df = pd.DataFrame(rows)
print(f"obs acc: {df.obs.mean():.3f} ± {df.obs.std():.3f}  "
      f"(range {df.obs.min():.3f}–{df.obs.max():.3f})")
print(f"stop point: {df.stop.mean():.0f} ± {df.stop.std():.0f} of {df.nprops.iloc[0]} proposals")
print(f"oracle {df.oracle.iloc[0]:.3f}, blind {df.blind.iloc[0]:.3f}, base {df.base.iloc[0]:.3f}")
df.to_csv(f"{OUT}/gate_variance.csv", index=False)
