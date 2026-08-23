"""Hardened rerun: leak-free candidates (GPS+recorded only), multi-seed, containment check,
observable gate via sampled-verification calibration, gated thermal capture."""
import re
import numpy as np, pandas as pd
from lib import corr_matrix, degrade, CKT

def load_feeder(which):
    if which == "ckt5":
        meta = pd.read_csv("/home/user/xfmr/meta.csv"); key = "xfmr"
        V = np.load("/home/user/xfmr/V15_90d.npy")
        ck, bc, xt = CKT, "Buscoords_ckt5.dss", "XFR_Loads_ckt5.dss"
        pat = r"New Transformer\.(\S+).*?wdg=1 bus=(\S+)"
    else:
        meta = pd.read_csv("/home/user/xfmr/meta24.csv"); key = "xf"
        V = np.load("/home/user/xfmr/V24.npy")
        ck = "/home/user/electricdss-tst/Version8/Distrib/EPRITestCircuits/ckt24"
        bc, xt = "buscoords_ckt24.dss", "transformers_ckt24.dss"
        pat = r"New\s+Transformer\.(\S+).*?wdg=1\s+bus=(\S+)"
    xfmrs = sorted(meta[key].unique())
    gid = {x: i for i, x in enumerate(xfmrs)}
    true_g = meta[key].map(gid).values
    coords = {}
    for ln in open(f"{ck}/{bc}"):
        p = ln.replace(",", " ").split()
        if len(p) >= 3:
            try: coords[p[0].lower()] = (float(p[1]), float(p[2]))
            except ValueError: pass
    xy = {}
    for m in re.finditer(pat, open(f"{ck}/{xt}").read()):
        b = m.group(2).split(".")[0].lower()
        if b in coords: xy[m.group(1)] = coords[b]
    XY = np.array([xy.get(x, (np.nan, np.nan)) for x in xfmrs])
    D = np.sqrt(((XY[:, None, :]-XY[None, :, :])**2).sum(-1))
    np.fill_diagonal(D, np.inf); D = np.nan_to_num(D, nan=np.inf)
    nearest = np.argsort(D, axis=1)[:, :15]
    sp = np.median(np.sort(D, axis=1)[:, 0][np.isfinite(np.sort(D, axis=1)[:, 0])])
    return V, true_g, XY, nearest, sp

def pipeline(C, true_g, XY, nearest, spacing, frac, cseed, jseed, K=5, verify_n=40):
    n = len(true_g)
    rj = np.random.default_rng(jseed)
    mxy = XY[true_g] + rj.normal(0, 0.35*spacing, (n, 2))
    Dm = np.nan_to_num(np.sqrt(((mxy[:, None, :]-XY[None, :, :])**2).sum(-1)), nan=np.inf)
    gps = np.argsort(Dm, axis=1)[:, :K]
    contain = np.mean([true_g[i] in gps[i] for i in range(n)])
    rc = np.random.default_rng(cseed)
    rec = true_g.copy()
    bad = rc.choice(n, size=int(frac*n), replace=False)
    for i in bad: rec[i] = rc.choice(nearest[true_g[i]])
    is_bad = np.zeros(n, bool); is_bad[bad] = True
    within = np.empty(n)
    for i in range(n):
        mem = np.where(rec == rec[i])[0]; mem = mem[mem != i]
        within[i] = C[i, mem].mean() if len(mem) else np.nan
    ok = ~np.isnan(within)
    med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826
    props = []; flagged = np.zeros(n, bool); corrected = np.zeros(n, int)-1
    for i in range(n):
        cands = set(gps[i].tolist()) | {rec[i]}          # LEAK-FREE: gps + recorded only
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
            flagged[i] = True; corrected[i] = gb
            props.append((sb-(w if not np.isnan(w) else med-3*mad), i, gb))
    fixable = np.array([np.sum((rec == true_g[i]) & (np.arange(n) != i) &
                               (true_g == true_g[i])) > 0 for i in bad])
    det = flagged[bad].mean(); fpr = flagged[~is_bad].mean()
    cfix = (corrected[bad][fixable] == true_g[bad][fixable]).mean() if fixable.any() else np.nan
    props.sort(reverse=True)
    lab = rec.copy(); base = np.mean(lab == true_g)
    accs = []; best = (base, 0)
    for k, (cf, i, gb) in enumerate(props, 1):
        lab[i] = gb; a = np.mean(lab == true_g); accs.append(a)
        if a > best[0]: best = (a, k)
    blind = accs[-1] if accs else base
    # OBSERVABLE gate: field-verify a random sample of proposals top-down;
    # estimate precision in sliding blocks; stop where est. precision < 0.5
    rv = np.random.default_rng(cseed+1000)
    ver_idx = set(rv.choice(len(props), size=min(verify_n, len(props)), replace=False).tolist())
    correct_flags = [(true_g[i] != rec[i]) and (gb == true_g[i]) for (cf, i, gb) in props]
    stop = len(props)
    seen = []
    for k in range(len(props)):
        if k in ver_idx: seen.append((k, correct_flags[k]))
        recent = [c for (kk, c) in seen if kk > k-60]
        if len(recent) >= 8 and np.mean(recent) < 0.5:
            stop = k; break
    obs_acc = accs[stop-1] if stop > 0 and accs else base
    return dict(contain=contain, det=det, fpr=fpr, corr_fix=cfix, base=base,
                oracle_gate=best[0], oracle_at=best[1], obs_gate=obs_acc,
                obs_at=stop, blind=blind, nprops=len(props))

results = {}
for feeder in ("ckt5", "ckt24"):
    V, true_g, XY, nearest, sp = load_feeder(feeder)
    C = corr_matrix(degrade(V))
    rows = []
    seeds = [(7, 5), (17, 15), (27, 25), (37, 35), (47, 45)] if feeder == "ckt5" else [(7, 5), (17, 15), (27, 25)]
    for cs, js in seeds:
        rows.append(pipeline(C, true_g, XY, nearest, sp, 0.10, cs, js))
    df = pd.DataFrame(rows)
    results[feeder] = df
    m, s = df.mean(), df.std()
    print(f"\n== {feeder} (10% corruption, {len(rows)} seeds, LEAK-FREE) ==")
    for k in ("contain", "det", "fpr", "corr_fix", "base", "oracle_gate", "obs_gate", "blind"):
        print(f"  {k:12s} {m[k]:.3f} ± {s[k]:.3f}")
    df.to_csv(f"/home/user/xfmr/audit_{feeder}.csv", index=False)

# leak-free length curve, ckt5, seed (7,5)
print("\n== leak-free length curve (ckt5) ==")
V, true_g, XY, nearest, sp = load_feeder("ckt5")
for days in (7, 14, 28, 56, 90):
    C = corr_matrix(degrade(V[:days*96]))
    r = pipeline(C, true_g, XY, nearest, sp, 0.10, 7, 5)
    print(f"  {days:3d}d det={r['det']:.3f} fpr={r['fpr']:.3f} corr_fix={r['corr_fix']:.3f} "
          f"oracle_gate={r['oracle_gate']:.3f} obs_gate={r['obs_gate']:.3f}")
