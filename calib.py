"""Open problem #2: calibrated confidence.
Train a proposal-scoring model on ckt5 simulation (labels free), freeze it,
evaluate on ckt24 (different feeder, never seen). Questions:
  (a) does better ordering raise the oracle peak?
  (b) do calibrated probabilities give a ZERO-truck-roll gate (apply while P>0.5)?
"""
import re
import numpy as np, pandas as pd
from lib import corr_matrix, degrade, CKT, OUT, CKT24
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

def build(feeder):
    if feeder == "ckt5":
        meta = pd.read_csv(f"{OUT}/meta.csv"); key = "xfmr"
        V = np.load(f"{OUT}/V15_90d.npy")
        ck, bc, xt = CKT, "Buscoords_ckt5.dss", "XFR_Loads_ckt5.dss"
        pat = r"New Transformer\.(\S+).*?wdg=1 bus=(\S+)"
    else:
        meta = pd.read_csv(f"{OUT}/meta24.csv"); key = "xf"
        V = np.load(f"{OUT}/V24.npy")
        ck = CKT24
        bc, xt = "buscoords_ckt24.dss", "transformers_ckt24.dss"
        pat = r"New\s+Transformer\.(\S+).*?wdg=1\s+bus=(\S+)"
    xfmrs = sorted(meta[key].unique()); gid = {x: i for i, x in enumerate(xfmrs)}
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
    fin = np.sort(D, axis=1)[:, 0]
    sp = np.median(fin[np.isfinite(fin)])
    C = corr_matrix(degrade(V))
    return dict(C=C, true_g=true_g, nearest=nearest, XY=XY, sp=sp, n=len(true_g))

def proposals(ctx, cseed, jseed):
    C, true_g, nearest, XY, sp, n = (ctx[k] for k in ("C","true_g","nearest","XY","sp","n"))
    rj = np.random.default_rng(jseed)
    mxy = XY[true_g] + rj.normal(0, 0.35*sp, (n, 2))
    Dm = np.nan_to_num(np.sqrt(((mxy[:, None, :]-XY[None, :, :])**2).sum(-1)), nan=np.inf)
    gps = np.argsort(Dm, axis=1)[:, :5]
    rc = np.random.default_rng(cseed)
    rec = true_g.copy()
    bad = rc.choice(n, size=int(0.10*n), replace=False)
    for i in bad: rec[i] = rc.choice(nearest[true_g[i]])
    within = np.empty(n)
    for i in range(n):
        mem = np.where(rec == rec[i])[0]; mem = mem[mem != i]
        within[i] = C[i, mem].mean() if len(mem) else np.nan
    ok = ~np.isnan(within)
    med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826
    rows = []
    for i in range(n):
        cands = set(gps[i].tolist()) | {rec[i]}
        sc = {}
        for g in cands:
            mem = np.where(rec == g)[0]; mem = mem[mem != i]
            if len(mem): sc[g] = C[i, mem].mean()
        alt = {g: s for g, s in sc.items() if g != rec[i]}
        if not alt: continue
        srt = sorted(alt.items(), key=lambda kv: -kv[1])
        gb, sb = srt[0]
        second = srt[1][1] if len(srt) > 1 else sb - 0.05
        w = sc.get(rec[i], np.nan)
        flag = (not np.isnan(w) and w < med-3*mad and sb > w) or \
               (sb > med-1*mad and (np.isnan(w) or sb > w))
        if not flag: continue
        mem_b = np.where(rec == gb)[0]
        coh = np.nan
        if len(mem_b) >= 2:
            sub = C[np.ix_(mem_b, mem_b)]
            coh = (sub.sum()-len(mem_b))/(len(mem_b)*(len(mem_b)-1))
        w_eff = w if not np.isnan(w) else med-3*mad
        val = 1 if (rec[i] != true_g[i] and gb == true_g[i]) else (-1 if rec[i] == true_g[i] else 0)
        rows.append(dict(
            margin=(sb-w_eff)/mad, sb_z=(sb-med)/mad,
            w_z=((w-med)/mad if not np.isnan(w) else -6.0),
            gap2=(sb-second)/mad, n_grp=len(mem_b),
            coh_z=((coh-med)/mad if not np.isnan(coh) else 0.0),
            dist=Dm[i, gb]/sp, singleton=float(np.isnan(w)),
            gpsrank=float(np.where(gps[i] == gb)[0][0]) if gb in gps[i] else 5.0,
            val=val, i=i, gb=gb, rec_wrong=int(rec[i] != true_g[i])))
    return pd.DataFrame(rows), rec, true_g

FEATS = ["margin","sb_z","w_z","gap2","n_grp","coh_z","dist","singleton","gpsrank"]

def curve(df, order_col, rec, true_g):
    """apply proposals in descending order_col; return accuracy trajectory + peak."""
    d = df.sort_values(order_col, ascending=False)
    lab = rec.copy(); base = np.mean(lab == true_g)
    accs = []
    for r in d.itertuples():
        lab[r.i] = r.gb; accs.append(np.mean(lab == true_g))
    peak = max(accs) if accs else base
    return base, np.array(accs), peak, d

print("building feeders...", flush=True)
c5, c24 = build("ckt5"), build("ckt24")
train = pd.concat([proposals(c5, cs, js)[0] for cs, js in
                   [(7,5),(17,15),(27,25),(37,35),(47,45)]], ignore_index=True)
tr = train[train.val != 0]
X = StandardScaler().fit(tr[FEATS])
lr = LogisticRegression(max_iter=2000, C=1.0).fit(X.transform(tr[FEATS]), (tr.val > 0).astype(int))
iso = IsotonicRegression(out_of_bounds="clip").fit(
    lr.predict_proba(X.transform(tr[FEATS]))[:, 1], (tr.val > 0).astype(int))
print(f"trained on {len(tr)} labeled proposals ({(tr.val>0).mean():.2f} positive)")
print("feature weights:", dict(zip(FEATS, np.round(lr.coef_[0], 2))))

print(f"\n== held-out feeder ckt24, 3 seeds ==")
print(f"{'seed':>5} {'base':>6} | {'peak-margin':>11} {'peak-LR':>8} | {'P>0.5 gate':>10} {'P>0.6':>6} | {'blind':>6}")
res = []
for cs, js in [(7,5),(17,15),(27,25)]:
    df, rec, tg = proposals(c24, cs, js)
    p_lr = iso.predict(lr.predict_proba(X.transform(df[FEATS]))[:, 1])
    df = df.assign(p=p_lr)
    base, accs_m, peak_m, _ = curve(df, "margin", rec, tg)
    _, accs_l, peak_l, d_l = curve(df, "p", rec, tg)
    # zero-truck-roll gates: apply while calibrated P > threshold
    for th in (0.5, 0.6):
        k = int((d_l.p > th).sum())
        df.attrs[f"g{th}"] = accs_l[k-1] if k > 0 else base
    blind = accs_l[-1]
    res.append(dict(base=base, peak_m=peak_m, peak_l=peak_l,
                    g50=df.attrs["g0.5"], g60=df.attrs["g0.6"], blind=blind))
    print(f"{cs:>5} {base:6.3f} | {peak_m:11.3f} {peak_l:8.3f} | {df.attrs['g0.5']:10.3f} "
          f"{df.attrs['g0.6']:6.3f} | {blind:6.3f}")
r = pd.DataFrame(res)
print(f"\nmeans: base {r.base.mean():.3f} | oracle peak: margin {r.peak_m.mean():.3f} -> "
      f"LR {r.peak_l.mean():.3f} | zero-verification gate P>0.5: {r.g50.mean():.3f} "
      f"P>0.6: {r.g60.mean():.3f} | blind {r.blind.mean():.3f}")
# calibration check on held-out feeder: predicted vs actual precision by decile
allp = []
for cs, js in [(7,5),(17,15),(27,25)]:
    df, rec, tg = proposals(c24, cs, js)
    df = df.assign(p=iso.predict(lr.predict_proba(X.transform(df[FEATS]))[:, 1]))
    allp.append(df[df.val != 0])
ap = pd.concat(allp)
ap["bin"] = pd.cut(ap.p, [0, .3, .5, .7, .9, 1.0])
print("\ncalibration on held-out feeder (predicted -> actual fraction good):")
print(ap.groupby("bin", observed=True).agg(pred=("p","mean"), actual=("val", lambda v: (v>0).mean()),
      n=("val","size")).round(3).to_string())
