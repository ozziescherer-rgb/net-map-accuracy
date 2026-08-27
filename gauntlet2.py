"""Gauntlet v2: four pre-mortem stressors, scored with the frozen leak-free pipeline.
(a) heterogeneous service drops (real re-simulated voltages)
(b) block corruption: whole transformer groups mis-recorded (subdivision-scale errors)
(c) grid-snapped premise coordinates (block-level geocoding)
(d) regulator-style zone tap steps injected as common-mode
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import re
import numpy as np, pandas as pd
from lib import corr_matrix, degrade, OUT

src = open(f"{OUT}/audit.py").read()
cut = src.index("results = {}")
exec(src[:cut])   # defines load_feeder(), pipeline()

V0, true_g, XY, nearest, sp = load_feeder("ckt5")
seeds = [(7,5),(17,15),(27,25)]

def run_seeds(C, tag, **kw):
    rows = [pipeline(C, true_g, XY, nearest, sp, 0.10, cs, js, **kw) for cs, js in seeds]
    df = pd.DataFrame(rows)
    print(f"{tag:42s} det={df.det.mean():.3f}±{df.det.std():.3f} fpr={df.fpr.mean():.3f} "
          f"corr_fix={df.corr_fix.mean():.3f}±{df.corr_fix.std():.3f} "
          f"obs_gate={df.obs_gate.mean():.3f} blind={df.blind.mean():.3f} base={df.base.mean():.3f}", flush=True)
    return df

print("== baseline (uniform 100-ft drops, from paper) ==")
C0 = corr_matrix(degrade(V0))
run_seeds(C0, "baseline")

print("\n== (a) heterogeneous service drops (50-400 ft) ==")
Vh = np.load(f"{OUT}/V15_90d_hetero.npy")
Ch = corr_matrix(degrade(Vh))
run_seeds(Ch, "hetero drops")

print("\n== (d) regulator-style zone tap steps ==")
# 3 zones by y-coordinate of true transformer; each zone: persistent random +-0.75V steps ~4/day
n = len(true_g)
zy = XY[true_g][:,1]
zone = np.digitize(zy, np.nanquantile(zy, [1/3, 2/3]))
rngT = np.random.default_rng(9)
T = V0.shape[0]
steps = np.zeros((T,3), dtype=np.float32)
for z in range(3):
    ev = rngT.random(T) < (4/96)          # ~4 events/day
    sig = np.where(ev, rngT.choice([-1,1], T)*0.75/240.0, 0)
    lvl = np.cumsum(sig)
    lvl -= np.round(lvl/(5*0.75/240.0))*(5*0.75/240.0)*0  # let it wander (bounded by mean reversion below)
    lvl = lvl - np.convolve(lvl, np.ones(500)/500, mode="same")*0.5  # mild mean reversion
    steps[:,z] = lvl
Vt = V0 + steps[:, zone]
Ct = corr_matrix(degrade(Vt))
run_seeds(Ct, "zone tap steps")

print("\n== (b) block corruption (whole groups mis-recorded) ==")
def pipeline_block(C, cseed, jseed):
    """like pipeline() but corruption = whole transformer groups reassigned."""
    n = len(true_g)
    rj = np.random.default_rng(jseed)
    mxy = XY[true_g] + rj.normal(0, 0.35*sp, (n,2))
    Dm = np.nan_to_num(np.sqrt(((mxy[:,None,:]-XY[None,:,:])**2).sum(-1)), nan=np.inf)
    gps = np.argsort(Dm, axis=1)[:,:5]
    rc = np.random.default_rng(cseed)
    rec = true_g.copy()
    bad_list = []
    gsizes = np.bincount(true_g, minlength=true_g.max()+1)
    cand_groups = rc.permutation(np.where(gsizes >= 2)[0])
    moved = 0
    for g in cand_groups:
        if moved >= int(0.10*n): break
        tgt = rc.choice(nearest[g][:5])
        mem = np.where(true_g == g)[0]
        rec[mem] = tgt
        bad_list.extend(mem.tolist()); moved += len(mem)
    bad = np.array(bad_list)
    is_bad = np.zeros(n, bool); is_bad[bad] = True
    within = np.empty(n)
    for i in range(n):
        mem = np.where(rec == rec[i])[0]; mem = mem[mem != i]
        within[i] = C[i, mem].mean() if len(mem) else np.nan
    ok = ~np.isnan(within)
    med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826
    props = []; flagged = np.zeros(n,bool); corrected = np.zeros(n,int)-1
    for i in range(n):
        cands = set(gps[i].tolist()) | {rec[i]}
        sc = {}
        for g2 in cands:
            mem = np.where(rec == g2)[0]; mem = mem[mem != i]
            if len(mem): sc[g2] = C[i, mem].mean()
        alt = {g2:s for g2,s in sc.items() if g2 != rec[i]}
        if not alt: continue
        gb = max(alt, key=alt.get); sb = alt[gb]
        w = sc.get(rec[i], np.nan)
        if (not np.isnan(w) and w < med-3*mad and sb > w) or \
           (sb > med-1*mad and (np.isnan(w) or sb > w)):
            flagged[i] = True; corrected[i] = gb
            props.append((sb-(w if not np.isnan(w) else med-3*mad), i, gb))
    det = flagged[bad].mean(); fpr = flagged[~is_bad].mean()
    corr = (corrected[bad] == true_g[bad]).mean()
    props.sort(reverse=True)
    lab = rec.copy(); base = np.mean(lab == true_g); accs=[]
    for (cf,i,gb) in props: lab[i]=gb; accs.append(np.mean(lab==true_g))
    oracle = max(accs) if accs else base; blind = accs[-1] if accs else base
    return dict(det=det, fpr=fpr, corr=corr, base=base, oracle=oracle, blind=blind)
rows = [pipeline_block(C0, cs, js) for cs, js in seeds]
df = pd.DataFrame(rows)
print(f"block errors: det={df.det.mean():.3f} fpr={df.fpr.mean():.3f} corr={df['corr'].mean():.3f} "
      f"base={df.base.mean():.3f} -> oracle {df.oracle.mean():.3f} blind {df.blind.mean():.3f}")
print("(note: block-corrupted meters keep their true siblings WITH them — the whole family "
      "moved together, so within-group correlation stays high. Detection must come from "
      "the alternative/GPS door. corr here = overall, incl. structurally hard cases)")

print("\n== (c) grid-snapped premise coordinates (block-level geocoding) ==")
for gridmult in (1.0, 2.0):
    gsz = gridmult*sp
    def pipeline_grid(C, cseed):
        n = len(true_g)
        snapped = np.round(XY[true_g]/gsz)*gsz    # everyone snaps to grid intersections
        Dm = np.nan_to_num(np.sqrt(((snapped[:,None,:]-XY[None,:,:])**2).sum(-1)), nan=np.inf)
        gps = np.argsort(Dm, axis=1)[:,:5]
        contain = np.mean([true_g[i] in gps[i] for i in range(n)])
        rc = np.random.default_rng(cseed)
        rec = true_g.copy()
        bad = rc.choice(n, size=int(0.10*n), replace=False)
        for i in bad: rec[i] = rc.choice(nearest[true_g[i]])
        is_bad = np.zeros(n,bool); is_bad[bad]=True
        within = np.empty(n)
        for i in range(n):
            mem = np.where(rec==rec[i])[0]; mem=mem[mem!=i]
            within[i]=C[i,mem].mean() if len(mem) else np.nan
        ok=~np.isnan(within); med=np.median(within[ok]); mad=np.median(np.abs(within[ok]-med))*1.4826
        props=[]; flagged=np.zeros(n,bool); corrected=np.zeros(n,int)-1
        for i in range(n):
            cands=set(gps[i].tolist())|{rec[i]}
            sc={}
            for g2 in cands:
                mem=np.where(rec==g2)[0]; mem=mem[mem!=i]
                if len(mem): sc[g2]=C[i,mem].mean()
            alt={g2:s for g2,s in sc.items() if g2!=rec[i]}
            if not alt: continue
            gb=max(alt,key=alt.get); sb=alt[gb]; w=sc.get(rec[i],np.nan)
            if (not np.isnan(w) and w<med-3*mad and sb>w) or (sb>med-1*mad and (np.isnan(w) or sb>w)):
                flagged[i]=True; corrected[i]=gb
                props.append((sb-(w if not np.isnan(w) else med-3*mad),i,gb))
        fixable=np.array([np.sum((rec==true_g[i])&(np.arange(n)!=i)&(true_g==true_g[i]))>0 for i in bad])
        det=flagged[bad].mean(); fpr=flagged[~is_bad].mean()
        cfix=(corrected[bad][fixable]==true_g[bad][fixable]).mean()
        props.sort(reverse=True)
        lab=rec.copy(); base=np.mean(lab==true_g); accs=[]
        for (cf,i,gb) in props: lab[i]=gb; accs.append(np.mean(lab==true_g))
        blind=accs[-1] if accs else base
        return contain, det, fpr, cfix, base, blind
    res = [pipeline_grid(C0, cs) for cs,_ in seeds]
    c,d,f,cf,b,bl = np.mean(res, axis=0)
    print(f"grid={gridmult:.0f}x spacing: containment={c:.3f} det={d:.3f} fpr={f:.3f} "
          f"corr_fix={cf:.3f} base={b:.3f} blind={bl:.3f}")
EOF_MARKER = True
