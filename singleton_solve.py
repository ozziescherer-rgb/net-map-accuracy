"""Open problem #1: the singleton observability boundary.
Weapon: directional load-coupling. If meter i truly shares a transformer with group g,
then g's members' voltages must dip when i's load spikes (shared impedance).
A meter whose load moves NO nearby group's voltage is coupled to no one -> singleton
verdict -> assign to nearest EMPTY transformer by premise GPS.
Evaluated over 5 corruption seeds on ckt5, 90 days.
"""
import re
import numpy as np, pandas as pd
from lib import degrade, CKT, OUT

meta = pd.read_csv(f"{OUT}/meta.csv")
xfmrs = sorted(meta.xfmr.unique()); gid = {x: i for i, x in enumerate(xfmrs)}
true_g = meta.xfmr.map(gid).values; n = len(true_g); nx = len(xfmrs)
V = degrade(np.load(f"{OUT}/V15_90d.npy"))       # what meters report
P = np.load(f"{OUT}/P15_90d.npy")                # kW, also reported
T = V.shape[0]

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

# ---- residualize once: remove feeder-wide common mode from V and P ----
X = np.column_stack([np.ones(T), P.sum(1), V.mean(1)])
B = np.linalg.lstsq(X, V, rcond=None)[0]; rV = V - X @ B
Bp = np.linalg.lstsq(X, P, rcond=None)[0]; rP = P - X @ Bp
rVz = (rV - rV.mean(0)) / (rV.std(0) + 1e-12)
rPz = (rP - rP.mean(0)) / (rP.std(0) + 1e-12)

def coupling_score(i, members):
    """Pooled evidence that group members' voltage responds (negatively) to meter i's load.
    Mean per-member correlation, scaled by sqrt(T * n_members).

    NOT a z-score. The sqrt(T) scaling assumes T independent samples; measured lag-1
    autocorrelation of the residual load is rho ~ 0.68 (voltage rho ~ 0.24), giving an
    effective sample size of ~1650 rather than T = 8640. A reported value of -8 is
    roughly 3.5 sigma, not 8. Treat this as a ranking statistic only; THETA below was
    selected empirically from the score distribution, not from a normal tail probability."""
    mem = members[members != i]
    if len(mem) == 0: return np.nan
    c = (rVz[:, mem] * rPz[:, [i]]).mean(0)        # corr(rV_m, rP_i) per member
    return float(c.mean() * np.sqrt(T * len(mem)))  # ~N(0,1) under no coupling

# ---- calibrate the "coupled" threshold on observable data (clean-ish meters) ----
# true-family coupling distribution vs unrelated-group distribution
rng0 = np.random.default_rng(0)
zt, zu = [], []
for i in rng0.choice(n, 300, replace=False):
    mem = np.where(true_g == true_g[i])[0]
    z = coupling_score(i, mem)
    if not np.isnan(z): zt.append(z)
    gW = nearest[true_g[i]][rng0.integers(0, 5)]
    z = coupling_score(i, np.where(true_g == gW)[0])
    if not np.isnan(z): zu.append(z)
print(f"coupling score: true family mean {np.mean(zt):.1f} (p90 {np.percentile(zt,90):.1f}), "
      f"unrelated mean {np.mean(zu):.1f} (p10 {np.percentile(zu,10):.1f})")
THETA = -8.0   # coupled iff score < THETA (empirical threshold, not a sigma level) (deeply negative = voltage dips with my load)

print(f"\n{'seed':>5} {'singT_n':>7} {'sing_recovered':>14} {'sing_prec':>9} "
      f"{'famT_verdict_ok':>15} {'corr_fix_new':>12}")
allres = []
for cseed, jseed in [(7,5),(17,15),(27,25),(37,35),(47,45)]:
    rj = np.random.default_rng(jseed)
    mxy = XY[true_g] + rj.normal(0, 0.35*sp, (n, 2))
    Dm = np.nan_to_num(np.sqrt(((mxy[:, None, :]-XY[None, :, :])**2).sum(-1)), nan=np.inf)
    gps_full = np.argsort(Dm, axis=1)
    rc = np.random.default_rng(cseed)
    rec = true_g.copy()
    bad = rc.choice(n, size=int(0.10*n), replace=False)
    for i in bad: rec[i] = rc.choice(nearest[true_g[i]])
    occ = np.bincount(rec, minlength=nx) > 0
    fixable = np.array([np.sum((rec == true_g[i]) & (np.arange(n) != i) &
                               (true_g == true_g[i])) > 0 for i in bad])
    singT = bad[~fixable]          # true transformer is an unobserved singleton
    famT = bad[fixable]

    def decide(i):
        """returns (verdict, assignment): verdict 'family' or 'singleton'."""
        cands = [g for g in gps_full[i, :10] if occ[g] and g != rec[i]]
        zs = {g: coupling_score(i, np.where(rec == g)[0]) for g in cands}
        zs = {g: z for g, z in zs.items() if not np.isnan(z)}
        gbest = min(zs, key=zs.get) if zs else None
        if gbest is not None and zs[gbest] < THETA:
            return "family", gbest
        # no coupling anywhere nearby -> singleton: nearest empty transformer
        for g in gps_full[i]:
            if not occ[g] and Dm[i, g] < 3*sp:
                return "singleton", g
        return "singleton", -1

    sing_correct = sing_attempt = 0
    fam_ok = 0
    corrected = {}
    for i in singT:
        v, g = decide(i); corrected[i] = g
        sing_attempt += 1
        if v == "singleton" and g == true_g[i]: sing_correct += 1
    for i in famT:
        v, g = decide(i); corrected[i] = g
        if v == "family": fam_ok += 1
    cfx = np.mean([corrected[i] == true_g[i] for i in famT])
    allres.append(dict(nsing=len(singT), srec=sing_correct,
                       sprec=sing_correct/max(sing_attempt,1),
                       famok=fam_ok/max(len(famT),1), cfx=cfx))
    print(f"{cseed:>5} {len(singT):>7} {sing_correct:>14} "
          f"{sing_correct/max(sing_attempt,1):>9.2f} {fam_ok/max(len(famT),1):>15.2f} {cfx:>12.3f}")

df = pd.DataFrame(allres)
print(f"\nSINGLETON RECOVERY: {df.srec.sum()}/{df.nsing.sum()} "
      f"({df.srec.sum()/df.nsing.sum():.1%}) across 5 seeds — was 0% by construction before.")
print(f"family-truth meters correctly kept on 'family' path: {df.famok.mean():.1%}")
print(f"coupling-based correction accuracy on family-truth errors: {df.cfx.mean():.3f} "
      f"(correlation-based was ~0.93)")
