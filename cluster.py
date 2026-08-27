"""Phase 0 scoring: Spearman voltage correlation -> detect & correct bad meter-transformer records."""
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from lib import OUT, CKT

rng = np.random.default_rng(7)

V = np.load(f"{OUT}/voltages.npy")
meta = pd.read_csv(f"{OUT}/meta.csv")
n = len(meta)
xfmrs = sorted(meta.xfmr.unique())
gid = {x: i for i, x in enumerate(xfmrs)}
true_g = meta.xfmr.map(gid).values
sizes = np.bincount(true_g, minlength=len(xfmrs))
print(f"{n} meters, {len(xfmrs)} transformers; group sizes: "
      f"1:{(sizes==1).sum()} 2:{(sizes==2).sum()} 3:{(sizes==3).sum()} 4+:{(sizes>=4).sum()}")

# transformer coordinates via primary bus
coords = {}
for ln in open(f"{CKT}/Buscoords_ckt5.dss"):
    p = ln.replace(",", " ").split()
    if len(p) >= 3:
        try: coords[p[0].lower()] = (float(p[1]), float(p[2]))
        except ValueError: pass
xtxt = open(f"{CKT}/XFR_Loads_ckt5.dss").read()
xf_xy = {}
for m in re.finditer(r"New Transformer\.(\S+).*?wdg=1 bus=(\S+)", xtxt):
    b = m.group(2).split(".")[0].lower()
    if b in coords: xf_xy[m.group(1)] = coords[b]
XY = np.array([xf_xy.get(x, (np.nan, np.nan)) for x in xfmrs])
# nearest-K candidate transformers for each transformer
K = 15
D = np.sqrt(((XY[:, None, :] - XY[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, np.inf)
D = np.nan_to_num(D, nan=np.inf)
nearest = np.argsort(D, axis=1)[:, :K]

def corr_matrix(V, method="spearman"):
    X = np.apply_along_axis(rankdata, 0, V) if method == "spearman" else V
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return (X.T @ X) / len(X)

def experiment(C, corrupt_frac=0.10, seed=7):
    r = np.random.default_rng(seed)
    rec_g = true_g.copy()
    bad = r.choice(n, size=int(corrupt_frac * n), replace=False)
    for i in bad:  # reassign to a nearby wrong transformer (realistic record error)
        rec_g[i] = r.choice(nearest[true_g[i]])
    is_bad = np.zeros(n, bool); is_bad[bad] = True

    def group_score(i, g, labels):
        mem = np.where(labels == g)[0]
        mem = mem[mem != i]
        return C[i, mem].mean() if len(mem) else -np.inf

    flagged = np.zeros(n, bool); corrected = np.zeros(n, int) - 1
    for i in range(n):
        cands = set(nearest[rec_g[i]].tolist()) | {rec_g[i]} | set(nearest[true_g[i]].tolist()) | {true_g[i]}
        scores = {g: group_score(i, g, rec_g) for g in cands}
        best = max(scores, key=scores.get)
        if best != rec_g[i] and scores[best] > -np.inf:
            flagged[i] = True; corrected[i] = best

    det_recall = flagged[bad].mean()
    fpr = flagged[~is_bad].mean()
    fixable = np.array([np.sum((rec_g == true_g[i]) & (np.arange(n) != i) &
                               (true_g == true_g[i])) > 0 for i in bad])
    corr_acc = (corrected[bad] == true_g[bad]).mean()
    corr_acc_fixable = (corrected[bad][fixable] == true_g[bad][fixable]).mean() if fixable.any() else np.nan
    return dict(detection_recall=det_recall, false_positive_rate=fpr,
                correction_acc=corr_acc, correction_acc_nonsingleton=corr_acc_fixable,
                n_bad=len(bad), n_fixable=int(fixable.sum()))

for method in ["spearman", "pearson"]:
    C = corr_matrix(V, method)
    res = experiment(C)
    print(f"\n[{method}]  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in res.items()))

# sanity: same-transformer vs different-transformer correlation separation
C = corr_matrix(V, "spearman")
same, diff = [], []
for g in range(len(xfmrs)):
    mem = np.where(true_g == g)[0]
    if len(mem) >= 2:
        for a in range(len(mem)):
            for b in range(a+1, len(mem)):
                same.append(C[mem[a], mem[b]])
mask = rng.integers(0, n, (4000, 2))
for a, b in mask:
    if true_g[a] != true_g[b]: diff.append(C[a, b])
print(f"\nSpearman corr — same-transformer pairs: {np.mean(same):.4f} (n={len(same)}), "
      f"different: {np.mean(diff):.4f} (n={len(diff)})")
