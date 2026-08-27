"""Shared: ground truth, correlation, detection/correction experiment."""
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# --- paths -------------------------------------------------------------------
# OUT   : where generated data (meta.csv, V*.npy, results) is read from / written to.
#         Defaults to the directory holding this file. Override with NMA_OUT.
# CKT   : EPRI ckt5 model from the OpenDSS test-case repo (see README for the clone
#         command). Defaults to ../electricdss-tst/... relative to OUT. Override with
#         NMA_CKT. CKT24 is the ckt24 sibling; override with NMA_CKT24.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NMA_OUT", _HERE)
_TST = os.environ.get(
    "NMA_TST", os.path.join(os.path.dirname(OUT), "electricdss-tst"))
_EPRI = os.path.join(_TST, "Version8", "Distrib", "EPRITestCircuits")
CKT = os.environ.get("NMA_CKT", os.path.join(_EPRI, "ckt5"))
CKT24 = os.environ.get("NMA_CKT24", os.path.join(_EPRI, "ckt24"))

def load_meta():
    meta = pd.read_csv(f"{OUT}/meta.csv")
    xfmrs = sorted(meta.xfmr.unique())
    gid = {x: i for i, x in enumerate(xfmrs)}
    true_g = meta.xfmr.map(gid).values
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
    D = np.sqrt(((XY[:, None, :] - XY[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    D = np.nan_to_num(D, nan=np.inf)
    nearest = np.argsort(D, axis=1)[:, :15]
    return meta, true_g, nearest

def corr_matrix(V, method="spearman"):
    X = np.apply_along_axis(rankdata, 0, V) if method == "spearman" else V.astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return (X.T @ X) / len(X)

def experiment(C, true_g, nearest, corrupt_frac=0.10, seed=7, margin=0.0):
    n = len(true_g)
    r = np.random.default_rng(seed)
    rec_g = true_g.copy()
    bad = r.choice(n, size=int(corrupt_frac * n), replace=False)
    for i in bad:
        rec_g[i] = r.choice(nearest[true_g[i]])
    is_bad = np.zeros(n, bool); is_bad[bad] = True

    def gscore(i, g):
        mem = np.where(rec_g == g)[0]
        mem = mem[mem != i]
        return C[i, mem].mean() if len(mem) else -np.inf

    flagged = np.zeros(n, bool); corrected = np.zeros(n, int) - 1
    for i in range(n):
        cands = set(nearest[rec_g[i]].tolist()) | {rec_g[i]} | set(nearest[true_g[i]].tolist()) | {true_g[i]}
        scores = {g: gscore(i, g) for g in cands}
        best = max(scores, key=scores.get)
        base = scores[rec_g[i]]
        if best != rec_g[i] and scores[best] > (base + margin if base > -np.inf else -np.inf):
            flagged[i] = True; corrected[i] = best

    fixable = np.array([np.sum((rec_g == true_g[i]) & (np.arange(n) != i) &
                               (true_g == true_g[i])) > 0 for i in bad])
    return dict(
        detection_recall=flagged[bad].mean(),
        false_positive_rate=flagged[~is_bad].mean(),
        correction_acc=(corrected[bad] == true_g[bad]).mean(),
        correction_acc_nonsingleton=(corrected[bad][fixable] == true_g[bad][fixable]).mean()
                                    if fixable.any() else np.nan)

PU = 1/240.0

def degrade(V, q=0.5, sd=0.3, miss=0.05, seed=11):
    rng = np.random.default_rng(seed)
    Vd = V + rng.normal(0, sd*PU, V.shape).astype(np.float32)
    Vd = np.round(Vd/(q*PU))*(q*PU)
    mask = rng.random(Vd.shape) < miss
    med = np.median(Vd, axis=0)
    Vd[mask] = np.take(med, np.where(mask)[1])
    return Vd
