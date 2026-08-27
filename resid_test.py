"""Local common-mode residualization vs regulator tap steps (additive candidate).

Tap steps are common-mode within a regulator zone: every meter in the zone moves
together, which inflates cross-transformer correlations and costs ~2pts. Fix
candidate: estimate each meter's zone common-mode as the median voltage of its
40 nearest premises OUTSIDE its own likely secondary (>1.5*spacing away), and
subtract it before correlating. Uses only observables (premise GPS, voltages).

Scored on: (1) tap-step scenario (should recover the ~2pts)
           (2) clean baseline (regression: must not hurt the paper numbers)

RESULT (2026-08-24): REJECTED. Tap steps: gate +0.7pt but det -2.9pts, corr_fix
-4.6pts. Baseline regression: corr_fix 0.919->0.879, gate 0.923->0.916 -- it
removes genuine secondary-level signal along with the zone common-mode. The
observable gate already contains the tap-step damage (gate holds within ~2pts
of baseline without help), so no module ships for this. Kept as a documented
dead end per the frozen-baseline discipline.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np, pandas as pd
from lib import corr_matrix, degrade, OUT

src = open(f"{OUT}/audit.py").read()
exec(src[:src.index("results = {}")])   # load_feeder, pipeline

V0, true_g, XY, nearest, sp = load_feeder("ckt5")
n = len(true_g)
seeds = [(7,5),(17,15),(27,25)]

# --- rebuild zone tap steps exactly as gauntlet2 (seed 9) ---
zy = XY[true_g][:,1]
zone = np.digitize(zy, np.nanquantile(zy, [1/3, 2/3]))
rngT = np.random.default_rng(9)
T = V0.shape[0]
steps = np.zeros((T,3), dtype=np.float32)
for z in range(3):
    ev = rngT.random(T) < (4/96)
    sig = np.where(ev, rngT.choice([-1,1], T)*0.75/240.0, 0)
    lvl = np.cumsum(sig)
    lvl -= np.round(lvl/(5*0.75/240.0))*(5*0.75/240.0)*0
    lvl = lvl - np.convolve(lvl, np.ones(500)/500, mode="same")*0.5
    steps[:,z] = lvl
Vt = V0 + steps[:, zone]

def residualize(Vd, jseed, k=40, excl=1.5):
    """subtract per-meter local common-mode (median of k nearest premises
    farther than excl*sp away). Premise GPS modeled identically to pipeline."""
    rj = np.random.default_rng(jseed)
    mxy = XY[true_g] + rj.normal(0, 0.35*sp, (n,2))
    D = np.sqrt(((mxy[:,None,:]-mxy[None,:,:])**2).sum(-1))
    np.fill_diagonal(D, np.inf)
    R = np.empty_like(Vd)
    order = np.argsort(D, axis=1)
    for i in range(n):
        nb = [j for j in order[i] if D[i,j] > excl*sp][:k]
        R[:,i] = Vd[:,i] - np.median(Vd[:,nb], axis=1)
    return R

def run(tag, V):
    for resid in (False, True):
        rows = []
        for cs, js in seeds:
            Vd = degrade(V)
            C = corr_matrix(residualize(Vd, js) if resid else Vd)
            rows.append(pipeline(C, true_g, XY, nearest, sp, 0.10, cs, js))
        df = pd.DataFrame(rows)
        print(f"{tag:12s} resid={resid!s:5s} det={df.det.mean():.3f} fpr={df.fpr.mean():.3f} "
              f"corr_fix={df.corr_fix.mean():.3f} obs_gate={df.obs_gate.mean():.3f} "
              f"blind={df.blind.mean():.3f} base={df.base.mean():.3f}", flush=True)

run("tap steps", Vt)
run("baseline", V0)
