"""Block-error detector (additive module). Signature of a whole-group record error:
the receiving group's internal correlation structure is BIMODAL (two true families
sharing one label), and the migrated family's premises sit near an EMPTY transformer.
Detector: split each recorded group's corr matrix into 2 clusters; if within-cluster
corr >> cross-cluster corr and both clusters >=2 members, the cluster whose premises
are farther from the recorded transformer is reassigned to the nearest empty
transformer near its premise centroid.

Scored on: (1) block-corruption scenario (can it fix what the core misses?)
           (2) standard scattered scenario (regression check: does it misfire?)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np, pandas as pd
from lib import corr_matrix, degrade, OUT

src = open(f"{OUT}/audit.py").read()
exec(src[:src.index("results = {}")])   # load_feeder

V0, true_g, XY, nearest, sp = load_feeder("ckt5")
C = corr_matrix(degrade(V0))
n = len(true_g); nx = true_g.max()+1
seeds = [(7,5),(17,15),(27,25)]

def make_block(cseed):
    rc = np.random.default_rng(cseed)
    rec = true_g.copy(); bad = []
    gs = np.bincount(true_g, minlength=nx)
    for g in rc.permutation(np.where(gs >= 2)[0]):
        if len(bad) >= int(0.10*n): break
        mem = np.where(true_g == g)[0]
        rec[mem] = rc.choice(nearest[g][:5]); bad.extend(mem.tolist())
    return rec, np.array(bad)

def make_scattered(cseed):
    rc = np.random.default_rng(cseed)
    rec = true_g.copy()
    bad = rc.choice(n, size=int(0.10*n), replace=False)
    for i in bad: rec[i] = rc.choice(nearest[true_g[i]])
    return rec, bad

def detect_blocks(rec, jseed, theta=0.03):
    """returns proposals: list of (meter_i, new_group). Uses only observables."""
    rj = np.random.default_rng(jseed)
    mxy = XY[true_g] + rj.normal(0, 0.35*sp, (n,2))   # premise GPS (modeled as usual)
    occ = np.bincount(rec, minlength=nx) > 0
    props = []
    for g in range(nx):
        mem = np.where(rec == g)[0]
        if len(mem) < 4: continue
        sub = C[np.ix_(mem, mem)]
        # 2-way split by sign of leading eigenvector of centered corr
        A = sub - sub.mean()
        vals, vecs = np.linalg.eigh(A)
        lab = (vecs[:, -1] > 0)
        c1, c2 = mem[lab], mem[~lab]
        if len(c1) < 2 or len(c2) < 2: continue
        w1 = sub[np.ix_(lab, lab)]; w2 = sub[np.ix_(~lab, ~lab)]
        within = (w1.sum()-len(c1))/max(len(c1)*(len(c1)-1),1)/2*2
        within2 = (w2.sum()-len(c2))/max(len(c2)*(len(c2)-1),1)/2*2
        between = sub[np.ix_(lab, ~lab)].mean()
        gap = min(within, within2) - between
        if gap < theta: continue
        # migrated cluster = premises farther from recorded transformer g
        d1 = np.linalg.norm(mxy[c1] - XY[g], axis=1).mean()
        d2 = np.linalg.norm(mxy[c2] - XY[g], axis=1).mean()
        mig = c1 if d1 > d2 else c2
        cen = mxy[mig].mean(0)
        # nearest EMPTY transformer to the migrated cluster's centroid
        dists = np.linalg.norm(XY - cen, axis=1)
        dists[occ] = np.inf
        tgt = int(np.argmin(dists))
        if dists[tgt] < 3*sp:
            for i in mig: props.append((int(i), tgt, float(gap)))
    return props

print("== block-corruption scenario (core pipeline: det 13.5%, corr 0%) ==")
for cseed, jseed in seeds:
    rec, bad = make_block(cseed)
    is_bad = np.zeros(n, bool); is_bad[bad] = True
    props = detect_blocks(rec, jseed)
    flagged = np.array([i for i,_,_ in props])
    correct = sum(1 for i,tgt,_ in props if tgt == true_g[i])
    det = is_bad[flagged].mean() if len(flagged) else 0     # precision of flags
    recall = np.isin(bad, flagged).mean()
    # net accuracy if applied
    lab = rec.copy()
    for i,tgt,_ in props: lab[i] = tgt
    print(f"seed {cseed}: {len(props)} proposals | flag precision {det:.2f} | "
          f"recall of block-errored meters {recall:.2f} | correct target {correct}/{len(props)} | "
          f"net acc {np.mean(rec==true_g):.3f} -> {np.mean(lab==true_g):.3f}")

print("\n== regression check: scattered scenario (detector should stay quiet) ==")
for cseed, jseed in seeds:
    rec, bad = make_scattered(cseed)
    is_bad = np.zeros(n, bool); is_bad[bad] = True
    props = detect_blocks(rec, jseed)
    fp = sum(1 for i,_,_ in props if not is_bad[i])
    lab = rec.copy()
    for i,tgt,_ in props: lab[i] = tgt
    print(f"seed {cseed}: {len(props)} proposals ({fp} on clean meters) | "
          f"net acc {np.mean(rec==true_g):.3f} -> {np.mean(lab==true_g):.3f}")
