"""Block-detector v2: adds a DISPLACED-GROUP test to the v1 bimodal-split test.

Why v1 recall is capped at 25-36%: v1 only sees blocks that merged into an
OCCUPIED transformer's group (two families under one label -> bimodal corr).
A block that landed on an EMPTY transformer label is a lone coherent family
with the wrong name -- nothing bimodal to split. v2 catches those with
geography: a coherent group whose premise centroid sits far from its recorded
transformer and close to an EMPTY one is proposed to move there, whole.

Scored on: (1) block scenario (recall should rise, precision must hold)
           (2) scattered scenario + clean data (both tests must stay quiet)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np, pandas as pd
from lib import OUT

src = open(f"{OUT}/audit.py").read()
exec(src[:src.index("results = {}")])   # load_feeder
bsrc = open(f"{OUT}/block_detector.py").read()
exec(bsrc[bsrc.index("V0, true_g"):bsrc.index('print("== block')])  # data + v1 fns

def detect_displaced(rec, jseed, margin=0.75, minfar=1.25, cohfloor=None):
    # thresholds chosen from sweep: (0.75,1.25) keeps flag precision >=0.97 with
    # recall .40-.44; looser (0.5,0.75) reaches .46-.51 but precision drops to .91
    """Whole-group displacement test. Only observables: C, rec, premise GPS, XY."""
    rj = np.random.default_rng(jseed)
    mxy = XY[true_g] + rj.normal(0, 0.35*sp, (n,2))
    occ = np.bincount(rec, minlength=nx) > 0
    # coherence distribution across recorded groups (for a robust floor)
    cohs = {}
    for g in range(nx):
        mem = np.where(rec == g)[0]
        if len(mem) < 2: continue
        sub = C[np.ix_(mem, mem)]
        cohs[g] = (sub.sum()-len(mem))/(len(mem)*(len(mem)-1))
    vals = np.array(list(cohs.values()))
    med = np.median(vals); mad = np.median(np.abs(vals-med))*1.4826
    floor = cohfloor if cohfloor is not None else med - 2*mad
    props = []
    for g, coh in cohs.items():
        if coh < floor: continue                    # not a single coherent family
        mem = np.where(rec == g)[0]
        cen = mxy[mem].mean(0)
        d_rec = np.linalg.norm(cen - XY[g])
        if d_rec < minfar*sp: continue              # centroid plausibly at recorded xfmr
        dists = np.linalg.norm(XY - cen, axis=1)
        dists[occ] = np.inf
        tgt = int(np.argmin(dists))
        if d_rec - dists[tgt] > margin*sp:
            for i in mem: props.append((int(i), tgt, float(d_rec - dists[tgt])))
    return props

def merge(p1, p2):
    got = {i for i,_,_ in p1}
    return p1 + [p for p in p2 if p[0] not in got]

def score(tag, mk):
    print(f"== {tag} ==")
    for cseed, jseed in seeds:
        rec, bad = mk(cseed)
        is_bad = np.zeros(n, bool)
        if len(bad): is_bad[bad] = True
        p1 = detect_blocks(rec, jseed)
        p2 = detect_displaced(rec, jseed)
        props = merge(p1, p2)
        flagged = np.array([i for i,_,_ in props], dtype=int)
        prec = is_bad[flagged].mean() if len(flagged) else float("nan")
        recall = np.isin(bad, flagged).mean() if len(bad) else float("nan")
        correct = sum(1 for i,t,_ in props if t == true_g[i])
        lab = rec.copy()
        for i,t,_ in props: lab[i] = t
        print(f"seed {cseed}: v1 {len(p1)} + displaced {len(p2)} -> {len(props)} props | "
              f"flag prec {prec:.2f} | recall {recall:.2f} | target {correct}/{len(props)} | "
              f"net {np.mean(rec==true_g):.3f} -> {np.mean(lab==true_g):.3f}")

score("block scenario (v1 alone: recall .25-.36, net +1.5-3.3pt)", make_block)
score("scattered scenario (regression: should stay quiet)", make_scattered)
score("clean data (must be silent)", lambda cs: (true_g.copy(), np.array([], dtype=int)))
