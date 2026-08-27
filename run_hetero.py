"""Gauntlet v2a: heterogeneous service drops. Same feeder, same 90d/5-min profiles
(seed 1234, identical to run_90d), but service-drop lengths randomized 50-400 ft
(lognormal) instead of the stock uniform 100 ft. Saves V15_90d_hetero.npy."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import os, time, re
import numpy as np
import pandas as pd
import opendssdirect as dss
from lib import OUT, CKT

DAYS, STEP_MIN = 90, 5
SPD = 24*60//STEP_MIN; STEPS = DAYS*SPD
rng = np.random.default_rng(1234)

meta = pd.read_csv(f"{OUT}/meta.csv")
n = len(meta)
t = np.arange(STEPS); hod = (t % SPD)*STEP_MIN/60.0
dow = (t//SPD) % 7; weekend = np.isin(dow, [5,6]).astype(np.float32)
def gauss(x, mu, sig): return np.exp(-0.5*((x-mu)/sig)**2).astype(np.float32)
wd = np.cumsum(rng.normal(0, 0.07, DAYS)) + 0.8*np.sin(np.arange(DAYS)/90*np.pi)
w = np.repeat(wd, SPD) + 0.6*gauss(hod, 15.5, 3.0)
w = ((w - w.min())/(w.max()-w.min()+1e-9)).astype(np.float32)

P = np.zeros((STEPS, n), dtype=np.float32)
for i, row in meta.iterrows():
    scale = row.kw
    if str(row.cls).lower().startswith("residential"):
        mu_m = rng.normal(7.2,.7); mu_e = rng.normal(19.0,.9)
        base = (0.20+0.40*gauss(hod,mu_m,rng.uniform(.8,1.4))
                     +0.80*gauss(hod,mu_e,rng.uniform(1.2,2.0))
                     +weekend*0.22*gauss(hod,13.0,3.0))
        nc = STEPS//3+2
        eps = np.zeros(nc); innov = rng.normal(0,.18,nc)
        for k in range(1,nc): eps[k]=.7*eps[k-1]+innov[k]
        slow = np.interp(np.arange(STEPS)/3.0, np.arange(nc), eps).astype(np.float32)
        prof = scale*base*np.exp(slow-slow.std()**2/2)
        susc = rng.uniform(0,1)**2
        if susc > 0.15:
            ac_kw = rng.uniform(2.0,4.0)
            period = max(2,int(rng.uniform(10,22)/STEP_MIN)); phase = rng.integers(0,period)
            duty = np.clip(susc*(0.15+0.85*w),0,0.92)
            cyc = ((t+phase)%period) < np.maximum(1,(duty*period).astype(int))
            prof = prof + ac_kw*cyc
        for _ in range(rng.poisson(2.2*DAYS)):
            s = rng.integers(0,STEPS-6); d = rng.integers(1,6)
            prof[s:s+d] += rng.uniform(1.0,5.0)
        if rng.random() < 0.15:
            for day in range(DAYS):
                if rng.random() < 0.75:
                    s = day*SPD + int((rng.normal(22.5,1.2)%24)*60/STEP_MIN)
                    d = int(rng.uniform(120,240)/STEP_MIN)
                    e = min(s+d,STEPS); s = max(0,s)
                    prof[s:e] += 7.2
    else:
        openh = ((hod>8)&(hod<18)).astype(np.float32)
        base = 0.25+0.85*openh*(1-0.6*weekend)
        base = base*(1+0.15*gauss(hod,13,2.5))+0.2*w*rng.uniform(0,1)
        prof = scale*base*np.exp(rng.normal(0,.06,STEPS).astype(np.float32))
        rt = rng.uniform(3,8)*scale/5
        period = max(2,int(rng.uniform(12,25)/STEP_MIN)); phase = rng.integers(0,period)
        duty = np.clip(0.2+0.6*w,0,.9)
        cyc = ((t+phase)%period) < np.maximum(1,(duty*period).astype(int))
        prof = prof + rt*cyc
    P[:, i] = np.clip(prof, 0.05, None)
print("profiles regenerated (identical to run_90d)", flush=True)

os.chdir(CKT)
dss.Text.Command("Redirect Master_ckt5.dss")
# --- randomize service-drop lengths (lines named s_*) ---
rl = np.random.default_rng(555)
changed = 0
i = dss.Lines.First()
while i:
    if dss.Lines.Name().lower().startswith("s_"):
        L = float(np.clip(rl.lognormal(np.log(120), 0.55), 50, 400))
        dss.Lines.Length(L)   # units already ft on these lines
        changed += 1
    i = dss.Lines.Next()
print(f"randomized {changed} service drops (50-400 ft, lognormal median~120)", flush=True)

node_idx = {nm:k for k,nm in enumerate(dss.Circuit.AllNodeNames())}
midx = np.array([node_idx[nd] for nd in meta.node])
tanphi = np.tan(np.arccos(meta.pf.values))
order = []
i = dss.Loads.First()
while i:
    order.append(dss.Loads.Name().lower()); i = dss.Loads.Next()
pos = {nm:j for j,nm in enumerate(meta.load)}
sel = np.array([pos.get(nm,-1) for nm in order])

V = np.zeros((STEPS, n), dtype=np.float32)
t0 = time.time(); bad = 0
for s in range(STEPS):
    j = dss.Loads.First(); k = 0
    while j:
        m_i = sel[k]
        if m_i >= 0:
            kw = float(P[s, m_i]); dss.Loads.kW(kw); dss.Loads.kvar(kw*tanphi[m_i])
        k += 1; j = dss.Loads.Next()
    dss.Solution.Solve()
    bad += 0 if dss.Solution.Converged() else 1
    V[s,:] = np.asarray(dss.Circuit.AllBusMagPu())[midx]
    if s % 6000 == 0: print(f"step {s}/{STEPS} {time.time()-t0:.0f}s", flush=True)

V15 = V.reshape(-1,3,n).mean(1)
np.save(f"{OUT}/V15_90d_hetero.npy", V15)
print(f"done {time.time()-t0:.0f}s nonconv={bad}; V15 {V15.shape} mean {V15.mean():.4f} min {V15.min():.4f}")
