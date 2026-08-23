"""Generalization: full pipeline on EPRI ckt24 (820 service transformers, ~3890 customers).
56 days @ 15-min. Saves V24.npy, meta24.csv."""
import re, os, time
import numpy as np
import pandas as pd
import opendssdirect as dss

CKT = "/home/user/electricdss-tst/Version8/Distrib/EPRITestCircuits/ckt24"
OUT = "/home/user/xfmr"
DAYS, SPD = 56, 96
STEPS = DAYS*SPD
rng = np.random.default_rng(99)

# ---- topology: transformer wdg2 sec bus <name>_sec ; load bus <name>_sec_k ----
xtxt = open(f"{CKT}/transformers_ckt24.dss").read()
sec2xf = {}
for m in re.finditer(r"New\s+Transformer\.(\S+).*?wdg=2\s+bus=(\S+)", xtxt):
    sec2xf[m.group(2).split(".")[0].lower()] = m.group(1)
print(f"{len(sec2xf)} service transformers")

os.chdir(CKT)
dss.Text.Command("Redirect master_ckt24.dss")
dss.Text.Command("Set mode=snapshot")
dss.Solution.Solve()
print("compiled; converged:", dss.Solution.Converged())

# enumerate loads with allocated kW
rows = []
i = dss.Loads.First()
while i:
    name = dss.Loads.Name().lower()
    kw = dss.Loads.kW()
    dss.Circuit.SetActiveElement(f"Load.{name}")
    bus = dss.CktElement.BusNames()[0].lower()
    busbase = bus.split(".")[0]
    secbase = re.sub(r"_(\d+)$", "", busbase)
    xf = sec2xf.get(secbase)
    rows.append(dict(load=name, node=bus, busbase=busbase, kw=kw, xf=xf))
    i = dss.Loads.Next()
meta = pd.DataFrame(rows)
n_all = len(meta)
meta = meta.dropna(subset=["xf"]).reset_index(drop=True)
meta = meta[meta.kw > 0].reset_index(drop=True)
n = len(meta)
print(f"{n}/{n_all} loads mapped to {meta.xf.nunique()} transformers; "
      f"kw: med {meta.kw.median():.1f} p95 {meta.kw.quantile(.95):.1f}")

# ---- profiles (same family as ckt5 runs) ----
t = np.arange(STEPS); hod = (t % SPD)/4.0
dow = (t//SPD) % 7; weekend = np.isin(dow, [5, 6]).astype(np.float32)
def gauss(x, mu, sig): return np.exp(-0.5*((x-mu)/sig)**2).astype(np.float32)
wd = np.cumsum(rng.normal(0, .07, DAYS)) + 0.8*np.sin(np.arange(DAYS)/DAYS*np.pi)
w = np.repeat(wd, SPD) + 0.6*gauss(hod, 15.5, 3.0)
w = ((w-w.min())/(w.max()-w.min()+1e-9)).astype(np.float32)

P = np.zeros((STEPS, n), dtype=np.float32)
for i, row in meta.iterrows():
    scale = row.kw
    if scale <= 20:   # residential
        mu_m = rng.normal(7.2, .7); mu_e = rng.normal(19.0, .9)
        base = (0.20 + 0.40*gauss(hod, mu_m, rng.uniform(.8, 1.4))
                     + 0.80*gauss(hod, mu_e, rng.uniform(1.2, 2.0))
                     + weekend*0.22*gauss(hod, 13.0, 3.0))
        susc = rng.uniform(0, 1)**2
        base = base + 0.55*susc*w
        eps = np.zeros(STEPS); innov = rng.normal(0, .18, STEPS)
        for k in range(1, STEPS): eps[k] = .7*eps[k-1] + innov[k]
        prof = scale*base*np.exp(eps - eps.std()**2/2)
        for _ in range(rng.poisson(0.9*DAYS)):
            s = rng.integers(0, STEPS-8); d = rng.integers(2, 8)
            prof[s:s+d] += rng.uniform(1.5, 4.5)
        if rng.random() < 0.15:
            for day in range(DAYS):
                if rng.random() < 0.75:
                    s = day*SPD + int((rng.normal(22.5, 1.2) % 24)*4)
                    d = int(rng.uniform(8, 16))
                    prof[max(0, s):min(s+d, STEPS)] += 7.2
    else:
        openh = ((hod > 8) & (hod < 18)).astype(np.float32)
        base = 0.25 + 0.85*openh*(1-0.6*weekend)
        base = base*(1+0.15*gauss(hod, 13, 2.5)) + 0.2*w*rng.uniform(0, 1)
        prof = scale*base*np.exp(rng.normal(0, .08, STEPS).astype(np.float32))
    P[:, i] = np.clip(prof, 0.05, None)
print("profiles done", flush=True)

node_idx = {nm: k for k, nm in enumerate(dss.Circuit.AllNodeNames())}
def first_node(bus):
    parts = bus.split(".")
    return f"{parts[0]}.{parts[1] if len(parts) > 1 else '1'}"
midx = np.array([node_idx[first_node(nd)] for nd in meta.node])
tanphi = np.tan(np.arccos(0.98))
order = []
i = dss.Loads.First()
while i:
    order.append(dss.Loads.Name().lower()); i = dss.Loads.Next()
pos = {nm: j for j, nm in enumerate(meta.load)}
sel = np.array([pos.get(nm, -1) for nm in order])

V = np.zeros((STEPS, n), dtype=np.float32)
t0 = time.time(); nc = 0
for s in range(STEPS):
    j = dss.Loads.First(); k = 0
    while j:
        m_i = sel[k]
        if m_i >= 0:
            kw = float(P[s, m_i])
            dss.Loads.kW(kw); dss.Loads.kvar(kw*tanphi)
        k += 1; j = dss.Loads.Next()
    dss.Solution.Solve()
    nc += 0 if dss.Solution.Converged() else 1
    V[s, :] = np.asarray(dss.Circuit.AllBusMagPu())[midx]
    if s % 800 == 0: print(f"step {s}/{STEPS} {time.time()-t0:.0f}s", flush=True)

np.save(f"{OUT}/V24.npy", V)
meta.to_csv(f"{OUT}/meta24.csv", index=False)
print(f"done {time.time()-t0:.0f}s nonconv={nc}; V {V.shape} mean {V.mean():.4f} min {V.min():.4f}")
