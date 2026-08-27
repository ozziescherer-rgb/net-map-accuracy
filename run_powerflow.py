"""Phase 0: time-series power flow on EPRI ckt5 with diverse synthetic home profiles.

Outputs: voltages.npy (steps x meters), meta.csv (meter -> true transformer), profiles.npy
"""
import re, os, sys, time
import numpy as np
import pandas as pd
import opendssdirect as dss
from lib import OUT, CKT

DAYS = 14
SPD = 96  # 15-min steps/day
STEPS = DAYS * SPD
rng = np.random.default_rng(42)

# ---------- ground truth from model files ----------
xfr_txt = open(f"{CKT}/XFR_Loads_ckt5.dss").read()
# transformer name -> secondary bus (wdg=2 bus=X_....)
xfmr_secbus = {}
for m in re.finditer(r"New Transformer\.(\S+).*?wdg=2 bus=(\S+)", xfr_txt):
    name, bus = m.group(1), m.group(2).split(".")[0].lower()
    xfmr_secbus[bus] = name
loads_txt = open(f"{CKT}/Loads_ckt5.dss").read()
# service line: bus1=X_id.ph bus2=X_id_n.ph  -> loadbus -> xbus
svc = {}
for m in re.finditer(r"New Line\.\S+\s+bus1=(\S+)\s+bus2=(\S+)", loads_txt):
    b1, b2 = m.group(1).split(".")[0].lower(), m.group(2).split(".")[0].lower()
    svc[b2] = b1
loads = []
for m in re.finditer(r"New Load\.(\S+)\s+phases=1\s+bus1=(\S+)\s+kv=\S+\s+kW=([\d.]+)\s+pf=([\d.]+).*?yearly=(\S+)", loads_txt):
    name, bus, kw, pf, cls = m.group(1), m.group(2), float(m.group(3)), float(m.group(4)), m.group(5)
    busbase = bus.split(".")[0].lower()
    xbus = svc.get(busbase)
    xf = xfmr_secbus.get(xbus)
    loads.append(dict(load=name.lower(), node=bus.lower(), busbase=busbase,
                      kw=kw, pf=pf, cls=cls, xfmr=xf))
meta = pd.DataFrame(loads)
n = len(meta)
unmapped = meta.xfmr.isna().sum()
print(f"{n} loads, {meta.xfmr.nunique()} transformers, {unmapped} unmapped")
meta = meta.dropna(subset=["xfmr"]).reset_index(drop=True)
n = len(meta)

# ---------- synthetic 15-min profiles ----------
t = np.arange(STEPS)
hod = (t % SPD) / 4.0          # hour of day
dow = (t // SPD) % 7           # 0..6, treat 5,6 weekend
weekend = np.isin(dow, [5, 6]).astype(float)

def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)

# common weather component (temperature proxy): smooth random walk + diurnal
w = np.cumsum(rng.normal(0, 0.08, DAYS))
w = np.repeat(w, SPD) + 0.6 * gauss(hod, 15.5, 3.0)
w = (w - w.min()) / (w.max() - w.min() + 1e-9)

P = np.zeros((STEPS, n), dtype=np.float32)
for i, row in meta.iterrows():
    scale = row.kw
    if row.cls.lower().startswith("residential"):
        mu_m = rng.normal(7.2, 0.7); mu_e = rng.normal(19.0, 0.9)
        base = (0.22 + 0.45 * gauss(hod, mu_m, rng.uniform(0.8, 1.4))
                     + 0.95 * gauss(hod, mu_e, rng.uniform(1.2, 2.0))
                     + weekend * 0.25 * gauss(hod, 13.0, 3.0))
        susc = rng.uniform(0, 1) ** 2
        base = base + 0.55 * susc * w
        # AR(1) behavioral noise
        eps = np.zeros(STEPS); phi = 0.7
        innov = rng.normal(0, 0.18, STEPS)
        for k in range(1, STEPS): eps[k] = phi * eps[k-1] + innov[k]
        prof = scale * base * np.exp(eps - eps.std()**2/2)
        # appliance spikes (dryer/oven/EV-ish)
        nspk = rng.poisson(0.9 * DAYS)
        for _ in range(nspk):
            s = rng.integers(0, STEPS - 8); d = rng.integers(2, 8)
            prof[s:s+d] += rng.uniform(1.5, 4.5)
    else:  # commercial
        openh = (hod > 8) & (hod < 18)
        base = 0.25 + 0.85 * openh * (1 - 0.6 * weekend)
        base = base * (1 + 0.15 * gauss(hod, 13, 2.5)) + 0.2 * w * rng.uniform(0, 1)
        prof = scale * base * np.exp(rng.normal(0, 0.08, STEPS))
    P[:, i] = np.clip(prof, 0.05, None)

# ---------- power flow ----------
os.chdir(CKT)
dss.Text.Command("Redirect Master_ckt5.dss")
dss.Text.Command("Set mode=snapshot")
node_names = dss.Circuit.AllNodeNames()
node_idx = {nm: k for k, nm in enumerate(node_names)}
midx = np.array([node_idx[nd] for nd in meta.node])
tanphi = np.tan(np.arccos(meta.pf.values))

# order of dss.Loads iteration
order = []
i = dss.Loads.First()
while i:
    order.append(dss.Loads.Name().lower()); i = dss.Loads.Next()
pos = {nm: j for j, nm in enumerate(meta.load)}
sel = [pos.get(nm, -1) for nm in order]

V = np.zeros((STEPS, n), dtype=np.float32)
t0 = time.time()
for s in range(STEPS):
    j = dss.Loads.First(); k = 0
    while j:
        m_i = sel[k]
        if m_i >= 0:
            kw = float(P[s, m_i])
            dss.Loads.kW(kw); dss.Loads.kvar(kw * tanphi[m_i])
        k += 1; j = dss.Loads.Next()
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        print("non-converged step", s)
    V[s, :] = np.asarray(dss.Circuit.AllBusMagPu())[midx]
    if s % 200 == 0:
        print(f"step {s}/{STEPS}  {time.time()-t0:.0f}s", flush=True)

np.save(f"{OUT}/voltages.npy", V)
np.save(f"{OUT}/profiles.npy", P)
meta.to_csv(f"{OUT}/meta.csv", index=False)
print(f"done in {time.time()-t0:.0f}s; V shape {V.shape}, "
      f"V mean {V.mean():.4f} pu, min {V.min():.4f}, max {V.max():.4f}")
