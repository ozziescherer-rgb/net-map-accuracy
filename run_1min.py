"""28-day, 1-minute power flow with sub-15-min physics: AC thermostat cycling,
EV charging, minute-scale appliance events. The signal that 15-min averaging destroys."""
import os, time
import numpy as np
import pandas as pd
import opendssdirect as dss
from lib import OUT, CKT

DAYS = 28
SPD = 1440
STEPS = DAYS * SPD
rng = np.random.default_rng(42)

meta = pd.read_csv(f"{OUT}/meta.csv")
n = len(meta)
t = np.arange(STEPS)
hod = (t % SPD) / 60.0
dow = (t // SPD) % 7
weekend = np.isin(dow, [5, 6]).astype(np.float32)

def gauss(x, mu, sig): return np.exp(-0.5*((x-mu)/sig)**2).astype(np.float32)

# weather proxy: daily random walk + diurnal, minute-smooth
wd = np.cumsum(rng.normal(0, 0.08, DAYS))
w = np.repeat(wd, SPD) + 0.6*gauss(hod, 15.5, 3.0)
w = ((w - w.min())/(w.max()-w.min()+1e-9)).astype(np.float32)

P = np.zeros((STEPS, n), dtype=np.float32)
for i, row in meta.iterrows():
    scale = row.kw
    if str(row.cls).lower().startswith("residential"):
        mu_m = rng.normal(7.2, .7); mu_e = rng.normal(19.0, .9)
        base = (0.20 + 0.40*gauss(hod, mu_m, rng.uniform(.8, 1.4))
                     + 0.80*gauss(hod, mu_e, rng.uniform(1.2, 2.0))
                     + weekend*0.22*gauss(hod, 13.0, 3.0))
        # slow behavioral noise: AR(1) at 15-min scale, interpolated to 1-min
        nc = STEPS//15 + 2
        eps = np.zeros(nc); innov = rng.normal(0, .18, nc)
        for k in range(1, nc): eps[k] = .7*eps[k-1] + innov[k]
        slow = np.interp(np.arange(STEPS)/15.0, np.arange(nc), eps).astype(np.float32)
        prof = scale*base*np.exp(slow - slow.std()**2/2)
        # AC thermostat cycling: square wave, duty follows weather
        susc = rng.uniform(0, 1)**2
        if susc > 0.15:
            ac_kw = rng.uniform(2.0, 4.0)
            period = int(rng.uniform(10, 22))       # minutes per cycle
            phase = rng.integers(0, period)
            duty = np.clip(susc*(0.15 + 0.85*w), 0, 0.92)
            cyc = ((t + phase) % period) < np.maximum(1, (duty*period).astype(int))
            prof = prof + ac_kw*cyc
        # minute-scale appliances (kettle/dryer/oven): short high spikes
        for _ in range(rng.poisson(2.2*DAYS)):
            s = rng.integers(0, STEPS-30); d = rng.integers(2, 25)
            prof[s:s+d] += rng.uniform(1.0, 5.0)
        # EV: 15% of homes, 7 kW for 2-4 h starting 21:00-01:00
        if rng.random() < 0.15:
            for day in range(DAYS):
                if rng.random() < 0.75:
                    s = day*SPD + int(rng.normal(22.5, 1.2) % 24 * 60)
                    d = int(rng.uniform(120, 240))
                    e = min(s+d, STEPS); s = max(0, s)
                    prof[s:e] += 7.2
    else:
        openh = ((hod > 8) & (hod < 18)).astype(np.float32)
        base = 0.25 + 0.85*openh*(1 - 0.6*weekend)
        base = base*(1 + 0.15*gauss(hod, 13, 2.5)) + 0.2*w*rng.uniform(0, 1)
        prof = scale*base*np.exp(rng.normal(0, .06, STEPS).astype(np.float32))
        # HVAC cycling for commercial too
        rt = rng.uniform(3, 8)*scale/5
        period = int(rng.uniform(12, 25)); phase = rng.integers(0, period)
        duty = np.clip(0.2 + 0.6*w, 0, .9)
        cyc = ((t+phase) % period) < np.maximum(1, (duty*period).astype(int))
        prof = prof + rt*cyc
    P[:, i] = np.clip(prof, 0.05, None)

print(f"profiles done: mean home {P.mean():.2f} kW, p99 {np.quantile(P,0.99):.1f} kW", flush=True)

os.chdir(CKT)
dss.Text.Command("Redirect Master_ckt5.dss")
node_idx = {nm: k for k, nm in enumerate(dss.Circuit.AllNodeNames())}
midx = np.array([node_idx[nd] for nd in meta.node])
tanphi = np.tan(np.arccos(meta.pf.values))
order = []
i = dss.Loads.First()
while i:
    order.append(dss.Loads.Name().lower()); i = dss.Loads.Next()
pos = {nm: j for j, nm in enumerate(meta.load)}
sel = np.array([pos.get(nm, -1) for nm in order])

V = np.zeros((STEPS, n), dtype=np.float32)
t0 = time.time(); bad = 0
for s in range(STEPS):
    j = dss.Loads.First(); k = 0
    while j:
        m_i = sel[k]
        if m_i >= 0:
            kw = float(P[s, m_i])
            dss.Loads.kW(kw); dss.Loads.kvar(kw*tanphi[m_i])
        k += 1; j = dss.Loads.Next()
    dss.Solution.Solve()
    bad += 0 if dss.Solution.Converged() else 1
    V[s, :] = np.asarray(dss.Circuit.AllBusMagPu())[midx]
    if s % 5000 == 0:
        print(f"step {s}/{STEPS} {time.time()-t0:.0f}s", flush=True)

np.save(f"{OUT}/V_1min.npy", V)
print(f"done {time.time()-t0:.0f}s, nonconverged={bad}, "
      f"V mean {V.mean():.4f} min {V.min():.4f} max {V.max():.4f}")
