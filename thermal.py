"""IEEE C57.91 thermal aging engine, driven by recovered meter-to-transformer maps.

Three maps compared end-to-end:
  TRUE      - ground truth (what a perfect map yields)
  RECORDS   - utility records with 10% errors (status quo)
  CORRECTED - our GPS+correlation corrected map
Outputs per transformer: peak/avg loading, % insulation life consumed per year,
EV hosting headroom. Plus: how badly record errors distort the risk ranking.
"""
import re
import numpy as np, pandas as pd
from lib import load_meta, corr_matrix, degrade, CKT

meta, true_g, nearest = load_meta()
n = len(true_g); nx = true_g.max()+1
xfmrs = sorted(meta.xfmr.unique())
P = np.load("/home/user/xfmr/P15_90d.npy")     # kW per meter, 15-min, 90d
V90 = np.load("/home/user/xfmr/V15_90d.npy")
T, _ = P.shape
DAYS = 90

# ---- kVA ratings + coords ----
xtxt = open(f"{CKT}/XFR_Loads_ckt5.dss").read()
kva = {}
for m in re.finditer(r"New Transformer\.(\S+).*?kVA=([\d.]+)", xtxt):
    kva[m.group(1)] = float(m.group(2))
KVA = np.array([kva[x] for x in xfmrs])
coords = {}
for ln in open(f"{CKT}/Buscoords_ckt5.dss"):
    p = ln.replace(",", " ").split()
    if len(p) >= 3:
        try: coords[p[0].lower()] = (float(p[1]), float(p[2]))
        except ValueError: pass
xy = {}
for m in re.finditer(r"New Transformer\.(\S+).*?wdg=1 bus=(\S+)", xtxt):
    b = m.group(2).split(".")[0].lower()
    if b in coords: xy[m.group(1)] = coords[b]
XY = np.array([xy.get(x, (np.nan, np.nan)) for x in xfmrs])

# ---- the three maps ----
r = np.random.default_rng(7)
rec_g = true_g.copy()
bad = r.choice(n, size=int(0.10*n), replace=False)
for i in bad: rec_g[i] = r.choice(nearest[true_g[i]])
# corrected map: GPS-K5 two-threshold (reproduce premise_gps best point)
spacing = 178.0
rng2 = np.random.default_rng(5)
meter_xy = XY[true_g] + rng2.normal(0, 0.35*spacing, (n, 2))
Dm = np.nan_to_num(np.sqrt(((meter_xy[:, None, :]-XY[None, :, :])**2).sum(-1)), nan=np.inf)
gps_rank = np.argsort(Dm, axis=1)
C = corr_matrix(degrade(V90))
within = np.empty(n)
for i in range(n):
    mem = np.where(rec_g == rec_g[i])[0]; mem = mem[mem != i]
    within[i] = C[i, mem].mean() if len(mem) else np.nan
ok = ~np.isnan(within)
med = np.median(within[ok]); mad = np.median(np.abs(within[ok]-med))*1.4826
corr_g = rec_g.copy()
for i in range(n):
    cands = set(gps_rank[i, :5].tolist()) | {rec_g[i]}
    sc = {}
    for g in cands:
        mem = np.where(rec_g == g)[0]; mem = mem[mem != i]
        if len(mem): sc[g] = C[i, mem].mean()
    alt = {g: s for g, s in sc.items() if g != rec_g[i]}
    if not alt: continue
    gb = max(alt, key=alt.get); sb = alt[gb]
    w = sc.get(rec_g[i], np.nan)
    if (not np.isnan(w) and w < med-3*mad and sb > w) or \
       (sb > med-1*mad and (np.isnan(w) or sb > w)):
        corr_g[i] = gb
print(f"corrected map: {np.mean(corr_g==true_g):.3f} of meters right "
      f"(records: {np.mean(rec_g==true_g):.3f})")

# ---- ambient temperature (same weather that drove the AC load; scaled to deg C) ----
rgw = np.random.default_rng(1234)
wd = np.cumsum(rgw.normal(0, 0.07, DAYS)) + 0.8*np.sin(np.arange(DAYS)/90*np.pi)
hod15 = (np.arange(T) % 96)/4.0
w = np.repeat(wd, 96) + 0.6*np.exp(-0.5*((hod15-15.5)/3.0)**2)
w = (w - w.min())/(w.max()-w.min()+1e-9)
Tamb = 16 + 20*w          # 16..36 C summer quarter

# ---- C57.91 thermal model ----
DT = 0.25                  # hours per step
TAU_TO = 3.0               # top-oil time constant (h), small ONAN units
R_LOSS, N_EXP, M_EXP = 5.0, 0.8, 0.8
DTO_R, DHS_R = 55.0, 25.0  # rated top-oil rise, HS-over-TO rise (65C-rise design)
PF = 0.95
NORMAL_LIFE_H = 180000.0

def aging(labels, extra_kw=None):
    """Returns per-transformer: pct life/yr, peak K, avg K."""
    agg = np.zeros((T, nx))
    for g in range(nx):
        mem = np.where(labels == g)[0]
        if len(mem): agg[:, g] = P[:, mem].sum(1)
    if extra_kw is not None: agg = agg + extra_kw
    K = (agg/PF) / KVA[None, :]
    dto_ult = DTO_R * (((K**2)*R_LOSS + 1)/(R_LOSS + 1))**N_EXP
    dto = np.empty_like(dto_ult)
    dto[0] = dto_ult[0]
    a = 1 - np.exp(-DT/TAU_TO)
    for s in range(1, T):
        dto[s] = dto[s-1] + a*(dto_ult[s] - dto[s-1])
    ths = Tamb[:, None] + dto + DHS_R * K**(2*M_EXP)
    faa = np.exp(15000/383.0 - 15000/(ths + 273.15))
    eq_hours = (faa*DT).sum(0)
    pct_per_yr = eq_hours/NORMAL_LIFE_H * (365.0/DAYS) * 100
    return pct_per_yr, K.max(0), K.mean(0), ths.max(0)

res = {}
for name, lab in [("true", true_g), ("records", rec_g), ("corrected", corr_g)]:
    res[name] = aging(lab)
    p, kpk, kav, hs = res[name]
    print(f"{name:9s}: median %life/yr {np.median(p):.3f}  p95 {np.quantile(p,.95):.2f}  "
          f"max {p.max():.1f}  units>4%/yr {(p>4).sum()}  peakK>1 {(kpk>1).sum()}")

# ---- risk ranking distortion ----
for topN in (25, 50):
    t_true = set(np.argsort(-res["true"][0])[:topN])
    t_rec = set(np.argsort(-res["records"][0])[:topN])
    t_cor = set(np.argsort(-res["corrected"][0])[:topN])
    print(f"top-{topN} risk overlap with truth: records {len(t_true&t_rec)}/{topN}, "
          f"corrected {len(t_true&t_cor)}/{topN}")

# ---- EV hosting capacity (on corrected map) ----
night = ((hod15 >= 21) | (hod15 < 2))
lab = corr_g
members = [np.where(lab == g)[0] for g in range(nx)]
host = np.zeros(nx, int)
base_p, base_kpk, _, _ = res["corrected"]
for g in range(nx):
    if len(members[g]) == 0: continue
    for k in range(0, 9):
        extra = np.zeros((T, nx)); extra[night, g] = 7.2*k
        # cheap local recompute for this transformer only
        aggg = P[:, members[g]].sum(1) + extra[:, g]
        K = (aggg/PF)/KVA[g]
        dtu = DTO_R*(((K**2)*R_LOSS+1)/(R_LOSS+1))**N_EXP
        d = np.empty(T); d[0] = dtu[0]
        a = 1-np.exp(-DT/TAU_TO)
        for s in range(1, T): d[s] = d[s-1] + a*(dtu[s]-d[s-1])
        ths = Tamb + d + DHS_R*K**(2*M_EXP)
        pct = (np.exp(15000/383.0-15000/(ths+273.15))*DT).sum()/NORMAL_LIFE_H*(365/DAYS)*100
        if pct > 4.0 or K.max() > 1.5:
            host[g] = k-1 if k > 0 else 0
            break
    else:
        host[g] = 8
occupied = np.array([len(m) > 0 for m in members])
print(f"\nEV hosting (corrected map): 0 EVs: {(host[occupied]==0).sum()}, "
      f"1-2: {((host[occupied]>=1)&(host[occupied]<=2)).sum()}, "
      f"3+: {(host[occupied]>=3).sum()} of {occupied.sum()} occupied units")

out = pd.DataFrame(dict(
    xfmr=xfmrs, kva=KVA, x=XY[:, 0], y=XY[:, 1],
    n_meters=[len(m) for m in members],
    pct_life_yr_true=res["true"][0], pct_life_yr_records=res["records"][0],
    pct_life_yr_corrected=res["corrected"][0],
    peakK=res["corrected"][1], avgK=res["corrected"][2],
    peak_hotspot_C=res["corrected"][3], ev_headroom=host))
out.to_csv("/home/user/xfmr/thermal_results.csv", index=False)
print("saved thermal_results.csv")
