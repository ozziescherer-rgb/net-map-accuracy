"""Does true 15-min interval averaging destroy the signal? Plus FPR margin sweep + data-length curve."""
import numpy as np, pandas as pd
from lib import load_meta, corr_matrix, experiment, degrade

meta, true_g, nearest = load_meta()
V1 = np.load("/home/user/xfmr/V_1min.npy")   # (40320, n)
STEPS, n = V1.shape

V15avg = V1.reshape(-1, 15, n).mean(axis=1)            # true AMI: interval average
V15inst = V1[::15]                                     # what we simulated before

rows = []
def run(name, V, **kw):
    r = experiment(corr_matrix(V), true_g, nearest, **kw)
    rows.append(dict(scenario=name, **{k: round(float(v), 3) for k, v in r.items()}))
    print(f"{name:42s} det={r['detection_recall']:.3f} fpr={r['false_positive_rate']:.3f} "
          f"corr={r['correction_acc']:.3f} corr_fix={r['correction_acc_nonsingleton']:.3f}", flush=True)

print("== resolution & averaging (clean) ==")
run("1-min clean (28d)", V1[::3])            # subsample x3 for tractable ranking cost
run("15-min instantaneous (28d)", V15inst)
run("15-min TRUE AVERAGE (28d)", V15avg)

print("\n== degraded (0.5V quant, 0.3V noise, 5% missing) ==")
run("15-min instantaneous degraded", degrade(V15inst))
run("15-min TRUE AVERAGE degraded", degrade(V15avg))

print("\n== data length (true-average, degraded) ==")
for days in (7, 14, 28):
    k = days*96
    run(f"avg degraded {days}d", degrade(V15avg[:k]))

print("\n== FPR margin sweep (true-average degraded, 28d) ==")
Vd = degrade(V15avg)
C = corr_matrix(Vd)
for margin in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
    r = experiment(C, true_g, nearest, margin=margin)
    rows.append(dict(scenario=f"margin={margin}", **{k: round(float(v), 3) for k, v in r.items()}))
    print(f"margin={margin:<6} det={r['detection_recall']:.3f} fpr={r['false_positive_rate']:.3f} "
          f"corr={r['correction_acc']:.3f}", flush=True)

pd.DataFrame(rows).to_csv("/home/user/xfmr/compare_results.csv", index=False)
