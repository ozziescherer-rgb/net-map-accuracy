"""Phase 1 preview: degrade voltage data to AMI realism, re-score detection/correction."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from cluster import corr_matrix, experiment, V, meta  # reuse

PU = 1/240.0  # volts -> pu on 240 V base
rng = np.random.default_rng(11)

scenarios = {
    "clean (baseline)":              lambda V: V,
    "quantize 0.1 V":                lambda V: np.round(V/(0.1*PU))*(0.1*PU),
    "quantize 0.5 V":                lambda V: np.round(V/(0.5*PU))*(0.5*PU),
    "quantize 1.0 V":                lambda V: np.round(V/(1.0*PU))*(1.0*PU),
    "noise sd 0.1 V":                lambda V: V + rng.normal(0, 0.1*PU, V.shape),
    "noise sd 0.3 V":                lambda V: V + rng.normal(0, 0.3*PU, V.shape),
    "q0.1V + noise0.1V + 5% miss":   None,  # built below
    "q0.5V + noise0.3V + 5% miss":   None,
}

def with_missing(Vd, frac):
    Vd = Vd.copy()
    mask = rng.random(Vd.shape) < frac
    Vd[mask] = np.nan
    # rank/corr with nan: fill by column median (crude but standard first pass)
    med = np.nanmedian(Vd, axis=0)
    idx = np.where(mask)
    Vd[idx] = med[idx[1]]
    return Vd

def combo(q, sd, miss):
    def f(V):
        Vd = V + rng.normal(0, sd*PU, V.shape)
        Vd = np.round(Vd/(q*PU))*(q*PU)
        return with_missing(Vd, miss)
    return f
scenarios["q0.1V + noise0.1V + 5% miss"] = combo(0.1, 0.1, 0.05)
scenarios["q0.5V + noise0.3V + 5% miss"] = combo(0.5, 0.3, 0.05)

print(f"{'scenario':34s} {'det':>6s} {'fpr':>6s} {'corr':>6s} {'corr(fixable)':>13s}")
rows = []
for name, f in scenarios.items():
    Vd = f(V)
    C = corr_matrix(Vd, "spearman")
    r = experiment(C)
    print(f"{name:34s} {r['detection_recall']:6.3f} {r['false_positive_rate']:6.3f} "
          f"{r['correction_acc']:6.3f} {r['correction_acc_nonsingleton']:13.3f}")
    rows.append(dict(scenario=name, **r))
pd.DataFrame(rows).to_csv("/home/user/xfmr/degradation_results.csv", index=False)
