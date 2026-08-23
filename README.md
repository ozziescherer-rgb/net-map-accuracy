# Net Map Accuracy

Code for the preprint **"Net Map Accuracy: Why Meter-to-Transformer Correction Must Be
Evaluated Under Application"** (O. Scherer, 2026).

Utilities' meter-to-transformer records are ~5-20% wrong. Tools that fix them from AMI
voltage correlation are graded on detection/correction accuracy — but every false-positive
"correction" breaks a record that was right. This repo demonstrates, on public EPRI feeder
models, that a pipeline beating the published state of the art (93% vs 51-56% correction)
still *reduces* net record accuracy when applied blindly at realistic error rates, and that
confidence-gated application with ~40 field verifications recovers most of the achievable
value with a bounded worst case.

Patent pending (US provisional filed Aug 2026).

## Setup

```bash
pip install -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/dss-extensions/electricdss-tst
cd electricdss-tst && git sparse-checkout set Version8/Distrib/EPRITestCircuits && cd ..
```

Edit `lib.py` `CKT` path if your checkout lives elsewhere (default expects
`electricdss-tst/Version8/Distrib/EPRITestCircuits/ckt5`).

## Reproducing the paper

Run in order (times on a laptop-class machine):

| Step | Script | Produces | ~Time |
|---|---|---|---|
| 1 | `run_powerflow.py` | meta.csv + 14d baseline voltages (ckt5) | 10 s |
| 2 | `run_90d.py` | 90 days @5-min -> true 15-min averages, ckt5 (V15_90d, P15_90d) | 3 min |
| 3 | `run_ckt24.py` | 56 days @15-min, ckt24 (V24, meta24) | 2 min |
| 4 | `audit.py` | headline multi-seed results + leak-free length curve (paper §4.1-4.3) | 5 min |
| 5 | `harden2.py` | 5%/20% corruption sweeps + gate-variance study (§4.3) | 3 min |
| 6 | `thermal.py` | C57.91 aging, hosting capacity, ranking-fidelity (§4.6) | 1 min |
| 7 | `premise_gps.py` | premise-GPS quality sweep (§4.5) | 2 min |
| 8 | `calib.py` | simulation-pretrained confidence calibration (§6 preliminary) | 3 min |
| 9 | `singleton_solve.py` | directional load-coupling singleton test (§6 preliminary) | 3 min |

Earlier exploratory scripts (`cluster.py`, `degrade.py`, `compare.py`, `improved.py`,
`iterate.py`, `loadsig.py`, `gate.py`, `sens.py`, `score_ckt24.py`,
`length_frozen_singleton.py`, `run_1min.py`) are retained for provenance; the paper's
reported numbers come from steps 4-9, which use leak-free candidate construction.

## Data

All inputs are public: EPRI ckt5/ckt24 circuit models from the official OpenDSS test-case
repository. Load profiles are synthesized (generators embedded in the run scripts,
deterministic seeds). No utility data is included. If you are a utility interested in
field validation — the necessary next step named in the paper — contact the author:
ozziescherer@gmail.com.

## License

MIT. If you use this work, please cite the preprint.
