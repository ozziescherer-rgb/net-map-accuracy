# v2 experiments (work in progress — NOT the published pipeline)

The published, frozen v1 pipeline lives in the repo root and matches the paper
(DOI: 10.5281/zenodo.22063360). Everything in this folder is post-publication
work: additive candidate modules and stress tests, each scored against the
frozen baseline on both its target scenario and a regression scenario. Nothing
here modifies v1; validated modules only ever add proposals through the same
conservative gate.

Status as of 2026-08-24 (all on EPRI ckt5, seeds (7,5),(17,15),(27,25) unless noted):

| File | What it is | Status |
|---|---|---|
| `gauntlet2.py` | Four pre-mortem stressors: heterogeneous service drops, block corruption, grid-snapped geocoding, regulator zone tap steps | Findings: drops ~1pt, geocoding ~0pt, tap steps ~2pt (gate holds), block corruption is a real blind spot for v1 (det 13.5%, corr 0%) |
| `run_hetero.py` | Re-simulation of ckt5 with lognormal 50–400 ft service drops (produces V15_90d_hetero.npy) | Done; input to gauntlet2a |
| `block_detector.py` | Additive block-error detector v1: bimodal eigenvector split of a recorded group's correlation matrix, migrated cluster reassigned to nearest empty transformer | Validated: 100% flag precision, recall 25–36%, net +1.5–3.3 pts on block scenario; silent on scattered and clean data |
| `block_detector2.py` | Adds displaced-group test (coherent group whose premise centroid sits ≥1.25× spacing from its recorded transformer and ≥0.75× spacing closer to an empty one) | Validated: precision 0.97–1.00, recall 40–44%, net +2.7–3.8 pts; still silent on scattered and clean data |
| `review_probe.py`, `review_probe2.py`, `review_probe3.py` | Adversarial review probes: frozen-noise variance, corruption-model hardness, singleton false-positive rate, biased premise coordinates, coupling-statistic effective sample size, decoy empty transformers | Results written up in `review_response.md`; corrections and scope statements in the root `KNOWN_LIMITATIONS.md` |
| `resid_test.py` | Local common-mode residualization vs tap steps | REJECTED — fails the baseline regression check (corr_fix −4 pts). Kept as documented dead end |

Promotion rule: a module ships to a field deployment only after (a) the failure
mode it targets is observed in real data, or (b) it passes the full multi-seed,
both-feeder treatment. Until then the deployable product is exactly v1.

Note: `calib.py`, `singleton_solve.py` and `thermal.py` live in the repository root, not here —
they were part of the original publication. `singleton_solve.py` was corrected in Aug 2026
(its coupling statistic was wrongly described as a z-score); see `KNOWN_LIMITATIONS.md`.

Scripts in this folder expect to be run from the repository root, e.g.
`python v2-experiments/block_detector.py`. They put the repo root on `sys.path` themselves and
read data via `lib.OUT`.
