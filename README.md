Net Map Accuracy
Benchmark and evaluation code for correcting utilities' meter-to-transformer records from
smart-meter (AMI) voltage data, with the correction evaluated after application (net map
accuracy) rather than by detection metrics alone.
Preprint: Net Map Accuracy: Why Meter-to-Transformer Correction Must Be Evaluated Under
Application (Zenodo, DOI 10.5281/zenodo.22063360).
Code is MIT-licensed (see `LICENSE`). Everything here is simulation; no utility data has been
processed and no field validation has taken place.
> **Correction, October 2026.** The v1 preprint and earlier versions of this README compared the
> method only against an NREL study and described its correction rate as exceeding "the published
> state of the art." That comparison was incomplete. Blakely and Reno (Sandia National
> Laboratories) published a correlation-plus-regression method for the same problem on the same
> EPRI test feeder in 2021, and Sandia distributes it as open source. In a head-to-head on this
> repository's benchmark, the Sandia method, with sensible preprocessing and a tuned threshold,
> achieves **higher net map accuracy than the method here at every corruption rate tested, with no
> field checks**. The engine in this repository is therefore not state of the art. What survives is
> the evaluation frame: by conventional detection and correction metrics the method here looks
> stronger, and by net map accuracy it is weaker, which is the preprint's central argument made
> against its own engine. Details, numbers, and caveats are in [`CORRECTION.md`](CORRECTION.md).
> The preprint will be revised to cite the prior work and report the comparison.
What the problem is
A utility's records of which distribution transformer serves which meter are often wrong, and a
wrong map corrupts every per-transformer analysis built on it (loading, thermal aging, hosting
capacity). Meters that share a transformer share a secondary conductor, so their voltages move
together; correlation of AMI voltage histories reveals which meters are mis-recorded. The question
this repository is built around is not whether such errors can be detected but whether applying a
tool's corrections leaves the map better or worse than it started, and how a utility would know
without ground truth.
What is in the repository
Benchmark. OpenDSS quasi-static time-series simulation of EPRI test circuits ckt5 (1,379
meters, 584 transformers; `run_powerflow.py`, `run_90d.py`) and ckt24 (`run_ckt24.py`), with
synthetic load profiles, injected record errors (`lib.experiment`), and AMI-style degradation
(0.5 V quantization, 0.3 V noise, 5% missing; `lib.degrade`). Premise coordinates are modeled
(`premise_gps.py`).
Method v1. Spearman correlation over all meter pairs, robust median/MAD scoring, a
GPS-constrained candidate ballot, two flag rules, and margin ranking (`lib.py`, `audit.py`).
Verification gate. A sequential stopping rule over about 40 field checks at randomized
ranks that stops applying corrections when observed precision falls below break-even
(`gate.py`, `audit.py`). The gate is useful for engines whose ranked lists have low precision; a
high-precision engine does not need it (see `CORRECTION.md`).
Stress tests and probes. Rooftop PV, regulator tap steps, geocoding error, service-drop
variation, history length, sparsity, correlated block errors, adversarial review probes
(`gauntlet2.py`, `review_probe*.py`, `resid_test.py`); write-ups in `review_response.md` and
`KNOWN_LIMITATIONS.md`. Post-publication additive modules are described in `V2_NOTES.md`.
Downstream. IEEE C57.91 thermal aging ranking on the corrected map (`thermal.py`), and
confidence calibration (`calib.py`).
Headline numbers (ckt5, 10% corruption, 5 seeds; see the preprint for scenarios)
Method v1: detection recall 0.90, false-positive rate 0.076, correction accuracy on fixable
errors 0.93. Gated net map accuracy 0.901 to 0.921 at 10% corruption and 0.951 to 0.960 at 5%,
where applying the same list without the gate gives 0.912 and 0.947.
For how these compare with the Sandia method on identical data, see `CORRECTION.md`. Read the
two files together; the numbers above are correct for what they measure and are not the whole
picture.
Reproducing the preprint
Clone this repository and, as a sibling directory, the OpenDSS test circuits:
```
   git clone https://github.com/ozziescherer-rgb/net-map-accuracy.git
   git clone https://github.com/dss-extensions/electricdss-tst.git
   ```
`lib.py` expects `../electricdss-tst/Version8/Distrib/EPRITestCircuits/ckt5` relative to the
repository; override with the environment variables `NMA_TST`, `NMA_CKT`, `NMA_CKT24`.
`pip install -r requirements.txt` (Python 3.10+, `opendssdirect.py`).
Build the simulated data: `python run_powerflow.py`, then `python run_90d.py` (90 days at
5-minute resolution, averaged to 15 minutes; this takes a while) and `python run_ckt24.py`
for the transfer feeder.
`python audit.py` reproduces the preprint's tables (`audit_ckt5.csv`, `audit_ckt24.csv`).
`audit.py` keeps the preprint's frozen measurement-noise realization on purpose;
`audit_varied_noise.py` draws fresh noise per seed (see `KNOWN_LIMITATIONS.md`).
Other scripts are self-describing by docstring and run from the repository root.
Prior work
L. Blakely and M. J. Reno, "Identification and Correction of Errors in Pairing AMI Meters and
Transformers," Power and Energy Conference at Illinois (PECI), 2021. SAND2021-2140C,
OSTI 1847226.
Sandia National Laboratories, distribution-system-model-calibration (`sdsmc`, BSD-3), including
`MeterTransformerPairing`: github.com/sandialabs/distribution-system-model-calibration.
NREL Technical Report NREL/TP-5C00-83877, the only comparison used in the v1 preprint.
Citation
Scherer, O. (2026). Net Map Accuracy: Why Meter-to-Transformer Correction Must Be Evaluated
Under Application. Zenodo. https://doi.org/10.5281/zenodo.22063360
Questions: ozziescherer@gmail.com
