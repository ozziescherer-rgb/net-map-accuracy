# Known limitations and review notes

This file exists because the fastest way to lose an engineer's trust is for them to find
something you knew about and did not say. Everything below was found by adversarial review of
this repository (Aug 26, 2026) and tested; the probe scripts are in `v2-experiments/`.

None of these change a number reported in the preprint (DOI 10.5281/zenodo.22063360). Two are
corrections to how results were computed or described; the rest are scope statements.

## Corrections applied to this repository

**AMI noise was frozen across the multi-seed runs.** `audit.py` originally computed the degraded
correlation matrix once, outside the seed loop, and `degrade()` defaults to a fixed seed — so the
five reported "seeds" varied corruption and premise-GPS jitter but shared a single measurement
noise realization. Each seed now draws its own. Measured effect on ckt5: means move by at most
0.5 points and in the favorable direction (correction accuracy 0.930 -> 0.935), so the published
figures are the conservative ones; the standard deviation on detection recall widens from
+/-0.026 to +/-0.034.

**The singleton coupling statistic was described as a z-score.** It scales a mean correlation by
sqrt(T * n), which assumes independent samples. The residual load series has lag-1 autocorrelation
of about 0.68 (voltage about 0.24), giving an effective sample size near 1,650 against T = 8,640.
A value of -8 corresponds to roughly 3.5 sigma, not 8. The function is renamed `coupling_score`,
sigma language is removed, and its threshold is stated as what it always was: empirical. This
affects presentation only; the reported 15% singleton recovery at ~97% specificity stands.

## Scope statements (unchanged, now explicit)

**Premise coordinates are modeled, not real.** Each meter's premise is placed at its true
transformer's location plus Gaussian noise (0.35 x transformer spacing), so the true transformer
is always the expected nearest. Real premises can sit systematically closer to a neighbour.
Tested by pulling a fraction of premises 60% of the way toward a random neighbouring transformer:
with half the premises biased, nearest-transformer accuracy falls from 0.866 to 0.575, but top-5
containment holds at 0.992, correction accuracy loses 1.2 points, and the verification gate does
not move. The method uses containment, not nearest-1. In deployment this model is replaced by the
utility's own premise coordinates.

**Every indexed transformer serves at least one meter.** The transformer index is built from the
meter metadata, so unmetered transformers (spares, streetlights, decommissioned units that persist
in a real GIS) do not exist in these benchmarks. This does not affect the core pipeline. It did
matter for the additive block detector, which assigns a migrated group to the nearest *empty*
transformer: in simulation the only empty transformers are ones the corruption itself emptied.
Tested by injecting phantom always-empty transformers as decoys at 10/25/50% of the real
transformer count: flag precision stays at 1.00 and the worst-case net gain falls by 0.3 points.

**False positives are concentrated in singletons.** Meters recorded as the sole customer on their
transformer are about 11% of meters but carry a false-positive rate of 0.21-0.26 against
0.04-0.07 for the rest, and account for 30-46% of all false positives. Pooled false-positive rates
in the preprint understate this concentration; per-stratum rates are the honest presentation.

**Confidence calibration is conditional on the simulated corruption process.** The calibrated
probabilities in `calib.py` estimate P(proposal correct | features, this corruption model, this
10% corruption rate). The ckt5 -> ckt24 result demonstrates transfer across feeders, not across
corruption processes; both sides share the same corruption and degradation models. Deployment use
requires recalibration on field-verified samples.

**The verification budget of ~40 covers the core pipeline only.** Additive modules (block
detector, coupling test) generate proposals that would require their own field verification, and
deployment recalibration consumes verified samples as well. Budgets should be stated per module,
not as a single global figure.

**Corruption model.** Errors are drawn uniformly from a transformer's 15 nearest neighbours.
Restricting corruption to the nearest 5, 3, or 1 makes results *better* (correction accuracy rises
from 0.930 to 0.959), so this is the most conservative of the variants tested. The harder
structure is correlated error - whole groups mis-recorded together - which the core pipeline does
fail (detection 13.5%, correction 0%); that failure is documented and is what the block detector
in `v2-experiments/` addresses.

**Everything here is simulation.** No real utility data has been processed. Field validation is
the point of the current pilot program.
