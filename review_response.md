# Adversarial review — 8 criticisms, tested

Run Aug 26, 2026 on ckt5, frozen v1 pipeline. Scripts: `review_probe.py`, `review_probe2.py`,
`review_probe3.py`. Verdicts are CONFIRMED (real defect), PARTLY VALID (real in principle,
bounded in practice), or REFUTED (tested, doesn't hold).

---

### 1. "The GPS prior is centered on the truth" — PARTLY VALID, defused

Correct as stated: `mxy = XY[true_g] + N(0, 0.35·spacing)` places every premise symmetrically
around its true transformer, so the true transformer is always the *expected* nearest. Real
premises can sit systematically closer to a neighbor (corner lots, back-lot service).

Tested by pulling a fraction of premises 60% of the way toward a random neighboring transformer:

| biased premises | nearest-1 correct | containment (K=5) | corr_fix | obs_gate |
|---|---|---|---|---|
| 0% | 0.866 | 1.000 | 0.919 | 0.923 |
| 10% | 0.806 | 0.998 | 0.919 | 0.923 |
| 25% | 0.719 | 0.995 | 0.913 | 0.924 |
| 50% | 0.575 | 0.992 | 0.907 | 0.924 |

The pipeline never uses nearest-1; it uses "is the truth in the top 5." Even with half the
premises badly biased — nearest-1 collapsing from 87% to 58% — containment stays at 99.2%,
correction loses 1.2 points, and the gate does not move at all. The modeling assumption is real
but the results do not rest on it. (In deployment this model is replaced by the utility's actual
premise coordinates.) **Action: add this table to the paper's disclosure paragraph.**

### 2. "Empty transformers don't exist in the index" — CONFIRMED, and it matters for v2

Verified: 584 of 584 indexed transformers serve ≥1 meter; the index is built from
`meta.xfmr.unique()`, so genuinely unmetered transformers (spares, streetlights, decommissioned
units in GIS) cannot exist. For v1 this is harmless — candidates come from GPS-nearest indexed
transformers either way. For the **block detector it was a potentially fatal shortcut**: it
assigns a migrated group to "the nearest EMPTY transformer," and in the sim the only empty
transformers are ones the corruption itself emptied, making "empty" a near-perfect tell.

Tested by injecting phantom always-empty transformers (jittered copies of real positions):

| decoy empties | net gain, 3 seeds |
|---|---|
| 0% | +2.9 / +1.6 / +3.3 pts |
| 10% | +2.9 / +1.6 / +3.3 |
| 25% | +2.9 / +1.3 / +3.1 |
| 50% | +2.9 / +1.4 / +3.3 |

Flag precision stays 1.00 throughout; worst case loses 0.3 points. The migrated cluster's premise
centroid is precise enough that the right empty target still wins against decoys. **Criticism was
correct to raise and the module survives it — but this test now belongs in the v2 record, and the
decoy generator should stay in the harness permanently.**

### 3. "Singleton FPR is unmeasured" — CONFIRMED as a gap; now measured

Direct measurement (clean meters only, so these are pure false positives):

| seed | recorded singletons | FPR singletons | FPR non-singletons | share of all FPs that are singletons |
|---|---|---|---|---|
| 7 | 149 (10.8%) | 0.212 | 0.065 | 30% |
| 17 | 147 (10.7%) | 0.257 | 0.040 | 46% |
| 27 | 144 (10.4%) | 0.225 | 0.052 | 36% |

Singletons are ~11% of meters and produce 30–46% of all false positives, at a 4–5× elevated rate.
This is a sharper statement than the paper currently makes and it *strengthens* the motivation for
the coupling test. **Action: report per-stratum FPR explicitly rather than a pooled number.**

### 4. "The coupling z-score isn't a z" — CONFIRMED, language is wrong

`coupling_z` multiplies a mean correlation by √(T·n_members), which assumes T independent samples.
Measured lag-1 autocorrelation of the residuals: voltage ρ=0.236, load ρ=0.679. Bartlett-style
effective sample size is ~5,300 (voltage) and ~1,650 (load) against T=8,640. A reported "z = −8"
is roughly **3.5σ**, not 8σ.

This does not change the 15% recovery result — THETA was chosen empirically and the statistic is
still a valid *ranking* score — but the name and any σ interpretation are indefensible.
**Action: rename to `coupling_score`, drop σ language, state the threshold as empirical. Must be
fixed before this appears in paper 2.**

### 5. "Measurement noise is frozen across multi-seed runs" — CONFIRMED

`degrade()` defaults to `seed=11`, and `audit.py` computes `corr_matrix(degrade(V))` **once**,
outside the seed loop. All five "seeds" vary only corruption and GPS jitter; every run shares one
AMI noise realization. Re-run with one noise draw per seed:

| | det | fpr | corr_fix | obs_gate |
|---|---|---|---|---|
| frozen (as published) | 0.9036 ± 0.0260 | 0.0762 ± 0.0074 | 0.9301 ± 0.0327 | 0.9211 ± 0.0057 |
| varied | 0.9066 ± 0.0336 | 0.0771 ± 0.0074 | 0.9350 ± 0.0336 | 0.9218 ± 0.0033 |

Means move ≤0.5 pt and *upward*; the published numbers are the conservative ones, so nothing is
overclaimed. Detection variance widens (±0.026 → ±0.034), so the published error bar on `det` is
mildly understated. **Action: vary the noise seed in all future runs and say so; note in the repo
that published figures used a fixed noise draw.**

### 6. "Calibrated p means something narrower than it looks" — VALID, no test needed

Correct. The isotonic calibration yields P(proposal correct | features, *this corruption process,
this 10% corruption rate, this simulation's noise model*). Deployment base rates are unknown and
will differ, and a calibrated probability is only transportable if the base rate transports. The
ckt5→ckt24 transfer shows robustness across *feeders*, not across *corruption processes* — both
sides of that test share the same corruption and degradation model.
**Action: state the conditioning explicitly, and in deployment recalibrate on the 40 field checks —
which we are collecting anyway. That turns the criticism into a design feature.**

### 7. "Verification budget accounting" — PARTLY VALID

For v1 the accounting is conservative: 40 verifications are sampled uniformly across the whole
ranked list, but only those above the stop point inform the decision, so we charge for more than
we use. The real gap is the v2 stack — block-detector proposals and coupling-test verdicts would
each need their own field verification, and no additional budget was ever allocated for them;
likewise deployment recalibration (item 6) consumes verification data.
**Action: publish a per-module verification budget, not one global "40".**

### 8. "The corruption model may be too easy" — REFUTED

Hypothesis was that scattering errors uniformly over the 15 nearest transformers is easier than
real errors, which concentrate on immediate neighbours. Tested by restricting corruption:

| corruption target | det | fpr | corr_fix | obs_gate | blind |
|---|---|---|---|---|---|
| nearest-15 (published) | 0.904 | 0.076 | 0.930 | 0.921 | 0.912 |
| nearest-5 | 0.917 | 0.075 | 0.946 | 0.920 | 0.914 |
| nearest-3 | 0.924 | 0.075 | 0.956 | 0.921 | 0.915 |
| nearest-1 | 0.909 | 0.067 | 0.959 | 0.919 | 0.923 |

Tightening the corruption makes results **better**, not worse — correction accuracy rises from
0.930 to 0.959. The published model is the most conservative of the four. (Plausible mechanism,
offered as hypothesis not finding: a meter mis-recorded onto a distant transformer is a larger
outlier inside that group and creates more collateral confusion for its new group-mates.)
The genuinely harder corruption structure is *correlated* error — whole groups moving together —
which is the block scenario, already tested, and which v1 does fail (det 13.5%, corr 0%). That
failure is documented and is what the block detector addresses.

---

## Scoreboard

| # | Criticism | Verdict |
|---|---|---|
| 1 | GPS prior centered on truth | Partly valid — bounded, 1.2 pt worst case, gate unaffected |
| 2 | No empty transformers in index | **Confirmed** — harmless for v1, was a real risk for the block detector; survives decoy test |
| 3 | Singleton FPR unmeasured | **Confirmed gap** — now measured: 4–5× elevated, 30–46% of all FPs |
| 4 | Coupling z isn't a z | **Confirmed** — "8σ" is really ~3.5σ; must rename before publication |
| 5 | Noise frozen across seeds | **Confirmed** — impact ≤0.5 pt and in the conservative direction |
| 6 | Calibrated p is narrower than it looks | Valid — fix by recalibrating on field checks |
| 7 | Verification budget accounting | Partly valid — v1 conservative, v2 stack unbudgeted |
| 8 | Corruption model too easy | **Refuted** — tighter corruption scores better; published setup is the conservative one |

Nothing here invalidates a published number. Two items (4 and 5) are presentation defects that
would embarrass us in review; two (3 and 7) are gaps that make the work better once filled; one
(2) was a genuine hidden dependency in a v2 module that survived its stress test.
