# Correction: prior art and a head-to-head comparison (October 2026)

## What was wrong

The v1 preprint (DOI 10.5281/zenodo.22063360), the landing page, and earlier versions of this
README stated that the method here corrects about 93% of fixable record errors "against a
published state of the art around 55%," that no prior work reported a false-positive rate, and
that field validation was in progress. Those statements rested on a single comparison (an NREL
technical report) and on a literature search that was never properly done. They are withdrawn.

Specifically:

- Prior work on this exact problem was not cited. Blakely and Reno (Sandia National Laboratories)
  published "Identification and Correction of Errors in Pairing AMI Meters and Transformers" at
  PECI 2021 (SAND2021-2140C, OSTI 1847226). It uses a correlation stage followed by a regression
  stage, on the same EPRI ckt5 test feeder used here, and reports grouping accuracy above 99% in
  its setting. Sandia distributes the implementation as open source (BSD-3) in the
  `distribution-system-model-calibration` package (`sdsmc.MeterTransformerPairing`), and the
  same group has published further work since.
- "No net harm" was stated as if it were a mathematical property of the gate. The preprint shows
  it empirically up to 10% error; at 20% one seed fell below its baseline. It is an empirical
  result over a stated range, not a guarantee.
- "Field validation in progress" and "first validations being negotiated" overstated the
  situation. The method was presented once to a utility engineering team. No data has been
  received and no validation has occurred.

## Head-to-head on this repository's benchmark

Setup: the 90-day ckt5 simulation from `run_90d.py`, identical corrupted records and degraded
voltages for both methods, held-out seeds. The method here ("v1") was run exactly as published and
reproduced the preprint's numbers. The Sandia `MeterTransformerPairing` code was run unmodified
except for (a) interpolating missing readings instead of median-filling them and (b) a correlation
threshold of 0.45 rather than the package default of 0.7. The threshold was chosen using ground
truth on development seeds and then confirmed on fresh seeds. Sandia's method uses the kW interval
channel as well as voltage; v1 uses voltage only.

Net map accuracy after applying each method's corrections:

| Corruption rate | Sandia (tuned, blind) | v1 (gated, ~40 checks) |
|---|---|---|
| 5%  | 97.8% | 96.1% |
| 10% | 94.7% | 91.9% |
| 20% | 87.0% | 81.7% |

(v1 gated figures are from the same held-out runs as the comparison and differ slightly from the
preprint's seed set, which gives 96.0, 92.1 and 81.2. Applying v1's list blind, without the gate,
gives 94.7, 91.2 and 80.7 on the preprint's seeds.)

Edit precision of the applied corrections was roughly 70 to 93% for Sandia versus about 50% for
v1's full list. Blind application of the Sandia method never left the map worse than the records
it started from at any rate tested, so the verification gate adds nothing on top of it. The gate
only helps an engine whose ranked list has low precision, which v1's does.

By conventional metrics the ranking is reversed: v1 shows higher detection recall (about 0.90
versus 0.78) and higher correction accuracy on detected errors (about 0.91 versus 0.66). That is
the preprint's thesis, that detection-style metrics can rank methods opposite to net map accuracy
after application, demonstrated against the preprint's own engine.

Where v1 was better: when the Sandia defaults were left in place, median-filled gaps and coarse
0.5 V voltage at the 0.7 threshold made the Sandia method produce few or no corrections, and v1
won those runs. That is a statement about preprocessing and tuning, not about the engines.

Caveats: ckt5 only; one corruption model (random reassignment to one of the 15 nearest
transformers), no correlated block errors; simulated data throughout; the tuned threshold was
selected with ground truth before being confirmed out of sample. The comparison script will be
added to this repository; until it is, treat the table as reported rather than reproducible.

## What this repository is still for

- A physics-based benchmark with controlled record corruption and AMI-realistic degradation, on
  public EPRI circuits, that any method can be scored on.
- The net-map-accuracy evaluation frame and the measured false-positive rate, which this
  comparison supports.
- The verification gate, as a tool for engines with lower-precision lists, and as a way to
  estimate precision in the field without ground truth.
- The stress tests, probes, and documented failure modes.

## What changes

- This file, the README notice, and `KNOWN_LIMITATIONS.md` are updated as of October 2026.
- The preprint will be revised (Zenodo new version) to cite the prior work, report the
  comparison, and remove the state-of-the-art and no-net-harm claims. Until then the v1 record
  stands as published and this file is the correction of record.
- The landing page no longer makes the 93% versus 55% comparison or solicits pilots on that
  basis.

Ozzie Scherer, ozziescherer@gmail.com
