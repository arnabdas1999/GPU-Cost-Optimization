# Experiment: synthetic-paired-demo

Status: **synthetic demo**

> **SYNTHETIC** — runs used the scripted model client; no GPU-efficiency claim follows.

This is a saved example of an A/A calibration report. The referenced raw run directories
are local artifacts and are not included in the repository.

## Decision: A/A CALIBRATION (window spread 11.2%)

Trials: 2. Prototype decision rule over a small sample; not a statistical test of equivalence.

| Check | Result | Detail |
| --- | --- | --- |
| comparisons are valid | pass | protected settings and cohorts match |
| candidate differs from baseline | FAIL | no tunable setting differs: this is an A/A calibration; use its spread as the noise floor |
| enough paired trials | pass | 2 trial(s), 2 required |
| pooled successes | pass | baseline 16, candidate 16 (max drop 0) |
| task-level regressions | pass | 0 regressed task(s) (tolerance 0 per task, max 0) |
| infrastructure errors | pass | baseline 0, candidate 0 |
| wall-time backstop hits | pass | baseline 0, candidate 0 |
| pooled allocated-window reduction | pass | 7.5% (baseline 48.1s, candidate 44.5s; required 5.0%) |
| reduction in every paired trial | pass | 3.8%, 10.8% |
| reduction exceeds trial-to-trial spread | FAIL | reduction 7.5% vs largest within-config spread 11.2% |

## Pooled per-task passes

| Task | Baseline | Candidate |
| --- | --- | --- |
| config_env | 4 | 4 |
| config_merge | 0 | 0 |
| parse_duration | 4 | 4 |
| parse_port_range | 0 | 0 |
| retry_backoff | 4 | 4 |
| retry_permanent | 4 | 4 |

## Trial 1


- Baseline: `runs/20261024T024105Z-synthetic-a6a2`
- Candidate: `runs/20261024T024128Z-synthetic-admission-active-2-d469`

| Metric | Baseline | Candidate | Change |
| --- | --- | --- | --- |
| Allocated window (s) | 22.7 | 21.9 | -0.9 (-3.8%) |
| Window cost | 0.0095 | 0.0091 | -0.0004 (-3.8%) |
| Successes / valid / submitted | 8 / 12 / 12 | 8 / 12 / 12 | +0 |
| Infrastructure errors | 0 | 0 | +0 |
| Wall-time backstop hits | 0 | 0 | +0 |
| p90 episode service time (s) | 6.95 | 6.62 | -0.3 (-4.7%) |

Episode transitions (baseline → candidate, matched by task and repetition):

| Transition | Episodes |
| --- | --- |
| fail->fail | 4 |
| pass->pass | 8 |

| Task | Episodes | Baseline passes | Candidate passes |
| --- | --- | --- | --- |
| config_env | 2 | 2 | 2 |
| config_merge | 2 | 0 | 0 |
| parse_duration | 2 | 2 | 2 |
| parse_port_range | 2 | 0 | 0 |
| retry_backoff | 2 | 2 | 2 |
| retry_permanent | 2 | 2 | 2 |

## Trial 2


- Baseline: `runs/20261024T024213Z-synthetic-86c9`
- Candidate: `runs/20261024T024150Z-synthetic-admission-active-2-38f2`

| Metric | Baseline | Candidate | Change |
| --- | --- | --- | --- |
| Allocated window (s) | 25.4 | 22.7 | -2.8 (-10.8%) |
| Window cost | 0.0106 | 0.0094 | -0.0011 (-10.8%) |
| Successes / valid / submitted | 8 / 12 / 12 | 8 / 12 / 12 | +0 |
| Infrastructure errors | 0 | 0 | +0 |
| Wall-time backstop hits | 0 | 0 | +0 |
| p90 episode service time (s) | 8.09 | 7.36 | -0.7 (-9.1%) |

Episode transitions (baseline → candidate, matched by task and repetition):

| Transition | Episodes |
| --- | --- |
| fail->fail | 4 |
| pass->pass | 8 |

| Task | Episodes | Baseline passes | Candidate passes |
| --- | --- | --- | --- |
| config_env | 2 | 2 | 2 |
| config_merge | 2 | 0 | 0 |
| parse_duration | 2 | 2 | 2 |
| parse_port_range | 2 | 0 | 0 |
| retry_backoff | 2 | 2 | 2 |
| retry_permanent | 2 | 2 | 2 |
