# HiME-2 implementation status (plan v2, 20 Nov 2026)

Written on a dev machine without the dataset, without a GPU, and without running any code
(only a syntax compile). All tests are to be run on the execution machine.

## Run on the execution machine
```
bash scripts/setup_wsl.sh      # once: apt packages, uv, clone, Python 3.12, deps
bash scripts/run_tests.sh      # all test layers + smoke; logs in runs/test_logs/
```
Set `HIME_DATA=/path/to/release` to enable the real-data tests.

## Gates
| Gate | Status | Note |
|---|---|---|
| A engineering | Not attempted | Code and tests written, never executed |
| B fidelity | Not attempted | Proposition tests exist for 1, 2, 3', 4', 9, 10, 11, Lemma 1', 7', 8, 9', Theorem 1, Theorem 2 (uncapped) |
| C reproduction | Blocked | No dataset; baselines unimplemented |
| D independent | Not attempted | Never self-awarded |

## Phases
| Phase | Status |
|---|---|
| 0 Scaffold, RNG, provenance | Written. Claim/ambiguity/source ledgers folded into this file |
| 1 Data layer, imagery audit | Blocked: dataset and schema absent. Seed-role guard and TestVault written |
| 2 Oracle world | Written (`hime/oracle.py`) |
| 3 Features, experts | Written |
| 4 Fusion | Written. D7 missing-chip offset not attempted |
| 5 Placement | Written, incl. exact MILP and H6 ratio dump to `runs/` |
| 6 Participation | Written. O5 capped-case evidence printed by test |
| 7 Engine, hierarchy, bytes | Written. Smoke script in `scripts/smoke.py` |
| 8 Baselines B1-B12 | Not attempted: each needs its full paper read first (plan rule) |
| 9-12 V0 grid, tuning, freeze, runs, release | Blocked on data, baselines and GPU |

## Assumptions made where the plan or design v1 was not available (AMBIGUITY_LOG)
1. Design v1 Proposition 7 is not in hand. `schedule.kappas` minimises sum drift_m kappa_m^p subject to
   per-round backhaul sum 2 K P_m / kappa_m <= B_bh, then rounds up. Replace with the real form.
2. Design v1 section 9 step 1 (client-to-edge assignment) is not in hand. Clients go to the nearest covering UAV;
   capacity is enforced only through s_j <= C_j.
3. Al-Hourani suburban/urban constants are the commonly quoted ones; check against the full text.
4. Proposition 10 (kappa = 1 equals flat FL with correction) holds only when the edge projection is inactive.
   With the beta >= 0 projection on E_S, the average of projected edge models differs from the correction-free
   run. The test uses unconstrained experts; the plan should state this scope.
5. Byte model: fp32 payloads; probes 4 B per expert per available client; one model broadcast per active edge per
   round; backhaul per sync = up + down (+ correction down) per active UAV edge. Direct edge 0 is terrestrial
   and not counted on UAV backhaul. Check against design v1 Proposition 6.
6. Predicted importance for placement is the last probed a_i (staleness rule of design v1 section 10 not in hand).
7. The oracle's seismic features are Gaussian, not ShakeMap values; monotone constraints apply to blocks 0-5.

## Evidence from the first WSL run (24 Nov 2026, oracle world only)
| Check | Result |
|---|---|
| Theorem 2 when units are capped (O5) | 157 capped instances, dominance never violated |
| Greedy / exact optimum (H6, 200 instances) | min 0.73, mean 0.97; bound 0.5 holds |
| Stacker w_SI with assumption (a) violated | 0.64 (0 expected when (a) holds) |
| Drift correction at kappa = 4 (H10) | distance 11.012 vs 11.017 without: negligible here |
| Controlled convergence, E_S | distance falls every evaluation but slowly (9.61 to 8.51 in 80 rounds): ill-conditioned spline basis |
