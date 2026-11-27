# HiME-2 implementation status (plan v2, 20 Nov 2026)

Written on a dev machine; executed on the WSL machine. All test layers for Phases 0 and 2-7 and the
smoke run pass there on the synthetic oracle world (24 Nov 2026).

## Run on the execution machine
```
bash scripts/setup_wsl.sh      # once: apt packages, uv, clone, Python 3.12, deps
bash scripts/run_tests.sh      # all test layers + smoke; logs in runs/test_logs/
```
Set `HIME_DATA=/path/to/release` to enable the real-data tests.

## Gates
| Gate | Status | Note |
|---|---|---|
| A engineering | Passed (oracle world) | Tests for Phases 0, 2-7 and the smoke run pass on WSL; real-data layer Blocked |
| B fidelity | Partially verified | Proposition tests exist for 1, 2, 3', 4', 9, 10, 11, Lemma 1', 7', 8, 9', Theorem 1, Theorem 2 (uncapped) |
| C reproduction | Blocked | No dataset; baselines unimplemented |
| D independent | Not attempted | Never self-awarded |

## Phases
| Phase | Status |
|---|---|
| 0 Scaffold, RNG, provenance | Written. Claim/ambiguity/source ledgers folded into this file |
| 1 Data layer, imagery audit | Contracts Passed on WSL (24 Nov 2026). Imagery audit Blocked: comparison chips absent |
| 2 Oracle world | Written (`hime/oracle.py`) |
| 3 Features, experts | Written |
| 4 Fusion | Written. D7 missing-chip offset not attempted |
| 5 Placement | Written, incl. exact MILP and H6 ratio dump to `runs/` |
| 6 Participation | Written. O5 capped-case evidence printed by test |
| 7 Engine, hierarchy, bytes | Written. Smoke script in `scripts/smoke.py` |
| 8 Baselines B1-B12 | Waiting for PDFs listed in docs/PAPERS.md |
| 9 V0 grid | DINOv2 grid built; 9.1 leave-fold-out selection run (DINOv3 pending licence). K/M/C calibration and tuning not yet written |
| 10-12 | Blocked on baselines |

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
7. E_S spline columns are standardised with exact federated statistics. Monotonicity is unchanged
   (positive rescaling); the ridge now acts on rescaled coefficients.
8. MMI_shape is undefined in every source the release checked, so E_S uses six features, all monotone.
9. Imagery audit compares against every dated 2024 layer (latest wins), not only the 17 Jan layers.
10. Initial UAV positions are k-means centroids of non-direct client medoids.
11. The oracle's seismic features are Gaussian, not ShakeMap values; monotone constraints apply to blocks 0-5.

## Evidence from the first WSL run (24 Nov 2026, oracle world only)
| Check | Result |
|---|---|
| Theorem 2 when units are capped (O5) | 157 capped instances, dominance never violated |
| Greedy / exact optimum (H6, 200 instances) | min 0.73, mean 0.97; bound 0.5 holds |
| Stacker w_SI with assumption (a) violated | 0.64 (0 expected when (a) holds) |
| Drift correction at kappa = 4 (H10) | distance 2.919 vs 2.935 without: small here |
| Controlled convergence | Objective gap closed: E_S 96.5%, E_I 99.95% in 80 rounds; distance falls every evaluation |
| Seismic column standardisation | E_S start distance 9.61 to 1.24; gap ratio 0.23 to 0.035 |

## First real-data run (24 Nov 2026, seed 1000, N = 100, validation fold)
| Check | Result |
|---|---|
| Contracts | Counts, has_pre = 0, buffer, folds, embeddings, 63/100 single-class clients reproduced; all post chips `ok`; no non-positive intensities |
| Val AUPRC (prevalence 0.027) | HiME-2 0.267; central ensemble 0.285; central E_I 0.292; central E_S 0.048 |
| Placement | r = 1,397 m; K hit the cap of 30 with 38% coverage, so the 70% rule cannot be met at L_max 120 dB |
| Embedding throughput (A4500) | DINOv2 ViT-S 1,272 img/s; ViT-B 228 img/s; DINOv3 blocked by gated weights |
| Imagery audit | Blocked: dated 2024 and 2022 chips absent on WSL |

## V0 selection, all four encoders (26 Nov 2026, seeds 1000-1004, validation folds, GPU Newton fits)
Supersedes the DINOv2-only run of 25 Nov (scipy fits).

| Test fold | Chosen cell | r | lambda | val AUPRC (mean ± SE) |
|---|---|---|---|---|
| 0 | dinov3_vitb16 cls+mean d8 chip | 64 | 0.01 | 0.265 ± 0.053 |
| 1 | dinov3_vitb16 cls+mean d8 chip | 64 | 0.01 | 0.433 ± 0.118 |
| 2 | dinov3_vitb16 cls+mean d8 chip | 64 | 0.01 | 0.436 ± 0.117 |
| 3 | dinov3_vitb16 cls+mean d8 chip | 64 | 0.01 | 0.392 ± 0.148 |
| 4 | dinov3_vitb16 cls+mean d8 ctx | 64 | 0.01 | 0.291 ± 0.063 |

- DINOv3 ViT-B wins every fold; best mean 0.366 against 0.345 for the best DINOv2 cell. The satellite ViT-L is best at 0.353, not first.
- r = 64 and lambda = 0.01 are again at the edges of their grids.
- GPU Newton fits agree with scipy to 1e-6 (gpu_fit test, CPU and CUDA).

## Imagery audit (26 Nov 2026)
43,551 buildings and 171 blocks flagged: their seamlessphoto chip is closer to nendophoto2022 than to their latest dated 2024 chip. Not yet acted on; `world(exclude_blocks=...)` exists for excluding them.

## UAV coverage sweep (26 Nov 2026, N = 100, seeds 1000-1004, validation fold)
Coverage is that of the initial k-means placement. Gap = HiME-2 minus central AUPRC.

| Path loss (dB) | UAV cap | Mean K | Mean coverage | HiME-2 AUPRC | Mean gap | Worst gap |
|---|---|---|---|---|---|---|
| 117 | 30 | 30 | 0.44 | 0.267 | -0.018 | -0.034 |
| 117 | 40 | 37 | 0.52 | 0.257 | -0.028 | -0.120 |
| 117 | 60 | 38 | 0.53 | 0.258 | -0.027 | -0.120 |
| 120 | 30 | 30 | 0.49 | 0.270 | -0.014 | -0.050 |
| 120 | 60 | 32 | 0.51 | 0.291 | +0.006 | -0.019 |
| 123 | 30 | 23 | 0.47 | 0.294 | +0.010 | -0.017 |
| 126 | 30 | 19 | 0.52 | 0.290 | +0.006 | -0.010 |

- No setting reaches 70% coverage at the initial placement.
- From 120 dB with 60 UAVs upwards, HiME-2 matches central within seed noise, with fewer UAVs as the path-loss limit rises. Five seeds cannot separate those settings.

## Deviations from the plan (decided 26 Nov 2026)
- Path-loss limit 123 dB instead of 117 dB (120 dB minus 3 dB margin), chosen from the coverage sweep above.
- V0 grids extended: PCA r adds 16 and 32; lambda adds 0.1 and 1, because r = 64 and lambda = 0.01 sat at the grid edges.
- V0 1-SE rule uses a paired SE (per-seed differences from the best cell) instead of the across-seed SE, which mostly measured fold difficulty.
- Chips the imagery audit flags (pre-event-looking) are kept; a 0/1 flag is added as a feature of the image expert and results will be reported split by flag.
