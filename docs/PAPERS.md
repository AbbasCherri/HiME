# Papers still needed (Phase 8 and open theory items)

Put the PDFs in `docs/papers/` using the file names below. Identifiers were checked against Crossref and arXiv on 24 Nov 2026.

## Baselines (must be read in full before implementation)
| File | Paper | Venue | DOI | arXiv |
|---|---|---|---|---|
| B3_fedau.pdf | Wang, Ji. A Lightweight Method for Tackling Unknown Participation Statistics in Federated Averaging | ICLR 2024 | none | 2306.03401 |
| B4_fedcbs.pdf | Zhang et al. Fed-CBS: A Heterogeneity-Aware Client Sampling Mechanism for Federated Learning via Class-Imbalance Reduction | ICML 2023 | none | 2209.15245 |
| B5_hicsfl.pdf | Chen, Vikalo. Heterogeneity-Guided Client Sampling: Towards Fast and Efficient Non-IID Federated Learning | NeurIPS 2024 | 10.52202/079017-2093 | 2310.00198 |
| B6_corr_debias.pdf | Sun et al. Debiasing Federated Learning with Correlated Client Participation | ICLR 2025 | none | 2410.01209 |
| B7_mtgc.pdf | Fang et al. Hierarchical Federated Learning with Multi-Timescale Gradient Correction | NeurIPS 2024 | 10.52202/079017-2504 | 2409.18448 |
| B8_uav_unbiased_hfl.pdf | Zhagypar et al. UAV-Assisted Unbiased Hierarchical Federated Learning: Performance and Convergence Analysis | IEEE TMLCN 2025 | 10.1109/TMLCN.2025.3546181 | 2407.07739 |
| B9_khelf.pdf | Khelf, Driouch, Ajib. On the Optimization of UAV-Assisted Wireless Networks for Hierarchical Federated Learning (the PDF itself, not the Xplore viewer page) | IEEE PIMRC 2023 | 10.1109/PIMRC56721.2023.10293857 | none |
| B10_fedmultimodal.pdf | Feng et al. FedMultimodal: A Benchmark for Multimodal Federated Learning | KDD 2023 | 10.1145/3580305.3599825 | 2306.09486 |
| B11_pmcmfl.pdf | Bao et al. Multimodal Federated Learning with Missing Modality via Prototype Mask and Contrast | ICML 2024 | none | 2312.13508 |
| B12_fedloge.pdf | Xiao et al. FedLoGe: Joint Local and Generic Federated Learning under Long-tailed Data | ICLR 2024 | none | 2401.08977 |

## Theory and models the code currently assumes
| File | Paper | DOI | Needed for |
|---|---|---|---|
| gabler1981.pdf | Gabler. A Comparison of Sampford's Sampling Procedure versus Unequal Probability Sampling with Replacement. Biometrika 1981 | 10.2307/2335462 | Theorem 2 conditions (O5) |
| gabler1984.pdf | Gabler. On Unequal Probability Sampling: Sufficient Conditions for the Superiority of Sampling without Replacement. Biometrika 1984 | 10.2307/2336409 | Theorem 2 conditions (O5) |
| qualite2008.pdf | Qualité. A comparison of conditional Poisson sampling versus unequal probability sampling with replacement. JSPI 2008 | 10.1016/j.jspi.2007.04.027 | Theorem 2 |
| sampford1967.pdf | Sampford. On Sampling Without Replacement with Unequal Probabilities of Selection. Biometrika 1967 | 10.2307/2335041 | Sampford design formula |
| alhourani2014.pdf | Al-Hourani, Kandeepan, Lardner. Optimal LAP Altitude for Maximum Coverage. IEEE WCL 2014 | 10.1109/LWC.2014.2342736 | Radio constants in hime/uav.py |
