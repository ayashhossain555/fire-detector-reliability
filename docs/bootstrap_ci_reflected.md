# Reflected / BCa bootstrap intervals - identity map (B=1000, seed 3407, image-level)

Written by `bootstrap_ci_reflected.py` on 2026-09-14T14:20:20. Source: `bootstrap_ci_reflected.json`. Resampling copied verbatim from `bootstrap_ci.py`; per-cell artefacts in `calibration_ci_reflected/<cell>.json`.

Cells done: 56 / 56 (40 in-domain + 12 core cross + 4 extra cross). Values in percentage points (x100).

## LaECE_0

- percentile interval reproduces `calibration_ci/<cell>.json` ci95 to 1e-9 in 56 / 56 cells (max abs diff 0.00e+00) - key `counts.LaECE_0.n_reproduces_calibration_ci_within_1e-9`
- cells where the interval crosses zero: percentile 0, basic 1, BCa 0 (BCa undefined in 0) - keys `counts.LaECE_0.<kind>.n_crosses_zero`
- cells where the interval excludes the point: percentile 2, basic 2, BCa 2 - keys `counts.LaECE_0.<kind>.n_excludes_point`
- mean (boot_mean - point) = +0.402 pp; boot_mean above point in 49 / 56 cells
- mean width: percentile 3.953 pp, BCa 3.528 pp; mean BCa shift of lower / upper endpoint vs percentile: -0.375 / -0.800 pp (max |shift| 1.871 / 2.919 pp)
- max |a| = 0.0524, max |z0| = 2.120

| cell | role | n_img | point | boot mean | SE | percentile | basic | BCa | z0 | a | basic flags | BCa flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtdetrl_dfire_s1337__to__d_fire_test | in_domain | 4306 | 11.78 | 11.77 | 0.44 | [10.94, 12.59] | [10.96, 12.62] | [10.92, 12.59] | -0.020 | +0.0066 | ok | ok |
| rtdetrl_dfire_s1337__to__d_fire_test_dedup | in_domain | 1991 | 13.58 | 13.59 | 0.55 | [12.54, 14.74] | [12.43, 14.62] | [12.55, 14.81] | +0.025 | +0.0073 | ok | ok |
| rtdetrl_dfire_s3407__to__d_fire_test | in_domain | 4306 | 13.77 | 13.78 | 0.43 | [12.95, 14.65] | [12.89, 14.59] | [12.92, 14.63] | -0.033 | +0.0054 | ok | ok |
| rtdetrl_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 16.02 | 16.03 | 0.53 | [14.99, 17.12] | [14.91, 17.04] | [14.98, 17.11] | -0.023 | +0.0052 | ok | ok |
| rtdetrl_pyrosdis_s1337__to__pyro_sdis_caltest | in_domain | 2050 | 3.45 | 4.07 | 0.61 | [2.91, 5.34] | [1.56, 4.00] | [2.30, 4.04] | -0.999 | +0.0001 | ok | ok |
| rtdetrl_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 7.32 | 7.62 | 0.70 | [6.20, 8.95] | [5.68, 8.43] | [5.90, 8.31] | -0.479 | +0.0033 | ok | ok |
| rtdetrl_pyrosdis_s3407_p60__to__pyro_sdis_caltest | in_domain | 2050 | 4.53 | 4.80 | 0.64 | [3.65, 6.07] | [2.98, 5.40] | [3.08, 5.49] | -0.410 | -0.0036 | ok | ok |
| rtdetrx_dfire_s3407__to__d_fire_test | in_domain | 4306 | 14.28 | 14.28 | 0.42 | [13.48, 15.08] | [13.49, 15.09] | [13.46, 15.05] | -0.045 | +0.0063 | ok | ok |
| rtdetrx_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 16.63 | 16.63 | 0.54 | [15.62, 17.74] | [15.53, 17.64] | [15.66, 17.78] | +0.045 | +0.0067 | ok | ok |
| rtdetrx_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 4.10 | 4.47 | 0.67 | [3.18, 5.79] | [2.42, 5.02] | [2.82, 5.08] | -0.533 | -0.0037 | ok | ok |
| v8m_dfire_s3407__to__d_fire_test | in_domain | 4306 | 3.83 | 4.05 | 0.39 | [3.27, 4.82] | [2.83, 4.38] | [2.93, 4.34] | -0.601 | +0.0041 | ok | ok |
| v8m_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 5.98 | 6.09 | 0.53 | [5.10, 7.20] | [4.77, 6.87] | [4.98, 7.02] | -0.166 | +0.0038 | ok | ok |
| v8m_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 2.80 | 3.49 | 0.60 | [2.36, 4.70] | [0.91, 3.25] | [1.70, 3.27] | -1.150 | +0.0018 | ok | ok |
| v8n_dfire_s3407__to__d_fire_test | in_domain | 4306 | 2.80 | 3.31 | 0.30 | [2.73, 3.87] | [1.73, 2.87] | [2.30, 2.86] | -1.728 | -0.0005 | ok | ok |
| v8n_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 3.37 | 3.97 | 0.40 | [3.21, 4.73] | [2.01, 3.53] | [2.75, 3.59] | -1.468 | +0.0015 | ok | ok |
| v8n_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 2.41 | 3.41 | 0.60 | [2.26, 4.63] | [0.19, 2.56] | [1.64, 2.56] | -1.695 | +0.0029 | ok | ok |
| v8s_dfire_s1337__to__d_fire_test | in_domain | 4306 | 3.67 | 3.92 | 0.33 | [3.27, 4.56] | [2.78, 4.07] | [2.88, 4.05] | -0.765 | +0.0003 | ok | ok |
| v8s_dfire_s1337__to__d_fire_test_dedup | in_domain | 1991 | 3.86 | 4.46 | 0.43 | [3.61, 5.31] | [2.41, 4.11] | [3.25, 4.07] | -1.433 | +0.0005 | ok | ok |
| v8s_dfire_s3407__to__d_fire_test | in_domain | 4306 | 2.57 | 3.05 | 0.31 | [2.43, 3.64] | [1.50, 2.71] | [2.25, 2.67] | -1.563 | +0.0004 | ok | ok |
| v8s_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 3.78 | 4.29 | 0.47 | [3.36, 5.22] | [2.35, 4.21] | [2.97, 4.18] | -1.098 | +0.0018 | ok | ok |
| v8s_dfiredd_s3407__to__d_fire_test | in_domain | 4306 | 7.57 | 7.60 | 0.46 | [6.69, 8.51] | [6.64, 8.46] | [6.67, 8.47] | -0.035 | +0.0031 | ok | ok |
| v8s_dfiredd_s3407__to__d_fire_test_dedup | in_domain | 1991 | 6.39 | 6.64 | 0.51 | [5.67, 7.68] | [5.09, 7.11] | [5.43, 7.17] | -0.465 | +0.0034 | ok | ok |
| v8s_dfiresub_s3407__to__d_fire_test | in_domain | 4306 | 2.35 | 2.92 | 0.31 | [2.35, 3.56] | [1.14, 2.36] | [2.05, 2.36] | -1.943 | +0.0006 | ok | ok |
| v8s_dfiresub_s3407__to__d_fire_test_dedup | in_domain | 1991 | 3.06 | 3.92 | 0.42 | [3.13, 4.74] | [1.38, 2.99] | [2.60, 2.98] | -2.120 | +0.0019 | exclPt | exclPt |
| v8s_pyrosdis_s1337__to__pyro_sdis_caltest | in_domain | 2050 | 3.09 | 3.81 | 0.59 | [2.70, 4.94] | [1.23, 3.48] | [2.01, 3.44] | -1.270 | -0.0014 | ok | ok |
| v8s_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 2.02 | 3.04 | 0.52 | [2.09, 4.12] | [-0.08, 1.95] | [1.53, 1.98] | -2.075 | -0.0020 | cross0 exclPt | exclPt |
| y11m_dfire_s3407__to__d_fire_test | in_domain | 4306 | 2.73 | 3.07 | 0.35 | [2.38, 3.76] | [1.70, 3.08] | [1.87, 3.03] | -1.036 | +0.0050 | ok | ok |
| y11m_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 4.00 | 4.41 | 0.49 | [3.46, 5.42] | [2.58, 4.54] | [3.00, 4.52] | -0.852 | +0.0070 | ok | ok |
| y11m_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 2.29 | 3.12 | 0.55 | [2.07, 4.22] | [0.36, 2.51] | [1.44, 2.47] | -1.572 | +0.0023 | ok | ok |
| y11n_dfire_s3407__to__d_fire_test | in_domain | 4306 | 2.97 | 3.42 | 0.32 | [2.78, 4.02] | [1.91, 3.15] | [2.36, 3.14] | -1.454 | -0.0014 | ok | ok |
| y11n_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 3.57 | 4.25 | 0.42 | [3.42, 5.06] | [2.07, 3.72] | [2.94, 3.74] | -1.626 | +0.0010 | ok | ok |
| y11n_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 3.68 | 4.14 | 0.63 | [2.94, 5.45] | [1.91, 4.42] | [2.04, 4.42] | -0.726 | -0.0044 | ok | ok |
| y11s_dfire_s1337__to__d_fire_test | in_domain | 4306 | 3.27 | 3.61 | 0.37 | [2.91, 4.33] | [2.21, 3.62] | [2.50, 3.63] | -0.942 | +0.0027 | ok | ok |
| y11s_dfire_s1337__to__d_fire_test_dedup | in_domain | 1991 | 4.69 | 5.09 | 0.50 | [4.14, 6.12] | [3.26, 5.24] | [3.65, 5.22] | -0.827 | +0.0043 | ok | ok |
| y11s_dfire_s3407__to__d_fire_test | in_domain | 4306 | 3.46 | 3.93 | 0.32 | [3.30, 4.53] | [2.39, 3.63] | [3.08, 3.64] | -1.433 | +0.0010 | ok | ok |
| y11s_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 4.36 | 4.82 | 0.48 | [3.92, 5.78] | [2.94, 4.81] | [3.29, 4.87] | -0.919 | +0.0047 | ok | ok |
| y11s_dfiredd_s3407__to__d_fire_test | in_domain | 4306 | 4.56 | 4.75 | 0.43 | [3.91, 5.60] | [3.52, 5.21] | [3.55, 5.18] | -0.473 | +0.0023 | ok | ok |
| y11s_dfiredd_s3407__to__d_fire_test_dedup | in_domain | 1991 | 3.65 | 4.22 | 0.47 | [3.37, 5.16] | [2.15, 3.94] | [2.82, 4.00] | -1.211 | +0.0048 | ok | ok |
| y11s_pyrosdis_s1337__to__pyro_sdis_caltest | in_domain | 2050 | 2.35 | 3.24 | 0.56 | [2.20, 4.35] | [0.35, 2.50] | [1.75, 2.48] | -1.655 | -0.0035 | ok | ok |
| y11s_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 2.55 | 3.49 | 0.57 | [2.41, 4.61] | [0.50, 2.70] | [1.75, 2.75] | -1.635 | -0.0001 | ok | ok |
| v8s_dfire_s3407__to__pyro_sdis_caltest | core_cross | 2050 | 5.62 | 6.93 | 1.40 | [4.38, 9.86] | [1.38, 6.86] | [3.24, 7.00] | -0.938 | +0.0102 | ok | ok |
| v8s_dfire_s3407__to__thesis_test | core_cross | 628 | 21.34 | 21.79 | 1.87 | [18.09, 25.59] | [17.09, 24.59] | [17.50, 24.79] | -0.217 | +0.0124 | ok | ok |
| y11s_dfire_s3407__to__pyro_sdis_caltest | core_cross | 2050 | 12.52 | 12.99 | 1.55 | [10.02, 16.02] | [9.02, 15.02] | [9.32, 15.22] | -0.290 | +0.0035 | ok | ok |
| y11s_dfire_s3407__to__thesis_test | core_cross | 628 | 16.82 | 17.46 | 1.63 | [14.34, 20.80] | [12.83, 19.29] | [13.30, 19.32] | -0.396 | +0.0002 | ok | ok |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | core_cross | 2050 | 26.84 | 26.81 | 1.58 | [23.79, 29.94] | [23.73, 29.89] | [23.85, 30.03] | +0.010 | +0.0082 | ok | ok |
| rtdetrl_dfire_s3407__to__thesis_test | core_cross | 628 | 40.59 | 40.56 | 1.91 | [36.63, 44.15] | [37.04, 44.55] | [36.66, 44.16] | +0.003 | +0.0008 | ok | ok |
| v8s_pyrosdis_s3407__to__d_fire_test_smokeonly | core_cross | 4306 | 20.23 | 20.41 | 1.99 | [16.43, 24.28] | [16.19, 24.03] | [16.13, 23.76] | -0.090 | -0.0004 | ok | ok |
| v8s_pyrosdis_s3407__to__thesis_test_smokeonly | core_cross | 628 | 44.76 | 44.97 | 5.56 | [33.31, 54.06] | [35.45, 56.21] | [33.31, 54.06] | -0.120 | -0.0345 | ok | ok |
| y11s_pyrosdis_s3407__to__d_fire_test_smokeonly | core_cross | 4306 | 20.40 | 20.81 | 1.89 | [17.20, 24.66] | [16.15, 23.60] | [16.69, 23.75] | -0.194 | +0.0001 | ok | ok |
| y11s_pyrosdis_s3407__to__thesis_test_smokeonly | core_cross | 628 | 39.52 | 39.54 | 7.29 | [24.78, 53.27] | [25.77, 54.26] | [24.32, 53.25] | -0.010 | -0.0019 | ok | ok |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | core_cross | 4306 | 24.43 | 24.63 | 1.40 | [21.88, 27.48] | [21.38, 26.98] | [21.48, 26.89] | -0.166 | -0.0035 | ok | ok |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | core_cross | 628 | 51.93 | 51.98 | 1.97 | [48.24, 55.90] | [47.95, 55.62] | [48.29, 56.07] | -0.038 | +0.0286 | ok | ok |
| rtdetrx_dfire_s3407__to__pyro_sdis_caltest | extra_cross | 2050 | 24.86 | 24.84 | 1.33 | [22.32, 27.45] | [22.28, 27.40] | [22.43, 27.58] | +0.025 | +0.0119 | ok | ok |
| rtdetrx_dfire_s3407__to__thesis_test | extra_cross | 628 | 31.49 | 31.46 | 2.06 | [27.50, 35.65] | [27.33, 35.49] | [28.04, 36.05] | +0.068 | +0.0152 | ok | ok |
| rtdetrx_pyrosdis_s3407__to__d_fire_test_smokeonly | extra_cross | 4306 | 11.07 | 12.33 | 1.91 | [8.52, 16.24] | [5.89, 13.62] | [6.65, 13.33] | -0.690 | -0.0020 | ok | ok |
| rtdetrx_pyrosdis_s3407__to__thesis_test_smokeonly | extra_cross | 628 | 43.91 | 43.91 | 2.21 | [40.08, 48.71] | [39.10, 47.73] | [40.59, 49.74] | +0.065 | +0.0524 | ok | ok |

## D_ECE

- percentile interval reproduces `calibration_ci/<cell>.json` ci95 to 1e-9 in 56 / 56 cells (max abs diff 0.00e+00) - key `counts.D_ECE.n_reproduces_calibration_ci_within_1e-9`
- cells where the interval crosses zero: percentile 0, basic 2, BCa 0 (BCa undefined in 0) - keys `counts.D_ECE.<kind>.n_crosses_zero`
- cells where the interval excludes the point: percentile 0, basic 0, BCa 0 - keys `counts.D_ECE.<kind>.n_excludes_point`
- mean (boot_mean - point) = +0.209 pp; boot_mean above point in 48 / 56 cells
- mean width: percentile 5.110 pp, BCa 5.019 pp; mean BCa shift of lower / upper endpoint vs percentile: -0.299 / -0.390 pp (max |shift| 3.372 / 6.415 pp)
- max |a| = 0.0556, max |z0| = 1.499

| cell | role | n_img | point | boot mean | SE | percentile | basic | BCa | z0 | a | basic flags | BCa flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtdetrl_dfire_s1337__to__d_fire_test | in_domain | 4306 | 6.02 | 6.00 | 0.52 | [5.00, 6.98] | [5.06, 7.03] | [5.04, 6.98] | +0.020 | -0.0032 | ok | ok |
| rtdetrl_dfire_s1337__to__d_fire_test_dedup | in_domain | 1991 | 4.99 | 5.19 | 0.69 | [3.84, 6.50] | [3.48, 6.15] | [3.46, 6.09] | -0.303 | +0.0029 | ok | ok |
| rtdetrl_dfire_s3407__to__d_fire_test | in_domain | 4306 | 4.39 | 4.49 | 0.49 | [3.56, 5.45] | [3.33, 5.22] | [3.27, 5.19] | -0.228 | -0.0033 | ok | ok |
| rtdetrl_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 4.05 | 4.18 | 0.66 | [2.93, 5.56] | [2.55, 5.18] | [2.79, 5.28] | -0.169 | +0.0010 | ok | ok |
| rtdetrl_pyrosdis_s1337__to__pyro_sdis_caltest | in_domain | 2050 | 11.55 | 11.85 | 0.97 | [9.96, 13.76] | [9.34, 13.15] | [9.52, 13.17] | -0.287 | -0.0043 | ok | ok |
| rtdetrl_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 7.99 | 8.07 | 0.95 | [6.24, 9.96] | [6.02, 9.74] | [6.06, 9.88] | -0.075 | -0.0030 | ok | ok |
| rtdetrl_pyrosdis_s3407_p60__to__pyro_sdis_caltest | in_domain | 2050 | 14.94 | 15.05 | 0.91 | [13.31, 16.78] | [13.10, 16.56] | [13.10, 16.62] | -0.113 | -0.0048 | ok | ok |
| rtdetrx_dfire_s3407__to__d_fire_test | in_domain | 4306 | 4.95 | 4.97 | 0.49 | [4.05, 5.91] | [3.99, 5.85] | [4.03, 5.90] | -0.033 | -0.0023 | ok | ok |
| rtdetrx_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 4.24 | 4.33 | 0.71 | [2.98, 5.69] | [2.80, 5.51] | [2.87, 5.60] | -0.095 | +0.0008 | ok | ok |
| rtdetrx_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 16.59 | 16.64 | 0.90 | [14.99, 18.41] | [14.77, 18.19] | [14.93, 18.31] | -0.030 | -0.0047 | ok | ok |
| v8m_dfire_s3407__to__d_fire_test | in_domain | 4306 | 10.71 | 10.74 | 0.57 | [9.60, 11.86] | [9.55, 11.81] | [9.55, 11.80] | -0.055 | -0.0029 | ok | ok |
| v8m_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 7.66 | 7.84 | 0.70 | [6.45, 9.19] | [6.14, 8.87] | [6.07, 8.80] | -0.264 | -0.0025 | ok | ok |
| v8m_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 9.76 | 9.79 | 0.94 | [7.99, 11.61] | [7.90, 11.53] | [7.96, 11.57] | -0.030 | -0.0037 | ok | ok |
| v8n_dfire_s3407__to__d_fire_test | in_domain | 4306 | 13.45 | 13.49 | 0.56 | [12.39, 14.60] | [12.29, 14.50] | [12.34, 14.53] | -0.075 | -0.0029 | ok | ok |
| v8n_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 11.28 | 11.33 | 0.74 | [9.88, 12.73] | [9.82, 12.67] | [9.76, 12.65] | -0.060 | -0.0028 | ok | ok |
| v8n_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 10.39 | 10.50 | 0.95 | [8.59, 12.32] | [8.46, 12.18] | [8.30, 12.00] | -0.164 | -0.0044 | ok | ok |
| v8s_dfire_s1337__to__d_fire_test | in_domain | 4306 | 13.30 | 13.33 | 0.58 | [12.23, 14.51] | [12.09, 14.37] | [12.23, 14.51] | +0.003 | -0.0047 | ok | ok |
| v8s_dfire_s1337__to__d_fire_test_dedup | in_domain | 1991 | 11.10 | 11.07 | 0.73 | [9.63, 12.46] | [9.74, 12.56] | [9.65, 12.48] | +0.038 | -0.0047 | ok | ok |
| v8s_dfire_s3407__to__d_fire_test | in_domain | 4306 | 13.10 | 13.12 | 0.58 | [12.06, 14.23] | [11.98, 14.15] | [12.04, 14.22] | -0.010 | -0.0038 | ok | ok |
| v8s_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 10.56 | 10.60 | 0.75 | [9.05, 12.02] | [9.11, 12.07] | [8.86, 11.91] | -0.083 | -0.0035 | ok | ok |
| v8s_dfiredd_s3407__to__d_fire_test | in_domain | 4306 | 6.49 | 6.61 | 0.60 | [5.33, 7.81] | [5.17, 7.64] | [5.04, 7.50] | -0.235 | -0.0024 | ok | ok |
| v8s_dfiredd_s3407__to__d_fire_test_dedup | in_domain | 1991 | 6.82 | 7.11 | 0.64 | [5.92, 8.38] | [5.26, 7.72] | [5.37, 7.83] | -0.407 | -0.0043 | ok | ok |
| v8s_dfiresub_s3407__to__d_fire_test | in_domain | 4306 | 12.84 | 12.85 | 0.59 | [11.60, 14.00] | [11.68, 14.08] | [11.50, 13.93] | -0.058 | -0.0042 | ok | ok |
| v8s_dfiresub_s3407__to__d_fire_test_dedup | in_domain | 1991 | 10.22 | 10.39 | 0.76 | [8.90, 11.81] | [8.62, 11.54] | [8.57, 11.45] | -0.256 | -0.0047 | ok | ok |
| v8s_pyrosdis_s1337__to__pyro_sdis_caltest | in_domain | 2050 | 10.36 | 10.49 | 0.91 | [8.80, 12.24] | [8.49, 11.93] | [8.72, 12.16] | -0.065 | -0.0038 | ok | ok |
| v8s_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 9.91 | 9.94 | 0.93 | [8.14, 11.79] | [8.03, 11.68] | [8.06, 11.67] | -0.050 | -0.0039 | ok | ok |
| y11m_dfire_s3407__to__d_fire_test | in_domain | 4306 | 12.16 | 12.18 | 0.58 | [10.99, 13.33] | [10.99, 13.33] | [10.94, 13.29] | -0.060 | -0.0043 | ok | ok |
| y11m_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 9.61 | 9.64 | 0.77 | [8.19, 11.18] | [8.03, 11.02] | [8.12, 11.11] | -0.033 | -0.0048 | ok | ok |
| y11m_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 9.74 | 9.78 | 0.90 | [8.05, 11.52] | [7.96, 11.42] | [7.95, 11.39] | -0.048 | -0.0039 | ok | ok |
| y11n_dfire_s3407__to__d_fire_test | in_domain | 4306 | 13.84 | 13.86 | 0.57 | [12.77, 15.04] | [12.64, 14.92] | [12.75, 15.02] | -0.005 | -0.0042 | ok | ok |
| y11n_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 11.24 | 11.22 | 0.72 | [9.84, 12.60] | [9.87, 12.64] | [9.87, 12.63] | +0.030 | -0.0045 | ok | ok |
| y11n_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 13.74 | 13.80 | 0.97 | [11.93, 15.59] | [11.90, 15.56] | [11.73, 15.44] | -0.083 | -0.0038 | ok | ok |
| y11s_dfire_s1337__to__d_fire_test | in_domain | 4306 | 11.93 | 12.02 | 0.56 | [10.92, 13.10] | [10.77, 12.95] | [10.66, 12.94] | -0.174 | -0.0045 | ok | ok |
| y11s_dfire_s1337__to__d_fire_test_dedup | in_domain | 1991 | 9.39 | 9.39 | 0.71 | [8.00, 10.73] | [8.05, 10.78] | [8.00, 10.71] | +0.000 | -0.0053 | ok | ok |
| y11s_dfire_s3407__to__d_fire_test | in_domain | 4306 | 12.24 | 12.37 | 0.53 | [11.33, 13.40] | [11.07, 13.14] | [11.02, 13.15] | -0.269 | -0.0041 | ok | ok |
| y11s_dfire_s3407__to__d_fire_test_dedup | in_domain | 1991 | 9.62 | 9.82 | 0.70 | [8.50, 11.17] | [8.07, 10.73] | [8.00, 10.77] | -0.285 | -0.0055 | ok | ok |
| y11s_dfiredd_s3407__to__d_fire_test | in_domain | 4306 | 10.20 | 10.21 | 0.65 | [9.00, 11.45] | [8.96, 11.41] | [9.00, 11.45] | +0.010 | -0.0020 | ok | ok |
| y11s_dfiredd_s3407__to__d_fire_test_dedup | in_domain | 1991 | 10.55 | 10.58 | 0.77 | [9.06, 11.98] | [9.12, 12.05] | [8.92, 11.89] | -0.063 | -0.0047 | ok | ok |
| y11s_pyrosdis_s1337__to__pyro_sdis_caltest | in_domain | 2050 | 12.65 | 12.71 | 0.94 | [10.79, 14.55] | [10.76, 14.51] | [10.68, 14.41] | -0.058 | -0.0043 | ok | ok |
| y11s_pyrosdis_s3407__to__pyro_sdis_caltest | in_domain | 2050 | 11.51 | 11.53 | 0.93 | [9.69, 13.44] | [9.57, 13.33] | [9.58, 13.36] | -0.040 | -0.0043 | ok | ok |
| v8s_dfire_s3407__to__pyro_sdis_caltest | core_cross | 2050 | 11.17 | 11.44 | 2.47 | [6.65, 16.22] | [6.12, 15.69] | [6.05, 15.70] | -0.126 | -0.0072 | ok | ok |
| v8s_dfire_s3407__to__thesis_test | core_cross | 628 | 18.21 | 19.01 | 2.80 | [13.46, 24.74] | [11.69, 22.97] | [12.01, 22.82] | -0.311 | +0.0016 | ok | ok |
| y11s_dfire_s3407__to__pyro_sdis_caltest | core_cross | 2050 | 4.05 | 5.93 | 1.82 | [2.90, 9.87] | [-1.77, 5.20] | [1.16, 5.65] | -1.019 | -0.0004 | cross0  | ok |
| y11s_dfire_s3407__to__thesis_test | core_cross | 628 | 11.60 | 12.65 | 2.53 | [7.87, 17.79] | [5.40, 15.33] | [5.81, 15.71] | -0.380 | -0.0019 | ok | ok |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | core_cross | 2050 | 12.68 | 12.73 | 2.23 | [8.36, 17.33] | [8.04, 17.00] | [8.26, 17.17] | -0.028 | +0.0059 | ok | ok |
| rtdetrl_dfire_s3407__to__thesis_test | core_cross | 628 | 34.70 | 34.64 | 2.97 | [28.77, 40.40] | [29.00, 40.63] | [28.73, 40.25] | -0.013 | -0.0031 | ok | ok |
| v8s_pyrosdis_s3407__to__d_fire_test_smokeonly | core_cross | 4306 | 12.71 | 13.40 | 2.70 | [8.23, 18.88] | [6.54, 17.19] | [6.66, 17.12] | -0.264 | -0.0037 | ok | ok |
| v8s_pyrosdis_s3407__to__thesis_test_smokeonly | core_cross | 628 | 44.76 | 44.97 | 5.56 | [33.31, 54.06] | [35.45, 56.21] | [33.31, 54.06] | -0.120 | -0.0345 | ok | ok |
| y11s_pyrosdis_s3407__to__d_fire_test_smokeonly | core_cross | 4306 | 12.15 | 12.97 | 2.58 | [8.11, 18.12] | [6.18, 16.19] | [6.53, 16.03] | -0.361 | -0.0027 | ok | ok |
| y11s_pyrosdis_s3407__to__thesis_test_smokeonly | core_cross | 628 | 44.48 | 44.52 | 5.22 | [35.10, 55.49] | [33.46, 53.86] | [36.45, 57.59] | +0.023 | +0.0556 | ok | ok |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | core_cross | 4306 | 20.30 | 20.44 | 1.93 | [16.65, 24.25] | [16.35, 23.95] | [16.08, 23.74] | -0.093 | -0.0047 | ok | ok |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | core_cross | 628 | 51.93 | 51.98 | 1.97 | [48.24, 55.90] | [47.95, 55.62] | [48.29, 56.07] | -0.038 | +0.0286 | ok | ok |
| rtdetrx_dfire_s3407__to__pyro_sdis_caltest | extra_cross | 2050 | 13.77 | 13.74 | 2.00 | [10.01, 17.75] | [9.79, 17.54] | [10.10, 17.84] | +0.010 | +0.0064 | ok | ok |
| rtdetrx_dfire_s3407__to__thesis_test | extra_cross | 628 | 28.28 | 28.10 | 3.52 | [21.41, 34.83] | [21.74, 35.15] | [21.87, 35.42] | +0.053 | +0.0113 | ok | ok |
| rtdetrx_pyrosdis_s3407__to__d_fire_test_smokeonly | extra_cross | 4306 | 5.25 | 8.27 | 2.08 | [4.67, 12.50] | [-2.00, 5.82] | [1.30, 6.08] | -1.499 | -0.0098 | cross0  | ok |
| rtdetrx_pyrosdis_s3407__to__thesis_test_smokeonly | extra_cross | 628 | 43.93 | 43.93 | 2.20 | [40.13, 48.71] | [39.14, 47.73] | [40.66, 49.76] | +0.073 | +0.0526 | ok | ok |

Total wall time of the per-cell runs: 808 s.
