# Label-free threshold transfer - RT-DETR-x cells (machine-written by label_free_rtdetrx.py)

Created 2026-09-14T14:23:51. Refine protocol reused unchanged; verification on the four RT-DETR-l seed-3407 cells of the same directions passed (max |dt_M1c| = 0.00e+00, max |dF1| = 0.00e+00, max |d| over every checked quantity = 5.55e-17). Leakage check passed: True. Bootstrap B = 1000, seed 0. W1 cut-off (in-domain null, label_free_clean_check.json) = 0.0858.

## Thresholds

| cell | t_src | kappa | t M1c full | t M1c 100 | t M1a | t ORC | W1 imgmax (above cut-off) | |t_src - t_ORC| |
|---|---|---|---|---|---|---|---|---|
| rtdetrx_dfire_s3407__to__pyro_sdis_caltest | 0.595 | 1.162 | 0.141 | 0.080 | 0.229 | 0.175 | 0.1636 (yes) | 0.420 |
| rtdetrx_dfire_s3407__to__thesis_test | 0.595 | 1.162 | 0.364 | 0.405 | 0.363 | 0.405 | 0.0678 (no) | 0.191 |
| rtdetrx_pyrosdis_s3407__to__d_fire_test_smokeonly | 0.320 | 0.976 | 0.032 | 0.034 | 0.033 | 0.127 | 0.4299 (yes) | 0.193 |
| rtdetrx_pyrosdis_s3407__to__thesis_test_smokeonly | 0.320 | 0.976 | 0.027 | 0.030 | 0.029 | 0.037 | 0.4672 (yes) | 0.283 |

## F1 on the target test (box, IoU 0.5) with 95 % image-bootstrap CI; regret = F1(ORC) - F1(method)

| cell | method | threshold | F1 [CI] | regret [CI] | gain vs t_src [CI] | gap closed [CI] | sens | FAR | LaECE_0 (n kept) |
|---|---|---|---|---|---|---|---|---|---|
| rtdetrx_dfire_s3407__to__pyro_sdis_caltest | M0_t_src | 0.595 | 0.306 [0.282, 0.329] | 0.055 [0.037, 0.075] |   |  | 0.330 | 0.172 | 0.248 (677) |
| rtdetrx_dfire_s3407__to__pyro_sdis_caltest | D25_default | 0.250 | 0.364 [0.343, 0.384] | -0.003 [-0.012, 0.006] | 0.057 [0.042, 0.075] | 1.051 [0.894, 1.280] | 0.511 | 0.298 | 0.231 (1512) |
| rtdetrx_dfire_s3407__to__pyro_sdis_caltest | M1c_kept_rate | 0.141 | 0.353 [0.333, 0.372] | 0.008 [0.002, 0.014] | 0.046 [0.027, 0.067] | 0.848 [0.686, 0.967] | 0.587 | 0.368 | 0.186 (2290) |
| rtdetrx_dfire_s3407__to__pyro_sdis_caltest | M1c_kept_rate_100 | 0.080 | 0.322 [0.305, 0.342] | 0.038 [0.027, 0.049] | 0.016 [-0.005, 0.039] | 0.296 [-0.128, 0.555] | 0.665 | 0.425 | 0.141 (3436) |
| rtdetrx_dfire_s3407__to__pyro_sdis_caltest | M1a_quantile_pooled | 0.229 | 0.361 [0.341, 0.382] | -0.000 [-0.007, 0.008] | 0.055 [0.039, 0.073] | 1.003 [0.874, 1.163] | 0.524 | 0.309 | 0.225 (1637) |
| rtdetrx_dfire_s3407__to__pyro_sdis_caltest | ORC_target_calval | 0.175 | 0.361 [0.341, 0.381] | 0.000 [0.000, 0.000] | 0.055 [0.037, 0.075] | 1.000 [1.000, 1.000] | 0.561 | 0.339 | 0.203 (1960) |
| rtdetrx_dfire_s3407__to__thesis_test | M0_t_src | 0.595 | 0.290 [0.236, 0.349] | -0.006 [-0.023, 0.011] |   |  | 0.758 | 0.009 | 0.313 (216) |
| rtdetrx_dfire_s3407__to__thesis_test | D25_default | 0.250 | 0.270 [0.230, 0.315] | 0.013 [-0.004, 0.031] | -0.020 [-0.047, 0.003] |  [-3.446, 1.178] (unstable) | 0.839 | 0.101 | 0.298 (546) |
| rtdetrx_dfire_s3407__to__thesis_test | M1c_kept_rate | 0.364 | 0.289 [0.239, 0.340] | -0.005 [-0.016, 0.004] | -0.001 [-0.021, 0.017] |  [0.729, 3.212] (unstable) | 0.807 | 0.049 | 0.322 (364) |
| rtdetrx_dfire_s3407__to__thesis_test | M1c_kept_rate_100 | 0.405 | 0.283 [0.235, 0.336] | 0.000 [0.000, 0.000] | -0.006 [-0.023, 0.011] |  [1.000, 1.000] (unstable) | 0.801 | 0.043 | 0.328 (332) |
| rtdetrx_dfire_s3407__to__thesis_test | M1a_quantile_pooled | 0.363 | 0.289 [0.239, 0.340] | -0.005 [-0.016, 0.004] | -0.001 [-0.021, 0.017] |  [0.729, 3.212] (unstable) | 0.807 | 0.049 | 0.322 (364) |
| rtdetrx_dfire_s3407__to__thesis_test | ORC_target_calval | 0.405 | 0.283 [0.235, 0.336] | 0.000 [0.000, 0.000] | -0.006 [-0.023, 0.011] |  [1.000, 1.000] (unstable) | 0.801 | 0.043 | 0.328 (332) |
| rtdetrx_pyrosdis_s3407__to__d_fire_test_smokeonly | M0_t_src | 0.320 | 0.100 [0.084, 0.116] | 0.052 [0.041, 0.065] |   |  | 0.086 | 0.025 | 0.113 (254) |
| rtdetrx_pyrosdis_s3407__to__d_fire_test_smokeonly | D25_default | 0.250 | 0.116 [0.099, 0.132] | 0.036 [0.026, 0.047] | 0.016 [0.010, 0.024] | 0.312 [0.196, 0.443] | 0.107 | 0.040 | 0.102 (357) |
| rtdetrx_pyrosdis_s3407__to__d_fire_test_smokeonly | M1c_kept_rate | 0.032 | 0.111 [0.101, 0.122] | 0.041 [0.028, 0.052] | 0.012 [-0.002, 0.026] | 0.222 [-0.033, 0.447] | 0.407 | 0.304 | 0.040 (4261) |
| rtdetrx_pyrosdis_s3407__to__d_fire_test_smokeonly | M1c_kept_rate_100 | 0.034 | 0.114 [0.104, 0.126] | 0.038 [0.026, 0.048] | 0.015 [0.001, 0.028] | 0.278 [0.026, 0.498] | 0.401 | 0.288 | 0.041 (4022) |
| rtdetrx_pyrosdis_s3407__to__d_fire_test_smokeonly | M1a_quantile_pooled | 0.033 | 0.113 [0.102, 0.124] | 0.039 [0.027, 0.050] | 0.013 [-0.000, 0.027] | 0.247 [-0.004, 0.467] | 0.403 | 0.296 | 0.040 (4150) |
| rtdetrx_pyrosdis_s3407__to__d_fire_test_smokeonly | ORC_target_calval | 0.127 | 0.152 [0.134, 0.169] | 0.000 [0.000, 0.000] | 0.052 [0.041, 0.065] | 1.000 [1.000, 1.000] | 0.182 | 0.088 | 0.068 (815) |
| rtdetrx_pyrosdis_s3407__to__thesis_test_smokeonly | M0_t_src | 0.320 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |   |  | 0.009 | 0.021 | 0.424 (13) |
| rtdetrx_pyrosdis_s3407__to__thesis_test_smokeonly | D25_default | 0.250 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |   (unstable) | 0.009 | 0.035 | 0.366 (22) |
| rtdetrx_pyrosdis_s3407__to__thesis_test_smokeonly | M1c_kept_rate | 0.027 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |   (unstable) | 0.211 | 0.529 | 0.060 (1348) |
| rtdetrx_pyrosdis_s3407__to__thesis_test_smokeonly | M1c_kept_rate_100 | 0.030 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |   (unstable) | 0.175 | 0.484 | 0.066 (1136) |
| rtdetrx_pyrosdis_s3407__to__thesis_test_smokeonly | M1a_quantile_pooled | 0.029 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |   (unstable) | 0.184 | 0.504 | 0.063 (1210) |
| rtdetrx_pyrosdis_s3407__to__thesis_test_smokeonly | ORC_target_calval | 0.037 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |   (unstable) | 0.149 | 0.446 | 0.078 (819) |

## RT-DETR-x vs RT-DETR-l (same direction, seed 3407; l values from the refine / pilot artefacts)

| direction | model | t_src | kappa | t M1c | t ORC | F1 t_src | F1 D25 | F1 M1c | F1 M1c100 | F1 M1a | F1 ORC | regret M1c [CI] | gap closed | W1 imgmax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dfire_s3407 -> pyro_sdis_caltest | rtdetrx | 0.595 | 1.162 | 0.141 | 0.175 | 0.306 | 0.364 | 0.353 | 0.322 | 0.361 | 0.361 | 0.008 [0.002, 0.014] | 0.848 | 0.1636 |
| | rtdetrl | 0.592 | 1.184 | 0.137 | 0.238 | 0.263 | 0.357 | 0.338 | 0.296 | 0.355 | 0.357 | 0.019 [0.007, 0.029] | 0.802 | 0.1807 |
| dfire_s3407 -> thesis_test | rtdetrx | 0.595 | 1.162 | 0.364 | 0.405 | 0.290 | 0.270 | 0.289 | 0.283 | 0.289 | 0.283 | -0.005 [-0.016, 0.004] |  | 0.0678 |
| | rtdetrl | 0.592 | 1.184 | 0.338 | 0.489 | 0.274 | 0.208 | 0.236 | 0.240 | 0.236 | 0.264 | 0.028 [0.010, 0.047] |  | 0.0775 |
| pyrosdis_s3407 -> d_fire_test_smokeonly | rtdetrx | 0.320 | 0.976 | 0.032 | 0.127 | 0.100 | 0.116 | 0.111 | 0.114 | 0.113 | 0.152 | 0.041 [0.028, 0.052] | 0.222 | 0.4299 |
| | rtdetrl | 0.411 | 1.085 | 0.162 | 0.285 | 0.141 | 0.164 | 0.136 | 0.144 | 0.136 | 0.163 | 0.027 [0.016, 0.037] | -0.200 | 0.3409 |
| pyrosdis_s3407 -> thesis_test_smokeonly | rtdetrx | 0.320 | 0.976 | 0.027 | 0.037 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] |  | 0.4672 |
| | rtdetrl | 0.411 | 1.085 | 0.121 | 0.129 | 0.000 | 0.005 | 0.016 | 0.016 | 0.016 | 0.016 | -0.000 [-0.004, 0.003] | 1.011 | 0.4039 |

## Verification cells (RT-DETR-l seed 3407): max |difference| vs refine / pilot artefacts

| cell | quantity | abs diff | tolerance |
|---|---|---|---|
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | t_M1c | 0.00e+00 | 1e-06 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | t_ORC | 0.00e+00 | 1e-06 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | t_src | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | kappa | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | f1_max_over_t_src_D25_M1c_ORC | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | regret_M1c_ci95_max | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | gap_closed_M1c | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | LaECE_0_max_over_t_src_D25_M1c_ORC | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | t_M1c_100_vs_refine_draw0 | 0.00e+00 | 0e+00 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | regret_M1c_100_vs_refine_draw0 | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | t_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | f1_M1a_vs_pilot | 5.55e-17 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | LaECE_0_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | W1_imgmax_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | W1_pooled_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | t_M1c | 0.00e+00 | 1e-06 |
| rtdetrl_dfire_s3407__to__thesis_test | t_ORC | 0.00e+00 | 1e-06 |
| rtdetrl_dfire_s3407__to__thesis_test | t_src | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | kappa | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | f1_max_over_t_src_D25_M1c_ORC | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | regret_M1c_ci95_max | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | gap_closed_M1c | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | LaECE_0_max_over_t_src_D25_M1c_ORC | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | t_M1c_100_vs_refine_draw0 | 0.00e+00 | 0e+00 |
| rtdetrl_dfire_s3407__to__thesis_test | regret_M1c_100_vs_refine_draw0 | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | t_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | f1_M1a_vs_pilot | 2.78e-17 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | LaECE_0_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | W1_imgmax_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_dfire_s3407__to__thesis_test | W1_pooled_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | t_M1c | 0.00e+00 | 1e-06 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | t_ORC | 0.00e+00 | 1e-06 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | t_src | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | kappa | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | f1_max_over_t_src_D25_M1c_ORC | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | regret_M1c_ci95_max | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | gap_closed_M1c | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | LaECE_0_max_over_t_src_D25_M1c_ORC | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | t_M1c_100_vs_refine_draw0 | 0.00e+00 | 0e+00 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | regret_M1c_100_vs_refine_draw0 | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | t_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | f1_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | LaECE_0_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | W1_imgmax_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | W1_pooled_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | t_M1c | 0.00e+00 | 1e-06 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | t_ORC | 0.00e+00 | 1e-06 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | t_src | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | kappa | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | f1_max_over_t_src_D25_M1c_ORC | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | regret_M1c_ci95_max | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | gap_closed_M1c | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | LaECE_0_max_over_t_src_D25_M1c_ORC | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | t_M1c_100_vs_refine_draw0 | 0.00e+00 | 0e+00 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | regret_M1c_100_vs_refine_draw0 | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | t_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | f1_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | LaECE_0_M1a_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | W1_imgmax_vs_pilot | 0.00e+00 | 1e-09 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | W1_pooled_vs_pilot | 0.00e+00 | 1e-09 |
