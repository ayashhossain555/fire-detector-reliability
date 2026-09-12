# Label-free threshold transfer by direction

Written by `label_free_by_direction.py` on 2026-09-13T01:46:00 from `label_free_threshold_refine.json` (2026-09-12T11:39:46) and `label_free_threshold_pilot.json` (2026-09-12T11:33:32). All numbers: `label_free_by_direction.json`.

Cell-level bootstrap (resample cells with replacement, seed 0, B = 10,000, percentile 2.5-97.5, same resamples for every rule). gap closed = 1 - mean regret / mean gap (ratio of means). Ties: |dF1| <= 1e-12. Floor: target-calval oracle threshold reaching target-test box F1 >= 0.1.

Cells below the floor: 9 of 44 (so 35 cells at or above it) - key `cells_below_floor`.

| cell | direction | oracle F1 | F1 t_src | F1 0.25 | F1 kept-rate | F1 quantile-pooled | n test GT |
|---|---|---|---|---|---|---|---|
| rtdetrl_pyrosdis_s1337__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.012 | 0.000 | 0.021 | 0.014 | 0.014 | 245 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.016 | 0.000 | 0.005 | 0.016 | 0.016 | 245 |
| rtdetrl_pyrosdis_s3407_p60__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 245 |
| v8n_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.013 | 0.000 | 0.000 | 0.017 | 0.011 | 245 |
| v8s_pyrosdis_s1337__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.010 | 0.000 | 0.000 | 0.009 | 0.014 | 245 |
| v8s_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.021 | 0.000 | 0.000 | 0.017 | 0.011 | 245 |
| y11n_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.018 | 0.000 | 0.000 | 0.015 | 0.016 | 245 |
| y11s_pyrosdis_s1337__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.018 | 0.000 | 0.000 | 0.015 | 0.006 | 245 |
| y11s_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.010 | 0.000 | 0.000 | 0.014 | 0.014 | 245 |

## All cells

| direction | n | mean F1 t_src | mean F1 0.25 | mean F1 kept-rate | mean F1 q-pooled | mean F1 oracle | mean gap [cell CI] |
|---|---|---|---|---|---|---|---|
| D-Fire->Pyro-SDIS | 13 | 0.212 | 0.258 | 0.288 | 0.296 | 0.301 | 0.089 [0.079, 0.099] |
| D-Fire->thesis | 13 | 0.256 | 0.251 | 0.258 | 0.229 | 0.261 | 0.005 [-0.000, 0.010] |
| Pyro-SDIS->D-Fire (smoke) | 9 | 0.102 | 0.117 | 0.130 | 0.140 | 0.152 | 0.050 [0.041, 0.058] |
| Pyro-SDIS->thesis (smoke) | 9 | 0.000 | 0.003 | 0.013 | 0.011 | 0.013 | 0.013 [0.009, 0.017] |
| overall | 44 | 0.159 | 0.175 | 0.190 | 0.186 | 0.200 | 0.041 [0.030, 0.052] |

| direction | n | rule | mean regret [cell CI] | gap closed [cell CI] | mean dF1 vs t_src [cell CI] | beats / worse / ties vs t_src |
|---|---|---|---|---|---|---|
| D-Fire->Pyro-SDIS | 13 | default 0.25 | 0.043 [0.031, 0.052] | 0.520 [0.421, 0.638] | +0.046 [0.036, 0.058] | 13 / 0 / 0 |
| D-Fire->Pyro-SDIS | 13 | kept-rate (refine M1c) | 0.013 [0.010, 0.017] | 0.852 [0.797, 0.897] | +0.076 [0.063, 0.088] | 13 / 0 / 0 |
| D-Fire->Pyro-SDIS | 13 | quantile-pooled (pilot M1a) | 0.005 [0.001, 0.009] | 0.947 [0.908, 0.983] | +0.084 [0.075, 0.093] | 13 / 0 / 0 |
| D-Fire->thesis | 13 | default 0.25 | 0.010 [-0.002, 0.025] | -1.054 [-24.371, 15.020] | -0.005 [-0.024, 0.009] | 10 / 3 / 0 |
| D-Fire->thesis | 13 | kept-rate (refine M1c) | 0.002 [-0.003, 0.009] | 0.487 [-6.176, 4.694] | +0.002 [-0.008, 0.011] | 8 / 5 / 0 |
| D-Fire->thesis | 13 | quantile-pooled (pilot M1a) | 0.032 [0.023, 0.041] | -5.636 [-42.091, 19.379] | -0.027 [-0.036, -0.018] | 1 / 12 / 0 |
| Pyro-SDIS->D-Fire (smoke) | 9 | default 0.25 | 0.035 [0.020, 0.047] | 0.304 [0.171, 0.511] | +0.015 [0.010, 0.021] | 9 / 0 / 0 |
| Pyro-SDIS->D-Fire (smoke) | 9 | kept-rate (refine M1c) | 0.022 [0.017, 0.027] | 0.563 [0.358, 0.699] | +0.028 [0.015, 0.040] | 8 / 1 / 0 |
| Pyro-SDIS->D-Fire (smoke) | 9 | quantile-pooled (pilot M1a) | 0.012 [0.000, 0.024] | 0.768 [0.435, 0.998] | +0.038 [0.018, 0.056] | 7 / 2 / 0 |
| Pyro-SDIS->thesis (smoke) | 9 | default 0.25 | 0.010 [0.004, 0.016] | 0.216 [0.000, 0.628] | +0.003 [0.000, 0.007] | 2 / 0 / 7 |
| Pyro-SDIS->thesis (smoke) | 9 | kept-rate (refine M1c) | 0.000 [-0.002, 0.002] | 0.984 [0.869, 1.135] | +0.013 [0.009, 0.016] | 8 / 0 / 1 |
| Pyro-SDIS->thesis (smoke) | 9 | quantile-pooled (pilot M1a) | 0.002 [-0.001, 0.006] | 0.847 [0.629, 1.106] | +0.011 [0.008, 0.014] | 8 / 0 / 1 |
| overall | 44 | default 0.25 | 0.025 [0.017, 0.032] | 0.391 [0.212, 0.531] | +0.016 [0.007, 0.024] | 34 / 3 / 7 |
| overall | 44 | kept-rate (refine M1c) | 0.009 [0.006, 0.012] | 0.775 [0.680, 0.852] | +0.031 [0.022, 0.042] | 37 / 6 / 1 |
| overall | 44 | quantile-pooled (pilot M1a) | 0.014 [0.009, 0.019] | 0.667 [0.430, 0.815] | +0.027 [0.013, 0.041] | 29 / 14 / 1 |

Paired per-cell difference F1(kept-rate) - F1(quantile-pooled) (positive = kept-rate better):

| direction | n | mean regret kept-rate | mean regret q-pooled | mean diff [cell CI] | two-sided cell-bootstrap p | wins / losses / ties (kept-rate) |
|---|---|---|---|---|---|---|
| D-Fire->Pyro-SDIS | 13 | 0.013 | 0.005 | -0.0085 [-0.013, -0.004] | 0.000 | 2 / 11 / 0 |
| D-Fire->thesis | 13 | 0.002 | 0.032 | +0.0292 [0.019, 0.039] | 0.000 | 13 / 0 / 0 |
| Pyro-SDIS->D-Fire (smoke) | 9 | 0.022 | 0.012 | -0.0102 [-0.017, -0.003] | 0.006 | 3 / 6 / 0 |
| Pyro-SDIS->thesis (smoke) | 9 | 0.000 | 0.002 | +0.0018 [-0.001, 0.005] | 0.195 | 3 / 4 / 2 |
| overall | 44 | 0.009 | 0.014 | +0.0044 [-0.001, 0.011] | 0.142 | 21 / 21 / 2 |

## Cells with oracle F1 >= 0.1

| direction | n | mean F1 t_src | mean F1 0.25 | mean F1 kept-rate | mean F1 q-pooled | mean F1 oracle | mean gap [cell CI] |
|---|---|---|---|---|---|---|---|
| D-Fire->Pyro-SDIS | 13 | 0.212 | 0.258 | 0.288 | 0.296 | 0.301 | 0.089 [0.079, 0.099] |
| D-Fire->thesis | 13 | 0.256 | 0.251 | 0.258 | 0.229 | 0.261 | 0.005 [-0.000, 0.010] |
| Pyro-SDIS->D-Fire (smoke) | 9 | 0.102 | 0.117 | 0.130 | 0.140 | 0.152 | 0.050 [0.041, 0.058] |
| Pyro-SDIS->thesis (smoke) | 0 | | | | | | |
| overall | 35 | 0.200 | 0.219 | 0.236 | 0.231 | 0.248 | 0.048 [0.035, 0.061] |

| direction | n | rule | mean regret [cell CI] | gap closed [cell CI] | mean dF1 vs t_src [cell CI] | beats / worse / ties vs t_src |
|---|---|---|---|---|---|---|
| D-Fire->Pyro-SDIS | 13 | default 0.25 | 0.043 [0.031, 0.052] | 0.520 [0.421, 0.638] | +0.046 [0.036, 0.058] | 13 / 0 / 0 |
| D-Fire->Pyro-SDIS | 13 | kept-rate (refine M1c) | 0.013 [0.010, 0.017] | 0.852 [0.797, 0.897] | +0.076 [0.063, 0.088] | 13 / 0 / 0 |
| D-Fire->Pyro-SDIS | 13 | quantile-pooled (pilot M1a) | 0.005 [0.001, 0.009] | 0.947 [0.908, 0.983] | +0.084 [0.075, 0.093] | 13 / 0 / 0 |
| D-Fire->thesis | 13 | default 0.25 | 0.010 [-0.002, 0.025] | -1.054 [-24.371, 15.020] | -0.005 [-0.024, 0.009] | 10 / 3 / 0 |
| D-Fire->thesis | 13 | kept-rate (refine M1c) | 0.002 [-0.003, 0.009] | 0.487 [-6.176, 4.694] | +0.002 [-0.008, 0.011] | 8 / 5 / 0 |
| D-Fire->thesis | 13 | quantile-pooled (pilot M1a) | 0.032 [0.023, 0.041] | -5.636 [-42.091, 19.379] | -0.027 [-0.036, -0.018] | 1 / 12 / 0 |
| Pyro-SDIS->D-Fire (smoke) | 9 | default 0.25 | 0.035 [0.020, 0.047] | 0.304 [0.171, 0.511] | +0.015 [0.010, 0.021] | 9 / 0 / 0 |
| Pyro-SDIS->D-Fire (smoke) | 9 | kept-rate (refine M1c) | 0.022 [0.017, 0.027] | 0.563 [0.358, 0.699] | +0.028 [0.015, 0.040] | 8 / 1 / 0 |
| Pyro-SDIS->D-Fire (smoke) | 9 | quantile-pooled (pilot M1a) | 0.012 [0.000, 0.024] | 0.768 [0.435, 0.998] | +0.038 [0.018, 0.056] | 7 / 2 / 0 |
| overall | 35 | default 0.25 | 0.028 [0.020, 0.037] | 0.403 [0.215, 0.549] | +0.019 [0.008, 0.030] | 32 / 3 / 0 |
| overall | 35 | kept-rate (refine M1c) | 0.011 [0.008, 0.015] | 0.760 [0.654, 0.839] | +0.036 [0.024, 0.049] | 29 / 6 / 0 |
| overall | 35 | quantile-pooled (pilot M1a) | 0.016 [0.011, 0.023] | 0.654 [0.390, 0.814] | +0.031 [0.014, 0.048] | 21 / 14 / 0 |

Paired per-cell difference F1(kept-rate) - F1(quantile-pooled) (positive = kept-rate better):

| direction | n | mean regret kept-rate | mean regret q-pooled | mean diff [cell CI] | two-sided cell-bootstrap p | wins / losses / ties (kept-rate) |
|---|---|---|---|---|---|---|
| D-Fire->Pyro-SDIS | 13 | 0.013 | 0.005 | -0.0085 [-0.013, -0.004] | 0.000 | 2 / 11 / 0 |
| D-Fire->thesis | 13 | 0.002 | 0.032 | +0.0292 [0.019, 0.039] | 0.000 | 13 / 0 / 0 |
| Pyro-SDIS->D-Fire (smoke) | 9 | 0.022 | 0.012 | -0.0102 [-0.017, -0.003] | 0.006 | 3 / 6 / 0 |
| overall | 35 | 0.011 | 0.016 | +0.0051 [-0.002, 0.013] | 0.183 | 18 / 17 / 0 |

## Per-cell F1 (target test, box F1 at IoU 0.5)

| cell | direction | oracle | t_src | 0.25 | kept-rate | q-pooled | gap | regret kept-rate | regret q-pooled | kept - q-pooled | >= floor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rtdetrl_dfire_s1337__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.357 | 0.302 | 0.354 | 0.340 | 0.359 | 0.055 | 0.017 | -0.002 | -0.0188 | yes |
| rtdetrl_dfire_s1337__to__thesis_test | D-Fire->thesis | 0.230 | 0.241 | 0.157 | 0.220 | 0.220 | -0.011 | 0.010 | 0.010 | +0.0002 | yes |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.357 | 0.263 | 0.357 | 0.338 | 0.355 | 0.093 | 0.019 | 0.002 | -0.0170 | yes |
| rtdetrl_dfire_s3407__to__thesis_test | D-Fire->thesis | 0.264 | 0.274 | 0.208 | 0.236 | 0.236 | -0.009 | 0.028 | 0.029 | +0.0002 | yes |
| rtdetrl_pyrosdis_s1337__to__d_fire_test_smokeonly | Pyro-SDIS->D-Fire (smoke) | 0.202 | 0.166 | 0.199 | 0.169 | 0.160 | 0.036 | 0.033 | 0.041 | +0.0083 | yes |
| rtdetrl_pyrosdis_s1337__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.012 | 0.000 | 0.021 | 0.014 | 0.014 | 0.012 | -0.001 | -0.001 | -0.0000 | no |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | Pyro-SDIS->D-Fire (smoke) | 0.163 | 0.141 | 0.164 | 0.136 | 0.136 | 0.022 | 0.027 | 0.027 | +0.0001 | yes |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.016 | 0.000 | 0.005 | 0.016 | 0.016 | 0.016 | -0.000 | -0.000 | +0.0000 | no |
| rtdetrl_pyrosdis_s3407_p60__to__d_fire_test_smokeonly | Pyro-SDIS->D-Fire (smoke) | 0.139 | 0.094 | 0.118 | 0.102 | 0.099 | 0.044 | 0.037 | 0.039 | +0.0028 | yes |
| rtdetrl_pyrosdis_s3407_p60__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.0000 | no |
| v8m_dfire_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.338 | 0.213 | 0.286 | 0.332 | 0.325 | 0.124 | 0.005 | 0.013 | +0.0073 | yes |
| v8m_dfire_s3407__to__thesis_test | D-Fire->thesis | 0.266 | 0.271 | 0.273 | 0.272 | 0.243 | -0.004 | -0.005 | 0.023 | +0.0286 | yes |
| v8n_dfire_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.325 | 0.243 | 0.260 | 0.309 | 0.323 | 0.082 | 0.016 | 0.002 | -0.0138 | yes |
| v8n_dfire_s3407__to__thesis_test | D-Fire->thesis | 0.283 | 0.279 | 0.281 | 0.279 | 0.223 | 0.004 | 0.004 | 0.060 | +0.0557 | yes |
| v8n_pyrosdis_s3407__to__d_fire_test_smokeonly | Pyro-SDIS->D-Fire (smoke) | 0.127 | 0.076 | 0.087 | 0.118 | 0.128 | 0.051 | 0.009 | -0.001 | -0.0097 | yes |
| v8n_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.013 | 0.000 | 0.000 | 0.017 | 0.011 | 0.013 | -0.004 | 0.002 | +0.0068 | no |
| v8s_dfire_s1337__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.306 | 0.217 | 0.254 | 0.288 | 0.306 | 0.089 | 0.018 | -0.000 | -0.0177 | yes |
| v8s_dfire_s1337__to__thesis_test | D-Fire->thesis | 0.262 | 0.258 | 0.263 | 0.257 | 0.202 | 0.004 | 0.005 | 0.059 | +0.0540 | yes |
| v8s_dfire_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.329 | 0.216 | 0.276 | 0.323 | 0.325 | 0.113 | 0.006 | 0.004 | -0.0021 | yes |
| v8s_dfire_s3407__to__thesis_test | D-Fire->thesis | 0.254 | 0.231 | 0.249 | 0.253 | 0.219 | 0.023 | 0.001 | 0.035 | +0.0340 | yes |
| v8s_dfiredd_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.231 | 0.169 | 0.189 | 0.206 | 0.228 | 0.062 | 0.025 | 0.003 | -0.0218 | yes |
| v8s_dfiredd_s3407__to__thesis_test | D-Fire->thesis | 0.243 | 0.233 | 0.242 | 0.248 | 0.242 | 0.010 | -0.005 | 0.001 | +0.0067 | yes |
| v8s_dfiresub_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.317 | 0.227 | 0.274 | 0.307 | 0.310 | 0.090 | 0.010 | 0.008 | -0.0025 | yes |
| v8s_dfiresub_s3407__to__thesis_test | D-Fire->thesis | 0.227 | 0.220 | 0.248 | 0.246 | 0.198 | 0.006 | -0.020 | 0.028 | +0.0484 | yes |
| v8s_pyrosdis_s1337__to__d_fire_test_smokeonly | Pyro-SDIS->D-Fire (smoke) | 0.137 | 0.088 | 0.094 | 0.119 | 0.136 | 0.049 | 0.017 | 0.000 | -0.0172 | yes |
| v8s_pyrosdis_s1337__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.010 | 0.000 | 0.000 | 0.009 | 0.014 | 0.010 | 0.001 | -0.004 | -0.0049 | no |
| v8s_pyrosdis_s3407__to__d_fire_test_smokeonly | Pyro-SDIS->D-Fire (smoke) | 0.137 | 0.083 | 0.092 | 0.118 | 0.142 | 0.054 | 0.019 | -0.005 | -0.0237 | yes |
| v8s_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.021 | 0.000 | 0.000 | 0.017 | 0.011 | 0.021 | 0.004 | 0.010 | +0.0059 | no |
| y11m_dfire_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.335 | 0.229 | 0.270 | 0.314 | 0.313 | 0.106 | 0.020 | 0.022 | +0.0016 | yes |
| y11m_dfire_s3407__to__thesis_test | D-Fire->thesis | 0.275 | 0.270 | 0.275 | 0.273 | 0.230 | 0.005 | 0.002 | 0.046 | +0.0436 | yes |
| y11n_dfire_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.240 | 0.151 | 0.187 | 0.230 | 0.243 | 0.088 | 0.009 | -0.004 | -0.0128 | yes |
| y11n_dfire_s3407__to__thesis_test | D-Fire->thesis | 0.255 | 0.256 | 0.257 | 0.258 | 0.239 | -0.001 | -0.003 | 0.016 | +0.0196 | yes |
| y11n_pyrosdis_s3407__to__d_fire_test_smokeonly | Pyro-SDIS->D-Fire (smoke) | 0.150 | 0.085 | 0.100 | 0.131 | 0.149 | 0.065 | 0.019 | 0.001 | -0.0177 | yes |
| y11n_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.018 | 0.000 | 0.000 | 0.015 | 0.016 | 0.018 | 0.003 | 0.003 | -0.0005 | no |
| y11s_dfire_s1337__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.282 | 0.198 | 0.246 | 0.277 | 0.282 | 0.083 | 0.005 | -0.000 | -0.0049 | yes |
| y11s_dfire_s1337__to__thesis_test | D-Fire->thesis | 0.289 | 0.272 | 0.289 | 0.287 | 0.255 | 0.017 | 0.002 | 0.034 | +0.0330 | yes |
| y11s_dfire_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.282 | 0.181 | 0.230 | 0.276 | 0.279 | 0.101 | 0.006 | 0.003 | -0.0025 | yes |
| y11s_dfire_s3407__to__thesis_test | D-Fire->thesis | 0.276 | 0.263 | 0.260 | 0.276 | 0.236 | 0.013 | -0.000 | 0.040 | +0.0404 | yes |
| y11s_dfiredd_s3407__to__pyro_sdis_caltest | D-Fire->Pyro-SDIS | 0.213 | 0.143 | 0.173 | 0.198 | 0.203 | 0.070 | 0.015 | 0.010 | -0.0048 | yes |
| y11s_dfiredd_s3407__to__thesis_test | D-Fire->thesis | 0.263 | 0.256 | 0.259 | 0.249 | 0.234 | 0.007 | 0.014 | 0.029 | +0.0154 | yes |
| y11s_pyrosdis_s1337__to__d_fire_test_smokeonly | Pyro-SDIS->D-Fire (smoke) | 0.155 | 0.092 | 0.099 | 0.137 | 0.154 | 0.063 | 0.018 | 0.001 | -0.0173 | yes |
| y11s_pyrosdis_s1337__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.018 | 0.000 | 0.000 | 0.015 | 0.006 | 0.018 | 0.003 | 0.012 | +0.0092 | no |
| y11s_pyrosdis_s3407__to__d_fire_test_smokeonly | Pyro-SDIS->D-Fire (smoke) | 0.156 | 0.092 | 0.099 | 0.139 | 0.156 | 0.064 | 0.018 | -0.000 | -0.0178 | yes |
| y11s_pyrosdis_s3407__to__thesis_test_smokeonly | Pyro-SDIS->thesis (smoke) | 0.010 | 0.000 | 0.000 | 0.014 | 0.014 | 0.010 | -0.003 | -0.003 | -0.0001 | no |

Keys: `groups.<direction>.<subset>.rules.<rule>.{mean_regret, mean_regret_ci95_cell_bootstrap, gap_closed, gap_closed_ci95_cell_bootstrap, n_beats_t_src, n_worse_than_t_src, n_ties_t_src}`; `groups.<direction>.<subset>.paired_kept_rate_minus_quantile_pooled`; `cells.<cell>.f1.<rule>`. Seconds: 0.11.
