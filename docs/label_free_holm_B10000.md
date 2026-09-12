# Holm-Bonferroni over 44 cells at B = 10000 (machine-written by label_free_holm_B10000.py)

Created 2026-09-13T00:47:17; 58.3 s CPU. Seed 0, alpha 0.05, Holm strictest level alpha/m = 0.00114. Two-sided p resolution 2/B = 0.0002 (B = 10000) vs 0.002 (B = 1000). Cells with two-sided p = 0: 20 (B = 10000) vs 20 (B = 1000).

| group | n | Holm gain / loss B=10000 | B=1000 | Bonferroni gain / loss B=10000 | B=1000 | unadjusted gain / loss B=10000 | B=1000 | CI excl. 0 gain / loss B=10000 | B=1000 |
|---|---|---|---|---|---|---|---|---|---|
| all | 44 | 21 / 1 | 21 / 1 | 21 / 1 | 20 / 0 | 27 / 1 | 28 / 2 | 27 / 1 | 28 / 2 |
| yolo | 34 | 17 / 0 | 17 / 0 | 17 / 0 | 17 / 0 | 23 / 0 | 24 / 0 | 23 / 0 | 24 / 0 |
| rtdetr | 10 | 4 / 1 | 4 / 1 | 4 / 1 | 3 / 0 | 4 / 1 | 4 / 2 | 4 / 1 | 4 / 2 |

Cells whose Holm status changes between B = 1000 and B = 10000: 0 (none). Max |change in point diff F1| = 0.00e+00 (point estimates do not depend on B).

| cell | family | diff F1 | CI95 B=10000 | CI95 B=1000 | p two-sided B=10000 | B=1000 | p Holm B=10000 | B=1000 | Holm B=10000 | B=1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| rtdetrl_dfire_s1337__to__pyro_sdis_caltest | rtdetr | +0.038 | [+0.019, +0.057] | [+0.018, +0.058] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | rtdetr | +0.075 | [+0.055, +0.095] | [+0.056, +0.095] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| rtdetrl_pyrosdis_s1337__to__thesis_test_smokeonly | rtdetr | +0.014 | [+0.006, +0.022] | [+0.006, +0.023] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| v8m_dfire_s3407__to__pyro_sdis_caltest | yolo | +0.119 | [+0.098, +0.140] | [+0.099, +0.138] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| v8n_dfire_s3407__to__pyro_sdis_caltest | yolo | +0.066 | [+0.046, +0.086] | [+0.046, +0.087] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| v8n_pyrosdis_s3407__to__d_fire_test_smokeonly | yolo | +0.042 | [+0.029, +0.055] | [+0.030, +0.056] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| v8s_dfire_s1337__to__pyro_sdis_caltest | yolo | +0.072 | [+0.051, +0.092] | [+0.052, +0.091] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| v8s_dfire_s3407__to__pyro_sdis_caltest | yolo | +0.107 | [+0.087, +0.127] | [+0.087, +0.127] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| v8s_dfiredd_s3407__to__pyro_sdis_caltest | yolo | +0.037 | [+0.018, +0.056] | [+0.017, +0.055] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| v8s_dfiresub_s3407__to__pyro_sdis_caltest | yolo | +0.080 | [+0.061, +0.098] | [+0.061, +0.099] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| v8s_pyrosdis_s1337__to__d_fire_test_smokeonly | yolo | +0.032 | [+0.018, +0.046] | [+0.018, +0.046] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| v8s_pyrosdis_s3407__to__d_fire_test_smokeonly | yolo | +0.035 | [+0.021, +0.048] | [+0.022, +0.049] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| y11m_dfire_s3407__to__pyro_sdis_caltest | yolo | +0.085 | [+0.065, +0.106] | [+0.066, +0.106] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| y11n_dfire_s3407__to__pyro_sdis_caltest | yolo | +0.079 | [+0.061, +0.098] | [+0.062, +0.098] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| y11n_pyrosdis_s3407__to__d_fire_test_smokeonly | yolo | +0.046 | [+0.032, +0.060] | [+0.032, +0.061] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| y11s_dfire_s1337__to__pyro_sdis_caltest | yolo | +0.079 | [+0.060, +0.098] | [+0.061, +0.098] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| y11s_dfire_s3407__to__pyro_sdis_caltest | yolo | +0.095 | [+0.076, +0.114] | [+0.076, +0.115] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| y11s_dfiredd_s3407__to__pyro_sdis_caltest | yolo | +0.055 | [+0.037, +0.073] | [+0.037, +0.073] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| y11s_pyrosdis_s1337__to__d_fire_test_smokeonly | yolo | +0.045 | [+0.031, +0.059] | [+0.030, +0.060] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| y11s_pyrosdis_s3407__to__d_fire_test_smokeonly | yolo | +0.047 | [+0.032, +0.061] | [+0.032, +0.061] | 0.0000 | 0.000 | 0.0000 | 0.000 | gain | gain |
| rtdetrl_dfire_s3407__to__thesis_test | rtdetr | -0.038 | [-0.062, -0.017] | [-0.061, -0.016] | 0.0004 | 0.002 | 0.0096 | 0.048 | LOSS | LOSS |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | rtdetr | +0.016 | [+0.007, +0.027] | [+0.007, +0.026] | 0.0006 | 0.002 | 0.0138 | 0.048 | gain | gain |
| v8s_pyrosdis_s3407__to__thesis_test_smokeonly | yolo | +0.017 | [+0.004, +0.032] | [+0.004, +0.032] | 0.0100 | 0.010 | 0.2200 | 0.210 |  |  |
| y11s_pyrosdis_s1337__to__thesis_test_smokeonly | yolo | +0.015 | [+0.003, +0.029] | [+0.003, +0.029] | 0.0114 | 0.008 | 0.2394 | 0.176 |  |  |
| v8n_pyrosdis_s3407__to__thesis_test_smokeonly | yolo | +0.017 | [+0.004, +0.033] | [+0.003, +0.032] | 0.0128 | 0.024 | 0.2560 | 0.480 |  |  |
| v8s_dfiresub_s3407__to__thesis_test | yolo | +0.026 | [+0.004, +0.050] | [+0.003, +0.050] | 0.0192 | 0.028 | 0.3648 | 0.532 |  |  |
| y11s_pyrosdis_s3407__to__thesis_test_smokeonly | yolo | +0.014 | [+0.003, +0.028] | [+0.003, +0.028] | 0.0310 | 0.040 | 0.5580 | 0.720 |  |  |
| y11n_pyrosdis_s3407__to__thesis_test_smokeonly | yolo | +0.015 | [+0.003, +0.032] | [+0.003, +0.033] | 0.0338 | 0.040 | 0.5746 | 0.720 |  |  |
| rtdetrl_dfire_s1337__to__thesis_test | rtdetr | -0.021 | [-0.043, +0.000] | [-0.042, -0.000] | 0.0526 | 0.050 | 0.8416 | 0.750 |  |  |
| v8s_dfire_s3407__to__thesis_test | yolo | +0.022 | [-0.001, +0.045] | [+0.001, +0.046] | 0.0628 | 0.044 | 0.9420 | 0.720 |  |  |
| v8s_pyrosdis_s1337__to__thesis_test_smokeonly | yolo | +0.009 | [+0.000, +0.021] | [+0.000, +0.021] | 0.0964 | 0.080 | 1.0000 | 1.000 |  |  |
| y11s_dfire_s1337__to__thesis_test | yolo | +0.015 | [-0.008, +0.038] | [-0.008, +0.039] | 0.1982 | 0.188 | 1.0000 | 1.000 |  |  |
| y11s_dfire_s3407__to__thesis_test | yolo | +0.013 | [-0.009, +0.035] | [-0.008, +0.034] | 0.2514 | 0.234 | 1.0000 | 1.000 |  |  |
| rtdetrl_pyrosdis_s3407_p60__to__d_fire_test_smokeonly | rtdetr | +0.008 | [-0.006, +0.021] | [-0.006, +0.021] | 0.2532 | 0.220 | 1.0000 | 1.000 |  |  |
| v8s_dfiredd_s3407__to__thesis_test | yolo | +0.015 | [-0.013, +0.041] | [-0.013, +0.043] | 0.2886 | 0.306 | 1.0000 | 1.000 |  |  |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | rtdetr | -0.004 | [-0.018, +0.010] | [-0.018, +0.009] | 0.5224 | 0.484 | 1.0000 | 1.000 |  |  |
| y11s_dfiredd_s3407__to__thesis_test | yolo | -0.007 | [-0.035, +0.018] | [-0.035, +0.019] | 0.5854 | 0.622 | 1.0000 | 1.000 |  |  |
| rtdetrl_pyrosdis_s1337__to__d_fire_test_smokeonly | rtdetr | +0.003 | [-0.012, +0.018] | [-0.012, +0.019] | 0.7280 | 0.716 | 1.0000 | 1.000 |  |  |
| y11m_dfire_s3407__to__thesis_test | yolo | +0.003 | [-0.019, +0.025] | [-0.017, +0.026] | 0.8070 | 0.812 | 1.0000 | 1.000 |  |  |
| v8s_dfire_s1337__to__thesis_test | yolo | -0.001 | [-0.023, +0.019] | [-0.023, +0.020] | 0.8744 | 0.882 | 1.0000 | 1.000 |  |  |
| y11n_dfire_s3407__to__thesis_test | yolo | +0.002 | [-0.022, +0.025] | [-0.022, +0.028] | 0.8818 | 0.864 | 1.0000 | 1.000 |  |  |
| v8m_dfire_s3407__to__thesis_test | yolo | +0.001 | [-0.018, +0.021] | [-0.017, +0.021] | 0.9404 | 0.974 | 1.0000 | 1.000 |  |  |
| v8n_dfire_s3407__to__thesis_test | yolo | -0.000 | [-0.022, +0.021] | [-0.021, +0.021] | 0.9514 | 0.950 | 1.0000 | 1.000 |  |  |
| rtdetrl_pyrosdis_s3407_p60__to__thesis_test_smokeonly | rtdetr | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | 1.0000 | 1.000 | 1.0000 | 1.000 |  |  |
