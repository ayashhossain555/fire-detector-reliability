# Paired post-hoc map vs identity: B = 10000 vs B = 1000 (11 cells; machine-written by bootstrap_ci_B10000_compare.py, 2026-09-13T01:02:16)

LaECE_0 in percentage points (x100). Two-sided bootstrap p resolution 0.0002 (B = 10000) vs 0.002 (B = 1000). Significance-call changes (CI excludes 0): 0.

| cell | map | diff LaECE_0 (pp) | CI95 B=10000 | CI95 B=1000 | p B=10000 | p B=1000 | call B=10000 | B=1000 | diff D_ECE (pp) | p D_ECE B=10000 | B=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rtdetrl_dfire_s1337__to__pyro_sdis_caltest | temperature | -6.9 | [-7.1, -6.7] | [-7.1, -6.7] | 0.0000 | 0.000 | helps | helps | -6.3 | 0.0000 | 0.000 |
| rtdetrl_dfire_s1337__to__pyro_sdis_caltest | platt | -15.7 | [-15.9, -14.0] | [-15.9, -14.1] | 0.0000 | 0.000 | helps | helps | -11.0 | 0.0090 | 0.006 |
| rtdetrl_dfire_s1337__to__pyro_sdis_caltest | isotonic | -17.0 | [-18.0, -13.5] | [-18.0, -13.6] | 0.0000 | 0.000 | helps | helps | -8.8 | 0.0450 | 0.042 |
| rtdetrl_dfire_s1337__to__thesis_test | temperature | -6.3 | [-6.6, -6.0] | [-6.5, -6.0] | 0.0000 | 0.000 | helps | helps | -6.1 | 0.0000 | 0.000 |
| rtdetrl_dfire_s1337__to__thesis_test | platt | -14.5 | [-15.2, -13.2] | [-15.2, -13.2] | 0.0000 | 0.000 | helps | helps | -14.6 | 0.0000 | 0.000 |
| rtdetrl_dfire_s1337__to__thesis_test | isotonic | -15.6 | [-16.5, -14.2] | [-16.5, -14.3] | 0.0000 | 0.000 | helps | helps | -15.4 | 0.0000 | 0.000 |
| rtdetrl_dfire_s3407__to__d_fire_test | temperature | -7.9 | [-8.4, -7.1] | [-8.4, -7.1] | 0.0000 | 0.000 | helps | helps | +8.2 | 0.0000 | 0.000 |
| rtdetrl_dfire_s3407__to__d_fire_test | platt | -11.3 | [-12.0, -9.8] | [-12.0, -9.8] | 0.0000 | 0.000 | helps | helps | +10.1 | 0.0000 | 0.000 |
| rtdetrl_dfire_s3407__to__d_fire_test | isotonic | -11.1 | [-11.8, -9.7] | [-11.9, -9.7] | 0.0000 | 0.000 | helps | helps | +10.2 | 0.0000 | 0.000 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | temperature | -8.7 | [-8.9, -8.5] | [-8.9, -8.5] | 0.0000 | 0.000 | helps | helps | -8.2 | 0.0262 | 0.022 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | platt | -17.7 | [-17.8, -15.1] | [-17.8, -15.2] | 0.0000 | 0.000 | helps | helps | -1.5 | 0.7880 | 0.786 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | isotonic | -12.0 | [-15.0, -8.5] | [-15.0, -8.5] | 0.0000 | 0.000 | helps | helps | +2.8 | 0.3458 | 0.318 |
| rtdetrl_dfire_s3407__to__thesis_test | temperature | -9.3 | [-9.6, -8.6] | [-9.6, -8.6] | 0.0000 | 0.000 | helps | helps | -9.3 | 0.0000 | 0.000 |
| rtdetrl_dfire_s3407__to__thesis_test | platt | -16.2 | [-16.7, -13.4] | [-16.8, -13.5] | 0.0000 | 0.000 | helps | helps | -16.4 | 0.0000 | 0.000 |
| rtdetrl_dfire_s3407__to__thesis_test | isotonic | -17.4 | [-18.6, -15.4] | [-18.6, -15.5] | 0.0000 | 0.000 | helps | helps | -17.1 | 0.0000 | 0.000 |
| v8s_dfire_s3407__to__thesis_test | temperature | +0.4 | [-0.6, +2.2] | [-0.6, +2.2] | 0.2796 | 0.276 | ns | ns | +0.7 | 0.4248 | 0.398 |
| v8s_dfire_s3407__to__thesis_test | platt | -2.1 | [-3.1, -0.6] | [-3.1, -0.5] | 0.0110 | 0.014 | helps | helps | -3.1 | 0.0108 | 0.014 |
| v8s_dfire_s3407__to__thesis_test | isotonic | -3.0 | [-4.6, -1.6] | [-4.6, -1.6] | 0.0002 | 0.000 | helps | helps | -3.1 | 0.0348 | 0.036 |
| v8s_pyrosdis_s3407__to__thesis_test_smokeonly | temperature | -0.1 | [-0.4, +0.1] | [-0.4, +0.1] | 0.3938 | 0.398 | ns | ns | -0.1 | 0.3938 | 0.398 |
| v8s_pyrosdis_s3407__to__thesis_test_smokeonly | platt | -1.0 | [-1.4, -0.6] | [-1.4, -0.6] | 0.0000 | 0.000 | helps | helps | -1.0 | 0.0000 | 0.000 |
| v8s_pyrosdis_s3407__to__thesis_test_smokeonly | isotonic | -0.4 | [-6.2, +2.5] | [-6.2, +2.5] | 0.8866 | 0.904 | ns | ns | -0.4 | 0.8866 | 0.904 |
| y11s_dfire_s3407__to__d_fire_test | temperature | +0.1 | [-0.5, +0.4] | [-0.6, +0.4] | 0.8256 | 0.812 | ns | ns | -0.9 | 0.0000 | 0.000 |
| y11s_dfire_s3407__to__d_fire_test | platt | -0.9 | [-1.6, -0.4] | [-1.5, -0.4] | 0.0014 | 0.000 | helps | helps | +1.4 | 0.0016 | 0.000 |
| y11s_dfire_s3407__to__d_fire_test | isotonic | -1.5 | [-2.2, -0.8] | [-2.1, -0.8] | 0.0000 | 0.000 | helps | helps | +1.5 | 0.0012 | 0.000 |
| y11s_dfire_s3407__to__thesis_test | temperature | -0.2 | [-1.0, +1.5] | [-1.0, +1.6] | 0.7598 | 0.780 | ns | ns | +0.3 | 0.9156 | 0.868 |
| y11s_dfire_s3407__to__thesis_test | platt | -2.2 | [-2.4, +0.2] | [-2.4, +0.1] | 0.0788 | 0.066 | ns | ns | -2.5 | 0.2866 | 0.286 |
| y11s_dfire_s3407__to__thesis_test | isotonic | -2.2 | [-3.8, -0.5] | [-3.4, -0.6] | 0.0118 | 0.012 | helps | helps | -2.8 | 0.0834 | 0.058 |
| y11s_pyrosdis_s3407__to__d_fire_test_smokeonly | temperature | +0.0 | [-1.4, +0.6] | [-1.4, +0.5] | 0.9232 | 0.974 | ns | ns | +0.1 | 0.6082 | 0.634 |
| y11s_pyrosdis_s3407__to__d_fire_test_smokeonly | platt | +1.0 | [+0.7, +1.7] | [+0.7, +1.7] | 0.0036 | 0.000 | harms | harms | +0.8 | 0.3490 | 0.358 |
| y11s_pyrosdis_s3407__to__d_fire_test_smokeonly | isotonic | +0.3 | [-1.1, +1.5] | [-1.1, +1.5] | 0.5216 | 0.558 | ns | ns | +0.4 | 0.4554 | 0.478 |
| y11s_pyrosdis_s3407__to__thesis_test_smokeonly | temperature | -0.3 | [-0.7, +0.2] | [-0.7, +0.2] | 0.2422 | 0.252 | ns | ns | -0.3 | 0.2422 | 0.252 |
| y11s_pyrosdis_s3407__to__thesis_test_smokeonly | platt | +0.6 | [+0.3, +0.9] | [+0.3, +0.9] | 0.0000 | 0.000 | harms | harms | +0.6 | 0.0000 | 0.000 |
| y11s_pyrosdis_s3407__to__thesis_test_smokeonly | isotonic | -1.2 | [-3.6, +1.2] | [-3.6, +1.4] | 0.2922 | 0.298 | ns | ns | -1.2 | 0.2922 | 0.298 |
