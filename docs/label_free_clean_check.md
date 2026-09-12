# Label-free threshold transfer — clean-split check, shift-rule, Table-4 tallies (machine-written by label_free_clean_check.py)

Created 2026-09-13T00:47:58. Keys quoted as part_A_clean_split / part_B_shift_rule / part_C_table4_tallies of label_free_clean_check.json.

## Part A — kept-rate rule with the de-duplicated D-Fire calibration split as pool + oracle set, clean smoke-only test

Mapping check: {'n_dedup_images': 927, 'n_calval_images': 1722, 'ids_subset': True, 'same_file_name_for_same_id': True, 'annotations_identical_on_dedup_images': True, 'n_annotations_dedup': 1623, 'n_annotations_calval_restricted_to_dedup_ids': 1623}. Official-split reproduction of t_M1c / t_ORC vs the refine artefact: 9 cells, max |dt| M1c = 0.00e+00, ORC = 0.00e+00, max |dF1| = 2.78e-17 (passed); cells without a refine counterpart (official numbers computed here only): ['v8m_pyrosdis_s3407__to__d_fire_test_smokeonly', 'y11m_pyrosdis_s3407__to__d_fire_test_smokeonly']. Not evaluable: none. Bootstrap: B = 1000, seed 0, clean-test images.

| cell | t_src | t_M1c clean / official | t_ORC clean / official | F1 t_src clean [CI] / official | F1 M1c clean [CI] / official | F1 ORC clean [CI] / official | regret M1c clean [CI] / official | gap closed clean [CI] / official | F1 official t_M1c on clean test |
|---|---|---|---|---|---|---|---|---|---|
| v8n_pyrosdis_s3407__to__d_fire_test_smokeonly | 0.344 | 0.012 / 0.013 | 0.039 / 0.040 | 0.043 [0.028, 0.058] / 0.076 | 0.101 [0.086, 0.114] / 0.118 | 0.096 [0.079, 0.112] / 0.127 | -0.005 [-0.015, 0.006] / 0.009 | 1.091 [0.907, 1.359] / 0.831 | 0.100 |
| v8s_pyrosdis_s3407__to__d_fire_test_smokeonly | 0.312 | 0.004 / 0.005 | 0.005 / 0.036 | 0.033 [0.021, 0.047] / 0.083 | 0.098 [0.085, 0.111] / 0.118 | 0.100 [0.086, 0.113] / 0.137 | 0.002 [-0.003, 0.006] / 0.019 | 0.976 [0.914, 1.047] / 0.644 | 0.098 |
| v8m_pyrosdis_s3407__to__d_fire_test_smokeonly | 0.292 | 0.003 / 0.004 | 0.009 / 0.019 | 0.030 [0.019, 0.044] / 0.087 | 0.105 [0.092, 0.119] / 0.138 | 0.108 [0.091, 0.127] / 0.157 | 0.003 [-0.007, 0.012] / 0.019 | 0.960 [0.853, 1.096] / 0.725 | 0.108 |
| y11n_pyrosdis_s3407__to__d_fire_test_smokeonly | 0.316 | 0.010 / 0.012 | 0.010 / 0.032 | 0.037 [0.025, 0.051] / 0.085 | 0.111 [0.098, 0.125] / 0.131 | 0.112 [0.099, 0.126] / 0.150 | 0.001 [-0.001, 0.002] / 0.019 | 0.987 [0.967, 1.018] / 0.711 | 0.115 |
| y11s_pyrosdis_s3407__to__d_fire_test_smokeonly | 0.306 | 0.004 / 0.005 | 0.018 / 0.017 | 0.043 [0.028, 0.059] / 0.092 | 0.111 [0.096, 0.126] / 0.139 | 0.112 [0.094, 0.131] / 0.156 | 0.001 [-0.010, 0.011] / 0.018 | 0.986 [0.856, 1.153] / 0.725 | 0.111 |
| v8s_pyrosdis_s1337__to__d_fire_test_smokeonly | 0.295 | 0.003 / 0.005 | 0.005 / 0.026 | 0.029 [0.017, 0.041] / 0.088 | 0.084 [0.070, 0.096] / 0.119 | 0.084 [0.071, 0.098] / 0.137 | 0.000 [-0.004, 0.004] / 0.017 | 0.991 [0.921, 1.074] / 0.645 | 0.085 |
| y11s_pyrosdis_s1337__to__d_fire_test_smokeonly | 0.317 | 0.007 / 0.008 | 0.018 / 0.047 | 0.032 [0.020, 0.046] / 0.092 | 0.110 [0.095, 0.125] / 0.137 | 0.112 [0.095, 0.129] / 0.155 | 0.002 [-0.007, 0.011] / 0.018 | 0.975 [0.876, 1.088] / 0.712 | 0.110 |
| rtdetrl_pyrosdis_s1337__to__d_fire_test_smokeonly | 0.373 | 0.120 / 0.130 | 0.207 / 0.246 | 0.127 [0.106, 0.150] / 0.166 | 0.154 [0.139, 0.169] / 0.169 | 0.180 [0.158, 0.204] / 0.202 | 0.026 [0.012, 0.040] / 0.033 | 0.506 [0.181, 0.756] / 0.078 | 0.162 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | 0.411 | 0.148 / 0.162 | 0.192 / 0.285 | 0.065 [0.048, 0.083] / 0.141 | 0.107 [0.093, 0.122] / 0.136 | 0.112 [0.096, 0.127] / 0.163 | 0.004 [-0.004, 0.012] / 0.027 | 0.905 [0.723, 1.096] / -0.200 | 0.110 |
| rtdetrl_pyrosdis_s3407_p60__to__d_fire_test_smokeonly | 0.341 | 0.021 / 0.021 | 0.057 / 0.064 | 0.049 [0.034, 0.067] / 0.094 | 0.061 [0.050, 0.073] / 0.102 | 0.078 [0.062, 0.096] / 0.139 | 0.017 [0.008, 0.026] / 0.037 | 0.395 [-0.153, 0.689] / 0.174 | 0.062 |
| y11m_pyrosdis_s3407__to__d_fire_test_smokeonly | 0.298 | 0.003 / 0.004 | 0.008 / 0.031 | 0.045 [0.030, 0.060] / 0.105 | 0.113 [0.100, 0.126] / 0.143 | 0.116 [0.100, 0.132] / 0.163 | 0.003 [-0.006, 0.011] / 0.019 | 0.958 [0.850, 1.094] / 0.669 | 0.115 |

| group | n | F1 t_src clean / off | F1 M1c clean / off | F1 ORC clean / off | regret t_src clean / off | regret M1c clean [CI img] [CI cell] / off | gap closed M1c clean / off | M1c beats/worse t_src (clean) | regret D25 clean / off |
|---|---|---|---|---|---|---|---|---|---|
| all | 11 | 0.049 / 0.101 | 0.105 / 0.132 | 0.110 / 0.153 | 0.061 / 0.052 | 0.005 [0.001, 0.009] [0.001, 0.011] / 0.021 | 0.885 / 0.520 | 11/0 | 0.049 / 0.039 |
| core_9 | 9 | 0.049 / 0.101 | 0.109 / 0.134 | 0.113 / 0.154 | 0.064 / 0.053 | 0.004 [0.000, 0.007] [-0.000, 0.010] / 0.020 | 0.931 / 0.541 | 9/0 | 0.051 / 0.040 |
| in_refine_artefact | 9 | 0.051 / 0.102 | 0.104 / 0.130 | 0.109 / 0.152 | 0.059 / 0.050 | 0.005 [0.002, 0.009] [0.000, 0.012] / 0.022 | 0.868 / 0.480 | 9/0 | 0.044 / 0.035 |
| yolo | 8 | 0.037 / 0.088 | 0.104 / 0.130 | 0.105 / 0.148 | 0.068 / 0.059 | 0.001 [-0.003, 0.004] [-0.001, 0.002] / 0.017 | 0.991 / 0.708 | 8/0 | 0.063 / 0.050 |
| rtdetr | 3 | 0.081 / 0.134 | 0.107 / 0.136 | 0.123 / 0.168 | 0.043 / 0.034 | 0.016 [0.009, 0.023] [0.004, 0.026] / 0.032 | 0.602 / 0.017 | 3/0 | 0.011 / 0.007 |

## Part B — shift diagnostic W1 (per-image max) and an in-domain null cut-off

Pilot W1 reproduced (max |dW1| = 0.00e+00). In-domain null: n = 68 (24 own-split + 44 camera-fold values); max = 0.0858 (split max 0.0133 at v8s_dfiresub_s3407; camera-fold max 0.0858 at {'run': 'y11m_pyrosdis_s3407', 'fold': 0}), p95 = 0.0771, median = 0.0415; max including the secondary dedup-test values = 0.1185.

Cut-off = 0.0858: 43 of 44 cross cells above, 1 below (by target above/n: d_fire 9/9, pyro_sdis 13/13, thesis 21/22).

| side | n | regret t_src mean | median | max | abs drift mean | median | max | regret M1c mean | W1 min-max | n drift>0.1 | n regret>0.02 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| above | 43 | 0.042 | 0.023 | 0.124 | 0.242 | 0.270 | 0.354 | 0.009 | 0.087-0.506 | 41 | 24 |
| below | 1 | -0.009 | -0.009 | -0.009 | 0.103 | 0.103 | 0.103 | 0.028 | 0.078-0.078 | 1 | 0 |

ROC AUC of W1_imgmax: |drift| > 0.1: 0.917 (pos 42/neg 2); regret(t_src) > 0.02: 0.631 (pos 24/neg 20). W1_pooled: 0.619 / 0.471. Spearman W1_imgmax vs |drift| rho = 0.680, vs regret rho = 0.256.

- null cut-off vs abs_drift_gt_0.1: tp 41 fp 2 fn 1 tn 0, sens 0.976, spec 0.000, bal-acc 0.488; leave-one-target-out fitted cut-off (balanced accuracy): pooled acc 0.795, sens 0.786, spec 1.000 (fitted cut-offs: d_fire 0.100, pyro_sdis 0.100, thesis 0.148)
- null cut-off vs regret_t_src_gt_0.02: tp 24 fp 19 fn 0 tn 1, sens 1.000, spec 0.050, bal-acc 0.525; leave-one-target-out fitted cut-off (balanced accuracy): pooled acc 0.705, sens 0.875, spec 0.500 (fitted cut-offs: d_fire 0.114, pyro_sdis 0.187, thesis 0.148)

| cell | target | W1 imgmax | above | abs drift | regret t_src | regret M1c |
|---|---|---|---|---|---|---|
| v8s_pyrosdis_s3407__to__thesis_test_smokeonly | thesis | 0.5065 | yes | 0.311 | 0.021 | 0.004 |
| v8s_pyrosdis_s1337__to__thesis_test_smokeonly | thesis | 0.5035 | yes | 0.291 | 0.010 | 0.001 |
| v8n_pyrosdis_s3407__to__thesis_test_smokeonly | thesis | 0.4988 | yes | 0.334 | 0.013 | -0.004 |
| y11s_pyrosdis_s3407__to__thesis_test_smokeonly | thesis | 0.4922 | yes | 0.304 | 0.010 | -0.003 |
| y11s_pyrosdis_s1337__to__thesis_test_smokeonly | thesis | 0.4874 | yes | 0.314 | 0.018 | 0.003 |
| y11n_pyrosdis_s3407__to__thesis_test_smokeonly | thesis | 0.4807 | yes | 0.313 | 0.018 | 0.003 |
| rtdetrl_pyrosdis_s3407_p60__to__thesis_test_smokeonly | thesis | 0.4800 | yes | 0.326 | 0.000 | 0.000 |
| v8s_pyrosdis_s3407__to__d_fire_test_smokeonly | d_fire | 0.4745 | yes | 0.276 | 0.054 | 0.019 |
| v8n_pyrosdis_s3407__to__d_fire_test_smokeonly | d_fire | 0.4664 | yes | 0.304 | 0.051 | 0.009 |
| v8s_pyrosdis_s1337__to__d_fire_test_smokeonly | d_fire | 0.4650 | yes | 0.269 | 0.049 | 0.017 |
| y11s_pyrosdis_s3407__to__d_fire_test_smokeonly | d_fire | 0.4595 | yes | 0.289 | 0.064 | 0.018 |
| rtdetrl_pyrosdis_s3407_p60__to__d_fire_test_smokeonly | d_fire | 0.4559 | yes | 0.277 | 0.044 | 0.037 |
| y11n_pyrosdis_s3407__to__d_fire_test_smokeonly | d_fire | 0.4444 | yes | 0.284 | 0.065 | 0.019 |
| y11s_pyrosdis_s1337__to__d_fire_test_smokeonly | d_fire | 0.4421 | yes | 0.270 | 0.063 | 0.018 |
| rtdetrl_pyrosdis_s3407__to__thesis_test_smokeonly | thesis | 0.4039 | yes | 0.282 | 0.016 | -0.000 |
| rtdetrl_pyrosdis_s1337__to__thesis_test_smokeonly | thesis | 0.3816 | yes | 0.240 | 0.012 | -0.001 |
| v8s_dfiredd_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.3650 | yes | 0.298 | 0.062 | 0.025 |
| rtdetrl_pyrosdis_s1337__to__d_fire_test_smokeonly | d_fire | 0.3442 | yes | 0.127 | 0.036 | 0.033 |
| rtdetrl_pyrosdis_s3407__to__d_fire_test_smokeonly | d_fire | 0.3409 | yes | 0.126 | 0.022 | 0.027 |
| y11s_dfiredd_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.3369 | yes | 0.279 | 0.070 | 0.015 |
| y11n_dfire_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.2583 | yes | 0.267 | 0.088 | 0.009 |
| y11m_dfire_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.2530 | yes | 0.280 | 0.106 | 0.020 |
| v8s_dfire_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.2493 | yes | 0.282 | 0.113 | 0.006 |
| v8s_dfire_s1337__to__pyro_sdis_caltest | pyro_sdis | 0.2487 | yes | 0.251 | 0.089 | 0.018 |
| y11s_dfire_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.2462 | yes | 0.303 | 0.101 | 0.006 |
| v8n_dfire_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.2402 | yes | 0.194 | 0.082 | 0.016 |
| y11s_dfire_s1337__to__pyro_sdis_caltest | pyro_sdis | 0.2359 | yes | 0.263 | 0.083 | 0.005 |
| v8m_dfire_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.2309 | yes | 0.344 | 0.124 | 0.005 |
| v8s_dfiresub_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.2147 | yes | 0.244 | 0.090 | 0.010 |
| v8s_dfiredd_s3407__to__thesis_test | thesis | 0.1868 | yes | 0.116 | 0.010 | -0.005 |
| rtdetrl_dfire_s3407__to__pyro_sdis_caltest | pyro_sdis | 0.1807 | yes | 0.354 | 0.093 | 0.019 |
| y11s_dfiredd_s3407__to__thesis_test | thesis | 0.1687 | yes | 0.178 | 0.007 | 0.014 |
| rtdetrl_dfire_s1337__to__pyro_sdis_caltest | pyro_sdis | 0.1480 | yes | 0.306 | 0.055 | 0.017 |
| v8s_dfiresub_s3407__to__thesis_test | thesis | 0.1137 | yes | 0.228 | 0.006 | -0.020 |
| y11s_dfire_s1337__to__thesis_test | thesis | 0.1074 | yes | 0.213 | 0.017 | 0.002 |
| v8s_dfire_s3407__to__thesis_test | thesis | 0.1071 | yes | 0.176 | 0.023 | 0.001 |
| v8m_dfire_s3407__to__thesis_test | thesis | 0.1032 | yes | 0.170 | -0.004 | -0.005 |
| y11s_dfire_s3407__to__thesis_test | thesis | 0.1022 | yes | 0.176 | 0.013 | -0.000 |
| v8s_dfire_s1337__to__thesis_test | thesis | 0.0998 | yes | 0.051 | 0.004 | 0.005 |
| y11m_dfire_s3407__to__thesis_test | thesis | 0.0975 | yes | 0.129 | 0.005 | 0.002 |
| v8n_dfire_s3407__to__thesis_test | thesis | 0.0971 | yes | 0.079 | 0.004 | 0.004 |
| y11n_dfire_s3407__to__thesis_test | thesis | 0.0905 | yes | 0.169 | -0.001 | -0.003 |
| rtdetrl_dfire_s1337__to__thesis_test | thesis | 0.0866 | yes | 0.130 | -0.011 | 0.010 |
| rtdetrl_dfire_s3407__to__thesis_test | thesis | 0.0775 | no | 0.103 | -0.009 | 0.028 |

| in-domain null (own split) | source calval | test | W1 imgmax |
|---|---|---|---|
| rtdetrl_dfire_s1337 | d_fire_calval | d_fire_test | 0.0097 |
| rtdetrl_dfire_s3407 | d_fire_calval | d_fire_test | 0.0126 |
| rtdetrl_pyrosdis_s1337 | pyro_sdis_calval | pyro_sdis_caltest | 0.0068 |
| rtdetrl_pyrosdis_s3407 | pyro_sdis_calval | pyro_sdis_caltest | 0.0046 |
| rtdetrl_pyrosdis_s3407_p60 | pyro_sdis_calval | pyro_sdis_caltest | 0.0055 |
| v8m_dfire_s3407 | d_fire_calval | d_fire_test | 0.0106 |
| v8m_pyrosdis_s3407 | pyro_sdis_calval | pyro_sdis_caltest | 0.0036 |
| v8n_dfire_s3407 | d_fire_calval | d_fire_test | 0.0102 |
| v8n_pyrosdis_s3407 | pyro_sdis_calval | pyro_sdis_caltest | 0.0067 |
| v8s_dfire_s1337 | d_fire_calval | d_fire_test | 0.0105 |
| v8s_dfire_s3407 | d_fire_calval | d_fire_test | 0.0127 |
| v8s_dfiredd_s3407 | d_fire_calval_dedup | d_fire_test_dedup | 0.0115 |
| v8s_dfiresub_s3407 | d_fire_calval | d_fire_test | 0.0133 |
| v8s_pyrosdis_s1337 | pyro_sdis_calval | pyro_sdis_caltest | 0.0085 |
| v8s_pyrosdis_s3407 | pyro_sdis_calval | pyro_sdis_caltest | 0.0056 |
| y11m_dfire_s3407 | d_fire_calval | d_fire_test | 0.0116 |
| y11m_pyrosdis_s3407 | pyro_sdis_calval | pyro_sdis_caltest | 0.0076 |
| y11n_dfire_s3407 | d_fire_calval | d_fire_test | 0.0095 |
| y11n_pyrosdis_s3407 | pyro_sdis_calval | pyro_sdis_caltest | 0.0074 |
| y11s_dfire_s1337 | d_fire_calval | d_fire_test | 0.0092 |
| y11s_dfire_s3407 | d_fire_calval | d_fire_test | 0.0113 |
| y11s_dfiredd_s3407 | d_fire_calval_dedup | d_fire_test_dedup | 0.0059 |
| y11s_pyrosdis_s1337 | pyro_sdis_calval | pyro_sdis_caltest | 0.0074 |
| y11s_pyrosdis_s3407 | pyro_sdis_calval | pyro_sdis_caltest | 0.0054 |

| in-domain null (camera folds) | fold 0 | fold 1 | fold 2 | fold 3 | max |
|---|---|---|---|---|---|
| rtdetrl_pyrosdis_s1337 | 0.0364 | 0.0315 | 0.0201 | 0.0466 | 0.0466 |
| rtdetrl_pyrosdis_s3407 | 0.0224 | 0.0506 | 0.0080 | 0.0665 | 0.0665 |
| rtdetrl_pyrosdis_s3407_p60 | 0.0712 | 0.0378 | 0.0428 | 0.0800 | 0.0800 |
| v8m_pyrosdis_s3407 | 0.0701 | 0.0658 | 0.0637 | 0.0677 | 0.0701 |
| v8n_pyrosdis_s3407 | 0.0484 | 0.0554 | 0.0531 | 0.0462 | 0.0554 |
| v8s_pyrosdis_s1337 | 0.0576 | 0.0422 | 0.0416 | 0.0587 | 0.0587 |
| v8s_pyrosdis_s3407 | 0.0752 | 0.0414 | 0.0380 | 0.0719 | 0.0752 |
| y11m_pyrosdis_s3407 | 0.0858 | 0.0627 | 0.0442 | 0.0671 | 0.0858 |
| y11n_pyrosdis_s3407 | 0.0440 | 0.0091 | 0.0372 | 0.0755 | 0.0755 |
| y11s_pyrosdis_s1337 | 0.0780 | 0.0726 | 0.0516 | 0.0571 | 0.0780 |
| y11s_pyrosdis_s3407 | 0.0739 | 0.0781 | 0.0650 | 0.0606 | 0.0781 |

## Part C — Table-4 tallies (n = beats + worse + ties/undefined)

| rule | artefact | n | beats t_src | worse | ties/undefined | regret mean | regret median |
|---|---|---|---|---|---|---|---|
| M0_t_src | refine | 44 | 0 | 0 | 44 | 0.041 | 0.023 |
| D25_default | refine | 44 | 34 | 3 | 7 | 0.025 | 0.018 |
| M1c_kept_rate | refine | 44 | 37 | 6 | 1 | 0.009 | 0.006 |
| M1r_rank_mean | refine | 44 | 37 | 6 | 1 | 0.009 | 0.006 |
| M1c_top5 | refine | 44 | 36 | 7 | 1 | 0.010 | 0.005 |
| M1c_top10 | refine | 44 | 36 | 7 | 1 | 0.009 | 0.006 |
| M1c_top20 | refine | 44 | 37 | 6 | 1 | 0.009 | 0.006 |
| M1c_top50 | refine | 44 | 37 | 6 | 1 | 0.009 | 0.006 |
| M1d_rank_norm | refine | 44 | 27 | 16 | 1 | 0.022 | 0.010 |
| M1e_rank_log | refine | 44 | 36 | 8 | 0 | 0.014 | 0.008 |
| ORC_target_calval | refine | 44 | 39 | 4 | 1 | 0.000 | 0.000 |
| M1a_quantile_pooled | pilot | 44 | 29 | 14 | 1 | 0.014 | 0.004 |
| M1b_quantile_imgmax | pilot | 44 | 32 | 12 | 0 | 0.026 | 0.012 |
| M2_beta_mom_post05 | pilot | 44 | 3 | 32 | 9 | 0.135 | 0.146 |
| M2_beta_mom_mixf1 | pilot | 44 | 3 | 32 | 9 | 0.100 | 0.101 |
| M2_beta_mle_post05 | pilot | 44 | 22 | 15 | 7 | 0.046 | 0.023 |
| M2_beta_mle_mixf1 | pilot | 44 | 21 | 15 | 8 | 0.041 | 0.021 |
| M2_hist20_post05 | pilot | 44 | 0 | 35 | 9 | 0.166 | 0.160 |
| M2_hist20_mixf1 | pilot | 44 | 0 | 35 | 9 | 0.122 | 0.127 |
| M3_beta_mle_shift_post05 | pilot | 44 | 24 | 17 | 3 | 0.028 | 0.011 |
| M3_beta_mle_shift_mixf1 | pilot | 44 | 24 | 17 | 3 | 0.030 | 0.011 |
| M3_hist20_shift_post05 | pilot | 44 | 0 | 35 | 9 | 0.164 | 0.160 |
| M3_hist20_shift_mixf1 | pilot | 44 | 0 | 35 | 9 | 0.119 | 0.118 |
| TOPT_test_optimal | pilot | 44 | 44 | 0 | 0 | -0.005 | -0.005 |
