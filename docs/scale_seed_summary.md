# Scale / seed summary (from matrix_summary.json; LaECE ×100)

## A. Scale (seed 3407)

| run | val mAP50 | in-domain LaECE id | D-ECE | LRP-FN | cross LaECE id (per target) | sens / FAR @t_src (main cross) |
|---|---|---|---|---|---|---|
| rtdetrl_dfire_s3407 | 0.789 | 13.8 [12.9, 14.6] | 4.4 | 17.0 | →Pyro-SDIS 26.8; →thesis 40.6; →FIgLib boxes 28.5 | 0.272 / 0.137 |
| v8n_dfire_s3407 | 0.780 | 2.8 [2.7, 3.9] | 13.4 | 19.4 | →Pyro-SDIS 7.2; →thesis 17.8; →FIgLib boxes 14.8 | 0.232 / 0.172 |
| v8s_dfire_s3407 | 0.790 | 2.6 [2.4, 3.6] | 13.1 | 19.0 | →Pyro-SDIS 5.6; →thesis 21.3; →FIgLib boxes 12.0 | 0.197 / 0.099 |
| v8m_dfire_s3407 | 0.800 | 3.8 [3.3, 4.8] | 10.7 | 15.4 | →Pyro-SDIS 11.0; →thesis 18.0; →FIgLib boxes 11.3 | 0.212 / 0.142 |
| y11n_dfire_s3407 | 0.778 | 3.0 [2.8, 4.0] | 13.8 | 20.0 | →Pyro-SDIS 12.7; →thesis 16.4; →FIgLib boxes 13.0 | 0.169 / 0.145 |
| y11s_dfire_s3407 | 0.798 | 3.5 [3.3, 4.5] | 12.2 | 15.8 | →Pyro-SDIS 12.5; →thesis 16.8; →FIgLib boxes 14.6 | 0.191 / 0.124 |
| y11m_dfire_s3407 | 0.800 | 2.7 [2.4, 3.8] | 12.2 | 17.5 | →Pyro-SDIS 10.5; →thesis 14.9; →FIgLib boxes 19.5 | 0.212 / 0.142 |
| rtdetrl_pyrosdis_s3407 | 0.649 | 7.3 [6.2, 8.9] | 8.0 | 18.0 | →D-Fire smoke 24.4; →D-Fire 24.4; →thesis 51.9; →thesis smoke 51.9; →FIgLib boxes 22.4 | 0.175 / 0.086 |
| v8n_pyrosdis_s3407 | 0.716 | 2.4 [2.3, 4.6] | 10.4 | 25.8 | →D-Fire smoke 24.1; →D-Fire 24.1; →thesis 58.9; →thesis smoke 58.9; →FIgLib boxes 35.5 | 0.078 / 0.034 |
| v8s_pyrosdis_s3407 | 0.735 | 2.0 [2.1, 4.1] | 9.9 | 20.1 | →D-Fire smoke 20.2; →D-Fire 20.2; →thesis 44.8; →thesis smoke 44.8; →FIgLib boxes 14.8 | 0.078 / 0.035 |
| v8m_pyrosdis_s3407 | 0.746 | 2.8 [2.4, 4.7] | 9.8 | 19.1 | →D-Fire smoke 22.3; →D-Fire 22.3; →thesis 54.2; →thesis smoke 54.2; →FIgLib boxes 16.6 | 0.082 / 0.036 |
| y11n_pyrosdis_s3407 | 0.716 | 3.7 [2.9, 5.4] | 13.7 | 22.0 | →D-Fire smoke 22.3; →D-Fire 22.3; →thesis 44.3; →thesis smoke 44.3; →FIgLib boxes 19.0 | 0.085 / 0.037 |
| y11s_pyrosdis_s3407 | 0.729 | 2.6 [2.4, 4.6] | 11.5 | 21.5 | →D-Fire smoke 20.4; →D-Fire 20.4; →thesis 39.5; →thesis smoke 39.5; →FIgLib boxes 13.4 | 0.092 / 0.031 |
| y11m_pyrosdis_s3407 | 0.749 | 2.3 [2.1, 4.2] | 9.7 | 18.2 | →D-Fire smoke 21.1; →D-Fire 21.1; →thesis 47.5; →thesis smoke 47.5; →FIgLib boxes 17.4 | 0.104 / 0.045 |

## B. Seeds (3407 vs 1337)

| model | val mAP50 | in-domain LaECE id | cross-main LaECE id | cross-main sens@t_src |
|---|---|---|---|---|
| rtdetr_dfire | 0.789 / 0.805 | 13.8 / 11.8 | 26.8 / 23.2 | 0.272 / 0.349 |
| rtdetr_pyrosdis | 0.649 / 0.652 | 7.3 / 3.5 | 24.4 / 16.6 | 0.175 / 0.173 |
| v8s_dfire | 0.790 / 0.800 | 2.6 / 3.7 | 5.6 / 8.9 | 0.197 / 0.213 |
| v8s_pyrosdis | 0.735 / 0.742 | 2.0 / 3.1 | 20.2 / 22.3 | 0.078 / 0.083 |
| y11s_dfire | 0.798 / 0.795 | 3.5 / 3.3 | 12.5 / 13.4 | 0.191 / 0.203 |
| y11s_pyrosdis | 0.729 / 0.733 | 2.6 / 2.4 | 20.4 / 21.0 | 0.092 / 0.094 |

|Δ| summary: {"val_mAP50": {"n": 6, "median": 0.005320000000000047, "max": 0.015920000000000045}, "LaECE_id_in": {"n": 6, "median": 0.010827904991318622, "max": 0.03863109888223263}, "LaECE_id_cross_main": {"n": 6, "median": 0.02660027979848309, "max": 0.07843795162260025}}

## C. Spread per (source → target) over all plain runs

| group | n | val mAP50 min/med/max | LaECE id min/med/max |
|---|---|---|---|
| dfire__to__d_fire_test | 10 | 0.778 / 0.797 / 0.805 | 2.6 / 3.4 / 13.8 |
| dfire__to__d_fire_test_dedup | 10 | 0.778 / 0.797 / 0.805 | 3.4 / 4.2 / 16.0 |
| dfire__to__figlib_bb | 10 | 0.778 / 0.797 / 0.805 | 10.6 / 13.8 / 33.6 |
| dfire__to__pyro_sdis_caltest | 10 | 0.778 / 0.797 / 0.805 | 5.6 / 11.8 / 26.8 |
| dfire__to__thesis_test | 10 | 0.778 / 0.797 / 0.805 | 14.9 / 17.9 / 41.1 |
| pyrosdis__to__d_fire_test | 10 | 0.649 / 0.731 / 0.749 | 16.6 / 21.7 / 24.4 |
| pyrosdis__to__d_fire_test_smokeonly | 10 | 0.649 / 0.731 / 0.749 | 16.6 / 21.7 / 24.4 |
| pyrosdis__to__d_fire_test_smokeonly_dedup | 10 | 0.649 / 0.731 / 0.749 | 9.5 / 23.7 / 27.0 |
| pyrosdis__to__figlib_bb | 10 | 0.649 / 0.731 / 0.749 | 12.5 / 16.1 / 35.5 |
| pyrosdis__to__pyro_sdis_caltest | 10 | 0.649 / 0.731 / 0.749 | 2.0 / 2.7 / 7.3 |
| pyrosdis__to__thesis_test | 10 | 0.649 / 0.731 / 0.749 | 36.2 / 46.4 / 58.9 |
| pyrosdis__to__thesis_test_smokeonly | 10 | 0.649 / 0.731 / 0.749 | 36.2 / 46.4 / 58.9 |