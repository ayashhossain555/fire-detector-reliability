# Local patches to analysis/detection_calibration (fiveai, pinned bf59c92)

The clone is gitignored; every local modification is recorded here with its
diff so the paper's release repo can reproduce it (and so upstream can be
offered the fix).

## Patch 1 — single-class threshold indexing (2026-09-01)
File: `src/detection_calibration/utils.py`, `threshold_detections()`.
Symptom: `IndexError: too many indices for array: array is 0-dimensional`
when the dataset has ONE category (Pyro-SDIS, smoke-only): the per-class
LRP-optimal threshold array is squeezed to 0-d before indexing.
Fix (behaviour-preserving for C>=2):
```diff
     del_items = []
+    detection_level_threshold = np.atleast_1d(detection_level_threshold)
     for idx, detection in enumerate(detections):
         if detection['score'] < detection_level_threshold[dataset_classes.index(detection['category_id'])]:
```
Verification (2026-09-01): [x] run_matrix single-class cell completes
(v8s_pyrosdis_s3407__to__pyro_sdis_caltest artefact written); [x] two-class
self-test passes post-patch with identical numbers (identity 14.2 /
isotonic 0.6 LaECE_0).

## 2026-09-03 — class-space guard in run_calibration.py (hub-side, toolbox untouched)
Repair cells fit a fire+smoke model's calibrator on the smoke-only Pyro-SDIS
calval; the toolbox's `calibrate()` does `coco_classes.index(category_id)`
and raised `ValueError: 1 is not in list` for fire detections. Fix: before
fitting/transforming, drop detections whose category is absent from the
calval GT (they are unevaluable on that GT anyway); the count is recorded in
the cell artefact under `inputs.{calval,test}_dets_dropped_offclass`. Forward
cells are unaffected (0 dropped) — verified by identical artefacts.

## 2026-09-03 — category ORDER alignment (hub-side data fix; toolbox untouched)
`CalibrationCOCO.dataset_classes = list(COCO(val).cats.keys())` follows the
GT file's category order, while `get_detection_thresholds` returns
`lrp_opt_thr` in COCOeval's SORTED catId order. With the D-Fire GTs written
as [smoke=2, fire=1] the per-class thresholds were swapped for every fit on
a D-Fire calval. For the D-Fire-trained models the two class thresholds were
within 0.001 of each other (negligible); for the smoke-only Pyro-SDIS models
fitted on `d_fire_calval` ("repair" cells) the fire slot was NaN, so the
smoke threshold became NaN and NO detection was removed — the first
"repair" numbers for those two cells (LaECE_0 1.8 / 2.0 at LRP-FN 42.7 /
36.2) were computed on all conf ≥ 0.001 detections and are WITHDRAWN.
Fix: all `coco_gt/*.json` now carry categories sorted by id
(`make_coco_gt.py` sorts at build time); every cell whose calval-gt was a
D-Fire GT was deleted and recomputed (exports unaffected — categories are
mapped by name). Caught by `threshold_decomposition.py`'s NaN thresholds.
