# Calibration-toolbox patch

The calibration metrics (D-ECE, LaECE, LRP-optimal thresholds, post-hoc
calibrators) are computed with `fiveai/detection_calibration`
(https://github.com/fiveai/detection_calibration, ECCV 2024), pinned at
commit `bf59c92f5416ac4b66a7c2c67b44487777ff2ea4`. The toolbox is not
redistributed here. One patch was applied locally; it is documented in
`docs/TOOLBOX-PATCHES.md` (Patch 1) and stored in this folder as
`detection_calibration_bf59c92.patch`.

What it does: `threshold_detections()` in `src/detection_calibration/utils.py`
indexes a per-class threshold array that is squeezed to 0-d when the dataset
has a single category (the smoke-only Pyro-SDIS ground truth); `np.atleast_1d`
restores the C=1 case without changing behaviour for C>=2 (verified: the
two-class self-test gives identical numbers before and after).

The two other items in `docs/TOOLBOX-PATCHES.md` (off-class detection guard,
category-order alignment) are hub-side fixes in `scripts/run_calibration.py`
and `scripts/make_coco_gt.py`; the toolbox is untouched by them.

## Apply

```bash
git clone https://github.com/fiveai/detection_calibration
cd detection_calibration
git checkout bf59c92f5416ac4b66a7c2c67b44487777ff2ea4
git apply /path/to/toolbox_patch/detection_calibration_bf59c92.patch
pip install -e .
```

The scripts in `scripts/` expect the clone at `analysis/detection_calibration/`
relative to the paper folder (see `docs/TOOLBOX-PATCHES.md`), i.e. next to the
scripts when the layout of this repository is used as `analysis/`.

## Licence

Upstream `detection_calibration` is licensed under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0),
https://creativecommons.org/licenses/by-nc-sa/4.0/. This patch is a
derivative work and is released under the same CC BY-NC-SA 4.0 licence.
