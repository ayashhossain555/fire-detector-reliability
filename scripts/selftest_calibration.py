"""End-to-end self-test of the calibration pipeline on synthetic data.

Creates a synthetic COCO GT pair (calval/caltest) + detections with a known
miscalibration (scores raised to a power), runs run_calibration's battery,
and asserts (a) it completes for all four calibrators and (b) isotonic
regression reduces LaECE_0 vs identity. No images touched — the toolbox
reads only JSONs. Scratch output only; not a paper artefact.
"""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
rng = random.Random(7)


def synth(n_imgs, tag, out_dir):
    imgs, anns, dets = [], [], []
    aid = 1
    for i in range(1, n_imgs + 1):
        imgs.append({"id": i, "file_name": f"{tag}_{i}.jpg", "width": 640, "height": 480})
        for _ in range(rng.randint(1, 4)):
            cat = rng.choice([1, 2])
            x, y, w, h = rng.uniform(0, 400), rng.uniform(0, 300), rng.uniform(40, 200), rng.uniform(40, 150)
            anns.append({"id": aid, "image_id": i, "category_id": cat,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            aid += 1
            # matched detection with jitter, overconfident score
            if rng.random() < 0.8:
                iou_quality = rng.uniform(0.5, 1.0)
                dets.append({"image_id": i, "category_id": cat,
                             "bbox": [x + rng.uniform(-8, 8), y + rng.uniform(-8, 8),
                                      w * rng.uniform(0.85, 1.15), h * rng.uniform(0.85, 1.15)],
                             "score": round(min(0.999, iou_quality ** 0.25), 6)})
        # background false positives with mid confidence
        for _ in range(rng.randint(0, 2)):
            dets.append({"image_id": i, "category_id": rng.choice([1, 2]),
                         "bbox": [rng.uniform(0, 500), rng.uniform(0, 380), 60, 60],
                         "score": round(rng.uniform(0.3, 0.8), 6)})
    gt = {"images": imgs, "annotations": anns,
          "categories": [{"id": 1, "name": "fire"}, {"id": 2, "name": "smoke"}]}
    (out_dir / f"{tag}_gt.json").write_text(json.dumps(gt), encoding="utf-8")
    (out_dir / f"{tag}_dets.json").write_text(json.dumps(dets), encoding="utf-8")


def main():
    out_dir = Path(tempfile.mkdtemp(prefix="calib_selftest_"))
    synth(120, "calval", out_dir)
    synth(120, "caltest", out_dir)
    r = subprocess.run([sys.executable, str(HERE / "run_calibration.py"),
                        "--calval-gt", str(out_dir / "calval_gt.json"),
                        "--calval-dets", str(out_dir / "calval_dets.json"),
                        "--test-gt", str(out_dir / "caltest_gt.json"),
                        "--test-dets", str(out_dir / "caltest_dets.json"),
                        "--cell", "SELFTEST"], capture_output=True, text=True, cwd=HERE)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:]); sys.exit("SELFTEST FAILED: nonzero exit")
    art = json.loads((HERE / "calibration" / "SELFTEST.json").read_text())
    ident = art["results"]["identity"]["LaECE_0"]
    iso = art["results"]["isotonic_regression"]["LaECE_0"]
    print(f"identity LaECE_0={ident*100:.1f} vs isotonic={iso*100:.1f}")
    assert iso < ident, "isotonic did not improve calibration on synthetic overconfident data"
    (HERE / "calibration" / "SELFTEST.json").unlink()  # scratch, not an artefact
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
