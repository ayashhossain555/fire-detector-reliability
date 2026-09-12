"""Pyro-SDIS camera overlap between the train and val splits.

Records, as a machine-written artefact, how many validation cameras also
appear in the training split (the claim in Section 3 of the draft).

Input : analysis/datasets/pyro_sdis_image_meta.json  (file -> split, partner, camera, date)
Output: analysis/pyro_camera_overlap.json
"""
import json
from collections import Counter
from pathlib import Path

A = Path(__file__).resolve().parent
meta = json.load(open(A / "datasets" / "pyro_sdis_image_meta.json", encoding="utf-8"))

cams = {"train": Counter(), "val": Counter()}
for fname, m in meta.items():
    cams[m["split"]][m["camera"]] += 1

train_cams = set(cams["train"])
val_cams = set(cams["val"])
shared = sorted(val_cams & train_cams)
val_only = sorted(val_cams - train_cams)

out = {
    "source": "datasets/pyro_sdis_image_meta.json",
    "n_images": {"train": sum(cams["train"].values()), "val": sum(cams["val"].values())},
    "n_cameras": {"train": len(train_cams), "val": len(val_cams)},
    "n_val_cameras_also_in_train": len(shared),
    "val_cameras_not_in_train": val_only,
    "val_frames_on_shared_cameras": sum(cams["val"][c] for c in shared),
    "frac_val_frames_on_shared_cameras": sum(cams["val"][c] for c in shared) / sum(cams["val"].values()),
}
json.dump(out, open(A / "pyro_camera_overlap.json", "w", encoding="utf-8"), indent=2)
print(json.dumps(out, indent=2))
