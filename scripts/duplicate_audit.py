"""Near-duplicate / leakage audit across the paper's image sets (phase 1).
Perceptual hashes (pHash 64-bit + dHash 64-bit, imagehash 4.3.2, greyscale)
for every image in every split of every dataset; then counts images at
Hamming distance <= {0, 4, 8} on pHash (a) WITHIN a dataset between train
and its eval splits (train->test leakage), (b) ACROSS datasets (transfer-
matrix contamination). Writes analysis/duplicate_audit.json (counts + sample
pairs for manual spot-check) and analysis/datasets/phash_index.parquet.
CPU-only, single process; safe alongside GPU training.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SETS = {
    "d_fire":    {s: DATA / "d_fire_yolo" / s / "images" for s in ("train", "val", "test")},
    "pyro_sdis": {s: DATA / "pyro_sdis_yolo" / s / "images" for s in ("train", "val")},
    "thesis":    {s: DATA / "final_fire_project_v4" / s for s in ("train", "valid", "test")},
}
EXT = {".jpg", ".jpeg", ".png"}

def hash_all():
    rows, t0, n = [], time.time(), 0
    for ds, splits in SETS.items():
        for sp, d in splits.items():
            for p in sorted(x for x in d.iterdir() if x.suffix.lower() in EXT):
                try:
                    with Image.open(p) as im:
                        im = im.convert("L"); w, h = im.size
                        ph, dh = imagehash.phash(im), imagehash.dhash(im)
                except Exception as e:
                    rows.append({"dataset": ds, "split": sp, "file": p.name, "error": str(e)}); continue
                rows.append({"dataset": ds, "split": sp, "file": p.name, "w": w, "h": h,
                             "phash": int(str(ph), 16), "dhash": int(str(dh), 16)})
                n += 1
                if n % 5000 == 0:
                    print(f"  {n} hashed ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)

def popcount64(a):
    a = a.astype(np.uint64); c = np.zeros(a.shape, dtype=np.int64)
    for _ in range(64):
        c += (a & np.uint64(1)).astype(np.int64); a >>= np.uint64(1)
    return c

def match(hA, hB, thr):
    """For each hash in A: is there a hash in B within Hamming <= thr? Returns
    (bool mask over A, min distance over A, argmin B hash)."""
    hA = hA.astype(np.uint64); hB = np.unique(hB.astype(np.uint64))
    mind = np.full(len(hA), 99, dtype=np.int64); arg = np.zeros(len(hA), dtype=np.uint64)
    for i in range(0, len(hA), 1000):
        d = popcount64(np.bitwise_xor(hA[i:i+1000, None], hB[None, :]))
        mind[i:i+1000] = d.min(axis=1); arg[i:i+1000] = hB[d.argmin(axis=1)]
    return mind <= thr, mind, arg

def main():
    idx = HERE / "datasets" / "phash_index.parquet"
    if idx.exists():
        df = pd.read_parquet(idx); print("loaded", len(df))
    else:
        df = hash_all(); df.to_parquet(idx); print("hashed", len(df), "->", idx)
    if "error" in df:
        df = df[df["error"].isna()]
    df = df.copy(); df["phash"] = df["phash"].astype(np.uint64)
    file_of = {(r.dataset, r.split, int(r.phash)): r.file for r in df.itertuples()}
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "hash": "pHash 64-bit (imagehash 4.3.2, greyscale)", "thresholds_hamming": [0, 4, 8],
           "counts": {f"{ds}/{sp}": int(((df.dataset == ds) & (df.split == sp)).sum()) for ds in SETS for sp in SETS[ds]},
           "exact_within_split": {}, "within_dataset_train_to_eval": {}, "across_datasets": {}}
    for ds in SETS:
        for sp in SETS[ds]:
            g = df[(df.dataset == ds) & (df.split == sp)]; vc = g.phash.value_counts()
            out["exact_within_split"][f"{ds}/{sp}"] = {"n_images": int(len(g)),
                "n_images_in_exact_dup_groups": int(vc[vc > 1].sum()), "n_groups": int((vc > 1).sum())}
    def report(T, tds, tsp, S, sds, key, store):
        res = {"n_target": int(len(T))}
        m8, mind, arg = match(T.phash.values, S.phash.values, 8)
        for thr in (0, 4, 8):
            hit = mind <= thr
            res[f"ham<={thr}"] = {"n_matched": int(hit.sum()), "frac": round(float(hit.mean()), 4) if len(T) else None}
        samples = []
        for r in np.where(m8)[0][:8]:
            h = int(arg[r]); src = next((f for (d, s, ph), f in file_of.items() if d == sds and ph == h), None)
            samples.append({"target": T.file.iloc[r], "source": src, "dist": int(mind[r])})
        res["samples_ham<=8"] = samples
        store[key] = res
    for ds in SETS:
        tr = df[(df.dataset == ds) & (df.split == "train")]
        for sp in SETS[ds]:
            if sp != "train":
                report(df[(df.dataset == ds) & (df.split == sp)], ds, sp, tr, ds, f"{ds}:{sp}<-train", out["within_dataset_train_to_eval"])
    for src in SETS:
        S = df[df.dataset == src]
        for tgt in SETS:
            if src == tgt: continue
            for sp in SETS[tgt]:
                if sp != "train":
                    report(df[(df.dataset == tgt) & (df.split == sp)], tgt, sp, S, src, f"{tgt}:{sp}<-{src}:all", out["across_datasets"])
    (HERE / "duplicate_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("artefact ->", HERE / "duplicate_audit.json")
    for sec in ("within_dataset_train_to_eval", "across_datasets"):
        for k, v in out[sec].items():
            print(k, {t: v[t]["n_matched"] for t in ("ham<=0", "ham<=4", "ham<=8")}, "of", v["n_target"])

if __name__ == "__main__":
    main()
