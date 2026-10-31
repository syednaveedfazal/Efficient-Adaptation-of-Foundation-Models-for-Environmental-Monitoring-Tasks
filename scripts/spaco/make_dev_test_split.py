"""
scripts/spaco/make_dev_test_split.py — Split the 264 validation scenes once into dev (132) and test (132).

Stratified by burn-pixel fraction quartile, seed 0. Writes:
  splits/val_dev_test.json  {seed, strata, dev, test, burn_frac}
  splits/val_dev.json, splits/val_test.json  (the {"n_scenes", "scenes"} format data.val_json expects)
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

RAW = Path("data/raw/hls_burn_scars/validation")


def main():
    scenes = json.load(open("data/splits/val_scenes.json"))["scenes"]
    frac = {}
    for s in scenes:
        with rasterio.open(RAW / f"{s}.mask.tif") as src:
            m = src.read(1)
        frac[s] = float((m == 1).mean())   # -1 (no data) counts as not burned, as in the dataset

    order = sorted(scenes, key=lambda s: (frac[s], s))
    strata = np.array_split(np.array(order), 4)  # quartiles of burn fraction
    rng = np.random.default_rng(0)
    dev, test = [], []
    for q in strata:
        q = list(q)
        rng.shuffle(q)
        dev += q[: len(q) // 2]
        test += q[len(q) // 2:]
    assert len(dev) == len(test) == 132 and not set(dev) & set(test)

    Path("splits").mkdir(exist_ok=True)
    json.dump({"seed": 0, "stratified_by": "burn-pixel fraction quartiles",
               "quartile_edges": [frac[str(q[-1])] for q in strata],
               "dev": sorted(dev), "test": sorted(test), "burn_frac": frac},
              open("splits/val_dev_test.json", "w"), indent=1)
    for name, ids in (("dev", dev), ("test", test)):
        json.dump({"n_scenes": len(ids), "scenes": sorted(ids)}, open(f"splits/val_{name}.json", "w"), indent=1)
        f = np.array([frac[s] for s in ids])
        print(name, len(ids), "mean burn frac", round(f.mean(), 4), "median", round(float(np.median(f)), 4))


if __name__ == "__main__":
    sys.exit(main())
