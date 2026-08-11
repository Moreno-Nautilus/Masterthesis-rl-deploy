"""Compare dumped deploy observations against a sim rollout dump."""

from __future__ import annotations

import argparse

import numpy as np


def _load(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", required=True, help="npz with policy and image arrays from Isaac.")
    parser.add_argument("--deploy", required=True, help="npz with policy and image arrays from deploy preprocessing.")
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    sim = _load(args.sim)
    dep = _load(args.deploy)
    ok = True
    for key in ("policy", "image"):
        if key not in sim or key not in dep:
            print(f"[FAIL] missing key {key}")
            ok = False
            continue
        if sim[key].shape != dep[key].shape:
            print(f"[FAIL] {key}: shape sim={sim[key].shape} deploy={dep[key].shape}")
            ok = False
            continue
        err = np.max(np.abs(sim[key].astype(np.float32) - dep[key].astype(np.float32)))
        status = "OK" if err <= args.atol else "FAIL"
        print(f"[{status}] {key}: max_abs_err={err:.8g}")
        ok = ok and err <= args.atol
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

