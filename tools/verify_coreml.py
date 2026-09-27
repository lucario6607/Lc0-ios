#!/usr/bin/env python3
"""Check a converted Core ML model against ONNX Runtime (CPU, fp32).

Prints one line per (batch size, compute units) and emits GitHub annotations.
Exits non-zero if the policy disagrees badly, which means the conversion is wrong.
"""
import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import coremltools as ct


def random_planes(batch, rng):
    """Sparse 0/1 planes like real positions, with the constant planes set."""
    x = (rng.random((batch, 112, 8, 8)) > 0.9).astype(np.float32)
    x[:, 104:108] = rng.integers(0, 2, (batch, 4, 1, 1)).astype(np.float32)  # castling
    x[:, 108] = rng.integers(0, 2, (batch, 1, 1)).astype(np.float32)        # side to move
    x[:, 109] = rng.integers(0, 100, (batch, 1, 1)).astype(np.float32) / 99  # rule50
    x[:, 110] = 0
    x[:, 111] = 1
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--batch-sizes", default="1,64")
    args = p.parse_args()

    meta = json.load(open(args.meta))
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_in = sess.get_inputs()[0].name
    onnx_outs = [o.name for o in sess.get_outputs()]
    rng = np.random.default_rng(0)
    ok = True

    for b in [int(s) for s in args.batch_sizes.split(",")]:
        if b not in meta["batch_sizes"]:
            continue
        x = random_planes(b, rng)
        ref = dict(zip(onnx_outs, sess.run(None, {onnx_in: x})))
        for units in ("CPU_ONLY", "CPU_AND_NE", "ALL"):
            model = ct.models.MLModel(args.package, function_name=f"b{b}",
                                      compute_units=getattr(ct.ComputeUnit, units))
            out = model.predict({meta["input"]: x})
            report = []
            for short, mil_name in meta["outputs"].items():
                onnx_name = next(o for o in onnx_outs if o.split("/")[-1] == short)
                r, c = ref[onnx_name], np.asarray(out[mil_name], dtype=np.float32).reshape(ref[onnx_name].shape)
                diff = float(np.max(np.abs(r - c)))
                if short == "policy":
                    agree = float(np.mean(np.argmax(r, 1) == np.argmax(c, 1)))
                    report.append(f"policy top1 {agree:.0%} maxdiff {diff:.3g}")
                    if agree < 0.9:
                        ok = False
                else:
                    report.append(f"{short} maxdiff {diff:.3g}")
            line = f"batch {b} {units}: " + ", ".join(report)
            print(line, flush=True)
            if os.environ.get("GITHUB_ACTIONS"):
                print(f"::notice title=Core ML check::{line}")
            del model

    if not ok:
        print("::error::Core ML output disagrees with ONNX Runtime; conversion is wrong")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
