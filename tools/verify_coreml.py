#!/usr/bin/env python3
"""Check a converted Core ML model against ONNX Runtime (CPU, fp32).

Prints one line per (batch size, compute units) and emits GitHub annotations.

Random inputs give flat policies with many near-ties, so a raw top-1 match
understates fp16 accuracy. The gate uses the KL divergence of the policy
distributions and the WDL difference; top-1 is also reported for positions
where the reference has a clear favourite. Exits non-zero only on errors big
enough to mean the conversion itself is wrong.
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


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--batch-sizes", default="1,64")
    p.add_argument("--positions", default=None,
                   help="raw float32 [n,112,8,8] real positions (else random boards)")
    p.add_argument("--units", default="CPU_ONLY,ALL")
    args = p.parse_args()
    positions = None
    if args.positions and os.path.exists(args.positions):
        positions = np.fromfile(args.positions, dtype=np.float32).reshape(-1, 112, 8, 8)
        print(f"checking on {len(positions)} real positions", flush=True)

    meta = json.load(open(args.meta))
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_in = sess.get_inputs()[0].name
    onnx_outs = [o.name for o in sess.get_outputs()]
    rng = np.random.default_rng(0)
    ok = True

    for b in [int(s) for s in args.batch_sizes.split(",")]:
        if b not in meta["batch_sizes"]:
            continue
        if positions is not None:
            reps = -(-b // len(positions))
            x = np.tile(positions, (reps, 1, 1, 1))[:b] if b > len(positions) else positions[:b]
            if b < 64 and len(positions) >= 64:
                # Small batches: check several chunks so the numbers mean something.
                x = positions[:64 - 64 % b]
        else:
            x = random_planes(b, rng)
        ref = dict(zip(onnx_outs, sess.run(None, {onnx_in: x})))
        for units in args.units.split(","):
            model = ct.models.MLModel(args.package, function_name=f"b{b}",
                                      compute_units=getattr(ct.ComputeUnit, units))
            # The function takes exactly b positions; run x in chunks of b.
            chunks = [model.predict({meta["input"]: x[i:i + b]}) for i in range(0, len(x), b)]
            out = {k: np.concatenate([np.asarray(c[k]) for c in chunks]) for k in chunks[0]}
            report = []
            for short, mil_name in meta["outputs"].items():
                onnx_name = next(o for o in onnx_outs if o.split("/")[-1] == short)
                r, c = ref[onnx_name], np.asarray(out[mil_name], dtype=np.float32).reshape(ref[onnx_name].shape)
                diff = float(np.max(np.abs(r - c)))
                if short == "policy":
                    pr, pc = softmax(r), softmax(c)
                    kl = float(np.mean(np.sum(pr * (np.log(pr + 1e-12) - np.log(pc + 1e-12)), axis=1)))
                    top2 = np.sort(pr, axis=1)[:, -2:]
                    clear = (top2[:, 1] - top2[:, 0]) > 0.05
                    same = np.argmax(r, 1) == np.argmax(c, 1)
                    clear_agree = float(np.mean(same[clear])) if clear.any() else float("nan")
                    report.append(f"policy KL {kl:.4f}, top1 {np.mean(same):.0%} "
                                  f"({clear_agree:.0%} of {int(clear.sum())} clear)")
                    if kl > 0.2:
                        ok = False
                else:
                    report.append(f"{short} maxdiff {diff:.3g}")
                    if short == "wdl" and diff > 0.2:
                        ok = False
            source = "real" if positions is not None else "random"
            line = f"{meta.get('precision', '')} batch {b} {units} ({len(x)} {source}): " + ", ".join(report)
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
