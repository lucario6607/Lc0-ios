#!/usr/bin/env python3
"""Convert an lc0 network (exported to ONNX by `lc0 leela2onnx`) to Core ML.

The result is one multi-function .mlpackage: a function per batch size
("b1", "b8", ...) with static shapes, all sharing one copy of the weights,
plus lc0coreml.json describing inputs/outputs and the network format that
lc0's coreml backend needs (so the phone never loads the original net).

Usage:
  lc0_to_coreml.py --onnx net.onnx --net net.pb.gz --out outdir \
      --batch-sizes 1,8,16,32,64,128,256 [--precision fp16|fp32|w8|w8a8]
      [--calibration positions.f32]

Precisions: fp16/fp32 compute; w8 = int8 weights (per-channel) with fp16
compute, which halves weight traffic; w4 = int4 weights (per block of 32),
halving it again; w8a8 = int8 weights and activations,
calibrated on --calibration (raw float32 [n,112,8,8], e.g. from lc0's coreml
backend dump= option).
"""
import argparse
import gzip
import json
import os
import shutil
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

INT_MAX = 2**31 - 1


# ---------------------------------------------------------------------------
# ONNX graph -> MIL program (static batch size)
# ---------------------------------------------------------------------------

def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


class Converter:
    def __init__(self, model):
        self.graph = model.graph
        self.consts = {i.name: numpy_helper.to_array(i) for i in self.graph.initializer}
        for n in self.graph.node:
            if n.op_type == "Constant":
                self.consts[n.output[0]] = numpy_helper.to_array(attrs(n)["value"])
        self.input_name = self.graph.input[0].name
        self.output_names = [o.name for o in self.graph.output]

    def build(self, batch):
        spec = [mb.TensorSpec(shape=(batch, 112, 8, 8), dtype=types.fp32)]

        @mb.program(input_specs=spec, opset_version=ct.target.iOS18)
        def prog(planes):
            env = {self.input_name: planes}
            for node in self.graph.node:
                if node.op_type == "Constant":
                    continue
                outs = self.convert_node(node, env)
                if not isinstance(outs, (list, tuple)):
                    outs = [outs]
                for name, var in zip(node.output, outs):
                    env[name] = var
            return tuple(mb.identity(x=env[o], name=safe(o)) for o in self.output_names)

        return prog

    # -- helpers -------------------------------------------------------------

    def val(self, name, env):
        """A graph value: a MIL var, or a numpy constant."""
        if name in env:
            return env[name]
        if name in self.consts:
            c = self.consts[name]
            if c.dtype == np.float64:
                c = c.astype(np.float32)
            if c.dtype == np.int64:
                c = c.astype(np.int32)
            return c
        raise KeyError(f"unknown value {name}")

    def const(self, name):
        if name not in self.consts:
            raise ValueError(f"{name} must be a constant")
        return self.consts[name]

    @staticmethod
    def shape(x):
        return list(x.shape) if isinstance(x, np.ndarray) else list(x.shape)

    # -- ops -----------------------------------------------------------------

    def convert_node(self, node, env):
        op = node.op_type
        a = attrs(node)
        i = [self.val(n, env) if n else None for n in node.input]
        name = safe(node.output[0])

        if op == "MatMul":
            return mb.matmul(x=i[0], y=i[1], name=name)
        if op in ("Add", "Sub", "Mul", "Div"):
            fn = {"Add": mb.add, "Sub": mb.sub, "Mul": mb.mul, "Div": mb.real_div}[op]
            return fn(x=i[0], y=i[1], name=name)
        if op == "Reshape":
            return mb.reshape(x=i[0], shape=self.const(node.input[1]).astype(np.int32), name=name)
        if op == "Transpose":
            return mb.transpose(x=i[0], perm=list(a["perm"]), name=name)
        if op == "LayerNormalization":
            rank = len(self.shape(i[0]))
            axis = a.get("axis", -1) % rank
            beta = i[2] if len(i) > 2 and i[2] is not None else None
            return mb.layer_norm(x=i[0], axes=list(range(axis, rank)), gamma=i[1], beta=beta,
                                 epsilon=float(a.get("epsilon", 1e-5)), name=name)
        if op == "Softmax":
            return mb.softmax(x=i[0], axis=a.get("axis", -1), name=name)
        if op == "Sigmoid":
            return mb.sigmoid(x=i[0], name=name)
        if op == "Tanh":
            return mb.tanh(x=i[0], name=name)
        if op == "Relu":
            return mb.relu(x=i[0], name=name)
        if op == "Softplus":
            return mb.softplus(x=i[0], name=name)
        if op == "Exp":
            return mb.exp(x=i[0], name=name)
        if op == "Sqrt":
            return mb.sqrt(x=i[0], name=name)
        if op == "Reciprocal":
            return mb.inverse(x=i[0], name=name)
        if op == "Identity":
            return mb.identity(x=i[0], name=name)
        if op == "Mish":
            return mb.mul(x=i[0], y=mb.tanh(x=mb.softplus(x=i[0])), name=name)
        if op == "Selu":
            return mb.mul(x=mb.elu(x=i[0], alpha=float(a.get("alpha", 1.67326319))),
                          y=np.float32(a.get("gamma", 1.05070102)), name=name)
        if op == "Elu":
            return mb.elu(x=i[0], alpha=float(a.get("alpha", 1.0)), name=name)
        if op == "Cast":
            to = {onnx.TensorProto.FLOAT: "fp32", onnx.TensorProto.FLOAT16: "fp16",
                  onnx.TensorProto.INT32: "int32", onnx.TensorProto.INT64: "int32"}[a["to"]]
            return mb.cast(x=i[0], dtype=to, name=name)
        if op == "Slice":
            return self.slice(node, i, name)
        if op == "Concat":
            return mb.concat(values=i, axis=a["axis"], name=name)
        if op == "Split":
            sizes = self.const(node.input[1]).astype(np.int32) if len(node.input) > 1 and node.input[1] \
                else np.array(a["split"], dtype=np.int32)
            return mb.split(x=i[0], split_sizes=sizes, axis=a.get("axis", 0), name=name)
        if op == "Gather":
            return mb.gather(x=i[0], indices=self.const(node.input[1]).astype(np.int32),
                             axis=a.get("axis", 0), name=name)
        if op == "Squeeze":
            axes = self.const(node.input[1]) if len(node.input) > 1 else a.get("axes")
            return mb.squeeze(x=i[0], axes=[int(x) for x in axes], name=name)
        if op == "Conv":
            return self.conv(node, i, a, name)
        if op == "GlobalAveragePool":
            return mb.reduce_mean(x=i[0], axes=[2, 3], keep_dims=True, name=name)
        if op == "ReduceMean":
            axes = self.const(node.input[1]) if len(node.input) > 1 else a["axes"]
            return mb.reduce_mean(x=i[0], axes=[int(x) for x in axes],
                                  keep_dims=bool(a.get("keepdims", 1)), name=name)
        if op == "Greater":
            return mb.greater(x=i[0], y=i[1], name=name)
        if op == "Where":
            return mb.select(cond=i[0], a=i[1], b=i[2], name=name)
        if op == "Expand":
            target = self.const(node.input[1]).astype(np.int64)
            src = self.shape(i[0])
            src = [1] * (len(target) - len(src)) + src
            reps = [int(t) if s == 1 else 1 for s, t in zip(src, target)]
            x = mb.reshape(x=i[0], shape=np.array(src, dtype=np.int32)) if len(src) != len(self.shape(i[0])) else i[0]
            return mb.tile(x=x, reps=reps, name=name)
        if op == "Pad":
            pads = self.const(node.input[1]).astype(np.int32)
            rank = len(pads) // 2
            mil_pads = np.array([v for d in range(rank) for v in (pads[d], pads[d + rank])], dtype=np.int32)
            return mb.pad(x=i[0], pad=mil_pads, mode="constant", constant_val=0.0, name=name)
        raise NotImplementedError(f"ONNX op {op} ({node.name}) is not supported yet")

    def slice(self, node, i, name):
        x = i[0]
        shape = self.shape(x)
        rank = len(shape)
        starts = self.const(node.input[1]).astype(np.int64)
        ends = self.const(node.input[2]).astype(np.int64)
        axes = self.const(node.input[3]).astype(np.int64) if len(node.input) > 3 and node.input[3] \
            else np.arange(len(starts))
        steps = self.const(node.input[4]).astype(np.int64) if len(node.input) > 4 and node.input[4] \
            else np.ones(len(starts), dtype=np.int64)
        begin, end, stride = [0] * rank, list(shape), [1] * rank
        for s, e, ax, st in zip(starts, ends, axes, steps):
            ax = int(ax) % rank
            dim = shape[ax]
            s = int(s) + dim if s < 0 else min(int(s), dim)
            e = int(e) + dim if e < 0 else min(int(e), dim)
            begin[ax], end[ax], stride[ax] = s, e, int(st)
        return mb.slice_by_index(x=x, begin=np.array(begin, dtype=np.int32),
                                 end=np.array(end, dtype=np.int32),
                                 stride=np.array(stride, dtype=np.int32), name=name)

    def conv(self, node, i, a, name):
        pads = list(a.get("pads", [0, 0, 0, 0]))
        rank = len(pads) // 2
        mil_pads = np.array([v for d in range(rank) for v in (pads[d], pads[d + rank])], dtype=np.int32)
        kwargs = dict(x=i[0], weight=i[1], strides=list(a.get("strides", [1, 1])),
                      pad_type="custom", pad=mil_pads,
                      dilations=list(a.get("dilations", [1, 1])), groups=int(a.get("group", 1)))
        if len(i) > 2 and i[2] is not None:
            kwargs["bias"] = i[2]
        return mb.conv(name=name, **kwargs)


def safe(name):
    """MIL names can't contain '/'."""
    return name.strip("/").replace("/", "__").replace(".", "_") or "v"


# ---------------------------------------------------------------------------
# lc0 network format (read straight from the .pb.gz, without lc0)
# ---------------------------------------------------------------------------

def read_varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def proto_fields(buf):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            v, pos = read_varint(buf, pos)
        elif wire == 1:
            v, pos = buf[pos:pos + 8], pos + 8
        elif wire == 2:
            n, pos = read_varint(buf, pos)
            v, pos = buf[pos:pos + n], pos + n
        elif wire == 5:
            v, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, v


def network_format(net_path):
    """pblczero.Net.format.network_format as a dict of lc0 enum values.

    Field numbers from lc0's proto/net.proto: Net.format = 4,
    Format.network_format = 2; NetworkFormat: input 1, output 2, network 3,
    policy 4, value 5, moves_left 6, default_activation 7, smolgen_activation 8,
    ffn_activation 9, input_embedding 10.
    """
    with gzip.open(net_path, "rb") as f:
        data = f.read()
    names = {1: "input", 2: "output", 3: "network", 4: "policy", 5: "value",
             6: "moves_left", 7: "default_activation", 8: "smolgen_activation",
             9: "ffn_activation", 10: "input_embedding"}
    result = {}
    for field, wire, v in proto_fields(data):
        if field == 4 and wire == 2:  # Net.format
            for f2, w2, v2 in proto_fields(v):
                if f2 == 2 and w2 == 2:  # Format.network_format
                    for f3, w3, v3 in proto_fields(v2):
                        if w3 == 0 and f3 in names:
                            result[names[f3]] = v3
    del data
    return result


# ---------------------------------------------------------------------------

def load_positions(path, limit):
    """Real positions for calibration; random ones if none were given."""
    if path and os.path.exists(path):
        x = np.fromfile(path, dtype=np.float32).reshape(-1, 112, 8, 8)
        print(f"calibration: {len(x)} positions from {path}", flush=True)
    else:
        print("calibration: no positions file, using random planes", flush=True)
        rng = np.random.default_rng(0)
        x = (rng.random((limit, 112, 8, 8)) > 0.9).astype(np.float32)
        x[:, 111] = 1
    return x[:limit]


def quantize(mlmodel, path, batch, calibration, bits=8):
    """int8/int4 weights (and int8 activations, if calibration data is given)."""
    import coremltools.optimize as cto
    if calibration is not None:
        # Activation calibration runs the model, so it needs a loadable copy.
        mlmodel.save(path)
        mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
        n = max(1, min(8, len(calibration) // batch))
        samples = [{"planes": calibration[i * batch:(i + 1) * batch]} for i in range(n)
                   if len(calibration[i * batch:(i + 1) * batch]) == batch]
        if not samples:
            reps = -(-batch // len(calibration))
            samples = [{"planes": np.tile(calibration, (reps, 1, 1, 1))[:batch]}]
        print(f"batch {batch}: calibrating activations on {len(samples)} x {batch} positions", flush=True)
        act = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric"))
        mlmodel = cto.coreml.linear_quantize_activations(mlmodel, act, samples,
                                                          calibration_op_group_size=200)
    if bits == 4:
        # int4 needs finer scales to stay accurate: one per block of 32 inputs.
        op = cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32,
            weight_threshold=2048)
    else:
        op = cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=2048)
    return cto.coreml.linear_quantize_weights(mlmodel, cto.coreml.OptimizationConfig(global_config=op))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--net", required=True, help="original .pb.gz, for the network format")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-sizes", default="1,8,16,32,64,128,256")
    p.add_argument("--precision", choices=["fp16", "fp32", "w8", "w8a8", "w4"], default="fp16")
    p.add_argument("--calibration", default=None,
                   help="raw float32 [n,112,8,8] positions for w8a8 calibration")
    p.add_argument("--calibration-positions", type=int, default=256)
    p.add_argument("--name", default=None)
    p.add_argument("--build-only", action="store_true",
                   help="only build the MIL programs (works without macOS)")
    args = p.parse_args()

    sizes = sorted({int(s) for s in args.batch_sizes.split(",") if s.strip()})
    name = args.name or os.path.basename(args.net).replace(".pb.gz", "").replace(".pb", "")
    fmt = network_format(args.net)
    print("network format:", fmt, flush=True)

    model = onnx.load(args.onnx)
    conv = Converter(model)
    outputs = conv.output_names
    del model

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "_work")
    os.makedirs(work, exist_ok=True)
    precision = ct.precision.FLOAT32 if args.precision == "fp32" else ct.precision.FLOAT16
    calibration = None
    if args.precision == "w8a8":
        calibration = load_positions(args.calibration, args.calibration_positions)

    parts = []
    for b in sizes:
        print(f"batch {b}: building", flush=True)
        prog = conv.build(b)
        if args.build_only:
            print(f"batch {b}: ok ({len(list(prog.functions['main'].operations))} ops)")
            continue
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=precision,
                             minimum_deployment_target=ct.target.iOS18,
                             skip_model_load=True)
        path = os.path.join(work, f"b{b}.mlpackage")
        if args.precision in ("w8", "w8a8", "w4"):
            mlmodel = quantize(mlmodel, path, b, calibration, bits=4 if args.precision == "w4" else 8)
        mlmodel.save(path)
        parts.append((b, path))
        del mlmodel, prog
    if args.build_only:
        return

    desc = ct.utils.MultiFunctionDescriptor()
    for b, path in parts:
        desc.add_function(path, src_function_name="main", target_function_name=f"b{b}")
    desc.default_function_name = f"b{sizes[-1]}"
    package = os.path.join(args.out, f"{name}.mlpackage")
    ct.utils.save_multifunction(desc, package)
    shutil.rmtree(work)

    meta = {
        "name": name,
        "batch_sizes": sizes,
        "precision": args.precision,
        "input": "planes",  # the MIL program argument name
        "outputs": {o.split("/")[-1]: safe(o) for o in outputs},
        "format": fmt,
    }
    with open(os.path.join(args.out, "lc0coreml.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
