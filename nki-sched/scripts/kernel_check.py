"""Check an emitted NKI kernel against numpy, three ways:

  --mode sim       nki.simulate (CPU; needs the Neuron SDK, no device)
  --mode compile   compile with neuronx-cc but do not run -- catches what the hardware compiler rejects
                   and the simulator accepts (e.g. divide in tensor_scalar); works while the cores are busy
  --mode hw        compile, run on a NeuronCore and compare (correctness; wall time is launch-dominated)

    python kernel_check.py kernel.py nki_matmul_tiled_ --op matmul --mode sim
    python kernel_check.py kernel.py nki_attention_    --op attention --mode compile --shapes 256x64
    python kernel_check.py kernel.py nki_attention_    --op attention --mode hw --dtypes f32,bf16

Shapes: matmul KxMxN (lhsT is [K,M], rhs is [K,N]); attention SxD (q, k, v are [S,D]). Attention inputs
are also run scaled by 30 so scores are large enough to overflow a softmax that forgets the row max.
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

MATMUL_SHAPES = "128x128x512,256x256x1024,512x128x512,256x512x1024"
ATTENTION_SHAPES = "128x64,64x128,96x32,128x128"


def make_cases(op, shape, dtype, seed=0):
    """-> [(tag, args, expected float32 result)]"""
    r = np.random.default_rng(seed + sum(shape))

    def cast(x):
        return x.astype(dtype)

    if op == "matmul":
        K, M, N = shape
        a, b = cast(r.standard_normal((K, M))), cast(r.standard_normal((K, N)))
        return [("", (a, b), a.astype(np.float32).T @ b.astype(np.float32))]
    S, D = shape
    q, k, v = (cast(r.standard_normal((S, D))) for _ in range(3))

    def ref(q, k, v):
        s = (q.astype(np.float32) @ k.astype(np.float32).T) / np.sqrt(D)
        e = np.exp(s - s.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True)) @ v.astype(np.float32)

    big_q, big_k = cast(q.astype(np.float32) * 30.0), cast(k.astype(np.float32) * 30.0)
    return [("normal", (q, k, v), ref(q, k, v)), ("x30 scores", (big_q, big_k, v), ref(big_q, big_k, v))]


def tolerance(op, dtype_name):
    if dtype_name == "bf16":
        return 3e-2
    return 1e-3 if op == "matmul" else 2e-3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kernel")
    ap.add_argument("entry")
    ap.add_argument("--op", choices=("matmul", "attention"), required=True)
    ap.add_argument("--mode", choices=("sim", "compile", "hw"), default="sim")
    ap.add_argument("--shapes", help="comma separated, default: nkibench's shapes for the op")
    ap.add_argument("--dtypes", default="f32", help="comma separated: f32,bf16")
    a = ap.parse_args(argv)

    if a.mode in ("hw", "compile"):
        os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    import ml_dtypes
    import nki

    spec = importlib.util.spec_from_file_location("kernel_under_test", a.kernel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, a.entry)
    shapes = [tuple(int(x) for x in t.split("x")) for t in (a.shapes or (MATMUL_SHAPES if a.op == "matmul" else ATTENTION_SHAPES)).split(",")]
    dtypes = {"f32": np.float32, "bf16": ml_dtypes.bfloat16}
    if a.mode == "compile":
        return _compile_only(kernel, a.op, shapes, dtypes, a.dtypes.split(","))

    bad = 0
    for dname in a.dtypes.split(","):
        for shape in shapes:
            for tag, args, want in make_cases(a.op, shape, dtypes[dname]):
                t0 = time.perf_counter()
                try:
                    got = np.asarray(nki.simulate(kernel)(*args) if a.mode == "sim" else kernel(*args)).astype(np.float32)
                except Exception as e:  # noqa: BLE001
                    if "NRT" in type(e).__name__ or "initialize NRT" in str(e):
                        print("NeuronCores are not available (held by another process on this node?): " + str(e).splitlines()[0])
                        return 3
                    raise
                err = float(np.max(np.abs(got - want) / (np.abs(want) + 1.0)))
                ok = got.shape == want.shape and bool(np.isfinite(got).all()) and err < tolerance(a.op, dname)
                bad += not ok
                print(f"{dname:5s} {'x'.join(map(str, shape)):>13s} {tag:10s} err {err:.2e} {'ok' if ok else 'MISMATCH'}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    print("ALL OK" if not bad else f"{bad} MISMATCHES")
    return 1 if bad else 0


def _compile_only(kernel, op, shapes, dtypes, dnames) -> int:
    """Compile through the whole pipeline -- NKI front end, MLIR verification, BIR, then neuronx-cc to a NEFF --
    but never run it. The executor is handed the compiled kernel *after* the NEFF exists; it records the NEFF's size
    and the neuronx-cc time, and a case only counts as compiling if a non-empty NEFF was produced."""
    from nki.framework.compiled import StandaloneKernel

    sk = kernel._to_subclass(StandaloneKernel)
    seen = {}

    def observe(compiled, exec_inputs, output_arrays):  # compile only, never run
        seen["neff_bytes"] = os.path.getsize(compiled.neff_path) if os.path.exists(compiled.neff_path) else 0
        seen["neuronx_cc_s"] = compiled.neuronx_cc_time

    object.__setattr__(sk, "_executor", observe)
    bad = 0
    for dname in dnames:
        for shape in shapes:
            _, args, _ = make_cases(op, shape, dtypes[dname])[0]
            label = f"{dname:5s} {'x'.join(map(str, shape)):>13s}"
            seen.clear()
            t0 = time.perf_counter()
            try:
                sk(*args)
            except Exception as e:  # noqa: BLE001
                msg = [l.strip() for l in str(e).splitlines() if "error" in l.lower()]
                print(f"{label} COMPILE FAILED: {' | '.join(msg)[:300] or str(e)[:300]}", flush=True)
                bad += 1
                continue
            if not seen.get("neff_bytes"):
                print(f"{label} NO NEFF PRODUCED (the front end accepted the kernel but neuronx-cc output is missing)", flush=True)
                bad += 1
                continue
            print(f"{label} COMPILES: NEFF {seen['neff_bytes'] / 1024:.0f} KiB, neuronx-cc {seen['neuronx_cc_s']:.1f}s "
                  f"(total {time.perf_counter() - t0:.1f}s)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
