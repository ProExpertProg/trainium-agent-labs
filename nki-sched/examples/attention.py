"""Single-head attention, O = softmax(Q K^T / sqrt(D)) V with Q, K, V of shape [S, D].

The spec is plain torch. `trace` expands it into stages (scores matmul, scale, the five softmax stages,
the value matmul); a schedule then places every stage on the chip. Two schedules live here:

  single_tile   S, D <= 128: every stage is one hardware instruction on one tile. This is what the
                nkibench attention level checks (seq 64/96/128, dim 32/64/128 with one kernel).
  tiled         S a multiple of 128, D <= 128: 128 query rows at a time. The scores, softmax and value
                matmul of a query tile are computed in turn (compute_at), the key axis in chunks of 128
                (the transposes are 128x128 at most) and the value matmul accumulates over key chunks in
                PSUM. Softmax is exact (the whole [128, S] row is in SBUF), not the online/flash variant.

    python examples/attention.py                         # prints the IR and the NKI kernel of each schedule
    python examples/attention.py tiled kernel.py         # also writes the kernel
"""

import argparse

import torch

import _common as c
from nki_sched import NC_DEFAULT, HardwareConfig, Sched, arg, trace
from nki_sched.ir import PSUM, SBUF


def attention(q, k, v):
    """one head: the scale is 1/sqrt(D), written as torch would."""
    return torch.softmax((q @ k.T) / (q.shape[1] ** 0.5), dim=-1) @ v


def attention_proc(dtype="f32", name="nki_attention_"):
    """Trace the spec with symbolic S (sequence) and D (head dim); example shapes are only for tracing."""
    dims = ("S", "D")
    specs = {n: arg(dims, dtype, example=(8, 4)) for n in ("q", "k", "v")}
    return trace(attention, specs, name=name, output="O")


def new_attention_sched(hw: HardwareConfig = NC_DEFAULT, **kw) -> Sched:
    return Sched(attention_proc(**kw), hw)


def single_tile(s: Sched, hw: HardwareConfig):
    """S, D <= pmax. Every stage becomes one instruction on a single tile."""
    s.bound("S", hw.pmax)
    s.bound("D", hw.pmax)

    # Hardware multiplies by a scalar or a per-row vector but cannot divide by one: x / y -> x * (1/y).
    s.recip_divisor("div")          # the 1/sqrt(D) scale is a kernel scalar
    s.recip_divisor("softmax")      # the 1/rowsum is its own [S, 1] stage

    # Where every intermediate lives: matmul results in PSUM, the rest in SBUF.
    for buf in ("div", "softmax_max", "softmax_neg", "softmax_exp", "softmax_sum", "softmax_sum_rcp", "softmax"):
        s.set_memory(buf, SBUF)
    s.set_memory("matmul", PSUM)
    s.set_memory("matmul_1", PSUM)

    # Inputs into SBUF; a matmul wants the contraction axis on the partitions, so Q and K (and later the
    # probabilities) are transposed on the tensor engine into PSUM and copied back to SBUF.
    for t in ("q", "k", "v"):
        s.stage_in(t, None, SBUF, f"{t}_sb")
    s.stage_transposed("q_sb", "qT")
    s.stage_transposed("k_sb", "kT")
    s.stage_transposed("softmax", "pT")
    s.stage_out("O", None, SBUF, "O_sb")

    # Every stage nest is now an instruction.
    s.replace("matmul.s", "ns.tensor.matmul")
    s.fold_init("matmul")
    s.replace("div.s", "ns.vector.scale")
    s.replace("softmax_max.s", "ns.vector.reduce_max")
    s.replace("softmax_neg.s", "ns.vector.negate")
    s.replace("softmax_exp.s", "ns.scalar.exp_bias")
    s.replace("softmax_sum.s", "ns.vector.reduce_sum")
    s.replace("softmax_sum_rcp.d0", "ns.vector.reciprocal")
    s.replace("softmax.s", "ns.vector.scale_rows")
    s.replace("matmul_1.s", "ns.tensor.matmul")
    s.fold_init("matmul_1")
    # the PSUM -> SBUF copy of the result (O_sb = matmul_1) is recognised as a copy at emission


def tiled(s: Sched, hw: HardwareConfig):
    """S a multiple of pmax, D <= pmax. Per tile of pmax query rows: scores in key chunks of pmax, softmax
    over the whole row, then O accumulated over key chunks."""
    s.bound("D", hw.pmax)
    s.recip_divisor("div")
    s.recip_divisor("softmax")

    # --- tile the queries, then compute the whole chain per query tile (last stage first)
    qo, qi = s.split("O.s", hw.pmax, names=("qo", "qi"), perfect=True)
    for stage in ("matmul_1", "softmax", "softmax_sum_rcp", "softmax_sum", "softmax_exp", "softmax_neg", "softmax_max", "div"):
        s.compute_at(stage, at=qo)

    # --- scores in key chunks of pmax: a chunk of K^T needs a 128x128 transpose at most
    kc, kci = s.split("div.s1", hw.pmax, names=("kc", "kci"), perfect=True)
    s.reorder(kc, "div.s", kci)
    s.compute_at("matmul", at=kc)

    # --- value matmul: contract over the keys in chunks of pmax, accumulating in PSUM
    vo, vi = s.split("matmul_1.s1", hw.pmax, names=("vo", "vi"), perfect=True)
    s.reorder(vo, vi, "matmul_1.s", "matmul_1.d")

    for buf in ("div", "softmax_max", "softmax_neg", "softmax_exp", "softmax_sum", "softmax_sum_rcp", "softmax"):
        s.set_memory(buf, SBUF)
    s.set_memory("matmul", PSUM)
    s.set_memory("matmul_1", PSUM)

    s.stage_in("q", qo, SBUF, "q_sb")
    s.stage_transposed("q_sb", "qT", at=qo)
    s.stage_in("k", kc, SBUF, "k_sb")
    s.stage_transposed("k_sb", "kT", at=kc)
    s.stage_in("v", vo, SBUF, "v_sb")
    s.stage_transposed("softmax", "pT", at=vo)
    s.stage_out("O", qo, SBUF, "O_sb")

    s.replace("matmul.s", "ns.tensor.matmul")
    s.fold_init("matmul")
    s.replace("div.s", "ns.vector.scale")
    s.replace("softmax_max.s", "ns.vector.reduce_max")
    s.replace("softmax_neg.s", "ns.vector.negate")
    s.replace("softmax_exp.s", "ns.scalar.exp_bias")
    s.replace("softmax_sum.s", "ns.vector.reduce_sum")
    s.replace("softmax_sum_rcp.d0", "ns.vector.reciprocal")
    s.replace("softmax.s", "ns.vector.scale_rows")
    s.replace(vi, "ns.tensor.matmul")
    s.fold_init("matmul_1")


SCHEDULES = {"single_tile": single_tile, "tiled": tiled}


def build(name: str, hw: HardwareConfig = NC_DEFAULT) -> Sched:
    s = new_attention_sched(hw)
    SCHEDULES[name](s, hw)
    return s


def main(argv=None):
    """no arguments: print every schedule's IR and kernel. With a schedule name and a file: write that kernel."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("schedule", nargs="?", choices=sorted(SCHEDULES))
    ap.add_argument("out", nargs="?", help="write the emitted kernel here")
    a = ap.parse_args(argv)
    for name in [a.schedule] if a.schedule else SCHEDULES:
        c.banner(SCHEDULES[name])
        s = build(name)
        c.show(s)
        if a.out:
            with open(a.out, "w") as f:
                f.write(s.source())
            print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
