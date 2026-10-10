"""Shared fixtures. Tests use only numpy, torch and this repo (no Neuron SDK, no nkibench)."""

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nki_sched import NC_TINY, Sched, arg, ir, trace, verify  # noqa: E402
from nki_sched.expr import sym, var  # noqa: E402

# small shapes that exercise 1..3 tiles in every dimension under NC_TINY (tiles 4 x 8, contraction 4)
CHECK_SHAPES = [dict(K=4, M=4, N=8), dict(K=8, M=8, N=16), dict(K=12, M=4, N=24)]


@pytest.fixture
def tiny_hw():
    return NC_TINY


@pytest.fixture
def check_shapes():
    return CHECK_SHAPES


@pytest.fixture
def make_proc():
    """factory: traced naive matmul program (C = lhsT.T @ rhs) with symbolic K, M, N."""

    def make(dtype="f32", name="mm"):
        return trace(
            lambda lhsT, rhs: lhsT.T @ rhs,
            {"lhsT": arg(("K", "M"), dtype, example=(8, 4)), "rhs": arg(("K", "N"), dtype, example=(8, 8))},
            name=name,
        )

    return make


@pytest.fixture
def naive_proc(make_proc):
    return make_proc()


@pytest.fixture
def make_inputs():
    """factory: random float32 inputs {"lhsT": [K,M], "rhs": [K,N]}."""

    def make(K, M, N, seed=0):
        r = np.random.default_rng(seed)
        return {"lhsT": r.standard_normal((K, M)).astype(np.float32),
                "rhs": r.standard_normal((K, N)).astype(np.float32)}

    return make


@pytest.fixture
def np_ref():
    """numpy reference for the matmul spec."""

    def ref(inputs):
        return inputs["lhsT"].T.astype(np.float32) @ inputs["rhs"].astype(np.float32)

    return ref


@pytest.fixture
def new_sched(make_proc, tiny_hw, np_ref):
    """factory: a Sched on the traced matmul with tiny tiles; check=True runs the differential
    oracle (incl. reverse-affine) after every primitive."""

    def make(check=True, proc=None, hw=None, shapes=CHECK_SHAPES):
        chk = verify.make_checker(np_ref, shapes) if check else None
        return Sched(proc if proc is not None else make_proc(), hw or tiny_hw, check=chk)

    return make


@pytest.fixture(scope="session")
def load_example():
    """factory: import examples/<name>.py as a module (the examples dir is put on sys.path)."""
    sys.path.insert(0, os.path.join(ROOT, "examples"))

    def load(name):
        spec = importlib.util.spec_from_file_location(f"examples_{name}", os.path.join(ROOT, "examples", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return load


@pytest.fixture
def build_golden_l4():
    """factory: the level-4 program written out by hand (what the l4 schedule should produce)."""

    def build(name="mm", tm=128, tn=512, tk=128):
        K, M, N = sym("K"), sym("M"), sym("N")
        A = ir.Aff
        lhsT = ir.Buffer("lhsT", (K, M), "like:lhsT", ir.HBM, "arg")
        rhs = ir.Buffer("rhs", (K, N), "like:rhs", ir.HBM, "arg")
        C = ir.Buffer("C", (M, N), "like:lhsT", ir.HBM, "out")
        acc = ir.Buffer("matmul", (A(tm), A(tn)), "f32", ir.PSUM)
        lhsT_sb = ir.Buffer("lhsT_sb", (A(tk), A(tm)), "like:lhsT", ir.SBUF)
        rhs_sb = ir.Buffer("rhs_sb", (A(tk), A(tn)), "like:rhs", ir.SBUF)
        C_sb = ir.Buffer("C_sb", (A(tm), A(tn)), "like:lhsT", ir.SBUF)
        mo, no, ko = var("mo"), var("no"), var("ko")
        full = lambda b: ir.Window(b.name, (A(0), A(0)), b.shape)  # noqa: E731
        dma_l = ir.Call("ns.sync.dma_copy", (("dst", full(lhsT_sb)),
                                             ("src", ir.Window("lhsT", (ko * tk, mo * tm), (A(tk), A(tm))))))
        dma_r = ir.Call("ns.sync.dma_copy", (("dst", full(rhs_sb)),
                                             ("src", ir.Window("rhs", (ko * tk, no * tn), (A(tk), A(tn))))))
        mm = ir.Call("ns.tensor.matmul", (("dst", full(acc)), ("stationary", full(lhsT_sb)), ("moving", full(rhs_sb))))
        cp = ir.Call("ns.vector.tensor_copy", (("dst", full(C_sb)), ("src", full(acc))))
        st = ir.Call("ns.sync.dma_copy", (("dst", ir.Window("C", (mo * tm, no * tn), (A(tm), A(tn)))), ("src", full(C_sb))))
        k_loop = ir.For("ko", K // tk, (ir.Alloc(lhsT_sb), ir.Alloc(rhs_sb), dma_l, dma_r, mm))
        n_loop = ir.For("no", N // tn, (ir.Alloc(acc), k_loop, ir.Alloc(C_sb), cp, st))
        return ir.Proc(
            name=name, args=(lhsT, rhs, C),
            sizes=(("K", "lhsT", 0), ("M", "lhsT", 1), ("N", "rhs", 1)),
            body=(ir.For("mo", M // tm, (n_loop,)),),
            assumptions=(ir.Assumption(M, tm), ir.Assumption(N, tn), ir.Assumption(K, tk)),
        )

    return build


@pytest.fixture
def build_golden_attention():
    """factory: single-tile attention (S, D <= 128) written out by hand at the ns level -- the program the
    attention schedule must produce. Mirrors the NKI idiom: q^T/k^T/p^T by tensor-engine transposes, the
    row max as a [S,1] tile, exp with a per-row bias, reciprocal + per-row scale for the normalisation."""
    from nki_sched import scalar as sc
    from nki_sched.expr import Aff as A

    def build(name="nki_attention_"):
        S, D = sym("S"), sym("D")
        I = ir.Alloc

        def B(nm, *shape, mem=ir.SBUF, dtype="like:q"):
            return ir.Buffer(nm, tuple(shape), dtype, mem)

        def W(b):
            return ir.Window(b.name, (A(0),) * len(b.shape), b.shape)

        def call(instr, **roles):
            return ir.Call(instr, tuple((r, W(b)) for r, b in roles.items()), ())

        q = ir.Buffer("q", (S, D), "like:q", ir.HBM, "arg")
        k = ir.Buffer("k", (S, D), "like:k", ir.HBM, "arg")
        v = ir.Buffer("v", (S, D), "like:v", ir.HBM, "arg")
        out = ir.Buffer("O", (S, D), "like:q", ir.HBM, "out")
        q_sb, k_sb, v_sb = B("q_sb", S, D), B("k_sb", S, D, dtype="like:k"), B("v_sb", S, D, dtype="like:v")
        qT_ps, qT = B("qT_ps", D, S, mem=ir.PSUM), B("qT", D, S)
        kT_ps, kT = B("kT_ps", D, S, mem=ir.PSUM, dtype="like:k"), B("kT", D, S, dtype="like:k")
        s_ps, s_sb = B("s_ps", S, S, mem=ir.PSUM, dtype="f32"), B("s_sb", S, S)
        m, nm, e = B("m", S, A(1)), B("nm", S, A(1)), B("e", S, S)
        ssum, rsum, p = B("ssum", S, A(1)), B("rsum", S, A(1)), B("p", S, S)
        pT_ps, pT = B("pT_ps", S, S, mem=ir.PSUM), B("pT", S, S)
        o_ps, o_sb = B("o_ps", S, D, mem=ir.PSUM, dtype="f32"), B("o_sb", S, D)
        mm = lambda dst, st, mv: ir.Call("ns.tensor.matmul", (("dst", W(dst)), ("stationary", W(st)), ("moving", W(mv))), (("accumulate", None),))  # noqa: E731
        scale_call = ir.Call("ns.vector.scale", (("dst", W(s_sb)), ("src", W(s_ps))), (("c", ir.Param("scale")),))
        body = (
            I(q_sb), call("ns.sync.dma_copy", dst=q_sb, src=q), I(k_sb), call("ns.sync.dma_copy", dst=k_sb, src=k),
            I(v_sb), call("ns.sync.dma_copy", dst=v_sb, src=v),
            I(qT_ps), call("ns.tensor.transpose", dst=qT_ps, src=q_sb), I(qT), call("ns.vector.tensor_copy", dst=qT, src=qT_ps),
            I(kT_ps), call("ns.tensor.transpose", dst=kT_ps, src=k_sb), I(kT), call("ns.vector.tensor_copy", dst=kT, src=kT_ps),
            I(s_ps), mm(s_ps, qT, kT), I(s_sb), scale_call,
            I(m), call("ns.vector.reduce_max", dst=m, src=s_sb), I(nm), call("ns.vector.negate", dst=nm, src=m),
            I(e), call("ns.scalar.exp_bias", dst=e, src=s_sb, bias=nm),
            I(ssum), call("ns.vector.reduce_sum", dst=ssum, src=e), I(rsum), call("ns.vector.reciprocal", dst=rsum, src=ssum),
            I(p), call("ns.vector.scale_rows", dst=p, src=e, vec=rsum),
            I(pT_ps), call("ns.tensor.transpose", dst=pT_ps, src=p), I(pT), call("ns.vector.tensor_copy", dst=pT, src=pT_ps),
            I(o_ps), mm(o_ps, pT, v_sb), I(o_sb), call("ns.vector.tensor_copy", dst=o_sb, src=o_ps),
            call("ns.sync.dma_copy", dst=out, src=o_sb),
        )
        return ir.Proc(
            name=name, args=(q, k, v, out),
            sizes=(("S", "q", 0), ("D", "q", 1)), body=body,
            assumptions=(ir.Bound(S, 128), ir.Bound(D, 128)),
            fparams=(("scale", sc.SBin("div", sc.SConst(1.0), sc.SBin("pow", sc.SSize(D), sc.SConst(0.5)))),),
        )

    return build


# ---------------------------------------------------------------------------------------- attention
ATTENTION_ARGS = {n: ("S", "D") for n in ("q", "k", "v")}


@pytest.fixture
def attention_fn():
    """the attention spec as torch writes it: softmax(q k^T / sqrt(D)) v"""
    import torch

    def attention(q, k, v):
        return torch.softmax((q @ k.T) / (q.shape[1] ** 0.5), dim=-1) @ v

    return attention


@pytest.fixture
def make_attention_proc(attention_fn):
    """factory: traced attention (symbolic S, D)"""

    def make(dtype="f32", name="attn"):
        specs = {n: arg(("S", "D"), dtype, example=(8, 4)) for n in ("q", "k", "v")}
        return trace(attention_fn, specs, name=name, output="O")

    return make


@pytest.fixture
def naive_attention(make_attention_proc):
    return make_attention_proc()


@pytest.fixture
def attention_inputs():
    """factory: random q, k, v of shape [S, D], optionally scaled up (large scores overflow a naive exp)"""

    def make(S, D, seed=0, scale=1.0):
        r = np.random.default_rng(seed)
        return {n: (r.standard_normal((S, D)) * scale).astype(np.float32) for n in ("q", "k", "v")}

    return make


@pytest.fixture
def attention_oracle(attention_fn):
    return verify.torch_oracle(attention_fn, ("q", "k", "v"))


@pytest.fixture
def new_attention_sched(make_attention_proc, tiny_hw, attention_oracle):
    """factory: a Sched on the traced attention with tiny tiles; check=True runs the oracle (incl. x6 inputs and
    reverse-affine) after every primitive. Shapes keep S a multiple of pmax=4 so the tiled schedule applies."""

    def make(check=True, shapes=None, hw=None):
        shapes = shapes or [dict(S=4, D=2), dict(S=8, D=4), dict(S=12, D=3)]
        chk = verify.make_checker(attention_oracle, shapes, args=ATTENTION_ARGS, out="O", scales=(1.0, 6.0), seeds=(0,)) if check else None
        return Sched(make_attention_proc(), hw or tiny_hw, check=chk)

    return make
