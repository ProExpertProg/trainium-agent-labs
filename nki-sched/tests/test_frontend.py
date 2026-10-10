import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nki_sched import arg, interp, ir, trace  # noqa: E402
from nki_sched.expr import Aff, sym, var  # noqa: E402
from nki_sched.frontend import TraceError  # noqa: E402

K, M, N = sym("K"), sym("M"), sym("N")


def nest(stage, loops, stmt):
    for v, ext in reversed(loops):
        stmt = ir.For(v, ext, (stmt,), stage)
    return stmt


def expected_matmul_ir(out_rhs):
    """The naive IR for C = lhsT.T @ rhs, written out by hand."""
    v = var
    acc = ir.Buffer("matmul", (M, N), "f32", ir.HBM, "temp")
    init = nest("matmul.init", [("matmul.init.m", M), ("matmul.init.n", N)],
                ir.Assign("matmul", (v("matmul.init.m"), v("matmul.init.n")), ir.Lit(0.0)))
    update = nest("matmul", [("matmul.m", M), ("matmul.n", N), ("matmul.k", K)],
                  ir.Reduce("matmul", (v("matmul.m"), v("matmul.n")),
                            ir.Mul(ir.Read("lhsT", (v("matmul.k"), v("matmul.m"))), ir.Read("rhs", (v("matmul.k"), v("matmul.n"))))))
    idx = (v("C.m"), v("C.n"))
    out = nest("C", [("C.m", M), ("C.n", N)], ir.Assign("C", idx, out_rhs(ir.Read("matmul", idx))))
    return (ir.Alloc(acc), init, update, out)


def trace2(fn, dtype="f32", dims=(("K", "M"), ("K", "N")), shapes=((8, 4), (8, 8)), names=("lhsT", "rhs")):
    args = {n: arg(d, dtype, example=s) for n, d, s in zip(names, dims, shapes)}
    return trace(fn, args, name="k")


@pytest.mark.parametrize("dtype, out_rhs", [
    ("f32", lambda r: r),                                   # f32 accumulate == f32 result: no cast needed
    ("bf16", lambda r: ir.CastE("like:lhsT", r)),            # torch returns bf16: cast the f32 accumulator
])
def test_trace_matmul_matches_the_hand_written_naive_ir(dtype, out_rhs):
    p = trace2(lambda lhsT, rhs: lhsT.T @ rhs, dtype)
    assert p.body == expected_matmul_ir(out_rhs)
    assert [(b.name, b.role, b.dtype) for b in p.args] == [
        ("lhsT", "arg", "like:lhsT"), ("rhs", "arg", "like:rhs"), ("C", "out", "like:lhsT")]
    assert p.sizes == (("K", "lhsT", 0), ("M", "lhsT", 1), ("N", "rhs", 1))
    assert p.name == "k"


def test_widening_casts_on_operands_are_elided():
    explicit = trace2(lambda lhsT, rhs: (lhsT.float().T @ rhs.float()).to(lhsT.dtype), "bf16")
    assert explicit.body == expected_matmul_ir(lambda r: ir.CastE("like:lhsT", r))


def test_narrowing_cast_is_kept_and_output_dtype_is_concrete():
    p = trace2(lambda lhsT, rhs: (lhsT.T @ rhs).to(torch.bfloat16), "f32")
    assert p.args[-1].dtype == "bf16"
    assert p.body[-1].body[0].body[0].rhs == ir.CastE("bf16", ir.Read("matmul", (var("C.m"), var("C.n"))))


def test_output_name_and_stage_names_follow_arguments():
    p = trace(lambda lhsT, rhs: lhsT.T @ rhs, {"lhsT": arg(("K", "M"), example=(8, 4)), "rhs": arg(("K", "N"), example=(8, 8))},
              name="k", output="out")
    assert p.args[-1].name == "out" and ir.find_loop(p.body, "out.m") is not None


# ---------------------------------------------------------------- generality: many specs, torch is the oracle
SPECS = {
    "a.T @ b": (lambda a, b: a.T @ b, (("K", "M"), ("K", "N")), ((8, 4), (8, 6))),
    "a @ b": (lambda a, b: a @ b, (("M", "K"), ("K", "N")), ((4, 8), (8, 6))),
    "a.T @ b.T": (lambda a, b: a.T @ b.T, (("K", "M"), ("N", "K")), ((8, 4), (6, 8))),
    "permute": (lambda a, b: a.permute(1, 0) @ b, (("K", "M"), ("K", "N")), ((8, 4), (8, 6))),
    "transposed result": (lambda a, b: (a.T @ b).T, (("K", "M"), ("K", "N")), ((8, 4), (8, 6))),
    "transpose(0,1)": (lambda a, b: torch.transpose(a, 0, 1) @ b, (("K", "M"), ("K", "N")), ((8, 4), (8, 6))),
}


@pytest.mark.parametrize("name", sorted(SPECS))
def test_trace_is_correct_for_many_specs(name):
    fn, dims, shapes = SPECS[name]
    p = trace2(fn, dims=dims, shapes=shapes, names=("a", "b"))
    r = np.random.default_rng(0)
    a, b = (r.standard_normal(s).astype(np.float32) for s in shapes)
    got = interp.run(p, {"a": a, "b": b})["C"]
    np.testing.assert_allclose(got, fn(torch.from_numpy(a), torch.from_numpy(b)).numpy(), rtol=1e-4, atol=1e-4)


def test_two_chained_matmuls_materialize_two_stages():
    fn = lambda a, b, c: (a.T @ b) @ c  # noqa: E731
    p = trace(fn, {"a": arg(("K", "M"), example=(8, 4)), "b": arg(("K", "N"), example=(8, 6)), "c": arg(("N", "P"), example=(6, 3))}, name="k")
    stages = {s.stage for s in ir.walk(p.body) if isinstance(s, ir.For)}
    assert stages == {"matmul.init", "matmul", "matmul_1.init", "matmul_1", "C"}
    r = np.random.default_rng(1)
    a, b, c = (r.standard_normal(s).astype(np.float32) for s in ((8, 4), (8, 6), (6, 3)))
    got = interp.run(p, {"a": a, "b": b, "c": c})["C"]
    np.testing.assert_allclose(got, (a.T @ b) @ c, rtol=1e-4, atol=1e-4)
    assert [s[0] for s in p.sizes] == ["K", "M", "N", "P"]


def test_bf16_trace_matches_torch_bf16_after_rounding():
    fn = lambda lhsT, rhs: lhsT.T @ rhs  # noqa: E731
    p = trace2(fn, "bf16")
    r = np.random.default_rng(2)
    a, b = (interp.quantize("bf16", r.standard_normal(s).astype(np.float32)) for s in ((8, 4), (8, 8)))
    got = interp.run(p, {"lhsT": a, "rhs": b, "__dtypes__": {"lhsT": "bf16", "rhs": "bf16"}})["C"]
    want = fn(torch.from_numpy(a).to(torch.bfloat16), torch.from_numpy(b).to(torch.bfloat16)).float().numpy()
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------- errors
@pytest.mark.parametrize("fn, dims, shapes, message", [
    (lambda lhsT, rhs: torch.relu(lhsT.T @ rhs), (("K", "M"), ("K", "N")), ((8, 4), (8, 8)), "unsupported torch op in spec: aten.relu"),
    (lambda lhsT, rhs: lhsT + rhs, (("A", "B"), ("A", "B")), ((4, 4), (4, 4)), "unsupported torch op in spec: aten.add"),
    # numerically consistent example shapes, but the contraction dims are different symbols: torch itself
    # refuses (they would have to be equal), and the error tells the user to name them the same
    (lambda lhsT, rhs: lhsT.T @ rhs, (("K", "M"), ("J", "N")), ((8, 4), (8, 8)), "must always be equal"),
])
def test_trace_errors_name_the_problem(fn, dims, shapes, message):
    with pytest.raises(TraceError, match=message):
        trace2(fn, dims=dims, shapes=shapes)


def test_trace_needs_example_shapes():
    with pytest.raises(TraceError, match="example"):
        trace(lambda a, b: a @ b, {"a": arg(("M", "K")), "b": arg(("K", "N"))})


def test_trace_rejects_unsupported_dtype():
    with pytest.raises(TraceError, match="unsupported dtype"):
        trace(lambda a, b: (a.T @ b).to(torch.float64), {"a": arg(("K", "M"), example=(4, 4)), "b": arg(("K", "N"), example=(4, 4))})


# ================================================================ scalar ops, softmax, attention
from nki_sched import scalar  # noqa: E402
from nki_sched.expr import sym as _sym  # noqa: E402


def trace_qkv(fn, dtype="f32", names=("q", "k", "v"), shape=(8, 4)):
    return trace(fn, {n: arg(("S", "D"), dtype, example=shape) for n in names}, name="k", output="O")


def stmt_of(proc, buf):
    """the statement that finishes `buf` (the last one writing it: the update, not the init)"""
    return [st for st in ir.walk(proc.body) if isinstance(st, (ir.Assign, ir.Reduce)) and st.buf == buf][-1]


def stages(proc):
    return [st.stage for st in proc.body if isinstance(st, ir.For)]


@pytest.mark.parametrize("op, symbol", [(lambda x: x * 0.5, "mul"), (lambda x: x / 4.0, "div"), (lambda x: 0.5 * x, "mul")])
def test_scalar_literal_ops_become_pointwise_stages(op, symbol):
    p = trace(lambda a: op(a), {"a": arg(("S", "D"), example=(8, 4))}, name="k", output="O")
    rhs = stmt_of(p, p.body[0].buf.name).rhs
    assert rhs.op == symbol and isinstance(rhs.b, ir.Lit)
    r = np.random.default_rng(0)
    a = r.standard_normal((5, 3)).astype(np.float32)
    np.testing.assert_allclose(interp.run(p, {"a": a})["O"], op(torch.from_numpy(a)).numpy(), rtol=1e-6)


def test_shape_derived_scalars_stay_symbolic_not_baked_in_from_the_example():
    fn = lambda q: (q * (q.shape[1] ** -0.5))  # noqa: E731
    a = trace(fn, {"q": arg(("S", "D"), example=(8, 4))}, name="k", output="O")
    b = trace(fn, {"q": arg(("S", "D"), example=(6, 10))}, name="k", output="O")
    assert a == b  # the example shape leaves no trace in the program
    (name, expr), = a.fparams
    assert expr.eval({"D": 16}) == pytest.approx(0.25)


@pytest.mark.parametrize("fn, message", [
    (lambda a, b: a * b, "elementwise ops between two tensors"),
    (lambda a, b: 2.0 / a, "unsupported torch op in spec: aten.reciprocal"),   # 2/a lowers to reciprocal
    (lambda a, b: torch.softmax(a, dim=0), "last dim"),
])
def test_unsupported_scalar_and_softmax_forms_are_named(fn, message):
    with pytest.raises(TraceError, match=message):
        trace(fn, {"a": arg(("S", "D"), example=(8, 4)), "b": arg(("S", "D"), example=(8, 4))}, name="k", output="O")


def test_softmax_materializes_five_stages_with_keepdims_row_statistics():
    p = trace(lambda a: torch.softmax(a, dim=-1), {"a": arg(("S", "D"), example=(8, 4))}, name="k", output="O")
    assert stages(p) == ["softmax_max.init", "softmax_max", "softmax_neg", "softmax_exp", "softmax_sum.init", "softmax_sum", "softmax", "O"]
    bufs = ir.buffers_of(p)
    S, D = _sym("S"), _sym("D")
    assert bufs["softmax_max"].shape == (S, Aff(1)) and bufs["softmax_sum"].shape == (S, Aff(1))
    assert bufs["softmax_exp"].shape == (S, D)
    mx = stmt_of(p, "softmax_max")
    assert isinstance(mx, ir.Reduce) and mx.op == "max" and mx.idx[1] == Aff(0)
    init = next(st for st in ir.walk(p.body) if isinstance(st, ir.Assign) and st.buf == "softmax_max")
    assert init.rhs == ir.Lit(float("-inf"))
    exp = stmt_of(p, "softmax_exp").rhs
    assert exp.op == "exp" and exp.a.op == "add"            # exp(x + (-max)): the row max is subtracted
    assert stmt_of(p, "softmax_sum").op == "add"
    assert stmt_of(p, "softmax").rhs.op == "div"


@pytest.mark.parametrize("S, D, scale", [(4, 3, 1.0), (7, 2, 1.0), (5, 5, 40.0)])
def test_softmax_matches_torch_even_where_a_naive_exp_overflows(S, D, scale):
    p = trace(lambda a: torch.softmax(a, dim=-1), {"a": arg(("S", "D"), example=(8, 4))}, name="k", output="O")
    a = (np.random.default_rng(S).standard_normal((S, D)) * scale * 20).astype(np.float32)   # exp(1000s) would overflow
    got = interp.run(p, {"a": a})["O"]
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, torch.softmax(torch.from_numpy(a), -1).numpy(), rtol=1e-5, atol=1e-6)


def test_softmax_refuses_non_f32_with_an_explanation():
    with pytest.raises(TraceError, match="only supported for float32"):
        trace(lambda a: torch.softmax(a, dim=-1), {"a": arg(("S", "D"), "bf16", example=(8, 4))}, name="k", output="O")


# ------------------------------------------------------------------ attention
def test_explicit_attention_traces_to_the_expected_stages(naive_attention):
    assert stages(naive_attention) == [
        "matmul.init", "matmul", "div", "softmax_max.init", "softmax_max", "softmax_neg", "softmax_exp",
        "softmax_sum.init", "softmax_sum", "softmax", "matmul_1.init", "matmul_1", "O"]
    assert naive_attention.sizes == (("S", "q", 0), ("D", "q", 1))
    # q @ k.T: the scores accumulate over D, with k read transposed
    red = stmt_of(naive_attention, "matmul")
    assert [r.buf for r in ir.expr_reads(red.rhs)] == ["q", "k"]
    assert red.rhs.a.idx[1] == red.rhs.b.idx[1]             # both indexed by the contraction (head) dim


@pytest.mark.parametrize("S, D", [(4, 2), (8, 4), (6, 8), (1 + 2, 3)])
@pytest.mark.parametrize("scale", [1.0, 30.0])
def test_attention_matches_torch(naive_attention, attention_inputs, attention_fn, S, D, scale):
    inp = attention_inputs(S, D, seed=S + D, scale=scale)
    got = interp.run(naive_attention, inp)["O"]
    want = attention_fn(*(torch.from_numpy(inp[n]) for n in ("q", "k", "v"))).numpy()
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_sdpa_is_the_same_program_as_the_explicit_spec_up_to_the_scale_form(attention_inputs, naive_attention):
    sdpa = trace_qkv(lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    inp = attention_inputs(6, 4, seed=1)
    np.testing.assert_allclose(interp.run(sdpa, inp)["O"], interp.run(naive_attention, inp)["O"], rtol=1e-5, atol=1e-6)
    assert "mul" in stages(sdpa) and "div" not in stages(sdpa)          # sdpa multiplies by 1/sqrt(D): no divide to rewrite
    (_, expr), = sdpa.fparams
    assert expr.eval({"D": 16}) == pytest.approx(0.25)


def test_sdpa_honours_an_explicit_scale(attention_inputs):
    p = trace_qkv(lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.3))
    inp = attention_inputs(5, 3, seed=2)
    want = torch.nn.functional.scaled_dot_product_attention(*(torch.from_numpy(inp[n]) for n in ("q", "k", "v")), scale=0.3).numpy()
    np.testing.assert_allclose(interp.run(p, inp)["O"], want, rtol=1e-4, atol=1e-5)
    assert p.fparams == ()                                               # a literal scale needs no kernel parameter


@pytest.mark.parametrize("kwargs", [dict(is_causal=True), dict(dropout_p=0.1), dict(attn_mask=True)])
def test_sdpa_only_supports_default_arguments(kwargs):
    def fn(q, k, v):
        mask = torch.ones(q.shape[0], k.shape[0], dtype=torch.bool) if kwargs.get("attn_mask") else None
        kw = {k_: v_ for k_, v_ in kwargs.items() if k_ != "attn_mask"}
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kw)

    with pytest.raises(TraceError):
        trace_qkv(fn)
