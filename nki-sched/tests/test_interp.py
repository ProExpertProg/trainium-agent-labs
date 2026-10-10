import numpy as np
import pytest

from nki_sched import interp, ir
from nki_sched.expr import Aff, sym, var

A = Aff
i, j = var("i"), var("j")


def buf(name, *shape, mem=ir.HBM, role="temp", dtype="f32"):
    return ir.Buffer(name, tuple(A(s) if isinstance(s, int) else s for s in shape), dtype, mem, role)


def proc(args, body, sizes=(), assumptions=()):
    return ir.Proc("t", tuple(args), tuple(sizes), tuple(body), tuple(assumptions))


def win(name, shape, lo=None, points=()):
    lo = lo or (A(0),) * len(shape)
    return ir.Window(name, tuple(lo), tuple(A(s) for s in shape), points)


# ------------------------------------------------------------------ quantize
@pytest.mark.parametrize("value, expected", [
    (1.0, 1.0),
    (1.0 + 2 ** -8, 1.0),                 # exactly halfway to the next bf16 -> ties to even (down)
    (1.0 + 3 * 2 ** -9, 1.0 + 2 ** -7),    # above halfway -> up
    (3.14159265, 3.140625),
    (-3.14159265, -3.140625),
    (0.0, 0.0),
])
def test_quantize_bf16(value, expected):
    assert interp.quantize("bf16", np.float32(value)) == np.float32(expected)


@pytest.mark.parametrize("dtype, value, expected", [
    ("f32", 0.1, np.float32(0.1)),
    ("f16", 0.1, np.float32(np.float16(0.1))),
    ("f16", 70000.0, np.float32(np.inf)),
])
def test_quantize_other_dtypes(dtype, value, expected):
    assert interp.quantize(dtype, np.float32(value)) == expected


def test_quantize_preserves_shape_and_scalar_rank():
    assert interp.quantize("bf16", np.float32(1.5)).ndim == 0
    assert interp.quantize("bf16", np.ones((2, 3), np.float32)).shape == (2, 3)


def test_quantize_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        interp.quantize("f8", np.zeros(1, np.float32))


# ------------------------------------------------------------------ statements
def test_assign_reduce_and_loops():
    # y[i] = sum_j x[i, j]
    x, y = buf("x", 2, 3, role="arg"), buf("y", 2, role="out")
    body = [ir.For("i", A(2), (
        ir.Assign("y", (i,), ir.Lit(0.0)),
        ir.For("j", A(3), (ir.Reduce("y", (i,), ir.Read("x", (i, j))),)),
    ))]
    xv = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = interp.run(proc([x, y], body), {"x": xv})
    np.testing.assert_array_equal(out["y"], xv.sum(1))


def test_sizes_are_read_from_argument_shapes_and_extents_use_them():
    n = sym("N")
    x, y = buf("x", n, role="arg"), buf("y", n, role="out")
    body = [ir.For("i", n, (ir.Assign("y", (i,), ir.Mul(ir.Read("x", (i,)), ir.Lit(2.0))),))]
    out = interp.run(proc([x, y], body, sizes=[("N", "x", 0)]), {"x": np.arange(5, dtype=np.float32)})
    np.testing.assert_array_equal(out["y"], 2 * np.arange(5))


def test_params_are_evaluated_from_sizes():
    n = sym("N")
    x, y = buf("x", n, role="arg"), buf("y", Aff.of(sym("H")), role="out")
    body = [ir.For("i", sym("H"), (ir.Assign("y", (i,), ir.Read("x", (i,))),))]
    p = proc([x, y], body, sizes=[("N", "x", 0)]).with_(params=(("H", n // 2),))
    out = interp.run(p, {"x": np.arange(8, dtype=np.float32)})
    np.testing.assert_array_equal(out["y"], np.arange(4))


@pytest.mark.parametrize("dtype, stored", [("f32", 0.1), ("f16", float(np.float16(0.1))), ("bf16", 0.10009765625)])
def test_stores_round_to_the_buffer_dtype(dtype, stored):
    y = buf("y", 1, role="out", dtype=dtype)
    out = interp.run(proc([y], [ir.Assign("y", (A(0),), ir.Lit(0.1))]), {})
    assert out["y"][0] == pytest.approx(stored, rel=1e-6)


def test_like_dtype_follows_the_input_dtype_override():
    x, y = buf("x", 1, role="arg", dtype="like:x"), buf("y", 1, role="out", dtype="like:x")
    p = proc([x, y], [ir.Assign("y", (A(0),), ir.Read("x", (A(0),)))])
    xv = np.array([1.0 + 2 ** -9], np.float32)
    assert interp.run(p, {"x": xv})["y"][0] == xv[0]
    assert interp.run(p, {"x": xv, "__dtypes__": {"x": "bf16"}})["y"][0] == 1.0  # rounded to bf16


def test_cast_expression_rounds():
    y = buf("y", 1, role="out")
    p = proc([y], [ir.Assign("y", (A(0),), ir.CastE("bf16", ir.Lit(3.14159265)))])
    assert interp.run(p, {})["y"][0] == pytest.approx(3.140625)


# ------------------------------------------------------------------ errors
def test_missing_input_is_reported():
    x, y = buf("x", 1, role="arg"), buf("y", 1, role="out")
    with pytest.raises(interp.InterpError, match="missing input x"):
        interp.run(proc([x, y], []), {})


def test_assumption_violation_is_reported():
    n = sym("N")
    x = buf("x", n, role="arg")
    p = proc([x], [], sizes=[("N", "x", 0)], assumptions=[ir.Assumption(n, 4)])
    with pytest.raises(interp.InterpError, match="assumption violated"):
        interp.run(p, {"x": np.zeros(6, np.float32)})
    interp.run(p, {"x": np.zeros(8, np.float32)})  # divisible: fine


def test_out_of_bounds_scalar_index():
    y = buf("y", 2, role="out")
    with pytest.raises(interp.InterpError, match="index out of bounds"):
        interp.run(proc([y], [ir.Assign("y", (A(2),), ir.Lit(1.0))]), {})


def test_read_of_uninitialised_buffer():
    y, t = buf("y", 1, role="out"), buf("t", 1)
    body = [ir.Alloc(t), ir.Assign("y", (A(0),), ir.Read("t", (A(0),)))]
    with pytest.raises(interp.InterpError, match="uninitialised"):
        interp.run(proc([y], body), {})


def test_reduce_into_uninitialised_buffer():
    y = buf("y", 1, role="out")
    with pytest.raises(interp.InterpError, match="uninitialised"):
        interp.run(proc([y], [ir.Reduce("y", (A(0),), ir.Lit(1.0))]), {})


def test_unknown_instruction():
    y = buf("y", 1, role="out")
    with pytest.raises(interp.InterpError, match="unknown instruction"):
        interp.run(proc([y], [ir.Call("ns.tensor.bogus", ())]), {})


# ------------------------------------------------------------------ instructions
def copy_proc(instr, dst_win, src_win, out_shape, dtype="f32", src_dtype="f32"):
    x = buf("x", 4, 8, role="arg", dtype=src_dtype)
    y = buf("y", *out_shape, role="out", dtype=dtype)
    return proc([x, y], [ir.Call(instr, (("dst", dst_win), ("src", src_win)))])


@pytest.mark.parametrize("instr", ["ns.sync.dma_copy", "ns.vector.tensor_copy", "ns.scalar.tensor_copy", "ns.gpsimd.tensor_copy"])
def test_copy_instructions_copy_a_window(instr):
    xv = np.arange(32, dtype=np.float32).reshape(4, 8)
    p = copy_proc(instr, win("y", (2, 4)), win("x", (2, 4), lo=(A(1), A(2))), (2, 4))
    np.testing.assert_array_equal(interp.run(p, {"x": xv})["y"], xv[1:3, 2:6])


def test_copy_with_point_dim_drops_the_axis():
    xv = np.arange(32, dtype=np.float32).reshape(4, 8)
    src = ir.Window("x", (A(0), A(3)), (A(4), A(1)), (1,))  # x[0:4, 3] -> 1-D of 4
    p = copy_proc("ns.sync.dma_copy", win("y", (4,)), src, (4,))
    np.testing.assert_array_equal(interp.run(p, {"x": xv})["y"], xv[:, 3])


def test_copy_casts_to_destination_dtype():
    xv = np.full((4, 8), 1.0 + 2 ** -9, np.float32)
    p = copy_proc("ns.vector.tensor_copy", win("y", (4, 8)), win("x", (4, 8)), (4, 8), dtype="bf16")
    assert (interp.run(p, {"x": xv})["y"] == 1.0).all()


def test_copy_shape_mismatch_and_out_of_bounds():
    xv = np.zeros((4, 8), np.float32)
    bad = copy_proc("ns.sync.dma_copy", win("y", (2, 4)), win("x", (4, 4)), (2, 4))
    with pytest.raises(interp.InterpError, match="shape mismatch"):
        interp.run(bad, {"x": xv})
    oob = copy_proc("ns.sync.dma_copy", win("y", (4, 4)), win("x", (4, 4), lo=(A(0), A(6))), (4, 4))
    with pytest.raises(interp.InterpError, match="out of bounds"):
        interp.run(oob, {"x": xv})


def matmul_proc_with(attrs, repeat=1, psum_shape=(3, 5)):
    """acc = repeat x (st.T @ mv) on freshly allocated PSUM, copied to y."""
    st, mv = buf("st", 4, 3, role="arg"), buf("mv", 4, 5, role="arg")
    y = buf("y", *psum_shape, role="out")
    acc = buf("acc", *psum_shape, mem=ir.PSUM)
    call = ir.Call("ns.tensor.matmul", (("dst", win("acc", psum_shape)), ("stationary", win("st", (4, 3))),
                                        ("moving", win("mv", (4, 5)))), attrs)
    return proc([st, mv, y], [ir.Alloc(acc)] + [call] * repeat
                + [ir.Call("ns.vector.tensor_copy", (("dst", win("y", psum_shape)), ("src", win("acc", psum_shape))))])


@pytest.fixture
def mm_inputs():
    r = np.random.default_rng(0)
    return {"st": r.standard_normal((4, 3)).astype(np.float32), "mv": r.standard_normal((4, 5)).astype(np.float32)}


@pytest.mark.parametrize("attrs, repeat, factor", [
    ((("accumulate", None),), 1, 1),   # first write to a fresh region overwrites
    ((("accumulate", None),), 3, 3),   # then later writes add
    ((("accumulate", False),), 3, 1),  # always overwrite
])
def test_matmul_accumulate_semantics(mm_inputs, attrs, repeat, factor):
    out = interp.run(matmul_proc_with(attrs, repeat), mm_inputs)["y"]
    np.testing.assert_allclose(out, factor * (mm_inputs["st"].T @ mm_inputs["mv"]), rtol=1e-5)


def test_matmul_accumulate_true_needs_initialised_psum(mm_inputs):
    with pytest.raises(interp.InterpError, match="uninitialised PSUM"):
        interp.run(matmul_proc_with((("accumulate", True),)), mm_inputs)


def test_matmul_accumulate_none_on_partially_written_region_is_rejected(mm_inputs):
    st, mv = buf("st", 4, 3, role="arg"), buf("mv", 4, 5, role="arg")
    y = buf("y", 3, 5, role="out")
    acc = buf("acc", 3, 5, mem=ir.PSUM)
    half = ir.Window("acc", (A(0), A(0)), (A(3), A(2)))
    first = ir.Call("ns.tensor.matmul", (("dst", half), ("stationary", win("st", (4, 3))),
                                         ("moving", ir.Window("mv", (A(0), A(0)), (A(4), A(2))))), (("accumulate", None),))
    whole = ir.Call("ns.tensor.matmul", (("dst", win("acc", (3, 5))), ("stationary", win("st", (4, 3))),
                                         ("moving", win("mv", (4, 5)))), (("accumulate", None),))
    with pytest.raises(interp.InterpError, match="partially-written"):
        interp.run(proc([st, mv, y], [ir.Alloc(acc), first, whole]), mm_inputs)


def test_matmul_shape_mismatch(mm_inputs):
    with pytest.raises(interp.InterpError, match="matmul: result"):
        interp.run(matmul_proc_with((), psum_shape=(3, 5)).with_(body=(
            ir.Alloc(buf("acc", 3, 4, mem=ir.PSUM)),
            ir.Call("ns.tensor.matmul", (("dst", win("acc", (3, 4))), ("stationary", win("st", (4, 3))),
                                         ("moving", win("mv", (4, 5)))), (("accumulate", False),)))), mm_inputs)


def test_matmul_on_a_point_indexed_3d_operand():
    st = buf("st", 4, 2, 3, role="arg")      # tile index on axis 1, dropped by a point window
    mv, y = buf("mv", 4, 5, role="arg"), buf("y", 3, 5, role="out")
    acc = buf("acc", 3, 5, mem=ir.PSUM)
    sta = ir.Window("st", (A(0), A(1), A(0)), (A(4), A(1), A(3)), (1,))
    call = ir.Call("ns.tensor.matmul", (("dst", win("acc", (3, 5))), ("stationary", sta), ("moving", win("mv", (4, 5)))),
                   (("accumulate", None),))
    p = proc([st, mv, y], [ir.Alloc(acc), call, ir.Call("ns.vector.tensor_copy", (("dst", win("y", (3, 5))), ("src", win("acc", (3, 5)))))])
    r = np.random.default_rng(1)
    inp = {"st": r.standard_normal((4, 2, 3)).astype(np.float32), "mv": r.standard_normal((4, 5)).astype(np.float32)}
    np.testing.assert_allclose(interp.run(p, inp)["y"], inp["st"][:, 1, :].T @ inp["mv"], rtol=1e-5)


# ------------------------------------------------------------------ loop order
def test_reverse_affine_only_reverses_affine_loops():
    y = buf("y", 3, role="out")
    # y[0] = 1; for i: y[i+1] = y[i] * 2  -- order matters
    def body(kind):
        return [ir.Assign("y", (A(0),), ir.Lit(1.0)),
                ir.For("i", A(2), (ir.Assign("y", (i + 1,), ir.Mul(ir.Read("y", (i,)), ir.Lit(2.0))),), kind=kind)]
    assert list(interp.run(proc([y], body("sequential")), {}, reverse_affine=True)["y"]) == [1.0, 2.0, 4.0]
    with pytest.raises(interp.InterpError):
        interp.run(proc([y], body("affine")), {}, reverse_affine=True)
    assert list(interp.run(proc([y], body("affine")), {}, reverse_affine=False)["y"]) == [1.0, 2.0, 4.0]


# ================================================================ expressions, reductions, parameters
def scalar_proc(rhs, out_shape=(1,), idx=(A(0),)):
    return proc([buf("y", *out_shape, role="out")], [ir.Assign("y", idx, rhs)])


@pytest.mark.parametrize("op, a, b, expected", [
    ("add", 1.5, 2.0, 3.5), ("sub", 1.5, 2.0, -0.5), ("mul", 1.5, 2.0, 3.0), ("div", 3.0, 2.0, 1.5), ("max", -1.0, 2.0, 2.0),
])
def test_binary_operators(op, a, b, expected):
    out = interp.run(scalar_proc(ir.Bin(op, ir.Lit(a), ir.Lit(b))), {})["y"]
    assert out[0] == pytest.approx(expected)


@pytest.mark.parametrize("op, a, expected", [("neg", 2.0, -2.0), ("exp", 1.0, np.e), ("recip", 4.0, 0.25)])
def test_unary_operators(op, a, expected):
    assert interp.run(scalar_proc(ir.Un(op, ir.Lit(a))), {})["y"][0] == pytest.approx(expected)


@pytest.mark.parametrize("op, init, expected", [("add", 0.0, 6.0), ("max", float("-inf"), 3.0)])
def test_reductions_start_from_the_identity_and_combine_with_their_operator(op, init, expected):
    x, y = buf("x", 3, role="arg"), buf("y", 1, role="out")
    body = [ir.Assign("y", (A(0),), ir.Lit(init)), ir.For("i", A(3), (ir.Reduce("y", (A(0),), ir.Read("x", (i,)), op),))]
    assert interp.run(proc([x, y], body), {"x": np.array([1.0, 2.0, 3.0], np.float32)})["y"][0] == expected


def test_a_max_reduction_of_very_negative_values_needs_the_minus_inf_identity():
    x, y = buf("x", 2, role="arg"), buf("y", 1, role="out")
    body = [ir.Assign("y", (A(0),), ir.Lit(float("-inf"))), ir.For("i", A(2), (ir.Reduce("y", (A(0),), ir.Read("x", (i,)), "max"),))]
    assert interp.run(proc([x, y], body), {"x": np.array([-1e30, -2e30], np.float32)})["y"][0] == np.float32(-1e30)


def test_kernel_scalar_parameters_are_evaluated_from_the_sizes():
    from nki_sched import scalar
    n = sym("N")
    x, y = buf("x", n, role="arg"), buf("y", n, role="out")
    body = [ir.For("i", n, (ir.Assign("y", (i,), ir.Bin("mul", ir.Read("x", (i,)), ir.Param("s"))),))]
    p = proc([x, y], body, sizes=[("N", "x", 0)]).with_(fparams=(("s", scalar.SBin("div", scalar.SConst(1.0), scalar.SSize(n))),))
    np.testing.assert_allclose(interp.run(p, {"x": np.array([2.0, 4.0, 6.0, 8.0], np.float32)})["y"], [0.5, 1.0, 1.5, 2.0])


def test_an_unknown_parameter_is_reported_by_name():
    with pytest.raises(interp.InterpError, match=r"unknown scalar parameter 'nope'"):
        interp.run(scalar_proc(ir.Param("nope")), {})


def test_bounds_are_checked_like_divisibility_assumptions():
    n = sym("N")
    x = buf("x", n, role="arg")
    p = proc([x], [], sizes=[("N", "x", 0)], assumptions=[ir.Bound(n, 4)])
    interp.run(p, {"x": np.zeros(4, np.float32)})
    with pytest.raises(interp.InterpError, match="assumption violated: N <= 4"):
        interp.run(p, {"x": np.zeros(5, np.float32)})


# ================================================================ the new instructions, against numpy
def one_call(instr, out_shape, in_shapes, attrs=(), out_dtype="f32", out_mem="SBUF"):
    """a program with a single instruction call over full windows; inputs are named by role"""
    args = [buf(r, *shp, role="arg") for r, shp in in_shapes.items()]
    dst = buf("dst", *out_shape, role="out", dtype=out_dtype)
    full = lambda b: ir.Window(b.name, (A(0),) * len(b.shape), b.shape)  # noqa: E731
    call = ir.Call(instr, (("dst", full(dst)),) + tuple((r, full(b)) for r, b in zip(in_shapes, args)), tuple(attrs))
    return proc(args + [dst], [call])


X = np.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], np.float32)
VEC = np.array([[2.0], [0.5]], np.float32)


@pytest.mark.parametrize("instr, shapes, inputs, attrs, expected", [
    ("ns.tensor.transpose", {"src": (2, 3)}, {"src": X}, (), X.T),
    ("ns.vector.reduce_max", {"src": (2, 3)}, {"src": X}, (), X.max(1, keepdims=True)),
    ("ns.vector.reduce_sum", {"src": (2, 3)}, {"src": X}, (), X.sum(1, keepdims=True)),
    ("ns.vector.scale", {"src": (2, 3)}, {"src": X}, (("c", 0.5),), X * 0.5),
    ("ns.vector.scale_rows", {"src": (2, 3), "vec": (2, 1)}, {"src": X, "vec": VEC}, (), X * VEC),
    ("ns.vector.negate", {"src": (2, 3)}, {"src": X}, (), -X),
    ("ns.vector.reciprocal", {"src": (2, 3)}, {"src": X}, (), 1.0 / X),
    ("ns.scalar.exp", {"src": (2, 3)}, {"src": X}, (), np.exp(X)),
    ("ns.scalar.exp_bias", {"src": (2, 3), "bias": (2, 1)}, {"src": X, "bias": -VEC}, (), np.exp(X - VEC)),
])
def test_instruction_handlers_compute_what_their_name_says(instr, shapes, inputs, attrs, expected):
    out_shape = {"ns.tensor.transpose": (3, 2), "ns.vector.reduce_max": (2, 1), "ns.vector.reduce_sum": (2, 1)}.get(instr, (2, 3))
    out = interp.run(one_call(instr, out_shape, shapes, attrs), inputs)["dst"]
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_scale_takes_a_kernel_parameter_as_well_as_a_literal():
    from nki_sched import scalar
    p = one_call("ns.vector.scale", (2, 3), {"src": (2, 3)}, (("c", ir.Param("k")),)).with_(fparams=(("k", scalar.SConst(3.0)),))
    np.testing.assert_allclose(interp.run(p, {"src": X})["dst"], X * 3.0)


@pytest.mark.parametrize("instr, shapes, message", [
    ("ns.vector.scale_rows", {"src": (2, 3), "vec": (2, 1)}, "operand 'vec' .* uninitialised"),
])
def test_instruction_operands_must_be_initialised(instr, shapes, message):
    p = one_call(instr, (2, 3), {"src": shapes["src"]}).with_(body=())
    uninit = buf("vec", 2, 1)
    full = lambda b: ir.Window(b.name, (A(0),) * len(b.shape), b.shape)  # noqa: E731
    src = buf("src", 2, 3, role="arg")
    call = ir.Call(instr, (("dst", full(buf("dst", 2, 3, role="out"))), ("src", full(src)), ("vec", full(uninit))))
    p = proc([src, buf("dst", 2, 3, role="out")], [ir.Alloc(uninit), call])
    with pytest.raises(interp.InterpError, match=message):
        interp.run(p, {"src": X})


def test_an_instruction_result_must_fit_its_destination():
    with pytest.raises(interp.InterpError, match="does not fit dst"):
        interp.run(one_call("ns.tensor.transpose", (2, 3), {"src": (2, 3)}), {"src": X})     # needs (3, 2)


def test_instruction_stores_round_to_the_destination_dtype():
    out = interp.run(one_call("ns.vector.negate", (1, 1), {"src": (1, 1)}, out_dtype="bf16"), {"src": np.array([[1.0 + 2 ** -9]], np.float32)})["dst"]
    assert out[0, 0] == -1.0
