import numpy as np
import pytest

from nki_sched import ScheduleError, interp, ir, scalar
from nki_sched.expr import sym
from nki_sched.transforms import TransformError, recip_divisor

torch = pytest.importorskip("torch")


def stage_stmt(proc, stage):
    return next(s for s in ir.walk(proc.body) if isinstance(s, (ir.Assign, ir.Reduce)) and s.buf == stage)


def stage_names(proc):
    return [s.stage for s in proc.body if isinstance(s, ir.For)]


# ------------------------------------------------------------------ scalar divisor
def test_scalar_divisor_becomes_a_reciprocal_parameter(naive_attention):
    assert dict(naive_attention.fparams) == {"scale": scalar.SBin("pow", scalar.SSize(sym("D")), scalar.SConst(0.5))}
    out = recip_divisor(naive_attention, "div")
    rhs = stage_stmt(out, "div").rhs
    assert isinstance(rhs, ir.Bin) and rhs.op == "mul" and rhs.b == ir.Param("scale_rcp")
    # the unused parameter is dropped; the new one is 1 / (D ** 0.5)
    assert [n for n, _ in out.fparams] == ["scale_rcp"]
    assert dict(out.fparams)["scale_rcp"].eval({"D": 64}) == pytest.approx(0.125)


# ------------------------------------------------------------------ buffer divisor
def test_buffer_divisor_gets_its_own_reciprocal_stage_in_front_of_the_consumer(naive_attention):
    out = recip_divisor(naive_attention, "softmax")
    assert stage_names(out).index("softmax_sum_rcp") == stage_names(out).index("softmax") - 1
    # one reciprocal per divisor element: a [S, 1] buffer, not a [S, S] one
    rcp = ir.buffers_of(out)["softmax_sum_rcp"]
    assert rcp.shape == ir.buffers_of(out)["softmax_sum"].shape and rcp.mem == ir.HBM
    assert stage_stmt(out, "softmax_sum_rcp").rhs == ir.Un("recip", ir.Read("softmax_sum", stage_stmt(out, "softmax_sum_rcp").idx))
    consumer = stage_stmt(out, "softmax").rhs
    assert consumer.op == "mul" and isinstance(consumer.b, ir.Read) and consumer.b.buf == "softmax_sum_rcp"
    assert consumer.b.idx[1].is_const  # still read at the [row, 0] element of the divisor


@pytest.mark.parametrize("stage", ["div", "softmax"])
@pytest.mark.parametrize("S, D", [(4, 2), (8, 5)])
def test_the_rewrite_is_equal_up_to_rounding(naive_attention, attention_inputs, stage, S, D):
    inp = attention_inputs(S, D, seed=3)
    want = interp.run(naive_attention, inp)["O"]
    got = interp.run(recip_divisor(naive_attention, stage), inp)["O"]
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_removing_the_division_entirely_leaves_no_div_in_the_program(naive_attention):
    out = recip_divisor(recip_divisor(naive_attention, "div"), "softmax")
    divs = [e for s in ir.walk(out.body) if isinstance(s, (ir.Assign, ir.Reduce))
            for e in _bins(s.rhs) if e.op == "div"]
    assert divs == []


def _bins(e):
    if isinstance(e, ir.Bin):
        yield e
        yield from _bins(e.a)
        yield from _bins(e.b)
    elif isinstance(e, ir.Un):
        yield from _bins(e.a)


# ------------------------------------------------------------------ errors
@pytest.mark.parametrize("stage, message", [
    ("matmul", "contains no division"),
    ("nope", "no top-level nest for stage 'nope'"),
])
def test_recip_divisor_errors(naive_attention, stage, message):
    with pytest.raises(TransformError, match=message):
        recip_divisor(naive_attention, stage)


def test_divisor_must_be_an_element_or_a_parameter(naive_attention):
    def poison(s):
        if isinstance(s, ir.Assign) and s.buf == "softmax":
            return ir.Assign(s.buf, s.idx, ir.Bin("div", s.rhs.a, ir.Un("exp", s.rhs.b)))
        return None

    bad = naive_attention.with_(body=tuple(ir.map_stmts(naive_attention.body, poison)))
    with pytest.raises(TransformError, match="divisor must be a buffer element or a scalar parameter"):
        recip_divisor(bad, "softmax")


def test_reciprocal_stage_name_must_be_free(naive_attention):
    once = recip_divisor(naive_attention, "softmax")
    # divide by softmax_sum again while softmax_sum_rcp (the reciprocal stage's name) already exists
    again = once.with_(body=tuple(ir.map_stmts(once.body, lambda s: ir.Assign(s.buf, s.idx, ir.Bin("div", s.rhs.a, ir.Read("softmax_sum", s.rhs.b.idx)))
                                               if isinstance(s, ir.Assign) and s.buf == "softmax" else None)))
    with pytest.raises(TransformError, match="already in use"):
        recip_divisor(again, "softmax")


# ------------------------------------------------------------------ the primitive
def test_sched_primitive_records_the_numeric_caveat_and_wraps_errors(new_attention_sched):
    s = new_attention_sched(check=True)
    s.recip_divisor("div")
    assert any("not bit-exact" in n for n in s.notes)
    assert s.history[-1][0] == "recip_divisor(div)"
    with pytest.raises(ScheduleError, match=r"recip_divisor\(matmul\): .*contains no division"):
        s.recip_divisor("matmul")
