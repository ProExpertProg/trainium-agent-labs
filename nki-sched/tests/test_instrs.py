"""The instruction table is the single source of truth for what each ns instruction means. These tests make
the three places that use it agree: the semantic pattern (what `replace` recognises), the matcher, and
the interpreter's handler. For every instruction with a pattern we build the loop nest the pattern
describes, run it as scalar code, run it again as the matched instruction call, and compare."""

import itertools

import numpy as np
import pytest

from nki_sched import interp, ir
from nki_sched.expr import Aff, var
from nki_sched.instrs import INSTRS, PBin, PHole, PRead, PUn
from nki_sched.match import MatchError, match_instr

A = Aff
PATTERNED = sorted(n for n, i in INSTRS.items() if i.pattern is not None)
EXTENTS = {"i": 3, "j": 4, "k": 2, "m": 3, "n": 4}


def roles_of(expr, out_role):
    seen = [out_role]

    def walk(e):
        if isinstance(e, PRead) and e.role not in seen:
            seen.append(e.role)
        elif isinstance(e, PBin):
            walk(e.a), walk(e.b)
        elif isinstance(e, PUn):
            walk(e.a)

    walk(expr)
    return seen


def shape_of(idx, ext):
    return tuple(A(ext[x] if isinstance(x, str) else 1) for x in idx)


def data_expr(p, ext):
    """the data expression a pattern describes, with loop variables named like the formals"""
    if isinstance(p, PRead):
        return ir.Read(p.role, tuple(var(x) if isinstance(x, str) else A(x) for x in p.idx))
    if isinstance(p, PBin):
        return ir.Bin(p.op, data_expr(p.a, ext), data_expr(p.b, ext))
    if isinstance(p, PUn):
        return ir.Un(p.op, data_expr(p.a, ext))
    if isinstance(p, PHole):
        return ir.Lit(0.5)
    raise TypeError(p)


def loop_nest(order, ext, stmt):
    body = (stmt,)
    for v in reversed(order):
        body = (ir.For(v, A(ext[v]), body),)
    return body[0]


def build(instr, order):
    """(scalar program, its out buffer) for the pattern, loops nested in `order`"""
    pat = instr.pattern
    ext = {v: EXTENTS[v] for v in pat.loops}
    out_role, out_idx = pat.out
    roles = roles_of(pat.expr, out_role)
    reads = {r: next(e for e in _reads(pat.expr) if e.role == r) for r in roles if r != out_role}
    bufs = [ir.Buffer(r, shape_of(reads[r].idx, ext), "f32", ir.HBM, "arg") for r in roles if r != out_role]
    out = ir.Buffer(out_role, shape_of(out_idx, ext), "f32", ir.HBM, "out")
    oidx = tuple(var(x) if isinstance(x, str) else A(x) for x in out_idx)
    rhs = data_expr(pat.expr, ext)
    stmt = ir.Assign(out_role, oidx, rhs) if pat.reduce is None else ir.Reduce(out_role, oidx, rhs, pat.reduce)
    nest = loop_nest(order, ext, stmt)
    # reductions start from an identity value (the instruction either consumes this nest or accumulates onto it)
    init_value = pat.init if pat.init is not None else 0.0
    init = None
    if pat.reduce is not None:
        out_vars = [v for v in order if v in out_idx]
        init = loop_nest(out_vars, ext, ir.Assign(out_role, oidx, ir.Lit(init_value)))
    return tuple(bufs), out, init, nest


def _reads(p):
    if isinstance(p, PRead):
        yield p
    elif isinstance(p, PBin):
        yield from _reads(p.a)
        yield from _reads(p.b)
    elif isinstance(p, PUn):
        yield from _reads(p.a)


def inputs_for(bufs, seed=0):
    r = np.random.default_rng(seed)
    return {b.name: r.uniform(0.5, 2.0, size=tuple(int(d.const) for d in b.shape)).astype(np.float32) for b in bufs}


def run(bufs, out, body, inputs):
    proc = ir.Proc("t", bufs + (out,), (), tuple(body))
    return interp.run(proc, inputs)[out.name]


def orders(pat):
    return list(itertools.permutations(pat.loops))


@pytest.mark.parametrize("name", PATTERNED)
def test_every_instruction_has_an_interpreter_handler_and_a_spelling(name):
    ins = INSTRS[name]
    assert name in interp.CALLS and ins.emit is not None
    assert set(ins.mems) == set(ins.roles)


@pytest.mark.parametrize("name", PATTERNED)
def test_pattern_roles_are_instruction_roles(name):
    ins = INSTRS[name]
    assert set(roles_of(ins.pattern.expr, ins.pattern.out[0])) <= set(ins.roles)


@pytest.mark.parametrize("name", PATTERNED)
def test_matched_instruction_computes_what_its_pattern_says(name):
    """for every loop order the pattern allows: scalar nest == the matched instruction call"""
    ins = INSTRS[name]
    for order in orders(ins.pattern):
        bufs, out, init, nest = build(ins, order)
        chain = []
        s = nest
        while isinstance(s, ir.For):
            chain.append(s)
            s = s.body[0]
        m = match_instr(ins, chain, s)
        inp = inputs_for(bufs, seed=hash((name, order)) % 1000)
        scalar_body = ([init] if init is not None else []) + [nest]
        want = run(bufs, out, scalar_body, inp)
        # the instruction overwrites its output when it has an init (consumed); otherwise it accumulates onto it
        call_body = [] if ins.pattern.init is not None else ([init] if init is not None else [])
        got = run(bufs, out, call_body + [m.call], inp)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=f"{name} order={order}")


@pytest.mark.parametrize("name", PATTERNED)
def test_match_windows_cover_exactly_the_nest(name):
    ins = INSTRS[name]
    bufs, out, init, nest = build(ins, ins.pattern.loops)
    chain, s = [], nest
    while isinstance(s, ir.For):
        chain.append(s)
        s = s.body[0]
    m = match_instr(ins, chain, s)
    assert [r for r, _ in m.call.args] == list(ins.roles)
    for role, w in m.call.args:
        buf = next(b for b in bufs + (out,) if b.name == role)
        assert w.is_full({role: buf}) and w.buf == role
    assert m.out_window.buf == ins.pattern.out[0]


# ------------------------------------------------------------------ matcher specifics
def nest2(stmt, extents=(("i", 3), ("j", 4))):
    return [ir.For(v, A(e), ()) for v, e in extents], stmt


def test_hole_matches_a_literal_or_a_kernel_parameter_and_is_kept_as_an_attribute():
    ins = INSTRS["ns.vector.scale"]
    for scalar, expected in ((ir.Lit(0.25), 0.25), (ir.Param("scale"), ir.Param("scale"))):
        chain, _ = nest2(None)
        stmt = ir.Assign("d", (var("i"), var("j")), ir.Bin("mul", ir.Read("s", (var("i"), var("j"))), scalar))
        assert dict(match_instr(ins, chain, stmt).call.attrs)["c"] == expected


def test_commutative_operands_match_in_either_order():
    ins = INSTRS["ns.vector.scale"]
    chain, _ = nest2(None)
    stmt = ir.Assign("d", (var("i"), var("j")), ir.Bin("mul", ir.Lit(2.0), ir.Read("s", (var("i"), var("j")))))
    assert dict(match_instr(ins, chain, stmt).call.attrs)["c"] == 2.0


def test_offsets_become_window_origins():
    ins = INSTRS["ns.vector.tensor_copy"] if INSTRS["ns.vector.tensor_copy"].pattern else INSTRS["ns.vector.negate"]
    chain, _ = nest2(None)
    stmt = ir.Assign("d", (var("i") + var("o") * 3, var("j")), ir.Un("neg", ir.Read("s", (var("i"), var("j") + 8))))
    m = match_instr(INSTRS["ns.vector.negate"], chain, stmt)
    assert m.call.arg("dst").lo == (var("o") * 3, A(0)) and m.call.arg("src").lo == (A(0), A(8))


@pytest.mark.parametrize("instr, chain, stmt, message", [
    # wrong operator
    ("ns.vector.negate", nest2(None)[0],
     ir.Assign("d", (var("i"), var("j")), ir.Un("exp", ir.Read("s", (var("i"), var("j"))))), "expected neg"),
    # a reduction instruction needs a reduction statement
    ("ns.vector.reduce_max", nest2(None)[0],
     ir.Assign("d", (var("i"), A(0)), ir.Read("s", (var("i"), var("j")))), "reduction; the nest has a plain assignment"),
    # the reduction operator must agree
    ("ns.vector.reduce_max", nest2(None)[0],
     ir.Reduce("d", (var("i"), A(0)), ir.Read("s", (var("i"), var("j"))), "add"), "'max' reduction"),
    # a constant dim that is not the constant the instruction needs
    ("ns.vector.reduce_max", nest2(None)[0],
     ir.Reduce("d", (var("i"), A(1)), ir.Read("s", (var("i"), var("j"))), "max"), "must be the constant 0"),
    # the wrong number of loops
    ("ns.vector.negate", nest2(None, (("i", 3),))[0],
     ir.Assign("d", (var("i"), A(0)), ir.Un("neg", ir.Read("s", (var("i"), A(0))))), "needs a nest of 2 loops, found 1"),
    # both of the instruction's loops would be bound to the same target loop: loop j is left unused
    ("ns.vector.negate", nest2(None)[0],
     ir.Assign("d", (var("i"), var("i")), ir.Un("neg", ir.Read("s", (var("i"), var("i"))))), "not used by the accesses"),
    # a constant where the instruction needs a loop variable
    ("ns.vector.negate", nest2(None)[0],
     ir.Assign("d", (var("i"), A(0)), ir.Un("neg", ir.Read("s", (var("i"), A(0))))), "must be one loop variable"),
    # a dim that is not a unit-stride loop variable
    ("ns.vector.negate", nest2(None)[0],
     ir.Assign("d", (var("i") * 2, var("j")), ir.Un("neg", ir.Read("s", (var("i") * 2, var("j"))))), "one loop variable"),
    # a scalar that is neither a literal nor a parameter
    ("ns.vector.scale", nest2(None)[0],
     ir.Assign("d", (var("i"), var("j")), ir.Bin("mul", ir.Read("s", (var("i"), var("j"))), ir.Read("t", (var("i"), var("j"))))),
     "must be a literal or a kernel parameter"),
])
def test_match_failures_say_why(instr, chain, stmt, message):
    with pytest.raises(MatchError, match=message):
        match_instr(INSTRS[instr], chain, stmt)


def test_transposed_matmul_operand_gets_a_layout_hint():
    chain = [ir.For(v, A(e), ()) for v, e in (("k", 2), ("m", 3), ("n", 4))]
    stmt = ir.Reduce("d", (var("m"), var("n")), ir.Bin("mul", ir.Read("a", (var("m"), var("k"))), ir.Read("b", (var("k"), var("n")))))
    with pytest.raises(MatchError, match="'a' is laid out transposed"):
        match_instr(INSTRS["ns.tensor.matmul"], chain, stmt)


def test_matmul_pattern_accumulates_onto_the_destination():
    chain = [ir.For(v, A(e), ()) for v, e in (("k", 2), ("m", 3), ("n", 4))]
    stmt = ir.Reduce("d", (var("m"), var("n")), ir.Bin("mul", ir.Read("a", (var("k"), var("m"))), ir.Read("b", (var("k"), var("n")))))
    assert dict(match_instr(INSTRS["ns.tensor.matmul"], chain, stmt).call.attrs) == {"accumulate": True}
