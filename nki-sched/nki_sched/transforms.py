"""Algebraic rewrites of a stage's arithmetic (as opposed to loop/memory rewrites). Each takes a Proc
and returns a new Proc; they are exposed as primitives on `Sched`."""

from __future__ import annotations

from dataclasses import replace

from . import ir, scalar
from .expr import Aff, var


class TransformError(Exception):
    pass


def _find_stage_nest(body: tuple, stage: str):
    for i, s in enumerate(body):
        if isinstance(s, ir.For) and s.stage == stage:
            return i
    return None


def _used_params(body) -> set:
    names = set()

    def visit(e):
        if isinstance(e, ir.Param):
            names.add(e.name)
        elif isinstance(e, ir.Bin):
            visit(e.a)
            visit(e.b)
        elif isinstance(e, (ir.Un,)):
            visit(e.a)
        elif isinstance(e, ir.CastE):
            visit(e.e)

    for s in ir.walk(body):
        if isinstance(s, (ir.Assign, ir.Reduce)):
            visit(s.rhs)
        elif isinstance(s, ir.Call):
            for _, v in s.attrs:
                if isinstance(v, ir.Param):
                    names.add(v.name)
    return names


def recip_divisor(proc: ir.Proc, stage: str) -> ir.Proc:
    """Rewrite every `a / b` in `stage` as `a * (1 / b)`.

    Hardware has no divide by a per-row vector or a scalar (nisa.tensor_scalar rejects `divide`), but it
    multiplies fast. The reciprocal is computed once: a divisor that is a kernel scalar parameter becomes
    a new parameter `<name>_rcp`; a divisor that is a buffer read gets its own reciprocal stage
    `<buf>_rcp` (one element per divisor element, not per output element).

    Not bit-exact: a * (1/b) rounds differently from a / b (about one ulp).
    """
    at = _find_stage_nest(proc.body, stage)
    if at is None:
        raise TransformError(f"no top-level nest for stage '{stage}' (apply recip_divisor before compute_at)")
    bufs = ir.buffers_of(proc)
    new_stages = []  # (Alloc, nest) for reciprocal stages
    fparams = dict(proc.fparams)
    made = {}

    def rewrite(e):
        if isinstance(e, ir.Bin):
            a, b = rewrite(e.a), rewrite(e.b)
            if e.op != "div":
                return ir.Bin(e.op, a, b)
            if isinstance(b, ir.Param):
                if b.name not in fparams:
                    raise TransformError(f"unknown scalar parameter {b.name}")
                new = f"{b.name}_rcp"
                fparams.setdefault(new, scalar.SBin("div", scalar.SConst(1.0), fparams[b.name]))
                return ir.Bin("mul", a, ir.Param(new))
            if isinstance(b, ir.Read):
                src = bufs[b.buf]
                name = f"{b.buf}_rcp"
                if name not in made:
                    if name in bufs:
                        raise TransformError(f"cannot create reciprocal stage: the name '{name}' is already in use")
                    loops = [(f"{name}.d{i}", ext) for i, ext in enumerate(src.shape)]
                    idx = tuple(var(n) for n, _ in loops)
                    nest = ir.Assign(name, idx, ir.Un("recip", ir.Read(b.buf, idx)))
                    for lv, ext in reversed(loops):
                        nest = ir.For(lv, ext, (nest,), name)
                    made[name] = True
                    new_stages.append((ir.Alloc(ir.Buffer(name, src.shape, src.dtype, ir.HBM, "temp")), nest))
                return ir.Bin("mul", a, ir.Read(name, b.idx))
            raise TransformError("the divisor must be a buffer element or a scalar parameter")
        if isinstance(e, ir.Un):
            return ir.Un(e.op, rewrite(e.a))
        if isinstance(e, ir.CastE):
            return ir.CastE(e.dtype, rewrite(e.e))
        return e

    def stmt(s):
        if isinstance(s, ir.For):
            return replace(s, body=tuple(stmt(x) for x in s.body))
        if isinstance(s, ir.Assign):
            return ir.Assign(s.buf, s.idx, rewrite(s.rhs))
        if isinstance(s, ir.Reduce):
            return ir.Reduce(s.buf, s.idx, rewrite(s.rhs), s.op)
        return s

    nest = stmt(proc.body[at])
    if nest == proc.body[at]:
        raise TransformError(f"stage '{stage}' contains no division")
    body = proc.body[:at] + tuple(x for pair in new_stages for x in pair) + (nest,) + proc.body[at + 1:]
    used = _used_params(body)
    fp = tuple((n, e) for n, e in fparams.items() if n in used)
    return proc.with_(body=body, fparams=fp)
