"""Instruction selection by unification (Exo's `replace`): match a perfect loop nest against an
instruction's semantic `Pattern` and, on success, build the instruction call with the operand windows
the nest covers."""

from __future__ import annotations

from dataclasses import dataclass

from . import ir
from .expr import Aff, Var
from .instrs import Instr, PBin, PHole, PRead, PUn, Pattern
from .lower import nest_window

_COMMUTATIVE = ("add", "mul", "max")


class MatchError(Exception):
    pass


@dataclass
class _State:
    loops: dict  # target loop var -> extent
    bind: dict  # formal loop name -> target loop var
    roles: dict  # role -> (buffer, idx tuple)
    holes: dict  # hole name -> float | ir.Param

    def copy(self):
        return _State(self.loops, dict(self.bind), dict(self.roles), dict(self.holes))


def _loop_vars_in(e: Aff, loops: dict) -> list:
    return [(a.name, c) for a, c in e.terms if isinstance(a, Var) and a.name in loops]


def _unify_access(role: str, pat_idx: tuple, buf: str, idx: tuple, st: _State):
    if len(pat_idx) != len(idx):
        raise MatchError(f"operand '{role}' ({buf}) has {len(idx)} dims but the instruction needs {len(pat_idx)}")
    if role in st.roles and st.roles[role] != (buf, idx):
        raise MatchError(f"operand '{role}' is read at two different places ({st.roles[role][0]} and {buf})")
    st.roles[role] = (buf, idx)
    for d, (p, e) in enumerate(zip(pat_idx, idx)):
        if isinstance(p, int):
            if not (e.is_const and e.const == p):
                raise MatchError(f"operand '{role}': dim {d} of {buf}[{', '.join(str(i) for i in idx)}] must be the constant {p}")
            continue
        lv = _loop_vars_in(e, st.loops)
        if len(lv) != 1 or lv[0][1] != 1:
            raise MatchError(f"operand '{role}': dim {d} of {buf}[{', '.join(str(i) for i in idx)}] must be one loop variable "
                             f"(with unit stride) plus an offset")
        v = lv[0][0]
        if st.bind.setdefault(p, v) != v:
            raise MatchError(f"operand '{role}': dim {d} is indexed by loop '{v}' but the instruction pairs '{p}' with '{st.bind[p]}'")
    return st


def _unify_expr(p, e, st: _State):
    """unify pattern expression p with data expression e (backtracking over commutative operand order)"""
    if isinstance(p, PRead):
        if not isinstance(e, ir.Read):
            raise MatchError(f"expected a read for operand '{p.role}', found {type(e).__name__}")
        return _unify_access(p.role, p.idx, e.buf, e.idx, st)
    if isinstance(p, PHole):
        if isinstance(e, ir.Lit):
            val = float(e.value)
        elif isinstance(e, ir.Param):
            val = e
        else:
            raise MatchError(f"scalar '{p.name}' must be a literal or a kernel parameter, found {type(e).__name__}")
        if st.holes.setdefault(p.name, val) != val:
            raise MatchError(f"scalar '{p.name}' matches two different values")
        return st
    if isinstance(p, PUn):
        if not (isinstance(e, ir.Un) and e.op == p.op):
            raise MatchError(f"expected {p.op}(...), found {_describe(e)}")
        return _unify_expr(p.a, e.a, st)
    if isinstance(p, PBin):
        if not (isinstance(e, ir.Bin) and e.op == p.op):
            raise MatchError(f"expected a '{p.op}' expression, found {_describe(e)}")
        attempts = [(e.a, e.b)] + ([(e.b, e.a)] if p.op in _COMMUTATIVE else [])
        last = None
        for left, right in attempts:
            trial = st.copy()
            try:
                _unify_expr(p.a, left, trial)
                _unify_expr(p.b, right, trial)
            except MatchError as err:
                last = err
                continue
            st.bind, st.roles, st.holes = trial.bind, trial.roles, trial.holes
            return st
        raise last
    raise TypeError(p)


def _describe(e) -> str:
    return type(e).__name__ + (f"({e.op})" if hasattr(e, "op") else "")


def _unify_nest(pattern: Pattern, chain: list, stmt) -> _State:
    if len(chain) != len(pattern.loops):
        raise MatchError(f"the instruction needs a nest of {len(pattern.loops)} loops, found {len(chain)}")
    st = _State({l.var: l.extent for l in chain}, {}, {}, {})
    if pattern.reduce is None:
        if not isinstance(stmt, ir.Assign):
            raise MatchError(f"the instruction computes `out = expr`; the nest has a {type(stmt).__name__} statement")
    else:
        if not isinstance(stmt, ir.Reduce) or stmt.op != pattern.reduce:
            raise MatchError(f"the instruction is a '{pattern.reduce}' reduction; the nest has "
                             f"{'a reduction with op ' + stmt.op if isinstance(stmt, ir.Reduce) else 'a plain assignment'}")
    role, pidx = pattern.out
    _unify_access(role, pidx, stmt.buf, stmt.idx, st)
    _unify_expr(pattern.expr, stmt.rhs, st)
    if sorted(st.bind.values()) != sorted(st.loops) or len(st.bind) != len(pattern.loops):
        unused = sorted(set(st.loops) - set(st.bind.values()))
        raise MatchError(f"loop(s) {unused} are not used by the accesses the instruction describes")
    return st


@dataclass
class Match:
    call: ir.Call
    out_window: ir.Window


def match_instr(instr: Instr, chain: list, stmt) -> Match:
    """Match the perfect nest `chain` (outermost first) whose innermost body is the single statement
    `stmt` against `instr.pattern`. Raises MatchError with a reason (and a layout hint when swapping
    an operand's dimensions would have made it match)."""
    pattern = instr.pattern
    try:
        st = _unify_nest(pattern, chain, stmt)
    except MatchError as err:
        hint = _transposed_hint(instr, chain, stmt)
        raise MatchError(str(err) + hint) from None
    wins = {}
    for role, (buf, idx) in st.roles.items():
        used = {v: st.loops[v] for v in st.loops if any(v in e.free_vars() for e in idx)}
        w, _ = nest_window(buf, idx, used)
        if w is None:
            raise MatchError(f"operand '{role}' ({buf}) is not a rectangular window of the loop nest")
        wins[role] = w
    attrs = tuple(pattern.attrs) + tuple(
        (k, (v if isinstance(v, ir.Param) else v)) for k, v in sorted(st.holes.items()))
    args = tuple((r, wins[r]) for r in instr.roles)
    return Match(ir.Call(instr.name, args, attrs), wins[pattern.out[0]])


def _transposed_hint(instr: Instr, chain: list, stmt) -> str:
    """If reversing the dims of one operand access makes the nest match, say which operand is laid out
    the wrong way round (the usual cause: a matmul operand that is [m, k] instead of [k, m])."""
    reads = list(ir.expr_reads(stmt.rhs))
    for r in reads:
        if len(r.idx) != 2:
            continue
        flipped = ir.Read(r.buf, tuple(reversed(r.idx)))
        try:
            _unify_nest(instr.pattern, chain, _swap_read(stmt, r, flipped))
        except MatchError:
            continue
        return (f". Operand '{r.buf}' is laid out transposed relative to what {instr.name} needs: stage a transposed copy "
                f"first (stage_in(..., perm=(1, 0)) or stage_transposed)")
    return ""


def _swap_read(stmt, old, new):
    def sub(e):
        if e is old:
            return new
        if isinstance(e, ir.Bin):
            return ir.Bin(e.op, sub(e.a), sub(e.b))
        if isinstance(e, ir.Un):
            return ir.Un(e.op, sub(e.a))
        if isinstance(e, ir.CastE):
            return ir.CastE(e.dtype, sub(e.e))
        return e

    if isinstance(stmt, ir.Reduce):
        return ir.Reduce(stmt.buf, stmt.idx, sub(stmt.rhs), stmt.op)
    return ir.Assign(stmt.buf, stmt.idx, sub(stmt.rhs))
