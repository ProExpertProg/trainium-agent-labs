"""The ns-level instruction set.

For every instruction this table says, in one place:
  * which memory space each operand may live in,
  * its *semantics* as a loop-nest `Pattern` (what `replace` recognises, and what tests check the
    interpreter's handler against),
  * its hardware caps, and
  * how it is spelled in NKI source.

Naming follows `ns.<engine>.<inst>`; the NKI function it lowers to is in `Instr.nisa`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .analysis import upper_bound
from .hw import HardwareConfig, HardwareError

# ---------------------------------------------------------------- semantic patterns


@dataclass(frozen=True)
class PRead:
    """read of operand `role` at an index tuple whose entries are formal loop names or int constants"""

    role: str
    idx: tuple


@dataclass(frozen=True)
class PBin:
    op: str  # add | sub | mul | div | max
    a: object
    b: object


@dataclass(frozen=True)
class PUn:
    op: str  # neg | exp | recip
    a: object


@dataclass(frozen=True)
class PHole:
    """a compile-time scalar: matches a float literal or a kernel parameter"""

    name: str


@dataclass(frozen=True)
class Pattern:
    """`for loops: out (reduce)= expr` with formal loop variables, in any order in the target nest.
    A reduction instruction overwrites its output, so it consumes the `init` nest in front of it."""

    loops: tuple
    out: tuple  # (role, idx)
    expr: object
    reduce: Optional[str] = None  # None: out = expr; "add" / "max": out op= expr
    init: Optional[float] = None
    attrs: tuple = ()  # constant attributes of the resulting call, e.g. (("accumulate", True),)


def _ij(role):
    return PRead(role, ("i", "j"))


# ---------------------------------------------------------------- hardware checks
def _check_matmul(call, hw: HardwareConfig, ub: Callable):
    sta, mov, dst = (call.arg(r) for r in ("stationary", "moving", "dst"))
    try:
        k, m = (ub(z) for z in sta.shape())
        k2, n = (ub(z) for z in mov.shape())
        dshape = tuple(call.arg("dst").shape())
    except (ValueError, AttributeError):
        raise HardwareError("ns.tensor.matmul: operand windows must be 2-D") from None
    if None in (k, m, k2, n):
        raise HardwareError("ns.tensor.matmul: tile sizes must be bounded (add a bound on the size symbols)")
    if tuple(sta.shape())[0] != tuple(mov.shape())[0] or dshape != (tuple(sta.shape())[1], tuple(mov.shape())[1]):
        raise HardwareError(f"ns.tensor.matmul: inconsistent shapes stationary{[str(z) for z in sta.shape()]} "
                            f"moving{[str(z) for z in mov.shape()]} dst{[str(z) for z in dshape]}")
    if k > hw.pmax or m > hw.stationary_fmax or n > hw.moving_fmax:
        raise HardwareError(f"ns.tensor.matmul: tile k={k} m={m} n={n} exceeds caps ({hw.pmax}, {hw.stationary_fmax}, {hw.moving_fmax})")


def _check_transpose(call, hw: HardwareConfig, ub: Callable):
    src, dst = call.arg("src"), call.arg("dst")
    p, f = (ub(z) for z in src.shape())
    if p is None or f is None:
        raise HardwareError("ns.tensor.transpose: tile sizes must be bounded (add a bound on the size symbols)")
    if p > hw.pmax or f > hw.pmax:
        raise HardwareError(f"ns.tensor.transpose: a {p}x{f} tile exceeds the {hw.pmax}x{hw.pmax} tensor-engine transpose limit")
    if tuple(dst.shape()) != tuple(reversed(tuple(src.shape()))):
        raise HardwareError("ns.tensor.transpose: dst must have the transposed shape of src")


# ---------------------------------------------------------------- NKI spelling
def _fmt_attrs(attrs: dict) -> str:
    return "".join(f", {k}={v}" for k, v in attrs.items() if k in ("accumulate", "engine") and v is not None)


def _spell(nisa_name: str, *fixed: str):
    """default spelling: nisa.<name>(role=window, ..., <fixed kwargs>, <accumulate/engine attrs>)"""

    def emit(instr: "Instr", srcs: dict, attrs: dict) -> str:
        kw = ", ".join(f"{r}={srcs[r]}" for r in instr.roles)
        fixed_kw = "".join(f", {f}" for f in fixed)
        return f"nisa.{nisa_name}({kw}{fixed_kw}{_fmt_attrs(attrs)})"

    return emit


def _renamed(nisa_name: str, mapping: dict, *fixed: str, scalar: Optional[str] = None):
    """spelling with renamed keyword arguments, e.g. src -> data; `scalar` is the NKI kwarg for the
    matched compile-time scalar (operand0=...)"""

    def emit(instr: "Instr", srcs: dict, attrs: dict) -> str:
        parts = [f"{mapping.get(r, r)}={srcs[r]}" for r in instr.roles]
        parts += list(fixed)
        if scalar:
            parts.append(f"{scalar}={attrs['c']}")
        return f"nisa.{nisa_name}({', '.join(parts)})"

    return emit


@dataclass(frozen=True)
class Instr:
    name: str  # ns-level name, "ns.tensor.matmul"
    engine: str  # tensor | vector | scalar | gpsimd | sync
    nisa: str  # NKI function it lowers to
    roles: tuple  # operand roles in call order
    mems: dict  # role -> allowed memory spaces
    emit: Callable = None
    pattern: Optional[Pattern] = None
    check: Optional[Callable] = None


_ONCHIP = ("SBUF", "PSUM")

INSTRS = {i.name: i for i in [
    Instr("ns.tensor.matmul", "tensor", "nc_matmul", ("dst", "stationary", "moving"),
          {"dst": ("PSUM",), "stationary": ("SBUF",), "moving": ("SBUF",)},
          emit=_spell("nc_matmul"),
          pattern=Pattern(("k", "m", "n"), ("dst", ("m", "n")),
                          PBin("mul", PRead("stationary", ("k", "m")), PRead("moving", ("k", "n"))),
                          reduce="add", attrs=(("accumulate", True),)),
          check=_check_matmul),
    Instr("ns.tensor.transpose", "tensor", "nc_transpose", ("dst", "src"),
          {"dst": ("PSUM",), "src": ("SBUF",)},
          emit=_renamed("nc_transpose", {"src": "data"}),
          pattern=Pattern(("i", "j"), ("dst", ("j", "i")), _ij("src")),
          check=_check_transpose),
    Instr("ns.sync.dma_copy", "sync", "dma_copy", ("dst", "src"),
          {"dst": ("HBM", "SBUF"), "src": ("HBM", "SBUF")}, emit=_spell("dma_copy")),
    Instr("ns.vector.tensor_copy", "vector", "tensor_copy", ("dst", "src"),
          {"dst": _ONCHIP, "src": _ONCHIP}, emit=_spell("tensor_copy")),
    Instr("ns.vector.reduce_max", "vector", "tensor_reduce", ("dst", "src"),
          {"dst": _ONCHIP, "src": _ONCHIP},
          emit=_renamed("tensor_reduce", {"src": "data"}, "op=nl.maximum", "axis=1"),
          pattern=Pattern(("i", "j"), ("dst", ("i", 0)), _ij("src"), reduce="max", init=float("-inf"))),
    Instr("ns.vector.reduce_sum", "vector", "tensor_reduce", ("dst", "src"),
          {"dst": _ONCHIP, "src": _ONCHIP},
          emit=_renamed("tensor_reduce", {"src": "data"}, "op=nl.add", "axis=1"),
          pattern=Pattern(("i", "j"), ("dst", ("i", 0)), _ij("src"), reduce="add", init=0.0)),
    Instr("ns.vector.scale", "vector", "tensor_scalar", ("dst", "src"),
          {"dst": _ONCHIP, "src": _ONCHIP},
          emit=_renamed("tensor_scalar", {"src": "data"}, "op0=nl.multiply", scalar="operand0"),
          pattern=Pattern(("i", "j"), ("dst", ("i", "j")), PBin("mul", _ij("src"), PHole("c")))),
    Instr("ns.vector.scale_rows", "vector", "tensor_scalar", ("dst", "src", "vec"),
          {"dst": _ONCHIP, "src": _ONCHIP, "vec": _ONCHIP},
          emit=lambda ins, s, a: f"nisa.tensor_scalar(dst={s['dst']}, data={s['src']}, op0=nl.multiply, operand0={s['vec']})",
          pattern=Pattern(("i", "j"), ("dst", ("i", "j")), PBin("mul", _ij("src"), PRead("vec", ("i", 0))))),
    Instr("ns.vector.negate", "vector", "tensor_scalar", ("dst", "src"),
          {"dst": _ONCHIP, "src": _ONCHIP},
          emit=_renamed("tensor_scalar", {"src": "data"}, "op0=nl.multiply", "operand0=-1.0"),
          pattern=Pattern(("i", "j"), ("dst", ("i", "j")), PUn("neg", _ij("src")))),
    Instr("ns.vector.reciprocal", "vector", "reciprocal", ("dst", "src"),
          {"dst": _ONCHIP, "src": _ONCHIP},
          emit=_renamed("reciprocal", {"src": "data"}),
          pattern=Pattern(("i", "j"), ("dst", ("i", "j")), PUn("recip", _ij("src")))),
    Instr("ns.scalar.exp", "scalar", "activation", ("dst", "src"),
          {"dst": _ONCHIP, "src": _ONCHIP},
          emit=_renamed("activation", {"src": "data"}, "op=nl.exp"),
          pattern=Pattern(("i", "j"), ("dst", ("i", "j")), PUn("exp", _ij("src")))),
    Instr("ns.scalar.exp_bias", "scalar", "activation", ("dst", "src", "bias"),
          {"dst": _ONCHIP, "src": _ONCHIP, "bias": _ONCHIP},
          emit=_renamed("activation", {"src": "data"}, "op=nl.exp"),
          pattern=Pattern(("i", "j"), ("dst", ("i", "j")), PUn("exp", PBin("add", _ij("src"), PRead("bias", ("i", 0)))))),
]}


def instr_upper_bound(bounds: dict) -> Callable:
    """size-expression upper bound function for hardware checks, given the program's Bound assumptions"""
    return lambda e: upper_bound(e, bounds)
