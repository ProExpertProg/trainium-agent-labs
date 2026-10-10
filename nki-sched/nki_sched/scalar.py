"""Compile-time floating-point scalars derived from size symbols, e.g. the attention scale
`1 / sqrt(D)`. They are kernel-level Python values (computed once from the argument shapes when the
kernel is traced), never baked-in constants, so one emitted kernel serves every shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .expr import Aff, to_py


@dataclass(frozen=True)
class SConst:
    value: float

    def eval(self, env: dict) -> float:
        return float(self.value)

    def to_py(self, rename=lambda n: n) -> str:
        return repr(float(self.value))

    def __str__(self):
        return repr(float(self.value))


@dataclass(frozen=True)
class SSize:
    """an integer size expression (e.g. the head dim D) used as a float"""

    expr: Aff

    def eval(self, env: dict) -> float:
        return float(self.expr.eval(env))

    def to_py(self, rename=lambda n: n) -> str:
        return to_py(self.expr, rename)

    def __str__(self):
        return str(self.expr)


_BIN = {
    "add": (lambda a, b: a + b, "+"),
    "sub": (lambda a, b: a - b, "-"),
    "mul": (lambda a, b: a * b, "*"),
    "div": (lambda a, b: a / b, "/"),
    "pow": (lambda a, b: a ** b, "**"),
}
_UN = {"sqrt": math.sqrt, "neg": lambda a: -a}


@dataclass(frozen=True)
class SBin:
    op: str
    a: Any
    b: Any

    def eval(self, env: dict) -> float:
        return _BIN[self.op][0](self.a.eval(env), self.b.eval(env))

    def to_py(self, rename=lambda n: n) -> str:
        return f"({self.a.to_py(rename)} {_BIN[self.op][1]} {self.b.to_py(rename)})"

    def __str__(self):
        return f"({self.a} {_BIN[self.op][1]} {self.b})"


@dataclass(frozen=True)
class SUn:
    op: str
    a: Any

    def eval(self, env: dict) -> float:
        return _UN[self.op](self.a.eval(env))

    def to_py(self, rename=lambda n: n) -> str:
        return f"({self.a.to_py(rename)} ** 0.5)" if self.op == "sqrt" else f"(-{self.a.to_py(rename)})"

    def __str__(self):
        return f"{self.op}({self.a})"
