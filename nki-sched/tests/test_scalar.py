import math

import pytest

from nki_sched.expr import sym
from nki_sched.scalar import SBin, SConst, SSize, SUn

D = SSize(sym("D"))
ONE = SConst(1.0)
ATTENTION_SCALE = SBin("div", ONE, SBin("pow", D, SConst(0.5)))


@pytest.mark.parametrize("expr, env, value", [
    (SConst(2.5), {}, 2.5),
    (D, {"D": 64}, 64.0),
    (SBin("add", SConst(1.0), SConst(2.0)), {}, 3.0),
    (SBin("sub", SConst(1.0), SConst(2.0)), {}, -1.0),
    (SBin("mul", SConst(3.0), D), {"D": 4}, 12.0),
    (SBin("pow", D, SConst(0.5)), {"D": 64}, 8.0),
    (SUn("sqrt", D), {"D": 64}, 8.0),
    (SUn("neg", SConst(2.0)), {}, -2.0),
    (ATTENTION_SCALE, {"D": 64}, 0.125),
    (ATTENTION_SCALE, {"D": 32}, 1 / math.sqrt(32)),
])
def test_eval(expr, env, value):
    assert expr.eval(env) == pytest.approx(value)


@pytest.mark.parametrize("expr, src", [
    (SConst(0.5), "0.5"),
    (D, "D"),
    (ATTENTION_SCALE, "(1.0 / (D ** 0.5))"),
    (SUn("sqrt", D), "(D ** 0.5)"),
    (SUn("neg", D), "(-D)"),
])
def test_to_py_is_self_contained_python(expr, src):
    assert expr.to_py() == src
    # evaluates to the same number as the symbolic expression
    assert eval(src, {}, {"D": 16}) == pytest.approx(expr.eval({"D": 16}))


def test_to_py_applies_rename():
    assert SSize(sym("D")).to_py(lambda n: n.lower()) == "d"


def test_str_is_readable():
    assert str(ATTENTION_SCALE) == "(1.0 / (D ** 0.5))"
    assert str(SUn("sqrt", D)) == "sqrt(D)"


def test_expressions_are_hashable_and_comparable():
    assert ATTENTION_SCALE == SBin("div", ONE, SBin("pow", D, SConst(0.5)))
    assert len({ATTENTION_SCALE, SBin("div", ONE, SBin("pow", D, SConst(0.5)))}) == 1
