import pytest

from nki_sched import ir
from nki_sched.analysis import AnalysisError, bounds_of, const_int, dim_interval, dma_bytes, injective, loop_ranges, region, upper_bound
from nki_sched.expr import Aff, sym, var

A = Aff
i, j, k = var("i"), var("j"), var("k")
M = sym("M")


@pytest.mark.parametrize("expr, ranges, lo, size", [
    (i, {"i": A(4)}, A(0), A(4)),                       # plain sweep
    (i + 5, {"i": A(4)}, A(5), A(4)),
    (i * 2, {"i": A(4)}, A(0), A(7)),                   # stride 2 touches 0,2,4,6 -> span 7
    (i * 4 + j, {"i": A(3), "j": A(4)}, A(0), A(12)),   # mixed radix: contiguous
    (-i + 3, {"i": A(4)}, A(0), A(4)),                  # negative coefficient: lowest is 3-3
    (var("o") * 128 + i, {"i": A(128)}, var("o") * 128, A(128)),  # outer var stays in the base
    (i, {"i": M // 128}, A(0), M // 128),               # symbolic extent
    (A(7), {"i": A(4)}, A(7), A(1)),                    # independent of ranged vars
])
def test_dim_interval(expr, ranges, lo, size):
    assert dim_interval(expr, ranges) == (lo, size)


def test_dim_interval_rejects_nonaffine_dependence_on_a_ranged_var():
    with pytest.raises(AnalysisError, match="non-affine"):
        dim_interval((i + 1) // 2, {"i": A(4)})


def test_region_of_a_single_access_is_its_bounding_box():
    lo, size = region([(var("o") * 128 + i, j)], {"i": A(128), "j": A(8)})
    assert lo == (var("o") * 128, A(0)) and size == (A(128), A(8))


def test_region_unions_accesses_that_differ_by_constants():
    lo, size = region([(i,), (i + 2,)], {"i": A(4)})
    assert lo == (A(0),) and size == (A(6),)
    lo, size = region([(i + 2,), (i,)], {"i": A(4)})
    assert lo == (A(0),) and size == (A(6),)


def test_region_rejects_accesses_with_symbolic_offset_difference():
    with pytest.raises(AnalysisError, match="symbolic offset"):
        region([(i + var("a"),), (i + var("b"),)], {"i": A(4)})


def test_const_int():
    assert const_int(A(5), "x") == 5
    with pytest.raises(AnalysisError, match="not a compile-time constant"):
        const_int(M, "the tile")


def test_loop_ranges_collects_nested_loops():
    body = (ir.For("i", A(2), (ir.For("j", M, (ir.Assign("a", (i, j), ir.Lit(0.0)),)),)),)
    assert loop_ranges(body) == {"i": A(2), "j": M}


@pytest.mark.parametrize("idx, loops, expected", [
    ((i, j), {"i": A(2), "j": A(3)}, True),                    # one loop per dimension
    ((i * 4 + j,), {"i": A(2), "j": A(4)}, True),              # mixed radix, no overlap
    ((i * 3 + j,), {"i": A(2), "j": A(4)}, False),             # stride 3 < extent 4: overlap
    ((i + j,), {"i": A(2), "j": A(2)}, False),                 # classic aliasing
    ((i,), {"i": A(2), "j": A(2)}, False),                     # j does not index -> not injective in (i, j)
    ((i, i), {"i": A(2)}, False),                              # same loop in two dims
    ((i * 4 + j,), {"i": M, "j": A(4)}, True),                 # symbolic outer extent is fine
])
def test_injective(idx, loops, expected):
    assert injective(idx, loops) is expected


# ================================================================ upper bounds of size expressions
S = sym("S")


@pytest.mark.parametrize("expr, bounds, expected", [
    (S, {"S": 128}, 128),
    (S + 3, {"S": 128}, 131),
    (2 * S, {"S": 64}, 128),
    (S // 4, {"S": 128}, 32),
    (S % 8, {}, 7),                                  # a remainder is bounded without any assumption
    (Aff.min(S, 100), {"S": 128}, 100),              # a min is bounded by its tighter side
    (Aff.min(S, 100), {}, 100),                      # ... even if the other side is unbounded
    (S - 5, {"S": 128}, 123),
    (Aff(7), {}, 7),
    (S * sym("T"), {"S": 4, "T": 8}, 32),            # a product of two bounded sizes
    (-S + 10, {"S": 4}, 10),                         # sizes are >= 0, so a negative coefficient contributes at most 0
])
def test_upper_bound(expr, bounds, expected):
    assert upper_bound(expr, bounds) == expected


@pytest.mark.parametrize("expr, bounds", [(S, {}), (S + sym("T"), {"S": 4}), (S * sym("T"), {"S": 4}), (var("i"), {"i": 4})])
def test_upper_bound_is_none_when_unbounded(expr, bounds):
    assert upper_bound(expr, bounds) is None


def test_bounds_of_reads_the_programs_bound_assumptions(naive_proc):
    p = naive_proc.with_(assumptions=(ir.Bound(sym("K"), 128), ir.Bound(sym("K"), 64), ir.Bound(2 * sym("M"), 100),
                                      ir.Assumption(sym("N"), 4)))
    assert bounds_of(p) == {"K": 64, "M": 50}        # the tightest bound wins; divisibility assumptions are not bounds


# ================================================================ static HBM traffic
def test_dma_bytes_multiplies_trip_counts_by_window_sizes(naive_proc):
    from nki_sched import NC_TINY, Sched
    s = Sched(naive_proc, NC_TINY)
    s.split("C.m", 4, names=("mo", "mi"), perfect=True)
    s.split("C.n", 8, names=("no", "ni"), perfect=True)
    s.reorder("mo", "no", "mi", "ni")
    s.compute_at("matmul", at="no")
    s.split("matmul.k", 4, names=("ko", "ki"), perfect=True)
    s.reorder("ko", "ki", "matmul.m", "matmul.n")
    s.set_memory("matmul", ir.PSUM)
    s.stage_in("lhsT", at="ko", mem=ir.SBUF, name="lhsT_sb")
    s.stage_in("rhs", at="ko", mem=ir.SBUF, name="rhs_sb")
    K_, M_, N_ = 12, 8, 16
    # each (mo, no, ko) iteration loads a 4x4 lhsT tile and a 4x8 rhs tile
    tiles = (M_ // 4) * (N_ // 8) * (K_ // 4)
    assert dma_bytes(s.proc, {"K": K_, "M": M_, "N": N_}, itemsize=4) == tiles * (4 * 4 + 4 * 8) * 4
    assert dma_bytes(s.proc, {"K": K_, "M": M_, "N": N_}, itemsize=2) == tiles * (4 * 4 + 4 * 8) * 2
