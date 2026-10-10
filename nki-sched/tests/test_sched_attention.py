"""Primitives that attention needs, exercised on the traced attention program (tiny chip: pmax = 4)."""

import numpy as np
import pytest

from nki_sched import ScheduleError, interp, ir
from nki_sched.expr import Aff, sym, var
from nki_sched.ir import PSUM, SBUF

A = Aff
S, D = sym("S"), sym("D")


def stmts(proc, kind):
    return [st for st in ir.walk(proc.body) if isinstance(st, kind)]


def calls(proc, instr=None):
    return [c for c in stmts(proc, ir.Call) if instr is None or c.instr == instr]


# ------------------------------------------------------------------ bound
def test_bound_records_an_assumption_once_and_is_idempotent(new_attention_sched):
    s = new_attention_sched(check=False)
    s.bound("S", 4)
    s.bound("S", 4)
    assert s.proc.assumptions == (ir.Bound(S, 4),)
    assert s.history[-1][0] == "bound(S <= 4)"


def test_bound_unknown_size_is_rejected_with_the_known_ones(new_attention_sched):
    with pytest.raises(ScheduleError, match=r"unknown size 'X'.*\['S', 'D'\]"):
        new_attention_sched(check=False).bound("X", 4)


def test_a_symbolic_partition_dim_is_only_legal_under_a_bound(new_attention_sched):
    s = new_attention_sched(check=False)
    with pytest.raises(ScheduleError, match="partition dim .* is S, which is not bounded"):
        s.set_memory("div", SBUF)
    s.bound("S", 4)
    s.set_memory("div", SBUF)
    assert ir.buffers_of(s.proc)["div"].mem == SBUF


def test_a_bound_larger_than_the_chip_is_not_enough(new_attention_sched):
    s = new_attention_sched(check=False)
    s.bound("S", 5)                       # tiny chip has pmax = 4
    with pytest.raises(ScheduleError, match="partition dimension"):
        s.set_memory("div", SBUF)


def test_the_bound_is_checked_when_the_program_runs(new_attention_sched, attention_inputs):
    s = new_attention_sched(check=False)
    s.bound("S", 4)
    with pytest.raises(interp.InterpError, match="assumption violated: S <= 4"):
        interp.run(s.proc, attention_inputs(5, 2))
    interp.run(s.proc, attention_inputs(4, 2))


def test_psum_tiles_are_bounded_by_the_bank_size(new_attention_sched):
    from nki_sched import HardwareConfig
    small_bank = HardwareConfig(pmax=128, psum_bank_bytes=32)             # a bank of 8 f32 per partition
    s = new_attention_sched(check=False, hw=small_bank)
    s.bound("S", 128)
    with pytest.raises(ScheduleError, match=r"matmul @ PSUM needs 512 B/partition > one bank \(32 B\)"):
        s.set_memory("matmul", PSUM)                                       # [S, S] f32 with S <= 128: 512 B per partition
    s2 = new_attention_sched(check=False, hw=small_bank)
    s2.bound("S", 8)
    s2.set_memory("matmul", PSUM)                                          # S <= 8: 32 B, exactly one bank


# ------------------------------------------------------------------ staging at the kernel root
@pytest.fixture
def bounded(new_attention_sched):
    def make(check=False):
        s = new_attention_sched(check=check)
        s.bound("S", 4)
        s.bound("D", 4)
        return s

    return make


def test_stage_in_at_the_root_is_placed_just_before_the_first_reader(bounded):
    s = bounded()
    s.stage_in("q", None, SBUF, "q_sb")
    i = next(i for i, st in enumerate(s.proc.body) if isinstance(st, ir.Alloc) and st.buf.name == "q_sb")
    assert isinstance(s.proc.body[i + 1], ir.Call) and s.proc.body[i + 1].instr == "ns.sync.dma_copy"
    reader = s.proc.body[i + 2]
    assert isinstance(reader, ir.For) and reader.stage == "matmul"        # the scores nest is the first reader of q
    assert s.history[-1][0] == "stage_in(q, at the kernel root, SBUF, q_sb)"


def test_stage_in_redirects_readers_but_not_the_producer_of_the_tensor(bounded):
    s = bounded()
    s.stage_in("q", None, SBUF, "q_sb")
    s.stage_in("q_sb", None, PSUM, "qT_ps", perm=(1, 0))
    copy = calls(s.proc, "ns.sync.dma_copy")[0]
    assert copy.arg("dst").buf == "q_sb"                                  # the producer still fills q_sb
    tr = calls(s.proc, "ns.tensor.transpose")[0]
    assert (tr.arg("src").buf, tr.arg("dst").buf) == ("q_sb", "qT_ps")
    red = [st for st in stmts(s.proc, ir.Reduce) if st.buf == "matmul"][0]
    assert {r.buf for r in ir.expr_reads(red.rhs)} == {"qT_ps", "k"}


def test_a_transposed_stage_swaps_the_window_dims_and_the_index_order(bounded):
    s = bounded()
    s.stage_in("q", None, SBUF, "q_sb")
    s.stage_in("q_sb", None, PSUM, "qT_ps", perm=(1, 0))
    assert ir.buffers_of(s.proc)["qT_ps"].shape == (D, S)
    red = [st for st in stmts(s.proc, ir.Reduce) if st.buf == "matmul"][0]
    read = next(r for r in ir.expr_reads(red.rhs) if r.buf == "qT_ps")
    assert read.idx == (var("matmul.d"), var("matmul.s"))                 # q[s, d] became qT[d, s]


def test_stage_transposed_gives_the_matmul_operand_layout_and_computes_the_same_thing(new_attention_sched):
    s = new_attention_sched(check=True, shapes=[dict(S=2, D=3), dict(S=4, D=4)])   # oracle after every step; S, D <= 4
    s.bound("S", 4)
    s.bound("D", 4)
    s.stage_in("q", None, SBUF, "q_sb")
    assert s.stage_transposed("q_sb", "qT") == "qT"
    bufs = ir.buffers_of(s.proc)
    assert (bufs["qT_ps"].mem, bufs["qT"].mem, bufs["qT"].shape) == (PSUM, SBUF, (D, S))
    assert [c.instr for c in calls(s.proc)] == ["ns.sync.dma_copy", "ns.tensor.transpose", "ns.vector.tensor_copy"]


@pytest.mark.parametrize("src_mem, dst_mem, perm, message", [
    (None, PSUM, (1, 0), r"HBM -> PSUM"),           # q is in HBM: transposes only exist SBUF -> PSUM
    ("sbuf", SBUF, (1, 0), r"SBUF -> SBUF"),
    ("sbuf", PSUM, (0, 0), "not a permutation"),
])
def test_transposed_staging_is_only_the_tensor_engine_transpose(bounded, src_mem, dst_mem, perm, message):
    s = bounded()
    if src_mem:
        s.stage_in("q", None, SBUF, "q_sb")
    with pytest.raises(ScheduleError, match=message):
        s.stage_in("q_sb" if src_mem else "q", None, dst_mem, "t", perm=perm)


def test_stage_in_refuses_a_tensor_written_at_or_after_its_first_read(tiny_hw):
    from nki_sched import Sched
    i = var("i")
    x = ir.Buffer("x", (A(4),), "f32", ir.HBM, "temp")
    out = ir.Buffer("y", (A(4),), "f32", ir.HBM, "out")
    # one loop both reads x[i] and writes x[i]: no copy can be placed between the producer and the reader
    body = (ir.Alloc(x), ir.For("i", A(4), (ir.Assign("x", (i,), ir.Lit(1.0)), ir.Assign("y", (i,), ir.Read("x", (i,))))))
    s = Sched(ir.Proc("t", (out,), (), body), tiny_hw)
    with pytest.raises(ScheduleError, match="written inside the scope .* at or after its first read"):
        s.stage_in("x", None, SBUF, "x_sb")


def test_stage_in_of_an_intermediate_whose_writer_comes_first_is_fine(bounded):
    s = bounded()
    s.stage_in("matmul", None, SBUF, "scores_sb")            # matmul is fully written (init + update) before the scale stage reads it
    assert ir.buffers_of(s.proc)["scores_sb"].mem == SBUF


def test_stage_in_with_no_readers_is_an_error(bounded):
    with pytest.raises(ScheduleError, match="no accesses to 'O'"):
        bounded().stage_in("O", None, SBUF, "x")


def test_stage_out_at_the_root_collects_the_writes_and_copies_out_at_the_end(bounded):
    s = bounded()
    s.stage_out("O", None, SBUF, "O_sb")
    assert isinstance(s.proc.body[0], ir.Alloc) and s.proc.body[0].buf.name == "O_sb"
    last = s.proc.body[-1]
    assert isinstance(last, ir.Call) and (last.arg("dst").buf, last.arg("src").buf) == ("O", "O_sb")
    writes = [st for st in stmts(s.proc, ir.Assign) if st.buf in ("O", "O_sb")]
    assert [w.buf for w in writes] == ["O_sb"]


# ------------------------------------------------------------------ compute_at with constant dims
@pytest.fixture
def query_tiled(bounded):
    def make():
        s = bounded()
        qo, qi = s.split("O.s", 4, names=("qo", "qi"), perfect=True)
        for stage in ("matmul_1", "softmax", "softmax_sum", "softmax_exp", "softmax_neg", "softmax_max", "div"):
            s.compute_at(stage, at=qo)
        return s

    return make


def test_compute_at_handles_a_row_statistic_with_a_constant_dim(query_tiled):
    s = query_tiled()
    bufs = ir.buffers_of(s.proc)
    assert bufs["softmax_max"].shape == (A(4), A(1)) and bufs["softmax_sum"].shape == (A(4), A(1))
    assert bufs["softmax_exp"].shape == (A(4), S) and bufs["div"].shape == (A(4), S)      # whole row of keys per query tile
    upd = [st for st in stmts(s.proc, ir.Reduce) if st.buf == "softmax_max"][0]
    assert upd.idx[1] == A(0) and upd.op == "max"


def test_the_whole_softmax_chain_moves_inside_the_query_tile_loop(query_tiled):
    s = query_tiled()
    qo = ir.find_loop(s.proc.body, "qo")
    inside = [st.stage for st in qo.body if isinstance(st, ir.For)]
    assert inside[:2] == ["div", "softmax_max.init"] and inside[-2:] == ["matmul_1", "O"]
    assert not [st for st in s.proc.body if isinstance(st, ir.For) and st.stage.startswith("softmax")]


def test_compute_at_rejects_consumers_that_read_a_different_constant_element(bounded):
    s = bounded()
    s.split("O.s", 4, names=("qo", "qi"), perfect=True)
    s.compute_at("matmul_1", at="qo")
    s.compute_at("softmax", at="qo")
    # make the normalisation read element [i, 1] of the row sum while the sum stage writes [i, 0]
    bad = s.proc.with_(body=tuple(ir.map_stmts(s.proc.body, lambda st: ir.Assign(st.buf, st.idx, ir.Bin(
        "div", st.rhs.a, ir.Read(st.rhs.b.buf, (st.rhs.b.idx[0], A(1)))))
        if isinstance(st, ir.Assign) and st.buf == "softmax" and isinstance(st.rhs, ir.Bin) and st.rhs.op == "div" else None)))
    s.proc = bad
    with pytest.raises(ScheduleError, match="written at constant index 0"):
        s.compute_at("softmax_sum", at="qo")


# ------------------------------------------------------------------ replace consumes an initialisation nest
def test_a_reduction_instruction_replaces_the_init_and_the_update_nests(bounded):
    s = bounded()
    s.set_memory("div", SBUF)
    s.set_memory("softmax_max", SBUF)
    before = len(stmts(s.proc, ir.For))
    s.replace("softmax_max.s", "ns.vector.reduce_max")
    assert not [st for st in stmts(s.proc, ir.For) if st.stage.startswith("softmax_max")]
    assert before - len(stmts(s.proc, ir.For)) == 4                              # two 2-loop nests became one instruction
    (call,) = calls(s.proc, "ns.vector.reduce_max")
    assert (call.arg("dst").buf, call.arg("src").buf) == ("softmax_max", "div")


def test_a_reduction_instruction_needs_its_init_directly_in_front(bounded):
    s = bounded()
    s.set_memory("div", SBUF)
    s.set_memory("softmax_max", SBUF)
    no_init = s.proc.with_(body=tuple(st for st in s.proc.body if not (isinstance(st, ir.For) and st.stage == "softmax_max.init")))
    s.proc = no_init
    with pytest.raises(ScheduleError, match="must initialise softmax_max to -inf"):
        s.replace("softmax_max.s", "ns.vector.reduce_max")


def test_replacing_with_the_wrong_reduction_names_the_mismatch(bounded):
    s = bounded()
    s.set_memory("div", SBUF)
    s.set_memory("softmax_sum", SBUF)
    s.set_memory("softmax_exp", SBUF)
    with pytest.raises(ScheduleError, match="'max' reduction; the nest has a reduction with op add"):
        s.replace("softmax_sum.s", "ns.vector.reduce_max")


def test_instruction_memory_spaces_are_enforced_at_replace(bounded):
    s = bounded()
    with pytest.raises(ScheduleError, match=r"argument 'dst' \(softmax_neg\) is in HBM"):
        s.replace("softmax_neg.s", "ns.vector.negate")                            # still in HBM


# ------------------------------------------------------------------ the whole thing
@pytest.mark.parametrize("S_, D_", [(2, 2), (3, 4), (4, 3)])
def test_the_single_tile_schedule_lowers_every_stage_and_computes_attention(load_example, new_attention_sched, attention_inputs, attention_oracle, S_, D_):
    ex = load_example("attention")
    s = new_attention_sched(check=False)
    ex.single_tile(s, s.hw)
    from nki_sched.lower import select_copies
    assert not [st for st in ir.walk(select_copies(s.proc).body) if isinstance(st, (ir.Assign, ir.Reduce, ir.For))]   # only instructions left
    inp = attention_inputs(S_, D_, seed=S_, scale=2.0)
    np.testing.assert_allclose(interp.run(s.proc, inp)["O"], attention_oracle(inp), rtol=1e-4, atol=1e-4)
