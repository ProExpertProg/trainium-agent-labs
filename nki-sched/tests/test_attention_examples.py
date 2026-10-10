"""The attention schedules in examples/attention.py: they must keep producing correct, hardware-legal kernels."""

from collections import Counter

import numpy as np
import pytest

from nki_sched import NC_DEFAULT, interp, ir
from nki_sched.analysis import dma_bytes
from nki_sched.emit import emit_nki


@pytest.fixture(scope="module")
def attn(load_example):
    return load_example("attention")


def instr_counts(proc):
    return Counter(c.instr for c in ir.walk(proc.body) if isinstance(c, ir.Call))


def lowered(attn, schedule, hw=NC_DEFAULT):
    s = attn.new_attention_sched(hw)
    getattr(attn, schedule)(s, hw)
    return s


# ------------------------------------------------------------------ single tile
def test_single_tile_uses_exactly_the_instruction_mix_of_the_hand_written_kernel(attn, build_golden_attention):
    scheduled = lowered(attn, "single_tile").proc
    from nki_sched.lower import select_copies
    assert instr_counts(select_copies(scheduled)) == instr_counts(build_golden_attention())


def test_single_tile_emits_a_shape_generic_kernel_for_s_and_d_up_to_the_partition_size(attn):
    src = lowered(attn, "single_tile").source()
    assert "def nki_attention_(q, k, v):" in src
    assert 'assert S <= 128, "expected S to be at most 128"' in src and 'assert D <= 128' in src
    assert "scale_rcp = (1.0 / (D ** 0.5))" in src                       # the scale comes from the shape, not the example
    assert "nl.divide" not in src                                         # rejected by the hardware compiler
    for op in ("nc_matmul", "nc_transpose", "tensor_reduce", "activation", "reciprocal"):
        assert f"nisa.{op}(" in src
    assert src.count("nisa.nc_matmul(") == 2 and src.count("nisa.nc_transpose(") == 3
    compile(src, "single_tile", "exec")


@pytest.mark.parametrize("S, D", [(2, 2), (3, 4), (4, 3), (4, 4)])
def test_single_tile_is_correct_after_every_primitive_on_the_tiny_chip(attn, new_attention_sched, S, D):
    s = new_attention_sched(check=True, shapes=[dict(S=S, D=D)])
    attn.single_tile(s, s.hw)                                             # raises at the first step that breaks the oracle
    assert len(s.history) > 30


# ------------------------------------------------------------------ tiled
def test_tiled_structure_query_tiles_outside_key_chunks_and_value_chunks(attn):
    from nki_sched.lower import select_copies
    proc = select_copies(lowered(attn, "tiled").proc)        # emission turns the final PSUM -> SBUF copy nest into an instruction
    qo = ir.find_loop(proc.body, "qo")
    assert [s.var for s in qo.body if isinstance(s, ir.For)] == ["kc", "vo"]
    assert qo.extent == ir.Aff.of("S") // 128
    assert not [s for s in proc.body if isinstance(s, ir.For) and s.var != "qo"]


def test_tiled_emits_a_kernel_for_any_multiple_of_128(attn):
    src = lowered(attn, "tiled").source()
    assert 'assert S % 128 == 0, "expected S to be a multiple of 128"' in src and "assert D <= 128" in src
    assert src.count("nisa.nc_matmul(") == 2 and src.count("nisa.nc_transpose(") == 3
    assert "for qo in nl.affine_range(S // 128):" in src and "for kc in nl.affine_range(S // 128):" in src
    # the value matmul accumulates in PSUM across key chunks: one PSUM tile per query tile, no explicit zeroing
    assert src.count("nl.psum") == 5 and "accumulate" not in src
    compile(src, "tiled", "exec")


@pytest.mark.parametrize("S, D", [(4, 2), (8, 4), (12, 3)])
def test_tiled_is_correct_after_every_primitive_on_the_tiny_chip(attn, new_attention_sched, S, D):
    s = new_attention_sched(check=True, shapes=[dict(S=S, D=D)])
    attn.tiled(s, s.hw)
    assert any("reassociates" in n for n in s.notes) and any("not bit-exact" in n for n in s.notes)


@pytest.mark.parametrize("S, D", [(128, 64), (256, 64), (512, 128)])
def test_tiled_hbm_traffic_reads_q_and_o_once_and_k_and_v_once_per_query_tile(attn, S, D):
    """K and V are re-read for every 128-row query tile (they do not stay resident): the static traffic model
    makes that cost visible and pins it."""
    proc = lowered(attn, "tiled").proc
    tiles = S // 128
    expected = 4 * (S * D + tiles * (2 * S * D) + S * D)               # q, k and v per tile, o
    assert dma_bytes(proc, {"S": S, "D": D}) == expected


def test_single_tile_hbm_traffic_is_the_floor(attn):
    proc = lowered(attn, "single_tile").proc
    assert dma_bytes(proc, {"S": 96, "D": 32}) == 4 * 4 * 96 * 32       # q, k, v in, o out: each tensor once


# ------------------------------------------------------------------ the example script
def test_attention_example_prints_both_schedules(attn, capsys):
    attn.main([])
    out = capsys.readouterr().out
    assert "single_tile" in out and "tiled" in out
    assert out.count("-- emitted NKI source:") == 2 and out.count("@nki.jit") == 2


def test_attention_cli_writes_the_requested_kernel(attn, tmp_path, capsys):
    out = tmp_path / "k.py"
    attn.main(["tiled", str(out)])
    text = out.read_text()
    assert "def nki_attention_(q, k, v):" in text and "for qo in nl.affine_range(S // 128):" in text
    assert "single_tile" not in capsys.readouterr().out          # only the requested schedule was built
