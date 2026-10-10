"""scripts/kernel_check.py runs on a Trainium node, but its references (what every node-side check compares a
kernel against) are plain numpy and must be right: a wrong reference would silently weaken every check."""

import importlib.util
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def kc():
    spec = importlib.util.spec_from_file_location("kernel_check", os.path.join(ROOT, "scripts", "kernel_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("shape", [(4, 3, 5), (8, 8, 8)])
def test_matmul_reference_is_lhsT_transposed_times_rhs(kc, shape):
    ((tag, (a, b), want),) = kc.make_cases("matmul", shape, np.float32)
    assert tag == "" and a.shape == (shape[0], shape[1]) and b.shape == (shape[0], shape[2])
    np.testing.assert_allclose(want, torch.from_numpy(a).T.numpy() @ b, rtol=1e-5)


@pytest.mark.parametrize("shape", [(6, 3), (16, 8)])
def test_attention_reference_matches_torch_sdpa(kc, shape):
    cases = kc.make_cases("attention", shape, np.float32)
    assert [t for t, _, _ in cases] == ["normal", "x30 scores"]
    for _, (q, k, v), want in cases:
        ref = torch.nn.functional.scaled_dot_product_attention(*(torch.from_numpy(x) for x in (q, k, v))).numpy()
        np.testing.assert_allclose(want, ref, rtol=1e-4, atol=1e-5)


def test_the_large_attention_case_really_overflows_a_naive_softmax(kc):
    (_, _, _), (_, (q, k, _), _) = kc.make_cases("attention", (32, 16), np.float32)
    scores = q.astype(np.float32) @ k.astype(np.float32).T / np.sqrt(16)
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.exp(scores)).all()          # exp(x) without the row max would be inf here


def test_tolerances_are_looser_for_bf16_and_attention_than_for_f32_matmul(kc):
    assert kc.tolerance("matmul", "f32") < kc.tolerance("attention", "f32") < kc.tolerance("attention", "bf16")


def test_default_shapes_are_the_nkibench_ones(kc):
    assert kc.MATMUL_SHAPES.split(",") == ["128x128x512", "256x256x1024", "512x128x512", "256x512x1024"]
    assert kc.ATTENTION_SHAPES.split(",") == ["128x64", "64x128", "96x32", "128x128"]
