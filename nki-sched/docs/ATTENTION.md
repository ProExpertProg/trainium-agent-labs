# Attention in nki-sched

Single-head attention, `O = softmax(Q Kᵀ / √D) V` with `Q, K, V` of shape `[S, D]`, from a torch spec to a
verified NKI kernel. Everything here is in `examples/attention.py` (the schedules), `nki_sched/frontend.py`
(softmax and scale ops), `nki_sched/instrs.py` (the vector/scalar/tensor-engine instructions) and
`nki_sched/transforms.py` (`recip_divisor`).

## 1. The spec and what `trace` makes of it

```python
def attention(q, k, v):
    """one head: the scale is 1/sqrt(D), written as torch would."""
    return torch.softmax((q @ k.T) / (q.shape[1] ** 0.5), dim=-1) @ v
```

`trace` exports it with symbolic dims, so `q.shape[1] ** 0.5` stays an expression over `D` and becomes a kernel
parameter (`scale = D ** 0.5`), not a number from the example shape. `F.scaled_dot_product_attention(q, k, v)`
(default arguments only: no mask, causal flag, dropout or GQA) expands to the same stages, multiplying by
`1/√D` instead of dividing.

The naive program is breadth-first, one stage per op, every stage in HBM:

| stage | computes | notes |
|---|---|---|
| `matmul` | `S = q kᵀ` | f32 accumulator, init + update nests; `k` is read transposed |
| `div` | `S / scale` | pointwise stage |
| `softmax_max` | row max of `div` | `[S, 1]` (reductions keep the reduced dim), init `-inf`, `max=` update |
| `softmax_neg` | `-max` | `[S, 1]` |
| `softmax_exp` | `exp(div + (-max))` | the max subtraction that keeps `exp` from overflowing |
| `softmax_sum` | row sum of `softmax_exp` | `[S, 1]` |
| `softmax` | `softmax_exp / softmax_sum` | |
| `matmul_1` | `softmax @ v` | contraction over the keys |
| `O` | copy to the output | |

## 2. The instruction set

`replace(loop, instr)` matches a loop nest against the instruction's semantic pattern (one table,
`instrs.INSTRS`), so an instruction is recognised by what it *computes*, in any loop order. A reduction
instruction overwrites its output, so it also consumes the init nest in front of it. The interpreter's
handler for each instruction is checked against its pattern in `tests/test_instrs.py`.

| ns instruction | lowers to | semantics (the loop nest `replace` recognises) |
|---|---|---|
| `ns.tensor.matmul` | `nisa.nc_matmul` | `for k, m, n: dst[m, n] += stationary[k, m] * moving[k, n]` |
| `ns.tensor.transpose` | `nisa.nc_transpose` | `for i, j: dst[j, i] = src[i, j]` |
| `ns.sync.dma_copy` | `nisa.dma_copy` | copy `dst = src` (HBM<->SBUF DMA, or on-chip copy with dtype cast) |
| `ns.vector.tensor_copy` | `nisa.tensor_copy` | copy `dst = src` (HBM<->SBUF DMA, or on-chip copy with dtype cast) |
| `ns.vector.reduce_max` | `nisa.tensor_reduce` | `for i, j: dst[i, 0] max= src[i, j]` (consumes the preceding `dst = -inf` nest) |
| `ns.vector.reduce_sum` | `nisa.tensor_reduce` | `for i, j: dst[i, 0] += src[i, j]` (consumes the preceding `dst = 0.0` nest) |
| `ns.vector.scale` | `nisa.tensor_scalar` | `for i, j: dst[i, j] = src[i, j] * c` |
| `ns.vector.scale_rows` | `nisa.tensor_scalar` | `for i, j: dst[i, j] = src[i, j] * vec[i, 0]` |
| `ns.vector.negate` | `nisa.tensor_scalar` | `for i, j: dst[i, j] = neg(src[i, j])` |
| `ns.vector.reciprocal` | `nisa.reciprocal` | `for i, j: dst[i, j] = recip(src[i, j])` |
| `ns.scalar.exp` | `nisa.activation` | `for i, j: dst[i, j] = exp(src[i, j])` |
| `ns.scalar.exp_bias` | `nisa.activation` | `for i, j: dst[i, j] = exp(src[i, j] + bias[i, 0])` |

Constraints learned from the NKI 0.6.0 documentation and the compiler (not all are visible in the simulator):

* `nisa.tensor_scalar` has **no divide**: the simulator accepts `op0=nl.divide`, the hardware compiler rejects it
  (`unsupported operator 'divide'`). Division is therefore rewritten as multiplication by a reciprocal.
* `tensor_scalar`'s vector operand must be float32 and shaped `[P, 1]`; on trn2 the scalar engine only supports
  multiply/add forms, so these run on the vector engine.
* `nc_transpose` (tensor engine) reads SBUF and writes PSUM, at most 128×128, and the output keeps the input dtype.
  `dma_transpose` only handles 2-byte dtypes, so float32 transposes go through the tensor engine.

## 3. New primitives

| primitive | what it does |
|---|---|
| `bound(size, max)` | assume `size <= max`; the kernel asserts it. A symbolic extent is only a legal partition dim when bounded |
| `recip_divisor(stage)` | rewrite `a / b` as `a * (1/b)`; a scalar divisor becomes a new kernel parameter, a buffer divisor gets its own reciprocal stage (one element per divisor element). Not bit-exact; recorded in `Sched.notes` |
| `stage_in(tensor, at=None, ..., perm=(1, 0))` | `at=None` stages at the kernel root, placed just before the first reader; `perm=(1, 0)` stages the transpose (a tensor-engine transpose, SBUF → PSUM) |
| `stage_transposed(tensor, name, at=None)` | the layout a matmul wants: transpose into PSUM, copy to SBUF |
| `stage_out(tensor, at=None, ...)` | also at the kernel root |
| `compute_at` | now accepts producers whose write index has a constant dim (the `[S, 1]` row statistics) |
| `replace` | table-driven: any instruction in the table; reductions consume their init nest |

## 4. Two schedules

### `single_tile`: S, D ≤ 128

Every stage is one instruction on one tile. The kernel is shape-generic: `S <= 128` and `D <= 128` are asserted, so
one file serves every nkibench attention shape (64/96/128 × 32/64/128).

```python
def single_tile(s: Sched, hw: HardwareConfig):
    """S, D <= pmax. Every stage becomes one instruction on a single tile."""
    s.bound("S", hw.pmax)
    s.bound("D", hw.pmax)

    # Hardware multiplies by a scalar or a per-row vector but cannot divide by one: x / y -> x * (1/y).
    s.recip_divisor("div")          # the 1/sqrt(D) scale is a kernel scalar
    s.recip_divisor("softmax")      # the 1/rowsum is its own [S, 1] stage

    # Where every intermediate lives: matmul results in PSUM, the rest in SBUF.
    for buf in ("div", "softmax_max", "softmax_neg", "softmax_exp", "softmax_sum", "softmax_sum_rcp", "softmax"):
        s.set_memory(buf, SBUF)
    s.set_memory("matmul", PSUM)
    s.set_memory("matmul_1", PSUM)

    # Inputs into SBUF; a matmul wants the contraction axis on the partitions, so Q and K (and later the
    # probabilities) are transposed on the tensor engine into PSUM and copied back to SBUF.
    for t in ("q", "k", "v"):
        s.stage_in(t, None, SBUF, f"{t}_sb")
    s.stage_transposed("q_sb", "qT")
    s.stage_transposed("k_sb", "kT")
    s.stage_transposed("softmax", "pT")
    s.stage_out("O", None, SBUF, "O_sb")

    # Every stage nest is now an instruction.
    s.replace("matmul.s", "ns.tensor.matmul")
    s.fold_init("matmul")
    s.replace("div.s", "ns.vector.scale")
    s.replace("softmax_max.s", "ns.vector.reduce_max")
    s.replace("softmax_neg.s", "ns.vector.negate")
    s.replace("softmax_exp.s", "ns.scalar.exp_bias")
    s.replace("softmax_sum.s", "ns.vector.reduce_sum")
    s.replace("softmax_sum_rcp.d0", "ns.vector.reciprocal")
    s.replace("softmax.s", "ns.vector.scale_rows")
    s.replace("matmul_1.s", "ns.tensor.matmul")
    s.fold_init("matmul_1")
    # the PSUM -> SBUF copy of the result (O_sb = matmul_1) is recognised as a copy at emission
```

### `tiled`: S a multiple of 128, D ≤ 128

128 query rows at a time: the stages are `compute_at` the query-tile loop (last stage first), the scores are
produced in key chunks of 128 (a K chunk needs a ≤128×128 transpose), softmax runs on the whole `[128, S]` row,
and `O` accumulates over key chunks of 128 in a single PSUM tile (`fold_init` + the matmul's first-write
semantics, as in the matmul `l4`).

```python
def tiled(s: Sched, hw: HardwareConfig):
    """S a multiple of pmax, D <= pmax. Per tile of pmax query rows: scores in key chunks of pmax, softmax
    over the whole row, then O accumulated over key chunks."""
    s.bound("D", hw.pmax)
    s.recip_divisor("div")
    s.recip_divisor("softmax")

    # --- tile the queries, then compute the whole chain per query tile (last stage first)
    qo, qi = s.split("O.s", hw.pmax, names=("qo", "qi"), perfect=True)
    for stage in ("matmul_1", "softmax", "softmax_sum_rcp", "softmax_sum", "softmax_exp", "softmax_neg", "softmax_max", "div"):
        s.compute_at(stage, at=qo)

    # --- scores in key chunks of pmax: a chunk of K^T needs a 128x128 transpose at most
    kc, kci = s.split("div.s1", hw.pmax, names=("kc", "kci"), perfect=True)
    s.reorder(kc, "div.s", kci)
    s.compute_at("matmul", at=kc)

    # --- value matmul: contract over the keys in chunks of pmax, accumulating in PSUM
    vo, vi = s.split("matmul_1.s1", hw.pmax, names=("vo", "vi"), perfect=True)
    s.reorder(vo, vi, "matmul_1.s", "matmul_1.d")

    for buf in ("div", "softmax_max", "softmax_neg", "softmax_exp", "softmax_sum", "softmax_sum_rcp", "softmax"):
        s.set_memory(buf, SBUF)
    s.set_memory("matmul", PSUM)
    s.set_memory("matmul_1", PSUM)

    s.stage_in("q", qo, SBUF, "q_sb")
    s.stage_transposed("q_sb", "qT", at=qo)
    s.stage_in("k", kc, SBUF, "k_sb")
    s.stage_transposed("k_sb", "kT", at=kc)
    s.stage_in("v", vo, SBUF, "v_sb")
    s.stage_transposed("softmax", "pT", at=vo)
    s.stage_out("O", qo, SBUF, "O_sb")

    s.replace("matmul.s", "ns.tensor.matmul")
    s.fold_init("matmul")
    s.replace("div.s", "ns.vector.scale")
    s.replace("softmax_max.s", "ns.vector.reduce_max")
    s.replace("softmax_neg.s", "ns.vector.negate")
    s.replace("softmax_exp.s", "ns.scalar.exp_bias")
    s.replace("softmax_sum.s", "ns.vector.reduce_sum")
    s.replace("softmax_sum_rcp.d0", "ns.vector.reciprocal")
    s.replace("softmax.s", "ns.vector.scale_rows")
    s.replace(vi, "ns.tensor.matmul")
    s.fold_init("matmul_1")
```

Emitted for any `S % 128 == 0, D <= 128`:

```python
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_(q, k, v):
  S, D = q.shape
  assert D <= 128, "expected D to be at most 128"
  assert S % 128 == 0, "expected S to be a multiple of 128"
  scale_rcp = (1.0 / (D ** 0.5))
  assert 16 * D + 12 * S + 1552 <= 196608, "SBUF footprint (bytes/partition, f32 worst case) exceeds the budget; block the loops"
  O = nl.ndarray((S, D), dtype=q.dtype, buffer=nl.shared_hbm)

  for qo in nl.affine_range(S // 128):
    O_sb = nl.ndarray((128, D), dtype=q.dtype, buffer=nl.sbuf)
    div = nl.ndarray((128, S), dtype=q.dtype, buffer=nl.sbuf)
    q_sb = nl.ndarray((128, D), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=q[128 * qo:128 * qo + 128, 0:D])
    qT_ps = nl.ndarray((D, 128), dtype=q.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=qT_ps, data=q_sb)
    qT = nl.ndarray((D, 128), dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qT, src=qT_ps)
    for kc in nl.affine_range(S // 128):
      matmul = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
      k_sb = nl.ndarray((128, D), dtype=k.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=k_sb, src=k[128 * kc:128 * kc + 128, 0:D])
      kT_ps = nl.ndarray((D, 128), dtype=k.dtype, buffer=nl.psum)
      nisa.nc_transpose(dst=kT_ps, data=k_sb)
      kT = nl.ndarray((D, 128), dtype=k.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=kT, src=kT_ps)
      nisa.nc_matmul(dst=matmul, stationary=qT, moving=kT)
      nisa.tensor_scalar(dst=div[0:128, 128 * kc:128 * kc + 128], data=matmul, op0=nl.multiply, operand0=scale_rcp)
    softmax_max = nl.ndarray((128, 1), dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=softmax_max, data=div, op=nl.maximum, axis=1)
    softmax_neg = nl.ndarray((128, 1), dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=softmax_neg, data=softmax_max, op0=nl.multiply, operand0=-1.0)
    softmax_exp = nl.ndarray((128, S), dtype=q.dtype, buffer=nl.sbuf)
    nisa.activation(dst=softmax_exp, data=div, bias=softmax_neg, op=nl.exp)
    softmax_sum = nl.ndarray((128, 1), dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=softmax_sum, data=softmax_exp, op=nl.add, axis=1)
    softmax_sum_rcp = nl.ndarray((128, 1), dtype=q.dtype, buffer=nl.sbuf)
    nisa.reciprocal(dst=softmax_sum_rcp, data=softmax_sum)
    softmax = nl.ndarray((128, S), dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=softmax, data=softmax_exp, op0=nl.multiply, operand0=softmax_sum_rcp)
    matmul_1 = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
    for vo in nl.affine_range(S // 128):
      v_sb = nl.ndarray((128, D), dtype=v.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=v_sb, src=v[128 * vo:128 * vo + 128, 0:D])
      pT_ps = nl.ndarray((128, 128), dtype=q.dtype, buffer=nl.psum)
      nisa.nc_transpose(dst=pT_ps, data=softmax[0:128, 128 * vo:128 * vo + 128])
      pT = nl.ndarray((128, 128), dtype=q.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=pT, src=pT_ps)
      nisa.nc_matmul(dst=matmul_1, stationary=pT, moving=v_sb)
    nisa.tensor_copy(dst=O_sb, src=matmul_1)
    nisa.dma_copy(dst=O[128 * qo:128 * qo + 128, 0:D], src=O_sb)

  return O
```

### What this does *not* do

* **float32 only.** The frontend refuses softmax on other dtypes: bfloat16 would need float32 row statistics and an
  explicit cast of the probabilities before the second matmul (the compiler rejects a bf16 per-row operand), which is
  a mixed-precision design of its own.
* **Not online (flash) softmax.** The whole `[128, S]` score row lives in SBUF (the kernel asserts a conservative
  footprint bound), so very long sequences need the online-softmax rewrite, which is an algebraic change, not a
  loop transformation. Out of scope here.
* **K and V are re-read for every query tile** (they are not kept resident), and `Kᵀ` chunks are re-transposed per
  query tile. `tests/test_attention_examples.py` pins the resulting HBM traffic so an improvement shows up as a change.
* No tail handling: `S` must be a multiple of 128 for `tiled` (`split(tail="cut")` is not implemented).
* nkibench's attention level (8) is not used: its report code crashes on attention's case format (`KeyError: 'M'`,
  after the kernel has been simulated). `scripts/kernel_check.py` checks attention instead, with the same shapes plus
  inputs scaled by 30 so a softmax that forgets the row max overflows.

## 5. What was verified, and where

| layer | single_tile | tiled |
|---|---|---|
| unit tests (`pytest`, interpreter on a tiny fake chip, oracle after **every** primitive, affine loops also run in reverse) | pass | pass |
| `nki.simulate` vs numpy, incl. ×30 scores | 128×64, 64×128, 96×32, 128×128 | 128×64, 256×64, 384×128, 512×32 (and 256×128) |
| `neuronx-cc` compile only (`kernel_check.py --mode compile`, no core needed; requires a non-empty NEFF, observed 26-32 KiB in 1.5-2.4 s) | same shapes: compiles | same shapes: compiles |
| NeuronCore (`--mode hw`) | **not run for this change**: the node's cores were held by another job | not run, same reason |

The hand-written reference kernel (`tests/test_emit.py::GOLDEN_ATTENTION_SOURCE`, the exact source emitted from a
hand-built IR) passes the simulator and compiles; the `single_tile` schedule emits the same instruction mix
(`tests/test_attention_examples.py`).

## 6. The programs

`single_tile` after scheduling (before the final copy nest becomes an instruction at emission):

```
proc nki_attention_(q: like:q[S, D], k: like:k[S, D], v: like:v[S, D], O: like:q[S, D])
  assume S <= 128; D <= 128
  scalars scale_rcp = (1.0 / (D ** 0.5))
  alloc O_sb : like:q[S, D] @ SBUF
  alloc matmul : f32[S, S] @ PSUM
  alloc q_sb : like:q[S, D] @ SBUF
  ns.sync.dma_copy(dst=q_sb[0:+S, 0:+D], src=q[0:+S, 0:+D])
  alloc k_sb : like:k[S, D] @ SBUF
  ns.sync.dma_copy(dst=k_sb[0:+S, 0:+D], src=k[0:+S, 0:+D])
  alloc qT_ps : like:q[D, S] @ PSUM
  ns.tensor.transpose(dst=qT_ps[0:+D, 0:+S], src=q_sb[0:+S, 0:+D])
  alloc qT : like:q[D, S] @ SBUF
  ns.vector.tensor_copy(dst=qT[0:+D, 0:+S], src=qT_ps[0:+D, 0:+S])
  alloc kT_ps : like:k[D, S] @ PSUM
  ns.tensor.transpose(dst=kT_ps[0:+D, 0:+S], src=k_sb[0:+S, 0:+D])
  alloc kT : like:k[D, S] @ SBUF
  ns.vector.tensor_copy(dst=kT[0:+D, 0:+S], src=kT_ps[0:+D, 0:+S])
  ns.tensor.matmul(dst=matmul[0:+S, 0:+S], stationary=qT[0:+D, 0:+S], moving=kT[0:+D, 0:+S], accumulate=None)
  alloc div : like:q[S, S] @ SBUF
  ns.vector.scale(dst=div[0:+S, 0:+S], src=matmul[0:+S, 0:+S], c=Param(name='scale_rcp'))
  alloc softmax_max : like:q[S, 1] @ SBUF
  ns.vector.reduce_max(dst=softmax_max[0:+S, 0:+1], src=div[0:+S, 0:+S])
  alloc softmax_neg : like:q[S, 1] @ SBUF
  ns.vector.negate(dst=softmax_neg[0:+S, 0:+1], src=softmax_max[0:+S, 0:+1])
  alloc softmax_exp : like:q[S, S] @ SBUF
  ns.scalar.exp_bias(dst=softmax_exp[0:+S, 0:+S], src=div[0:+S, 0:+S], bias=softmax_neg[0:+S, 0:+1])
  alloc softmax_sum : like:q[S, 1] @ SBUF
  ns.vector.reduce_sum(dst=softmax_sum[0:+S, 0:+1], src=softmax_exp[0:+S, 0:+S])
  alloc softmax : like:q[S, S] @ SBUF
  alloc softmax_sum_rcp : like:q[S, 1] @ SBUF
  ns.vector.reciprocal(dst=softmax_sum_rcp[0:+S, 0:+1], src=softmax_sum[0:+S, 0:+1])
  ns.vector.scale_rows(dst=softmax[0:+S, 0:+S], src=softmax_exp[0:+S, 0:+S], vec=softmax_sum_rcp[0:+S, 0:+1])
  alloc matmul_1 : f32[S, D] @ PSUM
  alloc v_sb : like:v[S, D] @ SBUF
  ns.sync.dma_copy(dst=v_sb[0:+S, 0:+D], src=v[0:+S, 0:+D])
  alloc pT_ps : like:q[S, S] @ PSUM
  ns.tensor.transpose(dst=pT_ps[0:+S, 0:+S], src=softmax[0:+S, 0:+S])
  alloc pT : like:q[S, S] @ SBUF
  ns.vector.tensor_copy(dst=pT[0:+S, 0:+S], src=pT_ps[0:+S, 0:+S])
  ns.tensor.matmul(dst=matmul_1[0:+S, 0:+D], stationary=pT[0:+S, 0:+S], moving=v_sb[0:+S, 0:+D], accumulate=None)
  for O.s in 0..S:
    for O.d in 0..D:
      O_sb[O.s, O.d] = matmul_1[O.s, O.d]
  ns.sync.dma_copy(dst=O[0:+S, 0:+D], src=O_sb[0:+S, 0:+D])
```
