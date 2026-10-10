# nki-sched — schedule language draft (v0)

Status: **partly implemented.** The tiled matmul (§1, §2 partly, §3 partly, §4.1, §4.2) and single-head attention (§4.3, `docs/ATTENTION.md`) run end to end; the schedules shown are the real code in `examples/`. Sections or rows marked *(planned)* are design only: symbolic block factors, K-blocking, `multibuffer`/`pipeline`/`overlap`, `set_engine`, pattern cursors (`s.find`), `fold_partition`, `transpose_input`. In v0 a handle is just a unique loop/buffer name. Names are provisional. Items marked `[verify]` depend on NKI behaviour not yet confirmed (nki 0.6.0, see PLAN §3).

## 0. Two program levels (and their names)

| level | what it is | instruction spelling | where it lives |
|---|---|---|---|
| **ns** (nki-sched IR) | the explicit loop program the schedule rewrites; engine-explicit | `ns.<engine>.<inst>(...)`, e.g. `ns.tensor.matmul`, `ns.vector.tensor_copy`, `ns.sync.dma_copy` | what `print(k.ir)` shows; what `replace`/`stage_*` create |
| **NKI source** (emitted) | Python that `@nki.jit` compiles | `nisa.<inst>(...)`, `nl.<...>` (real NKI names, e.g. `nisa.nc_matmul`, `nisa.tensor_copy(..., engine=...)`) | `k.source`; what `nkibench` checks |

The engine is part of the ns-level call, not an annotation: `ns.tensor` (PE array), `ns.vector`, `ns.scalar`, `ns.gpsimd`, `ns.sync` (DMA queues; an IR label — NKI has no such name, it maps to `dma_copy(..., engine=…)`/`dge_mode`, `[verify]`). The lowering ns → NKI source is a table lookup (PLAN §8). Why the real NKI names are kept at the lowest level: the emitted code must be exactly what `reference_level4.py` looks like. (`nl` is *not* used as the name of this level: it is already `nki.language`, and emitted code uses both `nl.*` and `nisa.*`.)

Design rule: *the syntax is Halide/TVM-flavoured names (`split`, `reorder`, `compute_at`, `stage_in`) on top of Exo-style semantics (each call is a checked rewrite of an explicit loop IR, references are cursors that forward).*

---

## 1. The reference implementation (torch)

The high-level spec is plain torch. It is both the **input to the compiler** (traced) and the **correctness oracle** (run eagerly).

```python
from nki_sched import arg, trace

def matmul(lhsT, rhs):             # lhsT: [K, M], rhs: [K, N]
    return lhsT.T @ rhs            # -> [M, N]

# Dims are symbolic in the IR; the example shapes are only used for tracing.
proc = trace(matmul, {"lhsT": arg(("K", "M"), "bf16", example=(8, 4)),
                      "rhs":  arg(("K", "N"), "bf16", example=(8, 8))}, name="matmul_kernel")
print(proc)                        # the naive IR below
```

`trace` exports the function with `torch.export` and walks the ATen graph generically (`nki_sched/frontend.py`): transposes/permutes and exact (widening) casts are views that just remap indices, `mm`/`matmul` materializes a stage (an f32 accumulator with an init nest and a reduction nest), and the output becomes a final pointwise stage. Anything else fails at trace time with the op's name. The hand-written IR for this spec lives in `tests/test_frontend.py`.

Layout: the spec takes `lhsT` (K on axis 0), same as `nkibench`/NKI tutorial kernels, so the contraction axis is already the partition axis for both operands. `A[M,K]` specs also trace (`a @ b`), but `replace(ns.tensor.matmul)` will refuse the stationary operand until a transposed copy is staged (a layout primitive is planned).

### Naive IR that `trace` produces (Halide-style: stages, breadth-first, "root" allocs)

```
  alloc matmul : f32[M, N] @ HBM
  for matmul.init.m in 0..M:
    for matmul.init.n in 0..N:
      matmul[matmul.init.m, matmul.init.n] = 0.0
  for matmul.m in 0..M:
    for matmul.n in 0..N:
      for matmul.k in 0..K:
        matmul[matmul.m, matmul.n] += lhsT[matmul.k, matmul.m] * rhs[matmul.k, matmul.n]
  for C.m in 0..M:
    for C.n in 0..N:
      C[C.m, C.n] = cast_like:lhsT(matmul[C.m, C.n])
```

Loops are auto-named `stage.var`; renamed by `split`/`fuse`. This IR is directly executable by the numpy interpreter, so every intermediate program can be diffed against the torch function.

---

## 2. References: how to point at loops, tensors, statements

Three complementary mechanisms; all produce **cursors** (opaque handles into the *current* program) and all cursors **forward** across rewrites (PLAN §6.3).

```python
@nks.schedule
def sched(s: nks.Sched):
    # (a) by auto-name — Halide/TVM style, the common case
    m = s.loop("C.m");   k = s.loop("matmul.k")
    A = s.buf("matmul");    st = s.stage("matmul")          # tensor / stage references

    # (b) by return value — primitives hand back the new cursors
    mo, mi = s.split(m, 128)                          # old handle `m` now forwards to `mo` (outer)

    # (c) by pattern — Exo style, for things with no name
    mm_nest = s.find("for ki in _: _")                # first loop with that name
    first_upd = s.find("matmul[_] += _")                 # a statement pattern
    second = s.find("for _ in _: _ #1")               # `#n` picks the (n+1)-th match
    ld = s.find_all("lhsT_sb[_] = lhsT[_]")           # many=True

    # navigation / gaps (Exo cursor API)
    s.parent(mm_nest); s.body(mo); s.next(first_upd); s.before(mo); s.after(mo)

    # inspection (never mutates)
    s.extent(mo); s.is_reduction(k); s.window(ld[0]); s.footprint(s.buf("lhsT_sb"))
```

Forwarding rules (precise semantics in PLAN §6.3): after `mo, mi = s.split(m, 128)`, the old handle `m` forwards to the outer loop `mo` (its index var is now `mo*128+mi`); using `m` where the split matters (e.g. a later `reorder`) is an error with a hint ("`m` was split; use `mo` or `mi`"), never silently ambiguous. Handles to deleted statements become `Invalid` and raise on use.

---

## 3. Primitive catalogue (draft)

Every primitive: `(preconditions checked) → rewrite → (cursor forwarding fn)`. Derived ops are library code built from these (Exo 2 philosophy), so the trusted core stays small.

### Core (trusted, each with its own legality check — PLAN §7)

| Primitive | Meaning | Legality |
|---|---|---|
| `split(loop, f, tail="cut"\|"guard"\|"perfect")` | `i → (io, ii)`, `i = io*f+ii` | `perfect`: extent ≡ 0 mod f proven; else tail handling per option |
| `reorder(l1, l2, ...)` | permute perfectly-nested loops | no dependence reversed; **reduction loops of an associative op may move freely** (Halide §3); fp-reassoc flagged |
| `fuse(l1, l2)` / `unroll(loop)` | as usual | const extent for unroll |
| `fission(loop, at=stmt)` / `fuse_stmts` | split/merge loop bodies | stmt-level commutativity |
| `compute_at(stage, at=loop)` | Halide `f.compute_at(g, y)` / TVM `s[f].compute_at(s[g], ax)`: move the producer's loop nests from the root to *inside* the consumer loop `at`; each iteration computes only the region the consumer needs there (interval analysis), and by default its storage shrinks/moves with it | producer region ⊆ computed region (Halide bounds inference) |
| `store_at(stage_or_buf, at=loop)` | allocation granularity | enclosing; live range covers uses |
| `stage_in(tensor, at=loop\|None, mem=, name=, perm=None)` | cache_read: copy window of `tensor` into new buffer in `mem` just before its first reader inside `loop` (or anywhere, for `at=None`, the kernel root); rewrite uses. `perm=(1, 0)` stages the transpose (SBUF → PSUM, `ns.tensor.transpose`). **The staging copy is created directly as the right ns instruction** from (src mem → dst mem): HBM↔SBUF `ns.sync.dma_copy`, on-chip `ns.vector.tensor_copy`. A *compute* nest that happens to be a pure copy (the PSUM→SBUF cast-out) is recognised and converted at emission (`select_copies`) | window inferred (interval analysis) or checked ⊇ accesses; capacity check |
| `stage_out(tensor, at=loop\|None, mem=, name=)` | cache_write/write-back (same implicit instruction choice) | same; `accum=True` form = zero/accumulate/add-back (Exo `stage_mem(accum=)`) |
| `set_memory(buf, HBM\|SBUF\|PSUM)` | memory space of an allocation | partition-dim rule (axis 0 ≤ pmax on SBUF/PSUM, *proved* from the program's `bound`s when symbolic), PSUM free size ≤ one bank; per instruction: matmul writes f32, a transpose keeps its input dtype, operands in the memories the instruction allows |
| `expand_dim(buf, size, idx)` / `lift_alloc` / `sink_alloc` | give a buffer a leading rotating dim; move alloc | indexing in bounds |
| `fold_init(stage)` | drop the zero-init nest; first matmul overwrites (`accumulate=None`) | init is a full-tile store of 0 immediately preceding the accumulate nest, target in PSUM |
| `replace(loop, instr)` | **tensorize**: match the nest against the instruction's semantic pattern (`instrs.INSTRS`); emit the call. A reduction instruction also consumes the init nest in front of it | unifier succeeds ⇒ equivalence by construction; the interpreter's handler is tested against the same pattern |
| `bound(size, max)` | assume `size <= max` (the kernel asserts it); lets a symbolic extent be a legal partition dim | size symbol of the program |
| `recip_divisor(stage)` | rewrite `a / b` as `a * (1/b)` (hardware multiplies by a scalar/row vector but cannot divide by one); the reciprocal is computed once | `b` is a buffer element or a kernel scalar; **not bit-exact**, recorded in `Sched.notes` |
| `mark(loop, AFFINE\|SEQUENTIAL\|STATIC)` | **optional assertion/override.** Loop kind is *inferred* from the dependence analysis at emission (→ `nl.affine_range / sequential_range / static_range`); `mark` checks an expectation (error if the analysis disagrees) or forces a more conservative kind | AFFINE: no loop-carried dep other than PSUM-accumulate reduction; STATIC ⇒ unroll |

### Derived (library, built on the core)

| Op | Built from | Intent |
|---|---|---|
| `tile(l1, l2, f1, f2)` | `split`×2 + `reorder` | |
| `hoist(buf, to=loop)` | re-`stage_in` at an outer loop with larger inferred window + delete old | the "hoist loads" optimization |
| `multibuffer(buf, depth, along=loop)` | `expand_dim` + index by `loop % depth` | rotating buffers (SBUF double-buffer, PSUM ping-pong) |
| `pipeline(loop, stages=[[...],[...]], lag=1)` | `fission` + `shift_loop` + peel prologue/epilogue + `multibuffer` | software pipelining; legality = WAR/RAW distance < depth |
| `overlap(stmtA, stmtB)` | `reorder_stmts` + engine annotation | assert independent & on different engines/queues, interleave in program order |
| `set_engine(stmt, ns.vector\|ns.scalar\|ns.gpsimd)` | move a statement whose instruction exists on several engines (e.g. `tensor_copy`) | instr must be legal on that engine; verified in 0.6.0: `tensor_copy`/`dma_copy` accept `engine=`, `activation` does not (fixed engine) |
| `stage_transposed(tensor, name, at=None)` | `stage_in(perm=(1, 0))` into PSUM, then copy to SBUF | the layout a matmul wants (`[k, m]`, contraction on the partition axis) from a `[m, k]` SBUF tile |
| `transpose_input(arg, via="dma"\|"pe")` | insert layout stage | `A[M,K] → lhsT[K,M]` (`nisa.dma_transpose` / `nisa.nc_transpose` exist in 0.6.0) |

---

## 4. Worked schedules — the matmul ladder from `nkibench.py`

Each schedule is a Python function; constants (`BM`, `depth`…) are plain parameters, so a schedule is also a **tunable template** (TVM knobs).

### 4.1 Tiled matmul (implemented: `examples/matmul_ladder.py`, `l4`)

```python
def l4(s, hw):
    mo, mi = s.split("C.m", hw.stationary_fmax, names=("mo", "mi"), perfect=True)   # 128
    no, ni = s.split("C.n", hw.moving_fmax, names=("no", "ni"), perfect=True)       # 512
    s.reorder(mo, no, mi, ni)
    s.compute_at("matmul", at=no)           # accumulator tile computed per output tile; region inferred: [128, 512]
    ko, ki = s.split("matmul.k", hw.pmax, names=("ko", "ki"), perfect=True)         # 128
    s.reorder(ko, ki, "matmul.m", "matmul.n")   # legal: + is associative (reduction rule)
    s.set_memory("matmul", PSUM)            # f32, 128 partitions x 512 = one bank
    s.stage_in("lhsT", at=ko, mem=SBUF, name="lhsT_sb")   # window inferred [ko*128:+128, mo*128:+128]
    s.stage_in("rhs",  at=ko, mem=SBUF, name="rhs_sb")    # window inferred [ko*128:+128, no*512:+512]
    s.stage_out("C",   at=no, mem=SBUF, name="C_sb")      # cast-out staging, then DMA to HBM
    s.replace(ki, "ns.tensor.matmul")       # k=partition, m=stationary free, n=moving free
    s.fold_init("matmul")                   # first matmul into the fresh PSUM tile overwrites
    # The cast-out nest C_sb[mi,ni] = matmul[mi,ni] is recognised as a pure copy at emission and
    # becomes ns.vector.tensor_copy (implicit instruction selection for copies).
```

`perfect=True` records the assumption `M % 128 == 0` (etc.); the emitted kernel asserts it. Extents stay symbolic (`M`, `N`, `K`), so one emitted file serves every shape, like `reference_level4.py`.

ns-level IR after the schedule (`print(s.show())`, before emission turns the cast-out nest into `ns.vector.tensor_copy`):

```
  for mo in 0..(M // 128):
    for no in 0..(N // 512):
      alloc C_sb : like:lhsT[128, 512] @ SBUF
      alloc matmul : f32[128, 512] @ PSUM
      for ko in 0..(K // 128):
        alloc rhs_sb : like:rhs[128, 512] @ SBUF
        ns.sync.dma_copy(dst=rhs_sb[0:+128, 0:+512], src=rhs[128*ko:+128, 512*no:+512])
        alloc lhsT_sb : like:lhsT[128, 128] @ SBUF
        ns.sync.dma_copy(dst=lhsT_sb[0:+128, 0:+128], src=lhsT[128*ko:+128, 128*mo:+128])
        ns.tensor.matmul(dst=matmul[0:+128, 0:+512], stationary=lhsT_sb[0:+128, 0:+128], moving=rhs_sb[0:+128, 0:+512], accumulate=None)
      for mi in 0..128:
        for ni in 0..512:
          C_sb[mi, ni] = matmul[mi, ni]
      ns.sync.dma_copy(dst=C[128*mo:+128, 512*no:+512], src=C_sb[0:+128, 0:+512])
```

The emitter lowers each `ns.<engine>.<inst>` to its NKI form (`ns.tensor.matmul` → `nisa.nc_matmul`, `ns.vector.tensor_copy` → `nisa.tensor_copy`, `ns.sync.dma_copy` → `nisa.dma_copy`; `engine=` is only emitted after an explicit `set_engine`). The emitted kernel passes `nkibench --level 4` in the simulator on a Trainium node (HBM traffic 1.00 / 1.56 / 1.00 / 2.00× the floor on its four shapes) and `tests/test_examples.py` pins the same ratios statically.

### 4.2 Hoisting (examples only: `l5`–`l7`)

`hoist(buf, to=loop)` moves a staged buffer's fill out of the loops between `to` and the original site. A loop that strides the source window by exactly one tile *grows* the buffer (over the partition axis the tiles fold into a new axis: `[128, K/128, 512]`, the tile index being an integer-indexed "point" axis of the matmul operand); a loop that does not index the source reuses the data.

```python
def l5(s, hw):
    l4(s, hw)
    s.reorder("no", "mo")          # independent output-tile loops: any order is legal
    s.hoist("rhs_sb", to="no")     # rhs strip loaded once per `no` instead of once per (mo, no)

def l6(s, hw):                     # l7 is the same schedule
    l5(s, hw)
    s.hoist("lhsT_sb", to=None)    # all of lhsT resident at the top of the kernel
```

```
  for no in 0..(N // 512):
    alloc rhs_sb : like:rhs[128, (K // 128), 512] @ SBUF
    for rhs_sb_ko_ld in 0..(K // 128):
      ns.sync.dma_copy(dst=rhs_sb[0:+128, rhs_sb_ko_ld, 0:+512], src=rhs[128*rhs_sb_ko_ld:+128, 512*no:+512])
    for mo in 0..(M // 128):
      alloc C_sb : like:lhsT[128, 512] @ SBUF
      alloc matmul : f32[128, 512] @ PSUM
      for ko in 0..(K // 128):
        alloc lhsT_sb : like:lhsT[128, 128] @ SBUF
        ns.sync.dma_copy(dst=lhsT_sb[0:+128, 0:+128], src=lhsT[128*ko:+128, 128*mo:+128])
        ns.tensor.matmul(dst=matmul[0:+128, 0:+512], stationary=lhsT_sb[0:+128, 0:+128], moving=rhs_sb[0:+128, ko, 0:+512], accumulate=None)
      for mi in 0..128:
        for ni in 0..512:
          C_sb[mi, ni] = matmul[mi, ni]
      ns.sync.dma_copy(dst=C[128*mo:+128, 512*no:+512], src=C_sb[0:+128, 0:+512])
```

**These are not general.** They keep whole operand strips in SBUF, so they only work while K·M and K·N fit on chip; the emitted kernel asserts a conservative footprint bound and fails loudly otherwise. They exist to exercise `hoist` and are not a blocking strategy. Real blocking needs symbolic block factors (`split` by `min(4, M//128)` so one file serves every shape) and K-blocking with several live PSUM tiles; neither is implemented.

### 4.3 Attention (implemented: `examples/attention.py`, details in `docs/ATTENTION.md`)

```python
def single_tile(s, hw):                     # S, D <= pmax: every stage is one instruction on one tile
    s.bound("S", hw.pmax); s.bound("D", hw.pmax)
    s.recip_divisor("div"); s.recip_divisor("softmax")          # hardware cannot divide by a scalar / row vector
    for buf in ("div", "softmax_max", "softmax_neg", "softmax_exp", "softmax_sum", "softmax_sum_rcp", "softmax"):
        s.set_memory(buf, SBUF)
    s.set_memory("matmul", PSUM); s.set_memory("matmul_1", PSUM)
    for t in ("q", "k", "v"): s.stage_in(t, None, SBUF, f"{t}_sb")
    s.stage_transposed("q_sb", "qT"); s.stage_transposed("k_sb", "kT"); s.stage_transposed("softmax", "pT")
    s.stage_out("O", None, SBUF, "O_sb")
    s.replace("matmul.s", "ns.tensor.matmul"); s.fold_init("matmul")
    s.replace("div.s", "ns.vector.scale")                       # PSUM evacuation + 1/sqrt(D)
    s.replace("softmax_max.s", "ns.vector.reduce_max"); s.replace("softmax_neg.s", "ns.vector.negate")
    s.replace("softmax_exp.s", "ns.scalar.exp_bias");  s.replace("softmax_sum.s", "ns.vector.reduce_sum")
    s.replace("softmax_sum_rcp.d0", "ns.vector.reciprocal"); s.replace("softmax.s", "ns.vector.scale_rows")
    s.replace("matmul_1.s", "ns.tensor.matmul"); s.fold_init("matmul_1")
```

`tiled` does the same per 128-row query tile (`compute_at` the whole chain into the tile loop, scores in key chunks of
128, the value matmul accumulating over key chunks in PSUM), for any `S % 128 == 0`. Both emit shape-generic kernels.

### 4.5 Beyond the ladder — overlap (device-measured, not in the simulator)

```python
@nks.schedule
def l8(s, depth=2):
    l7(s)                                              # shape-derived BM,BN,BK defaults
    # SBUF double-buffering of loads against the matmul that consumes the previous block
    s.multibuffer("lhsT_sb", depth, along="kb"); s.multibuffer("rhs_sb", depth, along="kb")
    s.pipeline("kb", stages=[s.find_all("ns.sync.dma_copy(lhsT_sb, _)") + s.find_all("ns.sync.dma_copy(rhs_sb, _)"),
                             s.find_all("ns.tensor.matmul(_)")], lag=1)
    # PSUM ping-pong: evacuate tile t while TensorE accumulates t+1
    s.multibuffer("matmul", 2, along="nb")
    s.pipeline("nb", stages=[[s.find("for mo in _: _")], [s.find("tensor_copy(C_sb, _)"),
                                                        s.find("dma_copy(C[_], C_sb)")]], lag=1)
    s.set_engine(s.find("ns.vector.tensor_copy(C_sb, _)"), ns.scalar)   # e.g. rebalance evacuation off the vector engine; tensor_copy accepts engine= (verified)
```

What `pipeline` *means on NKI 0.6.0*: no explicit sync API exists, so it is a **program-order + buffer-rotation transform**: the stage-0 work of iteration *i+lag* is emitted before the stage-1 work of iteration *i*, with `depth` rotating buffers so no WAR hazard exists; NKI's own dependence tracking then lets DMA/PE/Vector run concurrently. Whether we actually get overlap is an empirical question answered on device (PLAN §11, M5), not by the simulator.

---

## 5. Errors are part of the language

Lesson from this repo (`STATE.md`): the quality of the checker's message is the product. Every legality failure names the primitive, the violated rule, and the offending program locations:

```
ScheduleError in reorder(matmul.k, matmul.m) at l4.py:14
  cannot move 'matmul.k' outside 'matmul.m': 'matmul.m' contains two statements in its body
  (the reduction nest and the cast nest 'C'), so the nest is not perfect.
  hint: compute_at("matmul", at=...) first, or fission 'matmul.m' so each nest is separate.

ScheduleError in reorder(mo, ko)
  would reverse dependence: acc_t[...] += ... (update, reads PSUM written in ko) vs tensor_copy(C_sb, matmul) (reads after last ko)
  hint: the copy-out must stay outside the 'ko' loop.

ScheduleError in set_memory("matmul", PSUM)
  matmul has shape [256,512] f32: axis 0 (partition) = 256 > 128
  hint: split 'matmul.m' by 128 and compute_at the inner tile.

ScheduleError in l6.check_capacity()
  PSUM: BM*BN = 8 live fp32 tiles of [128,512] need 8 banks; HardwareConfig.psum_banks = <probed value>   [hw numbers from the M0 probe, not assumed]
```

---

## 6. Debug/inspection surface (implemented)

```python
from nki_sched import Sched, verify

check = verify.make_checker(oracle, shapes)        # numpy-interpret the program against a torch/numpy oracle ...
s = Sched(proc, hw, check=check)                   # ... after EVERY primitive (and again with affine loops reversed)
print(s.show())                                    # the current ns-level IR
print(s.source())                                  # the emitted NKI kernel (every loop nest must be an instruction)
s.loops("matmul")                                  # loop names of a stage's update nest, outermost first
s.history                                          # [(label, proc)]: the program after each primitive
s.notes                                            # caveats the schedule introduced: fp reassociation, recip_divisor
```

`dma_bytes(proc, sizes)` (in `nki_sched.analysis`) gives the static HBM traffic of a scheduled program. The node-side
checks (`scripts/kernel_check.py`, `scripts/device_check.sh`) are described in the README.
