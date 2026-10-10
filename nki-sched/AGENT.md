# nki-sched: write a schedule, not a kernel

You do not write NKI. You write a **schedule**: a short Python function that rewrites a naive loop
program step by step. Each call is checked; the final program is emitted as a verified NKI kernel.
A failed call raises an error that says what rule you broke. Fix that call and nothing else.

```python
def schedule(s, hw):
    ...   # calls on s; use hw.pmax / hw.stationary_fmax / hw.moving_fmax, never the literals 128 / 512
```

`PSUM`, `SBUF` and `hw` are already in scope. No imports. Reply with one python block holding `schedule`.
Always take tile sizes from `hw`: the schedule is first replayed on a tiny fake chip (pmax=4,
fmax=4/8) and checked numerically after every call; a hardcoded 128 fails there.

## The program you start from (matmul `C[m,n] = cast(sum_k lhsT[k,m] * rhs[k,n])`)

Sizes K, M, N are symbolic. `lhsT` is `[K,M]` (contraction on axis 0), `rhs` is `[K,N]`, `C` is `[M,N]`.
Loop names are `stage.var`, and every handle is just such a name (a string):

```
alloc matmul : f32[M,N] @ HBM                    # accumulator stage
for matmul.init.m, matmul.init.n: matmul[m,n] = 0      # init nest
for matmul.m: for matmul.n: for matmul.k: matmul[m,n] += lhsT[k,m] * rhs[k,n]    # update nest
for C.m: for C.n: C[m,n] = cast(matmul[m,n])     # output nest
```

## Primitives (all take loop/buffer names as strings)

| call | effect |
|---|---|
| `o, i = s.split(loop, factor, names=("o","i"), perfect=True)` | `loop -> o*factor + i`; `perfect=True` asserts extent % factor == 0 (needed for symbolic M,N,K). Returns the new names; the old name stops existing |
| `s.reorder(a, b, c, ...)` | permute a perfectly nested chain of loops, outermost first. Legal for reductions (`matmul.k` may move freely); refused if it reverses a dependence |
| `s.compute_at(stage, at=loop)` | move stage `matmul` (alloc + init + update nests) inside loop `at`, shrunk to the region consumed there |
| `s.set_memory(buf, PSUM or SBUF)` | PSUM: f32, axis 0 <= `hw.pmax`, free size <= one bank (`hw.moving_fmax` f32). SBUF: axis 0 <= `hw.pmax` |
| `s.stage_in(tensor, at=loop, mem=SBUF, name="x_sb")` | copy the window of `tensor` read inside `at` into a new buffer at the top of `at` (HBM->SBUF DMA); reads are redirected |
| `s.stage_out(tensor, at=loop, mem=SBUF, name="c_sb")` | writes to `tensor` inside `at` go to a new buffer, DMA'd out after `at` |
| `s.replace(loop, "ns.tensor.matmul")` | tensorize: the perfect 3-loop nest rooted at `loop` (k, m, n in any order) around one `+=` becomes one hardware matmul. Needs: dst in PSUM, both operands in SBUF laid out `[k,m]` and `[k,n]`, k <= `hw.pmax`, m <= `hw.stationary_fmax`, n <= `hw.moving_fmax` |
| `s.fold_init(buf)` | drop the zero-init nest of a PSUM accumulator; the first matmul overwrites |
| `s.hoist(buf, to=loop_or_None)` | move the fill of a `stage_in` buffer out to loop `to` (`None` = top of kernel). If loops between grow the window the buffer grows to hold it all (K tiles fold into axis 1); loops that do not index the source just reuse the data. Fewer HBM reloads |
| `s.mark(loop, "affine"/"sequential"/"static")` | optional assertion about loop kind |
| `s.loops(stage)` -> names, `s.show()` -> current program text | inspection only |

The cast-out nest `C_sb[m,n] = cast(matmul[m,n])` between PSUM and SBUF is turned into a vector copy
automatically; you do not tensorize it.

## Worked example: tiled matmul (nkibench level 4)

```python
def schedule(s, hw):
    mo, mi = s.split("C.m", hw.stationary_fmax, names=("mo", "mi"), perfect=True)
    no, ni = s.split("C.n", hw.moving_fmax, names=("no", "ni"), perfect=True)
    s.reorder(mo, no, mi, ni)
    s.compute_at("matmul", at=no)                       # one output tile at a time
    ko, ki = s.split("matmul.k", hw.pmax, names=("ko", "ki"), perfect=True)
    s.reorder(ko, ki, "matmul.m", "matmul.n")              # k outermost: accumulate across ko
    s.set_memory("matmul", PSUM)
    s.stage_in("lhsT", at=ko, mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at=ko, mem=SBUF, name="rhs_sb")
    s.stage_out("C", at=no, mem=SBUF, name="C_sb")
    s.replace(ki, "ns.tensor.matmul")
    s.fold_init("matmul")
```

Order matters: `compute_at` before splitting `matmul.k`; `set_memory` before `replace`; `stage_in` after
`compute_at` (so the window is a tile); `replace` before `fold_init`.

## Optimising

The checker also counts HBM bytes against the floor (each input and the output moved once).
A load inside a loop that does not index it is re-read every iteration. To cut traffic: reorder the
output-tile loops so the loop that does *not* index an operand is innermost, then `hoist` that
operand's buffer to the loop outside it. Hoisting `to=None` keeps an operand fully resident in SBUF
(capacity-limited). After a failure, the `s.show()` output in the report is the program as of the last
successful call: read the loop and buffer names from it.

## Errors

A `ScheduleError` names the primitive and the violated rule; the report also gives the program state
before the failing call and the calls that succeeded. Names that were split no longer exist: use
the returned `o, i` names. Change only the failing call (and what depends on it).

# Attention (`softmax(q k^T / sqrt(D)) v`, one head, float32)

Same language, different starting program. `q`, `k`, `v` are `[S, D]`, output `O`; sizes `S`, `D` are symbolic.
`hw` as before; never write 128/512 literally.

```
matmul        S = q k^T                f32 accumulator (init nest + update nest), loops matmul.s, matmul.s1, matmul.d
div           S / scale                scale = sqrt(D) is a kernel scalar
softmax_max   row max     [S,1]        init nest softmax_max.init.*, update `max=`
softmax_neg   -max        [S,1]
softmax_exp   exp(div + (-max))
softmax_sum   row sum     [S,1]
softmax       softmax_exp / softmax_sum
matmul_1      softmax @ v              loops matmul_1.s, matmul_1.d, matmul_1.s1
O             O[s,d] = matmul_1[s,d]
```

Extra primitives:

| call | effect |
|---|---|
| `s.bound("S", hw.pmax)` | assume `S <= pmax` (the kernel asserts it). A symbolic extent can only be a partition dim if bounded |
| `s.recip_divisor(stage)` | `a / b -> a * (1/b)`; needed for `div` (scalar) and `softmax` (row sum): the hardware cannot divide. Creates stage `softmax_sum_rcp` |
| `s.stage_in(t, None, SBUF, name)` | `at=None` stages at the kernel root (just before the first reader); also works for intermediates already in SBUF |
| `s.stage_transposed(t, name, at=None)` | `t` (in SBUF) -> transposed tile `name` in SBUF via PSUM; what a matmul needs when the contraction axis is the free axis (use for `q_sb`, `k_sb`, `softmax`) |
| `s.stage_out(t, None, SBUF, name)` | `at=None` allowed |
| `s.replace(loop, instr)` | instrs: `ns.tensor.matmul` (k,m,n loops), `ns.vector.scale` (`x*c`), `ns.vector.reduce_max` / `reduce_sum` (row reduction; also consumes the init nest), `ns.vector.negate`, `ns.vector.reciprocal`, `ns.vector.scale_rows` (`x * vec[i,0]`), `ns.scalar.exp_bias` (`exp(x + bias[i,0])`). Operands must be in SBUF/PSUM; `matmul` stationary/moving in SBUF, dst in PSUM |

Worked example, `S, D <= pmax` (one tile per stage). For bigger `S` (multiple of pmax) split `O.s` by `hw.pmax`,
`compute_at` the stages into that loop last-stage-first, chunk the key axis by `hw.pmax` (`div.s1`, `matmul_1.s1`):
see `examples/attention.py` `tiled`.

```python
def schedule(s, hw):
    s.bound("S", hw.pmax); s.bound("D", hw.pmax)
    s.recip_divisor("div"); s.recip_divisor("softmax")
    for buf in ("div", "softmax_max", "softmax_neg", "softmax_exp", "softmax_sum", "softmax_sum_rcp", "softmax"):
        s.set_memory(buf, SBUF)
    s.set_memory("matmul", PSUM); s.set_memory("matmul_1", PSUM)
    for t in ("q", "k", "v"):
        s.stage_in(t, None, SBUF, f"{t}_sb")
    s.stage_transposed("q_sb", "qT"); s.stage_transposed("k_sb", "kT"); s.stage_transposed("softmax", "pT")
    s.stage_out("O", None, SBUF, "O_sb")
    s.replace("matmul.s", "ns.tensor.matmul"); s.fold_init("matmul")
    s.replace("div.s", "ns.vector.scale")
    s.replace("softmax_max.s", "ns.vector.reduce_max"); s.replace("softmax_neg.s", "ns.vector.negate")
    s.replace("softmax_exp.s", "ns.scalar.exp_bias"); s.replace("softmax_sum.s", "ns.vector.reduce_sum")
    s.replace("softmax_sum_rcp.d0", "ns.vector.reciprocal"); s.replace("softmax.s", "ns.vector.scale_rows")
    s.replace("matmul_1.s", "ns.tensor.matmul"); s.fold_init("matmul_1")
```

Order matters: `set_memory` before `stage_transposed("softmax", ...)` (the source must be in SBUF) and before `replace`;
`stage_in`/`stage_transposed` before the `replace` of the nest that reads the tensor. A softmax that forgets to
subtract the row max overflows on large scores: the checker uses such inputs, so do not drop `softmax_neg`/`exp_bias`.
