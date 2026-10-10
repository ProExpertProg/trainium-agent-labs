"""Schedule primitives. Each primitive is a checked rewrite of the ns-level loop IR (Exo style):
it validates its own precondition, rebuilds the affected part of the immutable program, and
records a history entry. v0 handles are unique loop / buffer names resolved against the current
program on every call (a renamed or consumed name raises an error that says what replaced it).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

from . import ir
from .analysis import AnalysisError, injective, loop_ranges, region
from .emit import emit_nki
from .expr import Aff, Var, var
from .hw import NC_DEFAULT, HardwareConfig, HardwareError
from .instrs import INSTRS
from .lower import check_hw, copy_nest_to_call, infer_kinds, nest_window
from . import transforms
from .match import MatchError, match_instr


class ScheduleError(Exception):
    pass


def _chain(loop: ir.For):
    """Maximal perfect loop chain starting at `loop` (each body exactly one For)."""
    out = [loop]
    while len(out[-1].body) == 1 and isinstance(out[-1].body[0], ir.For):
        out.append(out[-1].body[0])
    return out


def _rebuild_chain(loops, inner_body, order):
    body = inner_body
    for l in reversed(order):
        body = (replace(l, body=tuple(body)),)
    return body[0]


class Sched:
    def __init__(self, proc: ir.Proc, hw: HardwareConfig = NC_DEFAULT, check: Optional[Callable] = None):
        self.proc = proc
        self.hw = hw
        self.history = [("init", proc)]
        self._renamed = {}
        self._check = check
        self.notes = []  # e.g. reassociation notices

    # ------------------------------------------------------------------ plumbing
    def _commit(self, label: str, proc: ir.Proc):
        try:
            check_hw(proc, self.hw)
        except HardwareError as e:
            raise ScheduleError(f"{label}: {e}") from None
        if self._check is not None:
            try:
                self._check(proc)
            except Exception as e:  # semantic regression detected by differential testing
                raise ScheduleError(f"{label}: result no longer matches the oracle: {e}") from None
        self.proc = proc
        self.history.append((label, proc))

    def _find(self, var_name: str) -> ir.For:
        l = ir.find_loop(self.proc.body, var_name)
        if l is None:
            if var_name in self._renamed:
                raise ScheduleError(f"loop '{var_name}' no longer exists: it was split into {self._renamed[var_name]}; use one of those")
            names = ir.loop_vars(self.proc.body)
            raise ScheduleError(f"no loop named '{var_name}'. Loops: {names}")
        return l

    def _buf(self, name: str) -> ir.Buffer:
        b = ir.buffers_of(self.proc).get(name)
        if b is None:
            raise ScheduleError(f"no buffer named '{name}'. Buffers: {list(ir.buffers_of(self.proc))}")
        return b

    def _fresh(self, name: str):
        if name in ir.buffers_of(self.proc) or name in ir.loop_vars(self.proc.body):
            raise ScheduleError(f"name '{name}' is already in use")

    def _rewrite_loop(self, var_name, fn):
        """Replace the For named var_name by fn(For) -> tuple of statements."""
        done = []

        def rec(stmts):
            out = []
            for s in stmts:
                if isinstance(s, ir.For):
                    if s.var == var_name:
                        done.append(1)
                        out.extend(fn(s))
                        continue
                    s = replace(s, body=tuple(rec(s.body)))
                out.append(s)
            return out

        body = tuple(rec(self.proc.body))
        if not done:
            raise ScheduleError(f"no loop named '{var_name}'")
        return self.proc.with_(body=body)

    # ------------------------------------------------------------------ inspection
    def loops(self, stage: str):
        """Loop variable names of a stage's update nest, outermost first."""
        for s in ir.walk(self.proc.body):
            if isinstance(s, ir.For) and s.stage == stage:
                return [l.var for l in _chain(s)]
        raise ScheduleError(f"no stage '{stage}'")

    def show(self):
        return str(self.proc)

    def source(self) -> str:
        return emit_nki(self.proc, self.hw)

    # ------------------------------------------------------------------ split
    def split(self, loop: str, factor: int, names=None, perfect: bool = False):
        """i -> (io, ii) with i = io*factor + ii. If the extent is not provably divisible by
        `factor`, perfect=True records the assumption `extent % factor == 0` (emitted as an assert)."""
        if not isinstance(factor, int) or factor < 1:
            raise ScheduleError("split factor must be a positive int")
        l = self._find(loop)
        o, i = names or (f"{loop}.o", f"{loop}.i")
        self._fresh(o)
        self._fresh(i)
        ext = l.extent
        provable = (ext.is_const and ext.const % factor == 0) or (
            ext.terms and ext.const % factor == 0 and all(c % factor == 0 for _, c in ext.terms))
        assumptions = self.proc.assumptions
        if ext.is_const and ext.const % factor:
            raise ScheduleError(f"split({loop}, {factor}): the constant extent {ext.const} is not a multiple of {factor} (tail='cut' not implemented yet)")
        if not provable:
            if not perfect:
                raise ScheduleError(
                    f"split({loop}, {factor}): cannot prove {ext} is a multiple of {factor}. "
                    f"Pass perfect=True to assume it (the kernel will assert it), or use tail='cut' (not implemented yet)."
                )
            a = ir.Assumption(ext, factor)
            if a not in assumptions:
                assumptions = assumptions + (a,)

        def fn(f: ir.For):
            body = tuple(ir.subst_vars(b, {loop: var(o) * factor + var(i)}) for b in f.body)
            inner = ir.For(i, Aff(factor), body, f.stage, f.kind)
            return (ir.For(o, ext // factor, (inner,), f.stage, f.kind),)

        p = self._rewrite_loop(loop, fn).with_(assumptions=assumptions)
        self._renamed[loop] = (o, i)
        self._commit(f"split({loop}, {factor})", p)
        return o, i

    # ------------------------------------------------------------------ reorder
    def reorder(self, *loops: str):
        """Permute a perfectly nested chain of loops into the given order (outermost first)."""
        if len(set(loops)) != len(loops) or len(loops) < 2:
            raise ScheduleError("reorder needs two or more distinct loops")
        ls = [self._find(n) for n in loops]
        # outermost = the one that contains all the others
        top = None
        for cand in ls:
            if {l.var for l in _chain(cand)} >= set(loops):
                top = cand
                break
        if top is None:
            raise ScheduleError(f"reorder{loops}: loops are not a perfectly nested contiguous chain "
                                f"(each must be the only statement of the one above)")
        chain = _chain(top)
        names = [l.var for l in chain]
        n = len(loops)
        if set(names[:n]) != set(loops):
            raise ScheduleError(f"reorder{loops}: loops must be the outermost {n} loops of the chain {names}")
        head = chain[:n]
        inner_body = chain[n - 1].body
        by_name = {l.var: l for l in head}
        # rectangular: no loop extent depends on another loop var of the chain
        for l in head:
            if l.extent.free_vars() & set(loops):
                raise ScheduleError(f"reorder: extent of {l.var} ({l.extent}) depends on another loop in the nest")
        self._check_reorder_legal(head, inner_body, loops)
        new = _rebuild_chain(head, inner_body, [by_name[x] for x in loops])
        p = self._rewrite_loop(top.var, lambda f: (new,))
        self._commit(f"reorder{tuple(loops)}", p)

    def _check_reorder_legal(self, head, inner_body, new_order):
        P = {l.var: l.extent for l in head}
        if [l.var for l in head] == list(new_order):
            return
        bufs = ir.buffers_of(self.proc)
        inner_allocs = {s.buf.name for s in ir.walk(inner_body) if isinstance(s, ir.Alloc)}
        acc = {}
        for s in ir.walk(inner_body):
            for mode, buf, loc in ir.accesses(s):
                acc.setdefault(buf, []).append((mode, loc))
        for buf, lst in acc.items():
            if buf in inner_allocs or not any("w" in m for m, _ in lst):
                continue
            keys = {(loc.lo, loc.size) if isinstance(loc, ir.Window) else tuple(loc) for _, loc in lst}
            if len(keys) != 1:
                raise ScheduleError(f"reorder{tuple(new_order)}: cannot prove independence — '{buf}' is accessed at different indices "
                                    f"inside the nest: {sorted(str(k) for k in keys)}")
            key = next(iter(keys))
            idx = key[0] if isinstance(lst[0][1], ir.Window) else key
            appears = {v: P[v] for v in P if any(v in e.free_vars() for e in idx)}
            missing = [v for v in P if v not in appears]
            if appears and not injective(tuple(idx), appears):
                raise ScheduleError(f"reorder{tuple(new_order)}: cannot prove distinct iterations write distinct elements of '{buf}' (index {[str(e) for e in idx]})")
            if isinstance(lst[0][1], ir.Window):
                w = lst[0][1]
                for v in appears:
                    if not any(w.lo[d].coef(v) and w.size[d].is_const and abs(w.lo[d].coef(v)) >= w.size[d].const for d in range(len(w.lo))):
                        raise ScheduleError(f"reorder{tuple(new_order)}: windows written to '{buf}' by different iterations of '{v}' may overlap")
            if missing:
                bad = [m for m, _ in lst if m not in ("rw",)]
                if bad:
                    raise ScheduleError(
                        f"reorder{tuple(new_order)}: loop(s) {missing} do not index '{buf}', so they carry a dependence through it, "
                        f"and '{buf}' is not purely accumulated (+=) inside the nest. Reordering is only legal for associative reductions; "
                        f"separate the initialisation/readers of '{buf}' from this nest first (compute_at / fission)."
                    )
                self.notes.append(f"reorder{tuple(new_order)} reassociates the floating-point reduction into '{buf}'")

    # ------------------------------------------------------------------ compute_at
    def compute_at(self, stage: str, at: str):
        """Move the producer stage's loop nests (and its allocation) inside loop `at`, restricted to
        the region the consumers inside `at` actually read (interval analysis)."""
        at_loop = self._find(at)
        root = list(self.proc.body)
        prod_idx = [i for i, s in enumerate(root) if isinstance(s, ir.For) and s.stage in (stage, stage + ".init")]
        if not prod_idx:
            raise ScheduleError(f"stage '{stage}' has no loop nests at the top level (already computed_at somewhere?)")
        prods = [root[i] for i in prod_idx]
        buf = self._buf(stage)
        # consumer accesses
        reads = []
        for s in ir.walk(at_loop.body):
            if isinstance(s, (ir.Assign, ir.Reduce)):
                reads += [r.idx for r in ir.expr_reads(s.rhs) if r.buf == stage]
        if not reads:
            raise ScheduleError(f"compute_at({stage}, at={at}): no reads of '{stage}' inside loop '{at}'")
        try:
            lo, size = region(reads, loop_ranges(at_loop.body))
        except AnalysisError as e:
            raise ScheduleError(f"compute_at({stage}, at={at}): {e}") from None

        new_prods = []
        for nest in prods:
            writes = [loc for s in ir.walk((nest,)) for m, b, loc in ir.accesses(s) if b == stage and "w" in m]
            widx = {tuple(w) for w in writes}
            if len(widx) != 1:
                raise ScheduleError(f"compute_at: producer nest writes '{stage}' at several index tuples")
            widx = next(iter(widx))
            mapping, ext_of = {}, {}
            for d, e in enumerate(widx):
                if e.is_const:
                    # a constant dim, e.g. the reduced dim of a [S, 1] row statistic: the consumers must read that element
                    if lo[d] != e or size[d] != Aff(1):
                        raise ScheduleError(f"compute_at: '{stage}' is written at constant index {e} in dim {d}, but the consumers inside "
                                            f"'{at}' read the window [{lo[d]}, +{size[d]}) there")
                    continue
                if len(e.terms) != 1 or e.const != 0 or e.terms[0][1] != 1 or not isinstance(e.terms[0][0], Var):
                    raise ScheduleError(f"compute_at: producer write index {[str(x) for x in widx]} must be a plain loop variable "
                                        f"(or a constant) per dim")
                v = e.terms[0][0].name
                mapping[v] = lo[d] + var(v)
                ext_of[v] = size[d]
            nest2 = ir.subst_vars(nest, mapping)

            def setext(s):
                if isinstance(s, ir.For):
                    return replace(s, extent=ext_of.get(s.var, s.extent), body=tuple(setext(b) for b in s.body))
                return s

            nest2 = setext(nest2)
            new_prods.append(nest2)
        new_prods = [ir.remap_buf(n, stage, stage, lambda idx: tuple(i - l for i, l in zip(idx, lo))) for n in new_prods]
        new_body_stmts = tuple(ir.remap_buf(b, stage, stage, lambda idx: tuple(i - l for i, l in zip(idx, lo))) for b in at_loop.body)
        newbuf = buf.with_(shape=tuple(size))
        new_at = replace(at_loop, body=(ir.Alloc(newbuf),) + tuple(new_prods) + new_body_stmts)

        def rec(stmts):
            out = []
            for s in stmts:
                if isinstance(s, ir.For):
                    s = new_at if s.var == at else replace(s, body=tuple(rec(s.body)))
                out.append(s)
            return out

        rest = [s for i, s in enumerate(root) if i not in prod_idx and not (isinstance(s, ir.Alloc) and s.buf.name == stage)]
        p = self.proc.with_(body=tuple(rec(rest)))
        self._commit(f"compute_at({stage}, at={at})", p)

    # ------------------------------------------------------------------ memory
    def set_memory(self, buf: str, mem: str):
        if mem not in ir.MEMS:
            raise ScheduleError(f"unknown memory {mem}")
        b = self._buf(buf)
        if b.role != "temp":
            raise ScheduleError(f"set_memory: '{buf}' is a kernel argument/output and always lives in HBM")

        def f(s):
            if isinstance(s, ir.Alloc) and s.buf.name == buf:
                return ir.Alloc(s.buf.with_(mem=mem))
            return None

        p = self.proc.with_(body=tuple(ir.map_stmts(self.proc.body, f)))
        self._commit(f"set_memory({buf}, {mem})", p)

    def _container(self, at):
        """(statements, rebuild) for the scope a staging buffer lives in: loop `at`'s body, or the whole
        kernel when at is None. rebuild(new statements) -> the new Proc."""
        if at is None:
            return self.proc.body, lambda new: self.proc.with_(body=tuple(new))
        loop = self._find(at)
        return loop.body, lambda new: self._rewrite_loop(at, lambda f: (replace(f, body=tuple(new)),))

    def _staging(self, tensor, at, mem, name, direction, perm=None):
        stmts, rebuild = self._container(at)
        where = f"at={at}" if at is not None else "at the kernel root"
        self._fresh(name)
        tb = self._buf(tensor)
        reads = lambda st: any(r.buf == tensor for x in ir.walk((st,)) if isinstance(x, (ir.Assign, ir.Reduce)) for r in ir.expr_reads(x.rhs))  # noqa: E731
        writes = lambda st: any(b == tensor and "w" in m for x in ir.walk((st,)) if not isinstance(x, ir.For) for m, b, _ in ir.accesses(x))  # noqa: E731
        first_read = next((i for i, st in enumerate(stmts) if reads(st)), None)
        if direction == "in":
            idxs = [r.idx for st in ir.walk(stmts) if isinstance(st, (ir.Assign, ir.Reduce)) for r in ir.expr_reads(st.rhs) if r.buf == tensor]
            late = [i for i, st in enumerate(stmts) if writes(st) and first_read is not None and i >= first_read]
            if late:
                raise ScheduleError(f"stage_in({tensor}): '{tensor}' is written inside the scope ({where}) at or after its first read; "
                                    f"stage it where it is only read, or use stage_out")
        else:
            acc = [(m, loc) for st in ir.walk(stmts) for m, b, loc in ir.accesses(st) if b == tensor]
            if any(m != "w" for m, _ in acc):
                raise ScheduleError(f"stage_out({tensor}): '{tensor}' is also read inside '{at}' (needs a load as well; not supported yet)")
            idxs = [tuple(loc) for _, loc in acc]
        if not idxs:
            raise ScheduleError(f"stage_{direction}({tensor}, {where}): no accesses to '{tensor}' inside the scope")
        try:
            lo, size = region(idxs, loop_ranges(stmts))
        except AnalysisError as e:
            raise ScheduleError(f"stage_{direction}({tensor}): {e}") from None
        perm = tuple(perm) if perm is not None else tuple(range(len(size)))
        if sorted(perm) != list(range(len(size))):
            raise ScheduleError(f"perm {perm} is not a permutation of the {len(size)} dims of '{tensor}'")
        shape = tuple(size[i] for i in perm)
        nb = ir.Buffer(name, shape, tb.dtype, mem, "temp")
        full = ir.Window(name, tuple(Aff(0) for _ in shape), shape)
        win = ir.Window(tensor, tuple(lo), tuple(size))
        rebase = lambda idx: tuple(idx[i] - lo[i] for i in perm)  # noqa: E731
        # reads are redirected from the first reader on; whatever produced the tensor earlier is left alone
        start = first_read if direction == "in" else 0
        body = list(stmts[:start]) + [ir.remap_buf(st, tensor, name, rebase) for st in stmts[start:]]
        if direction == "in" and perm != tuple(range(len(perm))) and any(
                isinstance(x, ir.Call) and any(w.buf == tensor for _, w in x.args) for st in stmts[start:] for x in ir.walk((st,))):
            raise ScheduleError(f"stage_in({tensor}, perm={perm}): an instruction after the first reader already uses '{tensor}' directly; "
                                f"transposed staging can only redirect scalar loop nests")
        if direction == "in":
            cp = self._copy_call(full, win, tb.mem, mem, perm)
            body[first_read:first_read] = [ir.Alloc(nb), cp]
        else:
            cp = self._copy_call(win, full, mem, tb.mem, perm)
            body = [ir.Alloc(nb)] + body + [cp]
        self._commit(f"stage_{direction}({tensor}, {where}, {mem}, {name})", rebuild(body))

    @staticmethod
    def _copy_call(dst: ir.Window, src: ir.Window, src_mem: str, dst_mem: str, perm: tuple = ()):
        if perm and tuple(perm) != tuple(range(len(perm))):
            if tuple(perm) != (1, 0) or (src_mem, dst_mem) != (ir.SBUF, ir.PSUM):
                raise ScheduleError(
                    f"a transposed copy (perm={tuple(perm)}) is only available as a tensor-engine transpose of a 2-D tile from "
                    f"SBUF into PSUM; this staging goes {src_mem} -> {dst_mem}. Stage the operand into SBUF first "
                    f"(float32 cannot use the DMA transpose), then transpose it into PSUM -- or use stage_transposed")
            return ir.Call("ns.tensor.transpose", (("dst", dst), ("src", src)))
        if ir.HBM in (src_mem, dst_mem):
            if src_mem == dst_mem:
                raise ScheduleError("HBM -> HBM staging copies are not supported")
            return ir.Call("ns.sync.dma_copy", (("dst", dst), ("src", src)))
        return ir.Call("ns.vector.tensor_copy", (("dst", dst), ("src", src)))

    def stage_in(self, tensor: str, at: Optional[str], mem: str, name: str, perm: Optional[tuple] = None):
        """cache_read: copy the window of `tensor` read inside loop `at` (or anywhere, for at=None) into a
        new buffer in `mem`, placed just before the first reader; reads are redirected to it. The copy is
        created as the right ns instruction from the memory spaces (HBM->SBUF: ns.sync.dma_copy). With
        perm=(1, 0) the buffer is the transpose of the window (SBUF -> PSUM, ns.tensor.transpose)."""
        self._staging(tensor, at, mem, name, "in", perm)
        return name

    def stage_transposed(self, tensor: str, name: str, at: Optional[str] = None) -> str:
        """The operand layout a matmul needs ([k, m] with the contraction on the partition axis) from a
        [m, k] tile: transpose it on the tensor engine into PSUM (`<name>_ps`), then copy to SBUF `name`.
        `tensor` must already be in SBUF."""
        ps = self.stage_in(tensor, at=at, mem=ir.PSUM, name=f"{name}_ps", perm=(1, 0))
        return self.stage_in(ps, at=at, mem=ir.SBUF, name=name)

    def stage_out(self, tensor: str, at: Optional[str], mem: str, name: str):
        """cache_write: writes to `tensor` inside `at` go to a new buffer in `mem`; the window is
        copied out after `at`'s body."""
        self._staging(tensor, at, mem, name, "out")
        return name

    # ------------------------------------------------------------------ instruction selection
    def replace(self, loop: str, instr: str):
        """Tensorize: replace the loop nest rooted at `loop` by an ns instruction when the nest matches
        the instruction's semantics (see `instrs.INSTRS`). An instruction that overwrites its output
        (the reductions) also consumes the initialisation nest directly in front of the loop."""
        ins = INSTRS.get(instr)
        if ins is None:
            raise ScheduleError(f"unknown instruction {instr}; known: {list(INSTRS)}")
        l = self._find(loop)
        chain = _chain(l)
        init_value, out_window = None, None
        if ins.pattern is None:
            call = copy_nest_to_call(l, ir.buffers_of(self.proc))
            if call is None or call.instr != instr:
                raise ScheduleError(f"replace({loop}, {instr}): the nest is not a pure elementwise copy between compatible memories")
        else:
            body = chain[-1].body
            if len(body) != 1 or not isinstance(body[0], (ir.Assign, ir.Reduce)):
                raise ScheduleError(f"replace({loop}, {instr}): the innermost loop '{chain[-1].var}' must contain exactly one statement")
            try:
                m = match_instr(ins, chain, body[0])
            except MatchError as e:
                raise ScheduleError(f"replace({loop}, {instr}): {e}") from None
            call, init_value, out_window = m.call, ins.pattern.init, m.out_window

        def rec(stmts):
            out = []
            for st in stmts:
                if isinstance(st, ir.For):
                    if st.var == loop:
                        if init_value is not None:
                            if not (out and self._is_init_nest(out[-1], out_window, init_value)):
                                raise ScheduleError(
                                    f"replace({loop}, {instr}): the statement in front of the nest must initialise "
                                    f"{out_window.buf} to {init_value} over the whole output window (the instruction overwrites it)")
                            out.pop()
                        out.append(call)
                        continue
                    st = replace(st, body=tuple(rec(st.body)))
                out.append(st)
            return out

        self._commit(f"replace({loop}, {instr})", self.proc.with_(body=tuple(rec(self.proc.body))))

    @staticmethod
    def _is_init_nest(stmt, window: ir.Window, value: float) -> bool:
        """is `stmt` a perfect nest that stores `value` over exactly `window`?"""
        if not isinstance(stmt, ir.For):
            return False
        ch = _chain(stmt)
        inner = ch[-1].body
        if len(inner) != 1 or not isinstance(inner[0], ir.Assign) or inner[0].rhs != ir.Lit(value) or inner[0].buf != window.buf:
            return False
        w, _ = nest_window(window.buf, inner[0].idx, {c.var: c.extent for c in ch})
        return w is not None and w.lo == window.lo and w.size == window.size

    def recip_divisor(self, stage: str):
        """Rewrite `a / b` in `stage` as `a * (1/b)` with the reciprocal computed once (the hardware
        multiplies by a per-row vector or scalar but cannot divide by one). Not bit-exact; recorded in
        `notes`."""
        try:
            p = transforms.recip_divisor(self.proc, stage)
        except transforms.TransformError as e:
            raise ScheduleError(f"recip_divisor({stage}): {e}") from None
        self.notes.append(f"recip_divisor({stage}) rewrites a / b as a * (1 / b): equal up to rounding, not bit-exact")
        self._commit(f"recip_divisor({stage})", p)

    def bound(self, size: str, max: int):
        """Assume the size symbol `size` is at most `max`; the emitted kernel asserts it. This is what
        lets a symbolic extent (e.g. the sequence length S) be proven to fit a 128-partition tile."""
        if size not in {n for n, _, _ in self.proc.sizes}:
            raise ScheduleError(f"bound: unknown size '{size}'. Sizes: {[n for n, _, _ in self.proc.sizes]}")
        new = ir.Bound(Aff.of(size), int(max))
        if new in self.proc.assumptions:
            return
        self._commit(f"bound({size} <= {max})", self.proc.with_(assumptions=self.proc.assumptions + (new,)))

    def fold_init(self, buf: str):
        """Drop the zero-initialisation nest of an accumulator: the first matmul into a freshly
        allocated PSUM tile overwrites (accumulate=None)."""
        b = self._buf(buf)
        if b.mem != ir.PSUM:
            raise ScheduleError(f"fold_init({buf}): only valid for PSUM accumulators (it is in {b.mem})")
        bufs = ir.buffers_of(self.proc)
        found = {}

        def is_init(s):
            if not (isinstance(s, ir.For) and s.stage == buf + ".init"):
                return False
            ch = _chain(s)
            inner = ch[-1].body
            if len(inner) != 1 or not isinstance(inner[0], ir.Assign) or inner[0].buf != buf or inner[0].rhs != ir.Lit(0.0):
                return False
            w, _ = nest_window(buf, inner[0].idx, {c.var: c.extent for c in ch})
            return w is not None and w.is_full(bufs)

        def rec(stmts):
            out = []
            for s in stmts:
                if is_init(s):
                    found["init"] = found.get("init", 0) + 1
                    continue
                if isinstance(s, ir.For):
                    s = replace(s, body=tuple(rec(s.body)))
                out.append(s)
            return out

        body = tuple(rec(self.proc.body))
        if found.get("init") != 1:
            raise ScheduleError(f"fold_init({buf}): expected exactly one full zero-initialisation nest, found {found.get('init', 0)}")
        n_mm = []

        def f(s):
            if isinstance(s, ir.Call) and s.instr == "ns.tensor.matmul" and s.arg("dst").buf == buf:
                if not s.arg("dst").is_full(bufs):
                    raise ScheduleError(f"fold_init({buf}): matmul writes only part of '{buf}'; first-write-overwrite would be unsound")
                n_mm.append(1)
                return replace(s, attrs=tuple((k, v) for k, v in s.attrs if k != "accumulate") + (("accumulate", None),))
            return None

        body = tuple(ir.map_stmts(body, f))
        if not n_mm:
            raise ScheduleError(f"fold_init({buf}): no ns.tensor.matmul accumulates into '{buf}' (replace the update nest first)")
        # every other access to buf must be a read after the matmuls (checked by the oracle step too)
        self._commit(f"fold_init({buf})", self.proc.with_(body=body))

    def hoist(self, buf: str, to: Optional[str]):
        """Re-stage `buf` (created by stage_in) at the outer loop `to`: its fill moves out of every
        loop between `to` and the original site. Loops whose variable strides the source window by
        exactly one tile (e.g. ko over K) *grow* the buffer: along the partition axis the tiles are
        folded into a new axis 1 ([128, T, n]); along a free axis the buffer is enlarged. Loops that
        do not index the source (e.g. mo for an rhs tile) simply reuse the data. `to=None` hoists to
        the top of the kernel."""
        to_loop = self._find(to) if to is not None else None
        container = to_loop.body if to_loop is not None else self.proc.body

        def search(stmts, path):
            for s in stmts:
                if isinstance(s, ir.Alloc) and s.buf.name == buf:
                    return path, stmts
                if isinstance(s, ir.For):
                    r = search(s.body, path + [s])
                    if r:
                        return r
            return None

        r = search(container, [])
        if r is None:
            raise ScheduleError(f"hoist({buf}, to={to}): '{buf}' is not allocated inside loop '{to}'")
        path, block = r
        if not path:
            raise ScheduleError(f"hoist({buf}, to={to}): '{buf}' is already allocated directly in '{to}'")
        ob = self._buf(buf)
        writers = [s for s in ir.walk(self.proc.body) if not isinstance(s, ir.For) for m, b, _ in ir.accesses(s) if b == buf and "w" in m]
        if len(writers) != 1 or not isinstance(writers[0], ir.Call) or writers[0] not in block:
            raise ScheduleError(f"hoist({buf}): expected a single fill (copy instruction) next to the allocation; found {len(writers)} writers")
        fill = writers[0]
        if not fill.arg("dst").is_full(ir.buffers_of(self.proc)):
            raise ScheduleError(f"hoist({buf}): the fill must write the whole buffer")
        src = fill.arg("src")
        for s in ir.walk(container):
            if any(b == src.buf and "w" in m for m, b, _ in ir.accesses(s)):
                raise ScheduleError(f"hoist({buf}): source '{src.buf}' is written inside '{to}'")
        hv = {l.var: l.extent for l in path}
        growth = {}  # var -> dim
        for d, lo in enumerate(src.lo):
            vs = [v for v in hv if lo.coef(v)]
            if len(vs) > 1:
                raise ScheduleError(f"hoist({buf}): several hoisted loops {vs} index dim {d} of the source")
            if vs:
                v = vs[0]
                if not src.size[d].is_const or lo.coef(v) != src.size[d].const:
                    raise ScheduleError(f"hoist({buf}): loop '{v}' strides the source window by {lo.coef(v)} but the tile is {src.size[d]} — not a whole-tile stride")
                growth[v] = d
        for v in hv:
            if v not in growth and any(v in e.free_vars() for e in src.lo):
                raise ScheduleError(f"hoist({buf}): source offsets depend on '{v}' non-affinely")
        if sum(1 for d in growth.values() if d == 0) > 1:
            raise ScheduleError(f"hoist({buf}): more than one hoisted loop grows the partition axis")
        fold_v = next((v for v, d in growth.items() if d == 0), None)
        sizes = list(ob.shape)
        for v, d in growth.items():
            if d != 0:
                sizes[d] = sizes[d] * hv[v]
        if fold_v is not None:
            sizes.insert(1, hv[fold_v])
        newbuf = ob.with_(shape=tuple(sizes))

        def fidx(idx):
            base = list(idx)
            for v, d in growth.items():
                if d != 0:
                    base[d] = base[d] + var(v) * ob.shape[d].const
            if fold_v is not None:
                base.insert(1, var(fold_v))
            return tuple(base)

        def fwin(w):
            lo = list(w.lo)
            size = list(w.size)
            pts = list(w.points)
            for v, d in growth.items():
                if d != 0:
                    lo[d] = lo[d] + var(v) * ob.shape[d].const
            if fold_v is not None:
                lo.insert(1, var(fold_v))
                size.insert(1, Aff(1))
                pts = [p + 1 if p >= 1 else p for p in pts] + [1]
            return ir.Window(w.buf, tuple(lo), tuple(size), tuple(sorted(pts)))

        def remap(s):
            if isinstance(s, ir.For):
                return replace(s, body=tuple(remap(x) for x in s.body if not (x is fill or (isinstance(x, ir.Alloc) and x.buf.name == buf))))
            if isinstance(s, (ir.Assign, ir.Reduce)):
                rhs = ir._remap_expr(s.rhs, buf, buf, fidx)
                if s.buf == buf:
                    raise ScheduleError(f"hoist({buf}): scalar writes to the buffer are not supported")
                return type(s)(s.buf, s.idx, rhs)
            if isinstance(s, ir.Call):
                return ir.Call(s.instr, tuple((rl, fwin(w) if w.buf == buf else w) for rl, w in s.args), s.attrs)
            return s

        # the new fill: one copy per tile of every growing loop, executed once at `to`
        new_vars = {v: f"{buf}_{v}_ld" for v in growth}
        for nv in new_vars.values():
            self._fresh(nv)
        subs = {v: var(nv) for v, nv in new_vars.items()}
        dst_lo = [Aff(0)] * len(ob.shape)
        dst = ir.Window(buf, tuple(dst_lo), tuple(ob.shape))
        dst = fwin(dst)
        dst = ir.Window(dst.buf, tuple(l.subs(subs) for l in dst.lo), dst.size, dst.points)
        srcw = ir.Window(src.buf, tuple(l.subs(subs) for l in src.lo), src.size, src.points)
        stmt = ir.Call(fill.instr, (("dst", dst), ("src", srcw)), fill.attrs)
        for v in reversed(list(growth)):
            stmt = ir.For(new_vars[v], hv[v], (stmt,), "", "auto")
        new_body = (ir.Alloc(newbuf), stmt) + tuple(remap(x) for x in container)
        if to_loop is None:
            p = self.proc.with_(body=new_body)
        else:
            new_to = replace(to_loop, body=new_body)
            p = self._rewrite_loop(to, lambda f: (new_to,))
        self._commit(f"hoist({buf}, to={to})", p)

    def mark(self, loop: str, kind: str):
        """Assert (or force a more conservative) loop kind: affine | sequential | static."""
        if kind not in ("affine", "sequential", "static"):
            raise ScheduleError("kind must be affine|sequential|static")
        self._find(loop)
        if kind == "affine":
            inferred = infer_kinds(self.proc)
            got = ir.find_loop(inferred.body, loop).kind
            if got != "affine":
                raise ScheduleError(f"mark({loop}, affine): dependence analysis says this loop carries a dependence (inferred {got})")

        def fn(f):
            return (replace(f, kind=kind),)

        self._commit(f"mark({loop}, {kind})", self._rewrite_loop(loop, fn))
