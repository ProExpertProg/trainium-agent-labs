"""Torch frontend: a plain torch function is both the compiler input (traced with torch.export and
expanded into the loop IR) and the correctness oracle (run eagerly).

The exported ATen graph is walked node by node. Every tensor value is a *virtual* tensor: a symbolic
shape, a dtype and a function from indices to a data expression. Views and exact casts (transpose,
permute, widening `.to`) just compose that function, so they cost nothing and never become loops.
Ops that need storage *materialize* a stage -- a buffer plus the loop nests computing it, the naive
breadth-first program every schedule starts from:

    matmul                an f32 accumulator stage (init nest + reduction nest)
    x * s, x / s          a pointwise stage (s a literal or a kernel-level scalar parameter)
    softmax(x, -1)        five stages: row max, its negation, exp(x + (-max)), row sum, e / sum
    scaled_dot_product_attention   scores matmul, scale, softmax, value matmul (default arguments only)

The graph output becomes a final pointwise stage that writes the kernel output. Reductions keep their
reduced dim as size 1 (`[S, 1]`), the shape of an on-chip per-row vector.

Shapes are exported symbolically (one torch Dim per dim name), so scalars derived from shapes, like
`q.shape[1] ** 0.5`, stay expressions over the size symbols -- they become kernel parameters, not
numbers baked in from the example shape.

Dtypes: buffers whose dtype equals the dtype of a kernel argument are emitted as "like:<arg>", so one
traced kernel follows the dtype of its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import ir, scalar
from .expr import Aff, Sym, var

_TORCH_DT = {"torch.float32": "f32", "torch.bfloat16": "bf16", "torch.float16": "f16"}
# casts a -> b that are exact (every a value is representable in b), so they can be elided
_EXACT = {("f32", "f32"), ("bf16", "bf16"), ("f16", "f16"), ("bf16", "f32"), ("f16", "f32")}
_NEG_INF = float("-inf")


class TraceError(Exception):
    pass


@dataclass(frozen=True)
class ArgSpec:
    dims: tuple  # symbolic dim names (str) or constants (int), e.g. ("K", "M")
    dtype: str = "f32"
    example: tuple = None  # concrete example shape used for tracing / the oracle

    def torch_dtype(self):
        import torch

        return {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[self.dtype]


def arg(dims, dtype="f32", example=None) -> ArgSpec:
    return ArgSpec(tuple(dims), dtype, tuple(example) if example is not None else None)


@dataclass(frozen=True)
class _Val:
    """A virtual tensor: shape (tuple of Aff), dtype, and idx-tuple -> data expression."""

    shape: tuple
    dtype: str
    elem: Callable


def _dt(torch_dtype) -> str:
    try:
        return _TORCH_DT[str(torch_dtype)]
    except KeyError:
        raise TraceError(f"unsupported dtype {torch_dtype} (supported: float32, bfloat16, float16)") from None


class _Builder:
    """Accumulates the program: allocations and loop nests, kernel scalar parameters, unique names."""

    def __init__(self, args: dict, output: str):
        self.args = args
        self.output = output
        self.stmts = []
        self.fparams = []  # [(name, scalar expr)]
        self.used = set(args) | {output}

    # ---- naming ----------------------------------------------------------------
    def dtype_ref(self, dt: str) -> str:
        for name, spec in self.args.items():
            if spec.dtype == dt:
                return f"like:{name}"
        return dt

    def unique(self, base: str) -> str:
        name, i = base, 1
        while name in self.used:
            name = f"{base}_{i}"
            i += 1
        self.used.add(name)
        return name

    @staticmethod
    def dim_names(shape, fallback):
        out, seen = [], set()
        for d, fb in zip(shape, fallback):
            base = fb
            if len(d.terms) == 1 and d.const == 0 and d.terms[0][1] == 1 and isinstance(d.terms[0][0], Sym):
                base = d.terms[0][0].name.lower()
            n, i = base, 1
            while n in seen:
                n = f"{base}{i}"
                i += 1
            seen.add(n)
            out.append(n)
        return out

    def param(self, s) -> ir.Param:
        """register a kernel-level scalar parameter (reusing an identical one)"""
        for name, existing in self.fparams:
            if existing == s:
                return ir.Param(name)
        name = self.unique("scale")
        self.fparams.append((name, s))
        return ir.Param(name)

    # ---- loop nests ------------------------------------------------------------
    @staticmethod
    def nest(stage, loops, body):
        """loops: [(varname, extent)] outermost first."""
        for v, ext in reversed(loops):
            body = (ir.For(v, ext, tuple(body), stage),)
        return body[0]

    def _loops(self, stage, shape, prefix=""):
        """loop variables over every dim of `shape`: ([(name, extent)], index tuple)"""
        dims = self.dim_names(shape, tuple("ijkl"[: len(shape)]))
        loops = [(f"{stage}.{prefix}{d}", e) for d, e in zip(dims, shape)]
        return loops, tuple(var(n) for n, _ in loops)

    def pointwise(self, name: str, shape: tuple, dtype: str, rhs: Callable) -> str:
        """stage `name`: for all idx: name[idx] = rhs(idx)"""
        self.used.add(name)
        loops, idx = self._loops(name, shape)
        buf = ir.Buffer(name, shape, self.dtype_ref(dtype), ir.HBM, "temp")
        self.stmts += [ir.Alloc(buf), self.nest(name, loops, (ir.Assign(name, idx, rhs(idx)),))]
        return name

    def reduce_rows(self, name: str, shape: tuple, dtype: str, op: str, init: float, rhs: Callable) -> str:
        """stage `name` of shape [R, 1]: name[i, 0] = op over j of rhs(i, j); init nest + reduction nest"""
        self.used.add(name)
        R, C = shape
        out_shape = (R, Aff(1))
        buf = ir.Buffer(name, out_shape, self.dtype_ref(dtype), ir.HBM, "temp")
        iloops, iidx = self._loops(f"{name}.init", out_shape)
        init_nest = self.nest(f"{name}.init", iloops, (ir.Assign(name, iidx, ir.Lit(init)),))
        loops, idx = self._loops(name, shape)
        upd = self.nest(name, loops, (ir.Reduce(name, (idx[0], Aff(0)), rhs(idx), op),))
        self.stmts += [ir.Alloc(buf), init_nest, upd]
        return name

    # ---- ops -------------------------------------------------------------------
    def view_transpose(self, x: _Val) -> _Val:
        if len(x.shape) != 2:
            raise TraceError("transpose is only supported on 2-D values")
        return _Val(x.shape[::-1], x.dtype, lambda idx, x=x: x.elem(tuple(idx)[::-1]))

    def cast(self, x: _Val, dt: str) -> _Val:
        if (x.dtype, dt) in _EXACT:
            return _Val(x.shape, dt, x.elem)
        return _Val(x.shape, dt, lambda idx, x=x, dt=dt: ir.CastE(self.dtype_ref(dt), x.elem(idx)))

    def matmul(self, name: str, x: _Val, y: _Val, res_dt: str) -> _Val:
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise TraceError("matmul is only supported on 2-D operands")
        if x.shape[1] != y.shape[0]:
            raise TraceError(f"contraction dims differ: {x.shape[1]} vs {y.shape[0]}")
        M, K, N = x.shape[0], x.shape[1], y.shape[1]
        name = self.unique(name)
        m, n, k = self.dim_names((M, N, K), ("m", "n", "k"))
        buf = ir.Buffer(name, (M, N), "f32", ir.HBM, "temp")
        vm, vn, vk = (var(f"{name}.{d}") for d in (m, n, k))
        zm, zn = var(f"{name}.init.{m}"), var(f"{name}.init.{n}")
        init = self.nest(f"{name}.init", [(zm.terms[0][0].name, M), (zn.terms[0][0].name, N)],
                         (ir.Assign(name, (zm, zn), ir.Lit(0.0)),))
        upd = self.nest(name, [(f"{name}.{m}", M), (f"{name}.{n}", N), (f"{name}.{k}", K)],
                        (ir.Reduce(name, (vm, vn), ir.Mul(x.elem((vm, vk)), y.elem((vk, vn)))),))
        self.stmts += [ir.Alloc(buf), init, upd]
        # torch returns the operands' dtype; the accumulator is f32, so narrow if needed
        out = _Val((M, N), "f32", lambda idx: ir.Read(name, tuple(idx)))
        return out if res_dt == "f32" else self.cast(out, res_dt)

    def scale(self, name: str, op: str, x: _Val, s, res_dt: str) -> _Val:
        """x * s or x / s as a pointwise stage; s is a scalar expression (literal or kernel parameter)"""
        operand = ir.Lit(s.value) if isinstance(s, scalar.SConst) else self.param(s)
        name = self.unique(name)
        self.pointwise(name, x.shape, res_dt, lambda idx: ir.Bin(op, x.elem(idx), operand))
        return _Val(x.shape, res_dt, lambda idx: ir.Read(name, tuple(idx)))

    def softmax(self, name: str, x: _Val, res_dt: str) -> _Val:
        """softmax over the last dim of a 2-D value, written the way hardware does it: subtract the
        row max (so exp cannot overflow), exponentiate, divide by the row sum."""
        if len(x.shape) != 2:
            raise TraceError("softmax is only supported on 2-D values")
        if res_dt != "f32":
            raise TraceError(f"softmax is only supported for float32 (got {res_dt}): the hardware needs float32 row statistics and an "
                             f"explicit cast of the probabilities before the next matmul, which this frontend does not generate yet")
        mx, ng = self.unique(f"{name}_max"), self.unique(f"{name}_neg")
        ex, sm, out = self.unique(f"{name}_exp"), self.unique(f"{name}_sum"), self.unique(name)
        R, C = x.shape
        col = (R, Aff(1))
        self.reduce_rows(mx, x.shape, res_dt, "max", _NEG_INF, lambda idx: x.elem(idx))
        self.pointwise(ng, col, res_dt, lambda idx: ir.Un("neg", ir.Read(mx, idx)))
        self.pointwise(ex, x.shape, res_dt, lambda idx: ir.Un("exp", ir.Bin("add", x.elem(idx), ir.Read(ng, (idx[0], Aff(0))))))
        self.reduce_rows(sm, x.shape, res_dt, "add", 0.0, lambda idx: ir.Read(ex, idx))
        self.pointwise(out, x.shape, res_dt, lambda idx: ir.Bin("div", ir.Read(ex, idx), ir.Read(sm, (idx[0], Aff(0)))))
        return _Val(x.shape, res_dt, lambda idx: ir.Read(out, tuple(idx)))

    def finish(self, v: _Val) -> ir.Buffer:
        dims = self.dim_names(v.shape, tuple(f"d{i}" for i in range(len(v.shape))))
        idx = tuple(var(f"{self.output}.{d}") for d in dims)
        out = ir.Buffer(self.output, v.shape, self.dtype_ref(v.dtype), ir.HBM, "out")
        self.stmts.append(self.nest(self.output, [(f"{self.output}.{d}", e) for d, e in zip(dims, v.shape)],
                                    (ir.Assign(self.output, idx, v.elem(idx)),)))
        return out


class _Tracer:
    """Walks the exported graph. Tensor ops dispatch through TENSOR_OPS (by ATen op name), scalar
    nodes (sym_size, pow, truediv, ...) through SCALAR_FUNCS (by Python function name)."""

    def __init__(self, args: dict, output: str):
        from torch.fx import Node

        self._Node = Node
        self.b = _Builder(args, output)
        self.env = {}
        self.placeholders = iter(args.items())
        self.result = None

    # ---- value helpers -----------------------------------------------------------
    def val(self, a):
        return self.env[a] if isinstance(a, self._Node) else a

    def tensor(self, a, what="operand") -> _Val:
        v = self.val(a)
        if not isinstance(v, _Val):
            raise TraceError(f"expected a tensor {what}, found {v!r}")
        return v

    def scalar_of(self, a):
        v = self.val(a)
        if isinstance(v, bool):
            raise TraceError("boolean scalars are not supported")
        if isinstance(v, (int, float)):
            return scalar.SConst(float(v))
        if isinstance(v, (scalar.SConst, scalar.SSize, scalar.SBin, scalar.SUn)):
            return v
        raise TraceError(f"expected a scalar, found {v!r}")

    @staticmethod
    def res_dtype(node) -> str:
        return _dt(node.meta["val"].dtype)

    # ---- tensor ops ---------------------------------------------------------------
    def op_transpose(self, node):
        return self.b.view_transpose(self.tensor(node.args[0]))

    def op_transpose_int(self, node):
        if sorted(int(i) % 2 for i in node.args[1:3]) != [0, 1]:
            raise TraceError("transpose is only supported on 2-D values")
        return self.op_transpose(node)

    def op_permute(self, node):
        if list(node.args[1]) != [1, 0]:
            raise TraceError("permute is only supported as a 2-D transpose")
        return self.op_transpose(node)

    def op_cast(self, node):
        dt = node.args[1] if len(node.args) > 1 else node.kwargs.get("dtype")
        return self.b.cast(self.tensor(node.args[0]), _dt(dt))

    def op_matmul(self, node):
        return self.b.matmul(node.name, self.tensor(node.args[0]), self.tensor(node.args[1]), self.res_dtype(node))

    def _scalar_op(self, node, op):
        x, y = (self.val(a) for a in node.args[:2])
        if isinstance(x, _Val) and isinstance(y, _Val):
            raise TraceError(f"{node.target}: elementwise ops between two tensors are not supported yet (tensor with a scalar only)")
        if op == "div" and not isinstance(x, _Val):
            raise TraceError("a scalar divided by a tensor is not supported (tensor / scalar only)")
        t, s = (x, y) if isinstance(x, _Val) else (y, x)
        return self.b.scale(node.name, op, t, self.scalar_of(s), self.res_dtype(node))

    def op_mul(self, node):
        return self._scalar_op(node, "mul")

    def op_div(self, node):
        return self._scalar_op(node, "div")

    def op_softmax(self, node):
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        x = self.tensor(node.args[0])
        if len(x.shape) != 2 or int(dim) not in (-1, 1):
            raise TraceError("softmax is only supported over the last dim of a 2-D value")
        return self.b.softmax(node.name, x, self.res_dtype(node))

    def op_sdpa(self, node):
        q, k, v = (self.tensor(a) for a in node.args[:3])
        kw = dict(zip(("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"), node.args[3:]))
        kw.update(node.kwargs)
        if kw.get("attn_mask") is not None or kw.get("is_causal") or kw.get("dropout_p", 0.0) or kw.get("enable_gqa"):
            raise TraceError("scaled_dot_product_attention is only supported without a mask, causal flag, dropout or GQA")
        if kw.get("scale") is None:
            s = scalar.SBin("div", scalar.SConst(1.0), scalar.SBin("pow", scalar.SSize(q.shape[1]), scalar.SConst(0.5)))
        else:
            s = self.scalar_of(kw["scale"])
        dt = self.res_dtype(node)
        scores = self.b.matmul("matmul", q, self.b.view_transpose(k), dt)
        scaled = self.b.scale("mul", "mul", scores, s, dt)
        probs = self.b.softmax("softmax", scaled, dt)
        return self.b.matmul("matmul", probs, v, dt)

    TENSOR_OPS = {
        "aten.t.default": op_transpose,
        "aten.numpy_T.default": op_transpose,
        "aten.transpose.int": op_transpose_int,
        "aten.permute.default": op_permute,
        "aten.to.dtype": op_cast,
        "aten._to_copy.default": op_cast,
        "aten.mm.default": op_matmul,
        "aten.matmul.default": op_matmul,
        "aten.mul.Tensor": op_mul,
        "aten.mul.Scalar": op_mul,
        "aten.div.Tensor": op_div,
        "aten.div.Scalar": op_div,
        "aten.softmax.int": op_softmax,
        "aten._softmax.default": op_softmax,
        "aten.scaled_dot_product_attention.default": op_sdpa,
    }

    # ---- scalar nodes --------------------------------------------------------------
    def sym_size(self, node):
        x = self.tensor(node.args[0])
        d = x.shape[int(node.args[1])]
        return scalar.SConst(float(d.const)) if d.is_const else scalar.SSize(d)

    SCALAR_BIN = {"pow": "pow", "truediv": "div", "mul": "mul", "add": "add", "sub": "sub"}

    def scalar_func(self, node, name):
        a = [self.scalar_of(x) for x in node.args]
        if name == "sym_float":
            return a[0]
        if name in ("_sym_sqrt", "sqrt"):
            return scalar.SUn("sqrt", a[0])
        return scalar.SBin(self.SCALAR_BIN[name], a[0], a[1])

    # ---- driver ---------------------------------------------------------------------
    def visit(self, node):
        if node.op == "placeholder":
            aname, spec = next(self.placeholders)
            shape = tuple(Aff.of(d) for d in spec.dims)
            self.env[node] = _Val(shape, spec.dtype, lambda idx, aname=aname: ir.Read(aname, tuple(idx)))
        elif node.op == "output":
            self.result = self.env[node.args[0][0]]
        elif node.op == "call_function":
            t = str(node.target)
            fname = getattr(node.target, "__name__", "")
            if "assert" in t or "constrain" in t:
                return
            if t == "aten.sym_size.int":
                self.env[node] = self.sym_size(node)
            elif t in self.TENSOR_OPS:
                self.env[node] = self.TENSOR_OPS[t](self, node)
            elif not t.startswith("aten.") and (fname in self.SCALAR_BIN or fname in ("sym_float", "_sym_sqrt", "sqrt")):
                self.env[node] = self.scalar_func(node, fname)
            else:
                raise TraceError(f"unsupported torch op in spec: {t} (supported: mm/matmul, t/permute/transpose, dtype casts, "
                                 f"tensor*scalar, tensor/scalar, softmax, scaled_dot_product_attention)")


def trace(fn, args: dict, name: str = "kernel", output: str = "C") -> ir.Proc:
    """Trace `fn(*args)` with torch.export and expand it to the naive loop IR.

    args: ordered {argument name: ArgSpec}, in the order of fn's parameters.
    name: entry-point name of the kernel; output: name of the output buffer."""
    import torch
    from torch.export import Dim

    examples = []
    for spec in args.values():
        if spec.example is None:
            raise TraceError("ArgSpec.example (a concrete example shape) is required for tracing")
        examples.append(torch.randn(*spec.example).to(spec.torch_dtype()))

    class _Spec(torch.nn.Module):
        def forward(self, *a):
            return fn(*a)

    # Every symbolic dim name becomes one torch.export Dim (equal names are one Dim), so the exported
    # graph keeps shapes symbolic: shape-dependent scalars such as `q.shape[1] ** 0.5` stay expressions
    # over the dims instead of being baked in as the example's number.
    dims = {}
    shapes = []
    for spec in args.values():
        shapes.append({i: dims.setdefault(d, Dim(d, min=2)) for i, d in enumerate(spec.dims) if isinstance(d, str)})
    try:
        graph = torch.export.export(_Spec(), tuple(examples), dynamic_shapes=(tuple(shapes),)).graph_module.graph
    except Exception as e:  # noqa: BLE001  torch reports constraint violations, shape errors, ... as many types
        lines = [l.strip(" -") for l in str(e).strip().splitlines()]
        detail = [l for l in lines if "must always be equal" in l or "specialized" in l]
        if detail:
            raise TraceError("the spec forces some dims to be equal or constant; give dims that must be equal the "
                             "same name in arg(...). torch says: " + " ".join(detail)[:400]) from None
        raise TraceError(f"torch.export could not trace the spec: {lines[0][:300]}") from None

    tr = _Tracer(args, output)
    for node in graph.nodes:
        tr.visit(node)
    if tr.result is None:
        raise TraceError("the spec has no tensor output")

    out = tr.b.finish(tr.result)
    buffers = tuple(
        ir.Buffer(aname, tuple(Aff.of(d) for d in spec.dims), f"like:{aname}", ir.HBM, "arg") for aname, spec in args.items()
    ) + (out,)
    sizes, seen = [], set()
    for aname, spec in args.items():
        for i, d in enumerate(spec.dims):
            if isinstance(d, str) and d not in seen:
                seen.add(d)
                sizes.append((d, aname, i))
    return ir.Proc(name=name, args=buffers, sizes=tuple(sizes), body=tuple(tr.b.stmts), fparams=tuple(tr.b.fparams))
