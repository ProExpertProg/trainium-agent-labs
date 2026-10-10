# nki-sched

A scheduling language on top of NKI: write the algorithm in torch, transform it with a short schedule
(tile, stage into SBUF/PSUM, tensorize, hoist, ...), emit a verified NKI kernel.

**Status:** the tiled matmul and single-head attention work end to end (torch spec -> loop IR -> schedule ->
NKI source). Multibuffer / pipeline / overlap, symbolic blocking and online softmax are still plan. See
`PLAN.md` section 13 for exactly what was verified (simulator, compile-only, NeuronCore) and what was not.

* [`PLAN.md`](PLAN.md): architecture, IR, lowering, correctness strategy, MLIR decision, roadmap, status
* [`SYNTAX.md`](SYNTAX.md): schedule-language draft and the worked matmul schedules
* [`docs/ATTENTION.md`](docs/ATTENTION.md): attention: the instruction set, the two schedules, what was verified
* [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md): Halide / TVM / Exo digest
* [`AGENT.md`](AGENT.md): concise description of the schedule language, meant to be fed into a kernel agent's context

## Try it

```
pip install -r requirements.txt
python -m pytest                             # unit tests; no Neuron SDK needed
python examples/loops.py                     # split / reorder / compute_at: prints the IR after each step
python examples/memory.py                    # staging, set_memory, tensorize: prints the IR and the NKI kernel
python examples/attention.py                 # attention: single-tile and tiled schedules, IR and NKI kernel
python examples/matmul_ladder.py l4 out.py   # emit the tiled-matmul kernel (l5-l7 are hoisting examples)
python examples/attention.py tiled out.py    # emit the tiled attention kernel
```

On a Trainium node (integration; not part of the unit tests):

```
ENV_FILE=../.env scripts/device_check.sh l4            # matmul: compile-only, simulator f32/bf16, nkibench, NeuronCore
ENV_FILE=../.env scripts/device_check.sh tiled         # attention (also single_tile): compile-only, simulator, NeuronCore
python scripts/kernel_check.py out.py nki_attention_ --op attention --mode compile   # one check on a kernel file
python scripts/nkibench_rules.py out.py 4 --nkibench /path/to/nkibench.py            # just nkibench's static rule scan
```
