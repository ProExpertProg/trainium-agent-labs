#!/bin/bash
# Emit a schedule's kernel, copy it to a Trainium node and check it there, cheapest layer first:
#
#   compile   neuronx-cc compile only (no core needed -- catches what the hardware compiler rejects and
#             the simulator accepts; works while other jobs hold the NeuronCores)
#   sim       nki.simulate against numpy (float32; matmul also bfloat16)
#   nkibench  (matmul levels only) nkibench's rule scan, numerics and HBM-traffic bar
#   hw        compile and run on a NeuronCore, compare with numpy (wall time is launch-dominated)
#
#   ENV_FILE=/path/to/.env scripts/device_check.sh l4          # matmul ladder: l4 l5 l6 l7
#   ENV_FILE=/path/to/.env scripts/device_check.sh single_tile # attention: single_tile tiled
#
# Environment:
#   ENV_FILE      file exporting AWS credentials     (default: $ROOT/../.env; pass it explicitly from a git worktree)
#   POD           pod to use                         (default: seat-270)
#   PYTHON        interpreter with torch + numpy     (default: python3)
#   NKIBENCH_DIR  directory holding nkibench.py on the node (default: /workspace/projects/02-kernel-agent)
#   SKIP_HW=1     skip the NeuronCore step (e.g. when the cores are held by another job)
#
# nkibench's attention level (8) is not run: its report code crashes on attention's case format
# (KeyError: 'M'), so attention is checked by kernel_check.py instead.
set -euo pipefail
WHAT=${1:?usage: device_check.sh l4|l5|l6|l7|single_tile|tiled}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
POD=${POD:-seat-270}
PYTHON=${PYTHON:-python3}
NKIBENCH_DIR=${NKIBENCH_DIR:-/workspace/projects/02-kernel-agent}
ENV_FILE=${ENV_FILE:-$ROOT/../.env}
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
TMP=$(mktemp -d)
OUT=$TMP/kernel_$WHAT.py
NAME=$(basename "$OUT")

case $WHAT in
  l[4-7]) OP=matmul; SHAPES=""; DTYPES=f32,bf16
          "$PYTHON" "$ROOT/examples/matmul_ladder.py" "$WHAT" "$OUT" >/dev/null ;;
  single_tile|tiled) OP=attention; DTYPES=f32
          # the tiled kernel needs S % 128 == 0; the single-tile one needs S, D <= 128
          if [ "$WHAT" = tiled ]; then SHAPES="128x64,256x64,384x128,512x32"; else SHAPES="128x64,64x128,96x32,128x128"; fi
          "$PYTHON" "$ROOT/examples/attention.py" "$WHAT" "$OUT" >/dev/null ;;
  *) echo "unknown target $WHAT" >&2; exit 2 ;;
esac
ENTRY=$(grep -m1 '^def ' "$OUT" | sed 's/def \([a-zA-Z0-9_]*\).*/\1/')

kubectl cp -c app "$OUT" "$POD:/tmp/$NAME"
kubectl cp -c app "$ROOT/scripts/kernel_check.py" "$POD:/tmp/kernel_check.py"
check() {  # mode
  kubectl exec "$POD" -c app -- bash -c "cd /tmp && python /tmp/kernel_check.py /tmp/$NAME $ENTRY --op $OP --mode $1 --dtypes $DTYPES ${SHAPES:+--shapes $SHAPES}" 2>&1 | grep -v '^20[0-9][0-9]-'
}

echo "== compile only (neuronx-cc, no core) =="; check compile
echo "== simulator =="; check sim
if [ "$OP" = matmul ]; then
  LEVEL=${WHAT#l}
  echo "== nkibench level $LEVEL =="
  kubectl exec "$POD" -c app -- bash -c "cd $NKIBENCH_DIR && python nkibench.py --level $LEVEL --check /tmp/$NAME" | grep -E "rules|numerics|HBM traffic|VIOLATION|FAILED" | cut -c1-160
fi
if [ -z "${SKIP_HW:-}" ]; then
  echo "== NeuronCore =="; check hw
fi
