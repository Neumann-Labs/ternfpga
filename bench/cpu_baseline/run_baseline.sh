#!/usr/bin/env bash
# Measure the CPU ternary-inference baseline: decode tokens/sec + package
# energy-per-token (via RAPL). Run after setup_bitnet.sh.
#
#   bash bench/cpu_baseline/run_baseline.sh
#
# Energy is whole-package (RAPL intel-rapl:0, which on this AMD Ryzen reports
# the CPU package) integrated over the full run (prefill + decode) and divided
# by generated tokens — an honest upper bound on J/token; a decode-only number
# comes later. Needs sudo to read energy_uj.
set -euo pipefail

ROOT="${BITNET_ROOT:-/srv/fpga/bitnet}"
MODEL="${MODEL:-$ROOT/models/2B-4T-gguf/ggml-model-i2_s.gguf}"
CLI="${CLI:-$ROOT/build/bin/llama-cli}"
N="${N_TOKENS:-256}"
THREADS="${THREADS:-16}"
PROMPT="${PROMPT:-The future of efficient on-device AI inference is}"
RAPL="${RAPL:-/sys/class/powercap/intel-rapl:0}"
OUT="${OUT:-/tmp/bitnet_run.txt}"

[ -x "$CLI" ]   || { echo "ERROR: llama-cli not at $CLI — run setup_bitnet.sh first"; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model not at $MODEL"; exit 1; }

read_uj() { sudo cat "$RAPL/energy_uj"; }
maxuj=$(sudo cat "$RAPL/max_energy_range_uj" 2>/dev/null || echo 0)

echo "warm-up..."
"$CLI" -m "$MODEL" -p "$PROMPT" -n 32 -t "$THREADS" -ngl 0 --temp 0 >/dev/null 2>&1 || true

echo "measuring ($N tokens, $THREADS threads)..."
e0=$(read_uj); t0=$(date +%s.%N)
"$CLI" -m "$MODEL" -p "$PROMPT" -n "$N" -t "$THREADS" -ngl 0 --temp 0 >"$OUT" 2>&1
t1=$(date +%s.%N); e1=$(read_uj)

de=$(( e1 - e0 )); [ "$de" -lt 0 ] && de=$(( de + maxuj ))
joules=$(python3 -c "print(f'{$de/1e6:.2f}')")
secs=$(python3 -c "print(f'{$t1-$t0:.2f}')")
tps=$(grep -oE "[0-9.]+ tokens per second" "$OUT" | tail -1 | grep -oE "^[0-9.]+" || echo "")
jpt=$(python3 -c "print(f'{$de/1e6/$N:.4f}')")

echo "===== CPU BASELINE ====="
echo "model      : $(basename "$MODEL")"
echo "threads    : $THREADS    n_tokens: $N"
echo "wall_secs  : $secs"
echo "package_J  : $joules   (RAPL $RAPL, full run)"
echo "eval_tok/s : ${tps:-NA}   (llama eval timing)"
echo "J/token    : $jpt   (package J / generated tokens, prefill included)"
echo "========================"
