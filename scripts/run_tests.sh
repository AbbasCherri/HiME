#!/usr/bin/env bash
# Runs every test layer and the smoke run, logging each step to runs/test_logs/<timestamp>/.
# Keeps going after a failure; exits non-zero if any step failed.
# Usage: bash scripts/run_tests.sh [--pull] [--gpu] [--embed-all]
#   --gpu        also: embedding throughput test (2,000 buildings, every encoder) and the imagery audit
#   --embed-all  also: the full V0 embedding grid (hours)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

# Determinism controls are set by the launcher, not inside Python (plan section 7).
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PULL=0; GPU=0; EMBED=0; REST=()
for a in "$@"; do
  case "$a" in --pull) PULL=1 ;; --gpu) GPU=1; REST+=("$a") ;; --embed-all) EMBED=1; REST+=("$a") ;; esac
done
if [ $PULL = 1 ]; then
  git pull --ff-only && uv sync --python 3.12 --extra embed
  exec bash "$0" "${REST[@]}"   # restart: bash must not keep reading a file that pull just rewrote
fi
export HIME_DATA="${HIME_DATA:-$HOME/HiME-dataset}"

LOG="runs/test_logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG"
git rev-parse HEAD > "$LOG/commit.txt"
uv pip freeze > "$LOG/env.txt" 2>/dev/null || true

declare -A RESULT
step() {
  local name="$1"; shift
  echo "==> $name"
  local t0=$SECONDS
  if "$@" > "$LOG/$name.log" 2>&1; then RESULT[$name]="Passed"; else RESULT[$name]="Failed"; fi
  if grep -q "Blocked" "$LOG/$name.log" && [ "${RESULT[$name]}" = "Passed" ]; then RESULT[$name]="Passed (data Blocked)"; fi
  echo "    ${RESULT[$name]} in $((SECONDS - t0)) s  (log: $LOG/$name.log)"
}

PYT="uv run pytest -rfEs -s --tb=short --durations=10"
step p0_governance     $PYT tests/test_p0_governance.py
step p1_realdata       $PYT tests/test_p1_realdata.py
step p2_oracle         $PYT tests/test_p2_oracle.py
step p3_experts        $PYT tests/test_p3_experts.py
step p4_fusion         $PYT tests/test_p4_fusion.py
step p5_placement      $PYT tests/test_p5_uav.py
step p6_participation  $PYT tests/test_p6_participation.py
step p7_engine         $PYT tests/test_p7_engine.py
step smoke             uv run python scripts/smoke.py
step real_sanity       uv run python scripts/run_real.py --N 100 --seeds 1000 --T 100
STEPS="p0_governance p1_realdata p2_oracle p3_experts p4_fusion p5_placement p6_participation p7_engine smoke real_sanity"
if [ $GPU = 1 ]; then
  step embed_throughput uv run python scripts/embed_v0.py --limit 2000
  step imagery_audit    uv run python scripts/audit_imagery.py
  STEPS="$STEPS embed_throughput imagery_audit"
fi
if [ $EMBED = 1 ]; then
  step embed_all uv run python scripts/embed_v0.py
  STEPS="$STEPS embed_all"
fi

echo
echo "Summary ($LOG/summary.txt)"
FAIL=0
for k in $STEPS; do
  printf "%-18s %s\n" "$k" "${RESULT[$k]}" | tee -a "$LOG/summary.txt"
  [ "${RESULT[$k]}" = "Failed" ] && FAIL=1
done
F=$(grep -h "^FAILED\|^ERROR\|^E  " "$LOG"/*.log | head -60)
[ -n "$F" ] && printf "\nFailures:\n%s\n" "$F" >> "$LOG/summary.txt"
echo >> "$LOG/summary.txt"; echo "Measurements:" >> "$LOG/summary.txt"
grep -h "O5 evidence\|greedy/opt\|w_SI under\|distance to central\|^N=\|fold 0:\|clients without\|calib share\|floored\|has_post\|images/s\|FAILED (\|Blocked\|flagged" "$LOG"/*.log >> "$LOG/summary.txt" 2>/dev/null || true
[ -f runs/smoke.json ] && cp runs/smoke.json "$LOG/"
echo; sed -n "/^Failures:\|^Measurements:/,\$p" "$LOG/summary.txt"
exit $FAIL
