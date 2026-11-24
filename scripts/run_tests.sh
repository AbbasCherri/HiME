#!/usr/bin/env bash
# Runs every test layer and the smoke run, logging each step to runs/test_logs/<timestamp>/.
# Keeps going after a failure; exits non-zero if any step failed.
# Usage: bash scripts/run_tests.sh [--pull]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

# Determinism controls are set by the launcher, not inside Python (plan section 7).
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ "${1:-}" = "--pull" ]; then
  git pull --ff-only && uv sync --python 3.12
  exec bash "$0"          # restart: bash must not keep reading a file that pull just rewrote
fi

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

echo
echo "Summary ($LOG/summary.txt)"
FAIL=0
for k in p0_governance p1_realdata p2_oracle p3_experts p4_fusion p5_placement p6_participation p7_engine smoke; do
  printf "%-18s %s\n" "$k" "${RESULT[$k]}" | tee -a "$LOG/summary.txt"
  [ "${RESULT[$k]}" = "Failed" ] && FAIL=1
done
F=$(grep -h "^FAILED\|^ERROR\|^E  " "$LOG"/*.log | head -60)
[ -n "$F" ] && printf "\nFailures:\n%s\n" "$F" >> "$LOG/summary.txt"
echo >> "$LOG/summary.txt"; echo "Measurements:" >> "$LOG/summary.txt"
grep -h "O5 evidence\|greedy/opt\|w_SI under\|distance to central" "$LOG"/*.log >> "$LOG/summary.txt" 2>/dev/null || true
[ -f runs/smoke.json ] && cp runs/smoke.json "$LOG/"
echo; sed -n "/^Failures:\|^Measurements:/,\$p" "$LOG/summary.txt"
exit $FAIL
