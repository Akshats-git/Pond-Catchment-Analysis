#!/usr/bin/env bash
# A small, varied workload for the telemetry demo: fresh areas on different workers,
# a repeat (result-cache hit) and an area with no pond site (a failed job).
# Usage: tools/generate_load.sh [base_url]   (default http://localhost:5229)
set -u
U="${1:-http://localhost:5229}/api/v1"

run() {
  local id st
  id=$(curl -s -X POST "$U/jobs" -H 'Content-Type: application/json' -d "{\"bbox\": $1}" \
       | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])') || return
  for _ in $(seq 1 90); do
    st=$(curl -s "$U/jobs/$id?include_result=false" | python3 -c \
      'import sys,json; d=json.load(sys.stdin); print(d["status"], d.get("worker"), d.get("run_s"), d.get("error",{}).get("code",""))')
    case "$st" in done*|failed*) echo "$1 -> $st"; return;; esac
    sleep 1
  done
}

run "[81.2814, 21.2398, 81.3126, 21.2636]"   # the sample area
run "[81.2814, 21.2398, 81.3126, 21.2636]"   # same again: served from the cache
run "[81.2850, 21.2400, 81.3100, 21.2600]"
run "[81.2800, 21.2410, 81.3000, 21.2620]"
run "[81.2830, 21.2420, 81.3090, 21.2630]"
run "[81.2790, 21.2380, 81.3050, 21.2590]"
run "[81.2860, 21.2430, 81.3110, 21.2640]"
run "[0.0, 0.0, 0.01, 0.01]"                 # open ocean: no pond site, the job fails
