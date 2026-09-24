#!/usr/bin/env bash
# Strategy A of PLAN3 §6.2: the existing Go load balancer (~/lb on sys1), unmodified,
# round-robin over the three pond workers, on sys1:6000 -> public 6229.
#
# It was built for Lab 2/3 against the chat apps on :4000 and already speaks
# `-health-path /health`, which the pond workers serve. Nothing about it changes; only the
# backend list does. Compared against the gateway's least-busy dispatcher in docs/SCALING.md.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/hosts.sh
backends=$(worker_urls "${WORKERS[@]}")
on "$GATEWAY" "cd ~ && setsid ./start-lb.sh '$backends' 6000 >/dev/null 2>&1 </dev/null & sleep 2; curl -s -m 3 localhost:6000/lb/status || true"
