#!/usr/bin/env bash
# Strategy A of PLAN3 §6.2: the existing Go load balancer (~/lb on sys1), unmodified,
# round-robin over the pond workers, on sys1:6000 -> public 6229.
#
#   deploy/start-lb.sh              # start (or restart) it
#   deploy/start-lb.sh stop
#
# It was built for Lab 2/3 against the chat apps on :4000 and already speaks
# `-health-path /health`, which the pond workers serve. The binary is run directly rather
# than through ~/start-lb.sh, which always logs to ~/logs/lb.log: that is the log of the
# balancer already serving :3000, and that one is left alone. Two flags matter:
#   -sticky=false   its default pins a client to one backend by cookie, and a load
#                   generator is one client, so every request would go to one worker;
#   no -tls-*       plain HTTP, like the workers themselves.
# Compared against the gateway's least-busy dispatcher in docs/SCALING_LAB.md.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/hosts.sh
backends=$(worker_urls "${WORKERS[@]}")

stop='old=$(ss -tlnp 2>/dev/null | awk "/:6000 /" | grep -o "pid=[0-9]*" | cut -d= -f2 | head -1);
      [ -n "$old" ] && kill "$old" && sleep 1; true'
if [ "${1:-}" = "stop" ]; then
    on "$GATEWAY" "$stop"
    exit 0
fi
on "$GATEWAY" "$stop; mkdir -p ~/logs;
    (setsid ~/lb/lb -listen :6000 -backends '$backends' -health-path /health -sticky=false \
        > ~/logs/lb-pond.log 2>&1 < /dev/null &);
    sleep 3; curl -s -m 3 localhost:6000/lb/status || tail -5 ~/logs/lb-pond.log"
