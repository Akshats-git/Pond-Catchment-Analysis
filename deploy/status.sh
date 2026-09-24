#!/usr/bin/env bash
# Health and memory of each container. Memory is read from memory.stat's `anon`, not
# memory.current: the latter counts reclaimable page cache and understated the headroom on
# sys2/3/4 by more than 200 MB (PLAN3 §11.6).
set -uo pipefail
cd "$(dirname "$0")/.."
source deploy/hosts.sh
HOSTS=("$@")
[ ${#HOSTS[@]} -eq 0 ] && HOSTS=("$GATEWAY" "${WORKERS[@]}")
for host in "${HOSTS[@]}"; do
    on "$host" 'h=$(curl -s -m 3 localhost:5000/health || echo DOWN);
                cg=/sys/fs/cgroup;
                anon=$(awk "/^anon /{printf \"%.0f\", \$2/1048576}" $cg/memory.stat 2>/dev/null);
                max=$(awk "{printf \"%.0f\", \$1/1048576}" $cg/memory.max 2>/dev/null);
                peak=$(awk "{printf \"%.0f\", \$1/1048576}" $cg/memory.peak 2>/dev/null);
                echo "anon=${anon}MB peak=${peak:-?}MB cap=${max}MB  $h"' \
        | sed "s/^/$host  /" || echo "$host  unreachable"
done
