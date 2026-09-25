#!/usr/bin/env bash
# Peak memory of the pond workers while a load test runs.
#
#   deploy/memwatch.sh start     # begin sampling on every worker, twice a second
#   ... run the load ...
#   deploy/memwatch.sh stop      # print each worker's peaks and stop sampling
#
# Two numbers per host. `rss` is the pond worker process alone (the uvicorn under run.sh),
# which is what the 512 MB budget has to hold. `anon` is the whole container from
# memory.stat, chat app included, which is what the kernel enforces the cap against.
# memory.peak is no use here: it is the container's all-time high, chat app and all.
set -uo pipefail
cd "$(dirname "$0")/.."
source deploy/hosts.sh

SAMPLER=$(mktemp)
trap 'rm -f "$SAMPLER"' EXIT
cat > "$SAMPLER" <<'SAMPLE'
cd "$(dirname "$0")"
echo $$ > .memwatch.pid
peak_rss=0; peak_anon=0
while [ -f .memwatch.pid ]; do
    pid=$(pgrep -P "$(cat .run.lock)" | head -1)
    rss=$(awk '/^VmRSS/{print int($2/1024)}' "/proc/$pid/status" 2>/dev/null)
    anon=$(awk '/^anon /{print int($2/1048576)}' /sys/fs/cgroup/memory.stat)
    [ "${rss:-0}" -gt "$peak_rss" ] && peak_rss=$rss
    [ "${anon:-0}" -gt "$peak_anon" ] && peak_anon=$anon
    echo "rss=${peak_rss}MB anon=${peak_anon}MB" > .memwatch
    sleep 0.5
done
SAMPLE

case "${1:-}" in
start)
    for host in "${WORKERS[@]}"; do
        on_with "$SAMPLER" "$host" "cd ~/$APP_DIR && cat > .memwatch.sh && rm -f .memwatch &&
            (setsid bash .memwatch.sh >/dev/null 2>&1 </dev/null &)" && echo "$host sampling"
    done
    ;;
stop)
    for host in "${WORKERS[@]}"; do
        on "$host" "cd ~/$APP_DIR && rm -f .memwatch.pid; sleep 0.6; echo \"$host peak \$(cat .memwatch 2>/dev/null)\""
    done
    ;;
*)
    echo "usage: $0 start|stop" >&2
    exit 2
    ;;
esac
