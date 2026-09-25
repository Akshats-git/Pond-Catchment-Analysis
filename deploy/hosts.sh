# The four lab containers (PLAN3 §3.2). Sourced by the other scripts in this directory.
#
#   name  ssh port  docker bridge IP  public app port   role
#   sys1  2229      172.17.0.30       5229 -> 5000      gateway (the Phase 2 URL, unchanged)
#   sys2  2230      172.17.0.31       5230 -> 5000      worker
#   sys3  2231      172.17.0.32       5231 -> 5000      worker
#   sys4  2232      172.17.0.33       5232 -> 5000      worker
#
# Workers listen on 5000 inside their container, never 4000: 4000 on sys2/3/4 is somebody's
# running chat app, and 3000 on sys1 is the lab's own load balancer. Leave both alone.

LAB_HOST="${LAB_HOST:-10.1.75.53}"
LAB_USER="${LAB_USER:-student}"
APP_DIR="${APP_DIR:-PondCatchmentAnalysis}"

declare -A SSH_PORT=([sys1]=2229 [sys2]=2230 [sys3]=2231 [sys4]=2232)
declare -A BRIDGE_IP=([sys1]=172.17.0.30 [sys2]=172.17.0.31 [sys3]=172.17.0.32 [sys4]=172.17.0.33)
declare -A PUBLIC_PORT=([sys1]=5229 [sys2]=5230 [sys3]=5231 [sys4]=5232)

GATEWAY=sys1
# LAB_WORKERS="sys2 sys3" leaves a host out, e.g. one with no memory to spare.
read -r -a WORKERS <<< "${LAB_WORKERS:-sys2 sys3 sys4}"

worker_urls() {
    local urls=() w
    for w in "$@"; do urls+=("http://${BRIDGE_IP[$w]}:5000"); done
    (IFS=,; echo "${urls[*]}")
}

# The lab's network drops connections for a minute at a time, so a connection that fails
# (ssh exit 255) is retried. Anything a command reads from stdin must come from a file,
# `on_with <file> <host> ...`, so that a retry can send it again.
on_with() {  # on_with <stdin file> <host> <command...>
    local input=$1 host=$2; shift 2
    local attempt status
    for attempt in 1 2 3 4 5 6 7 8; do
        ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
            -p "${SSH_PORT[$host]}" "$LAB_USER@$LAB_HOST" "$@" < "$input" && return 0
        status=$?
        [ $status -ne 255 ] && return $status
        echo "  ($host: connection failed, retry $attempt)" >&2
        sleep $((attempt * 5))
    done
    return 255
}

on() {  # on <host> <command...>
    on_with /dev/null "$@"
}
