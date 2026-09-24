#!/usr/bin/env bash
# Ship the service to the four lab containers and start each in its role.
#
#   deploy/deploy.sh                 # code + venv + restart, all four
#   deploy/deploy.sh sys3            # just one host
#   STRATEGY=round_robin deploy/deploy.sh sys1   # gateway with the other dispatcher
#
# The containers are not git checkouts and have no rsync, so code goes over ssh as a tar
# stream. sys2/3/4 have Python 3.12 but no numpy and no python3-venv package, so the venv
# is made with --without-pip and bootstrapped with get-pip.py (outbound internet works but
# resets now and then, hence the retries).
#
# Each host runs the same `run.sh`. What makes sys1 the gateway is one line in its
# `.env.role`: POND_JOBS_WORKERS, the workers' Docker bridge addresses.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/hosts.sh

HOSTS=("$@")
[ ${#HOSTS[@]} -eq 0 ] && HOSTS=("$GATEWAY" "${WORKERS[@]}")
STRATEGY="${STRATEGY:-least_busy}"

FILES=(app static data/tiles data/contours_1m.kml requirements.txt run.sh)

role_env() {
    local host=$1
    # Every host: the contour path's 4-grid ensemble does not fit 512 MB; the map path's does.
    echo "export POND_API_DEFAULT_ENSEMBLE=false"
    echo "export POND_API_ALLOW_ENSEMBLE=false"
    if [ "$host" = "$GATEWAY" ]; then
        echo "export POND_JOBS_WORKERS=$(worker_urls "${WORKERS[@]}")"
        echo "export POND_JOBS_STRATEGY=$STRATEGY"
    else
        echo "export POND_JOBS_WORKERS="
    fi
}

for host in "${HOSTS[@]}"; do
    echo "== $host (${BRIDGE_IP[$host]}) =="
    tar czf - "${FILES[@]}" | on "$host" "mkdir -p ~/$APP_DIR && cd ~/$APP_DIR && tar xzf -"
    role_env "$host" | on "$host" "cat > ~/$APP_DIR/.env.role"

    on "$host" bash -s <<REMOTE
set -e
cd ~/$APP_DIR
if [ ! -x .venv/bin/python ]; then
    python3 -m venv --without-pip .venv
    curl -sS --retry 5 https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    .venv/bin/python /tmp/get-pip.py -q
fi
.venv/bin/pip install -q --retries 10 --timeout 60 -r requirements.txt
# Restart by PID, never pkill -f: that pattern matches this ssh command's own command line
# and kills the shell before the next line runs (PLAN3 §11.7).
if [ -f .run.lock ] && pid=\$(cat .run.lock 2>/dev/null) && kill -0 "\$pid" 2>/dev/null; then
    pkill -P "\$pid" || true
    kill "\$pid" || true
    sleep 1
fi
setsid ./run.sh >/dev/null 2>&1 </dev/null &
REMOTE
done

sleep 4
deploy/status.sh "${HOSTS[@]}"
