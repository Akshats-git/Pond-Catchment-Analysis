#!/usr/bin/env bash
# Ship the service to the four lab containers and start each in its role.
#
#   deploy/deploy.sh                 # code + venv + restart, all four
#   deploy/deploy.sh sys3            # just one host
#   STRATEGY=round_robin deploy/deploy.sh sys1   # gateway with the other dispatcher
#   LAB_WORKERS=sys2 MAX_QUEUED=4 deploy/deploy.sh sys1   # one worker, a short queue
#
# The containers are not git checkouts and have no rsync, so code goes over ssh as a tar
# stream. sys2/3/4 have Python 3.12 but no numpy and no python3-venv package, so the venv
# is made with --without-pip. The lab's own internet is too slow to fetch numpy and scipy
# (16 KB in 10 s from PyPI, 2026-09-25), so the wheels are downloaded here once, into
# .cache/wheels, and shipped with the code; pip installs them with --no-index. They are
# sent to a host only when requirements.txt has changed since its last install.
#
# Each host runs the same `run.sh`. What makes sys1 the gateway is one line in its
# `.env.role`: POND_JOBS_WORKERS, the workers' Docker bridge addresses.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/hosts.sh

HOSTS=("$@")
[ ${#HOSTS[@]} -eq 0 ] && HOSTS=("$GATEWAY" "${WORKERS[@]}")
STRATEGY="${STRATEGY:-least_busy}"

FILES=(app static data/tiles data/contours_1m.kml requirements.txt run.sh
       tools/__init__.py tools/loadtest.py tools/smoke.py)

role_env() {
    local host=$1
    # Every host: the contour path's 4-grid ensemble does not fit 512 MB; the map path's does.
    echo "export POND_API_DEFAULT_ENSEMBLE=false"
    echo "export POND_API_ALLOW_ENSEMBLE=false"
    if [ "$host" = "$GATEWAY" ]; then
        echo "export POND_JOBS_WORKERS=$(worker_urls "${WORKERS[@]}")"
        echo "export POND_JOBS_STRATEGY=$STRATEGY"
        [ -n "${MAX_QUEUED:-}" ] && echo "export POND_JOBS_MAX_QUEUED=$MAX_QUEUED"
    else
        echo "export POND_JOBS_WORKERS="
    fi
}

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
tar czf "$STAGE/code.tgz" "${FILES[@]}"

WHEELS=.cache/wheels
REQ_HASH=$(sha256sum requirements.txt | cut -c1-16)
if [ ! -f "$WHEELS/.req-hash" ] || [ "$(cat "$WHEELS/.req-hash")" != "$REQ_HASH" ]; then
    rm -rf "$WHEELS" && mkdir -p "$WHEELS"
    "$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )" -m pip download -q --only-binary=:all: --implementation cp --python-version 3.12 \
        --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 \
        -r requirements.txt pip -d "$WHEELS"
    echo "$REQ_HASH" > "$WHEELS/.req-hash"
fi
tar cf "$STAGE/wheels.tar" -C "$WHEELS" .

for host in "${HOSTS[@]}"; do
    echo "== $host (${BRIDGE_IP[$host]}) =="
    on_with "$STAGE/code.tgz" "$host" "mkdir -p ~/$APP_DIR && cd ~/$APP_DIR && tar xzf -"
    role_env "$host" > "$STAGE/env.role"
    on_with "$STAGE/env.role" "$host" "cat > ~/$APP_DIR/.env.role"
    # Installed already, or shipped by a run that stopped before installing.
    if [ "$(on "$host" "cat ~/$APP_DIR/.venv/.req-hash 2>/dev/null || cat ~/$APP_DIR/.wheels/.req-hash 2>/dev/null || true")" != "$REQ_HASH" ]; then
        echo "  shipping wheels ($(du -sh "$WHEELS" | cut -f1))"
        on_with "$STAGE/wheels.tar" "$host" "rm -rf ~/$APP_DIR/.wheels && mkdir -p ~/$APP_DIR/.wheels && tar xf - -C ~/$APP_DIR/.wheels"
    fi

    cat > "$STAGE/remote.sh" <<REMOTE
set -e
cd ~/$APP_DIR
if [ -d .wheels ]; then
    [ -x .venv/bin/python ] || python3 -m venv --without-pip .venv
    if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
        # A pip wheel can run itself to install pip.
        wheel=\$(ls .wheels/pip-*.whl | head -1)
        .venv/bin/python "\$wheel/pip" install -q --no-index --find-links .wheels pip
    fi
    .venv/bin/python -m pip install -q --no-index --find-links .wheels -r requirements.txt
    echo $REQ_HASH > .venv/.req-hash
    rm -rf .wheels
fi
# Restart by PID, never pkill -f: that pattern matches this ssh command's own command line
# and kills the shell before the next line runs (PLAN3 §11.7).
# Wait for the old uvicorn to let go of the port, or the new one fails to bind and
# run.sh loops on "address already in use" until it does.
if [ -f .run.lock ] && pid=\$(cat .run.lock 2>/dev/null) && kill -0 "\$pid" 2>/dev/null; then
    children=\$(pgrep -P "\$pid" || true)
    kill "\$pid" || true
    [ -n "\$children" ] && kill \$children 2>/dev/null || true
    for _ in \$(seq 30); do
        alive=0
        for c in \$pid \$children; do kill -0 "\$c" 2>/dev/null && alive=1; done
        [ \$alive -eq 0 ] && break
        sleep 0.5
    done
fi
setsid ./run.sh >/dev/null 2>&1 </dev/null &
REMOTE
    on_with "$STAGE/remote.sh" "$host" bash -s
done

sleep 4
deploy/status.sh "${HOSTS[@]}"
