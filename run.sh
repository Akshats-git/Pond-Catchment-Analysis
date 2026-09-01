#!/usr/bin/env bash
# Serve the Pond Catchment Analysis API, and keep serving it.
#
# Written for the lab container stu68_sys1, where the URL has to answer whenever someone
# decides to look rather than whenever it was last started by hand.
#
#   laptop -> 10.1.75.53:5229 -> container 172.17.0.30:5000 -> uvicorn (0.0.0.0:5000)
#
# Port 5000 is the container-side port the lab forwards to 5229. Bind 0.0.0.0, never
# 127.0.0.1, or the forward has nothing to reach.
#
# Start it detached so it outlives the ssh session that launched it:
#
#   setsid ~/PondCatchmentAnalysis/run.sh >/dev/null 2>&1 </dev/null &
#
# Safe to run again at any time: the lock below means a second copy exits rather than
# fighting the first for the port.

set -u
cd "$(dirname "$0")" || exit 1

LOG="server.log"
LOCK=".run.lock"
LOG_MAX_BYTES=$((20 * 1024 * 1024))

# ---------------------------------------------------------------------------- #
# One instance only.
# ---------------------------------------------------------------------------- #
# Without this, a second launch produces a second loop whose uvicorn cannot bind 5000,
# so it dies and respawns every two seconds forever, filling the log and leaving two
# supervisors racing over one port. `flock` makes a re-run a no-op instead.
exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
    echo "run.sh is already running (pid $(cat "$LOCK" 2>/dev/null || echo unknown))." >&2
    exit 0
fi
echo $$ >&9

# ---------------------------------------------------------------------------- #
# Settings this host needs.
# ---------------------------------------------------------------------------- #
# The container is capped at 512 MB. One analysis peaks near 300 MB and fits; the 4-grid
# ensemble peaks near 580 MB and does not.
#
#   default_ensemble=false  - an ordinary request does the analysis that fits.
#   allow_ensemble=false    - an explicit ensemble=true gets a 422 saying the host cannot
#                             do it, instead of taking the worker down with it.
#
# Concurrency is bounded in config.py (max_concurrent_analyses=1) rather than here,
# because it is a property of the code's memory use on any host, not of this container.
export POND_API_DEFAULT_ENSEMBLE=false
export POND_API_ALLOW_ENSEMBLE=false

# ---------------------------------------------------------------------------- #
# Serve, and restart if the kernel or anything else takes it out.
# ---------------------------------------------------------------------------- #
while true; do
    # An unbounded log on a shared 247 GB overlay is somebody else's outage. Truncated
    # rather than rotated: this log is a breadcrumb trail, not an audit record.
    if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt "$LOG_MAX_BYTES" ]; then
        printf '[%s] log passed %s bytes, truncated\n' "$(date -Is)" "$LOG_MAX_BYTES" > "$LOG"
    fi

    ./.venv/bin/python -m uvicorn app.main:app \
        --host 0.0.0.0 --port 5000 \
        --workers 1 --timeout-keep-alive 65 \
        >> "$LOG" 2>&1
    status=$?

    printf '[%s] uvicorn exited (%s), restarting in 2s\n' "$(date -Is)" "$status" >> "$LOG"
    sleep 2
done
