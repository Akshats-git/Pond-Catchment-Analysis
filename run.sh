#!/usr/bin/env bash
# Serve the Pond Catchment Analysis API inside stu68_sys1.
#
# Port 5000 is the container-side port that the lab forwards to 10.1.75.53:5229.
# Bind 0.0.0.0, never 127.0.0.1, or the forward has nothing to reach.
#
# The container has a 512 MB memory cap. The 4-grid ensemble peaks over that and gets
# OOM-killed, so it is off by default here; a single grid peaks near 300 MB and fits.
# The loop restarts uvicorn if the kernel does kill it, so the URL stays up.
cd "$(dirname "$0")" || exit 1

export POND_API_DEFAULT_ENSEMBLE=false

while true; do
    ./.venv/bin/python -m uvicorn app.main:app \
        --host 0.0.0.0 --port 5000 \
        --workers 1 --timeout-keep-alive 65 \
        >> server.log 2>&1
    echo "[$(date -Is)] uvicorn exited ($?), restarting in 2s" >> server.log
    sleep 2
done
