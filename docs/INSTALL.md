# Installing and running

Three ways to run the service, from the smallest to the whole system.

| | What you get | Needs |
|---|---|---|
| [1. One process](#1-one-process) | Everything in one uvicorn: page, API, jobs, CV, saved sites | Python 3.12 |
| [2. docker compose](#2-docker-compose) | Frontend, gateway, three workers, CV service, database volume; each capped at 512 MB | Docker |
| [3. The lab's four containers](#3-the-labs-four-containers) | Gateway on sys1, workers on sys2-4 | ssh to 10.1.75.53 |

The demo region around the sample sheet (81.18-81.42 E, 21.13-21.37 N) works in all three
**with no internet at all**: its elevation tiles are committed in `data/tiles/`. Anywhere
else, tiles are fetched from the public AWS bucket on first use and cached on disk.

---

## 1. One process

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Open **http://localhost:8000**. Press **Try the sample area**: the sheet's rectangle is
drawn and analysed, and the pond, its catchment and its yield appear on the map in about a
second. Or search for a village, pick **Box** or **Polygon**, draw around it, and press
**Analyse this area**. To analyse a surveyed contour sheet instead, switch to
**I have a survey sheet** and drop the `.kml` or `.kmz` on the map.

The API documentation is at http://localhost:8000/docs.

```bash
curl -X POST localhost:8000/api/v1/analyzeArea -H 'Content-Type: application/json' \
     -d '{"bbox": [81.2814, 21.2398, 81.3126, 21.2636]}'
curl -F contour_map=@data/contours_1m.kml localhost:8000/api/v1/analyzeContour
```

Tests: `pip install pytest` then `pytest` (588 tests, about three minutes, no network).

## 2. docker compose

```bash
docker compose up -d --build
```

Open **http://localhost:5229**. Six containers:

```
                       ┌───────────────┐
  browser ── :5229 ──► │ frontend      │ nginx: the page, and /api → gateway
                       └──────┬────────┘
                       ┌──────▼────────┐
                       │ api (gateway) │ jobs · result cache · least-busy dispatch · saved sites ── pond-db volume
                       └──┬─────┬───┬──┘
               ┌──────────┘     │   └──────────┐
        ┌──────▼─────┐   ┌──────▼─────┐ ┌──────▼─────┐
        │ worker1    │   │ worker2    │ │ worker3    │ one analysis at a time each, own tile cache
        └──────┬─────┘   └──────┬─────┘ └──────┬─────┘
               └───────────┬────┴──────────────┘
                     ┌─────▼──────┐
                     │ cv         │ satellite imagery: existing water, land availability
                     └────────────┘
```

Every service is capped at 512 MB, the lab containers' limit. Useful knobs, as
environment variables when running `docker compose up`:

```bash
POND_JOBS_STRATEGY=round_robin docker compose up -d api     # the other dispatcher
POND_WORKERS=http://worker1:5000 docker compose up -d api   # one worker only
```

`python -m tools.scaling` runs the stress matrix against this stack and rewrites
[SCALING.md](SCALING.md). `docker compose down` stops it; add `-v` to also delete the
saved-sites database and the tile caches.

## 3. The lab's four containers

```bash
deploy/deploy.sh            # all four: code, venv, restart in role
deploy/deploy.sh sys3       # one host
deploy/status.sh            # health and real memory use of each
deploy/start-lb.sh          # optional: the Go balancer on sys1:6000, round-robin over the workers
```

| Host | ssh | Role | URL |
|---|---|---|---|
| sys1 | 2229 | gateway: page, jobs, dispatch, saved sites | http://10.1.75.53:5229 |
| sys2 | 2230 | worker | 172.17.0.31:5000 (bridge) |
| sys3 | 2231 | worker | 172.17.0.32:5000 |
| sys4 | 2232 | worker | 172.17.0.33:5000 |

The same `run.sh` runs everywhere; `deploy.sh` writes each host's `.env.role`, and on sys1
that sets `POND_JOBS_WORKERS` to the three workers' bridge addresses. Workers use port
5000 and nothing else: 4000 on sys2-4 is someone's chat app and 3000 on sys1 the lab's own
balancer. sys2-4 have no numpy and no `python3-venv`, so the venv is bootstrapped with
`get-pip.py`, which `deploy.sh` does on first run.

Two things learned the hard way, both in the scripts:

- Read memory from `memory.stat`'s `anon`, never `memory.current`, which counts page
  cache and understated the headroom on sys2-4 by more than 200 MB.
- Restart by PID, never `pkill -f run.sh` over ssh: the pattern matches the ssh command's
  own command line and kills the shell before the next line runs.

## Configuration

Every setting in `app/config.py` is overridable as `POND_<GROUP>_<FIELD>`. The Phase 3 ones:

```bash
POND_ELEVATION_NETWORK_ENABLED=false   # tiles from disk only (tests; an offline demo)
POND_ELEVATION_CELL_BUDGET=1000000     # the grid coarsens rather than exceed this
POND_ELEVATION_AOI_BUFFER=0.25         # margin analysed around a selection
POND_ELEVATION_MAX_AOI_AREA_KM2=100    # larger selections are a 422
POND_ELEVATION_OPENTOPO_KEY=...        # enables the OpenTopography alternate source
POND_JOBS_WORKERS=http://a:5000,...    # makes this process a gateway
POND_JOBS_STRATEGY=least_busy          # or round_robin
POND_JOBS_MAX_QUEUED=32                # beyond this, 503 busy
POND_CV_SERVICE_URL=http://cv:5000     # call a separate CV service instead of in-process
POND_STORE_PATH=.data/ponds.sqlite3    # saved analyses
POND_PLACES_ENABLED=false              # no place search (no calls to Nominatim)
```

To seed tiles for another region so its demo also runs offline:

```bash
python -m tools.prewarm_tiles --bbox 80.9,21.0,81.2,21.3 --zooms 11 12 13
```
