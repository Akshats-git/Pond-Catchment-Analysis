# Installing and running

Three ways to run the service, from the smallest to the whole system.

| | What you get | Needs |
|---|---|---|
| [1. One process](#1-one-process) | Everything in one uvicorn: page, API, jobs, CV, saved sites | Python 3.12 |
| [2. docker compose](#2-docker-compose) | Frontend, gateway, three workers, CV service, database volume; each capped at 512 MB | Docker |
| [3. Four separate hosts](#3-four-separate-hosts) | Gateway on one host, a worker on each of the other three | ssh to the hosts |

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

Tests: `pip install pytest` then `pytest` (591 tests, about three minutes, no network).

## 2. docker compose

```bash
docker compose up -d --build
```

Open **http://localhost:5229**. Seven containers (the seventh, `otel`, is the Grafana
stack described below):

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

Every service is capped at 512 MB, the limit of the containers it was first deployed on. Useful knobs, as
environment variables when running `docker compose up`:

```bash
POND_JOBS_STRATEGY=round_robin docker compose up -d api     # the other dispatcher
POND_WORKERS=http://worker1:5000 docker compose up -d api   # one worker only
```

### Watching it

The `otel` container collects traces, metrics and logs from the other services. Open
**http://localhost:3000** (user `admin`, password `admin`) after `docker compose up`.

```bash
tools/generate_load.sh                       # sends a few varied analyses
curl -u admin:admin -H 'Content-Type: application/json' \
     -d @deploy/grafana/pond-dashboard.json http://localhost:3000/api/dashboards/db
```

- **Explore > Tempo** shows one trace per request, across the gateway, a worker and the CV service.
- **Explore > Loki** shows the logs. Try `{service_name="pond-worker"}`.
- **Dashboards > Pond Catchment - OpenTelemetry** shows the metrics.

Telemetry only runs when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Compose sets it. Without it
(one process, the tests, the Render deploy) nothing is sent. To turn it off in compose,
delete that line from `docker-compose.yml`. The `otel` image is about 2.7 GB and is not
part of the 512 MB limit.

If `otel` stays `unhealthy` after its image version changes, the `otel-data` volume was
written by the older release (its profiling store is not forward-compatible). Reset it
with `docker compose rm -sf otel && docker volume rm pondcatchmentanalysis_otel-data`,
then `docker compose up -d otel`; only past telemetry is lost.

`python -m tools.scaling` runs the stress matrix against this stack and rewrites
[SCALING.md](SCALING.md). `docker compose down` stops it; add `-v` to also delete the
saved-sites database and the tile caches.

## 3. Four separate hosts

```bash
deploy/deploy.sh            # all four: code, venv, restart in role
deploy/deploy.sh sys3       # one host
deploy/status.sh            # health and real memory use of each
deploy/start-lb.sh          # optional: the Go balancer on sys1:6000, round-robin over the workers
```

| Host | Role | Port |
|---|---|---|
| sys1 | gateway: page, jobs, dispatch, saved sites | 5229 |
| sys2-4 | worker | 5000, on the private bridge |

Host addresses and ssh ports live in `deploy/hosts.sh` and can be overridden from the
environment.

The same `run.sh` runs everywhere; `deploy.sh` writes each host's `.env.role`, and on sys1
that sets `POND_JOBS_WORKERS` to the three workers' bridge addresses. Workers use port
5000 because the other ports on those shared hosts were taken. sys2-4 have no numpy and
no `python3-venv`, so the venv is bootstrapped with `get-pip.py`, which `deploy.sh` does
on first run.

Two things learned the hard way, both in the scripts:

- Read memory from `memory.stat`'s `anon`, never `memory.current`, which counts page
  cache and understated the headroom on sys2-4 by more than 200 MB.
- Restart by PID, never `pkill -f run.sh` over ssh: the pattern matches the ssh command's
  own command line and kills the shell before the next line runs.

## Configuration

Every setting in `app/config.py` is overridable as `POND_<GROUP>_<FIELD>`. The ones you are most likely to change:

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
