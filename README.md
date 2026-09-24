# Pond Catchment Analysis

Draw an area on a map. Get back where a village pond should go, the ground that drains
into it, and how much water that ground delivers in an average year, drawn on the map.
Or upload a surveyed contour sheet for the same answer at survey accuracy.

```bash
docker compose up -d --build        # then open http://localhost:5229 and press "Try the sample area"
```

```bash
curl -X POST http://localhost:5229/api/v1/analyzeArea -H 'Content-Type: application/json' \
     -d '{"bbox": [81.2814, 21.2398, 81.3126, 21.2636]}'
```

The Village Pond Planning System: Phase 1 was the high-level design, Phase 2 the contour
analysis ([report](Pond_Catchment_Analysis_Report.pdf)), and Phase 3, this, the complete
map-first product. The Phase 3 report is [docs/REPORT_PHASE3.md](docs/REPORT_PHASE3.md);
the API reference [docs/API.md](docs/API.md); installation [docs/INSTALL.md](docs/INSTALL.md);
the method and its evidence [docs/METHODOLOGY.md](docs/METHODOLOGY.md),
[docs/VALIDATION.md](docs/VALIDATION.md) and [docs/SCALING.md](docs/SCALING.md).

## What it does

1. **Choose the land.** Search for a village, then drag a box or click a polygon around
   it. The area is measured as you draw and checked against the service's limits before
   anything is sent.
2. **Heights, from free data.** SRTM elevation tiles (AWS terrain tiles, keyless) for the
   selection plus a 25% margin, so streams that cross the drawn line are counted in full.
   The demo region is committed with the repository and needs no network.
3. **The analysis,** unchanged from Phase 2: smooth, fill pits, route water downhill,
   rank sites by how much drains to them, keep them 3 m clear of any watercourse, trace
   each catchment, cross-check it on three more grids for an error bar.
4. **Water.** Ten years of daily rainfall for the site from Open-Meteo, SCS-CN runoff per
   rain day, and a stage-storage curve for the pond.
5. **On the map.** The pond, its catchment, the flow network, the alternatives, contour
   lines generated from the DEM, and from satellite imagery the existing ponds and the
   land that is built on or under water. Rainfall and storage charts, export as GeoJSON
   or PNG, print, and save the analysis to reopen later.

A progress bar reports the stage the analysis is actually in, because jobs run
asynchronously and report where they are. The sample area takes about half a second.

## How far to trust the free-data answer

The provided 1 m contour survey and the free SRTM tiles were run over the same ground
([VALIDATION.md](docs/VALIDATION.md)): the DEMs agree to 0.26 m RMSE, the recommended site
is in the same valley 400 m apart, and one of the map path's catchments traces the
survey's recommended one with IoU 0.53. The map path's catchment is 31% larger (87 ha
against 66 ha) because its grid is 17.8 m, not 3.1 m. Use the map path to find the site;
use a survey to design the pond.

## Under load

Measured on the docker-compose stack with every container capped at 512 MB, the lab's
limit ([SCALING.md](docs/SCALING.md)):

| workers | throughput | p95 |
|---|---|---|
| 1 | 1.34 req/s | 7.65 s |
| 2 | 2.10 req/s | 4.50 s |
| 3 | 3.54 req/s | 2.99 s |

Peak worker memory 276 MiB of 512. A repeated area is answered from the cache in 10 ms.
Overload is a `503 busy` the client can retry, and a selection too big for the memory is
a `422 aoi_too_large` before any work starts, never an out-of-memory kill.

## Architecture

```
browser ── frontend (nginx) ── gateway ─┬─ worker ─┐
                                        ├─ worker ─┼── cv (imagery)
                                        └─ worker ─┘
          saved sites: SQLite on the gateway's volume
```

One codebase, one image. What a process is depends on its environment: with
`POND_JOBS_WORKERS` set it is a gateway (job store, result cache, least-busy dispatch,
saved sites) and hands analyses to workers over their own job API; without it, it analyses
itself. On the lab containers the gateway is sys1 and the workers sys2-4.

| Path | What lives there |
|---|---|
| `app/core/` | The analysis. Phase 2's modules unchanged; Phase 3 added `raster_dem.py` (area to DEM), `contouring.py` (contours from a DEM) and `network.py` (the stream layer) |
| `app/providers/` | Elevation tiles and rainfall, each behind one interface with a cache |
| `app/pipeline.py` | `analyse` (a sheet) and `analyse_area` (an area): different front doors, one analysis |
| `app/jobs.py` | Async jobs, the result cache, the dispatcher |
| `app/cv/` | Satellite imagery: existing water and land availability, in-process or as its own service |
| `app/store.py` | Saved analyses |
| `app/routers/` | HTTP only: validation and error mapping |
| `static/index.html` | The page. One file, no build step, no CDN |
| `deploy/` | The four lab containers, and nginx for compose |
| `tools/` | Tile pre-warming, the load generator and the scaling matrix |
| `data/` | The sample contour sheet and the committed demo-region tiles |

## Endpoints

| | |
|---|---|
| `POST /api/v1/analyzeArea` | An area in (bbox, polygon or GeoJSON), the answer out |
| `POST /api/v1/jobs`, `GET /api/v1/jobs/{id}` | The same, asynchronously, with per-stage progress |
| `POST /api/v1/analyzeContour` | A contour sheet (KML/KMZ) in, the same answer out |
| `POST /api/v1/renderMap` | Either, drawn as a PNG |
| `POST /api/v1/contours` | Contour lines from a sheet, or generated for an area |
| `POST /api/v1/imagery/detectPonds` | Existing water bodies |
| `POST /api/v1/land/available` | Water, buildings and trees, and the share left free |
| `/api/v1/ponds` | Save, list, reopen, delete analyses; `ponds.geojson` for GIS |
| `GET /api/v1/places?q=` | Village search |
| `GET /api/v1/rainfall` | Ten years of rainfall for a point, by month and year |
| `GET /api/v1/cluster`, `GET /health` | Workers, queue, cache; liveness and limits |

Errors always come back as `{"status": "error", "code", "detail", "hint"}`. Full
reference: [docs/API.md](docs/API.md), or `/docs` on a running service.

## Running it

See [docs/INSTALL.md](docs/INSTALL.md). In short: `uvicorn app.main:app` for one process,
`docker compose up` for the whole system, `deploy/deploy.sh` for the lab's four containers.

The Phase 2 service is deployed at **http://10.1.75.53:5229** on `stu68_sys1`.
`deploy/deploy.sh` replaces it with the Phase 3 gateway at the same address and starts
workers on sys2-4.

## Tests

```bash
pytest
```

588 tests, about three minutes, no network: the live rainfall fetch is off, elevation
tiles come from the committed seed or from a formula, and imagery is synthesised with a
known answer. Among them: the analytic valley and the mass balance re-run through the
raster path, the two-path agreement with the survey, and the gateway dispatching to real
worker processes, routing around a dead one.

## Status

- [x] Phase 2 (0-12): contour analysis, API, demo page, deployment, report
- [x] 13 Elevation provider · 14 Raster DEM · 15 `analyzeArea` · 16 Contours from the DEM
- [x] 17 Two-path validation · 18 Area selection UI · 19 Async jobs · 20 Four-system deploy scripts
- [x] 21 Stress and scaling (measured on compose) · 22 Front-end completion · 23 Imagery (CV)
- [x] 24 Saved analyses · 25 docker-compose · 26 Docs
- [ ] The Phase 3 system deployed to the lab containers, and the scaling matrix re-run there
