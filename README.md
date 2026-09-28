# Pond Catchment Analysis

Draw a box on a map. Get back where to dig a village pond, the ground that drains into
it, and how much water it'll hold in an average year — all overlaid on the map.

**[Try the live demo →](https://pond-catchment-analysis-vt7g.onrender.com)**
*(free hosting, so it naps after 15 minutes idle — give it ~30s to wake up)*

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Akshats-git/Pond-Catchment-Analysis)

![Pond Catchment Analysis screenshot](docs/figures/phase3_area_result.jpg)

## What it does

- **Pick the land** — search a village, then drag a box or draw a polygon around it.
- **Free elevation** — SRTM terrain tiles fill in the heights, no upload needed. Have a
  surveyed contour sheet instead? Upload the KML/KMZ for the same answer at survey accuracy.
- **Finds the pond site** — routes water downhill across the terrain, ranks locations by
  how much drains into them, and keeps a safe distance from watercourses.
- **Estimates the water** — ten years of real rainfall for that spot, run through a
  standard runoff model, turned into a fill curve for the pond.
- **Shows the work** — catchment boundary, flow paths, contours, and (from satellite
  imagery) existing ponds and buildable land, all on the map. Export as GeoJSON or PNG,
  save an analysis, reopen it later.

A sample area near Raipur, Chhattisgarh is one click away and resolves in about half a second.

## Running it locally

```bash
docker compose up -d --build
# open http://localhost:5229 and press "Try the sample area"
```

or hit the API directly:

```bash
curl -X POST http://localhost:5229/api/v1/analyzeArea -H 'Content-Type: application/json' \
     -d '{"bbox": [81.2814, 21.2398, 81.3126, 21.2636]}'
```

Full setup instructions, including a single-process dev server, are in
[docs/INSTALL.md](docs/INSTALL.md).

## How it works

```
browser ── frontend (nginx) ── gateway ─┬─ worker ─┐
                                        ├─ worker ─┼── cv (imagery)
                                        └─ worker ─┘
          saved sites: SQLite on the gateway's volume
```

One codebase, one Docker image — whether a process is a gateway, a worker, or the
imagery service is decided entirely by its environment variables. Run it with nothing
set and it just analyzes itself, which is what the live demo does.

| Path | What lives there |
|---|---|
| `app/core/` | The analysis: DEM building, hydrology, catchment delineation, pond siting, contouring |
| `app/providers/` | Elevation tiles and rainfall, each behind a cached interface |
| `app/cv/` | Satellite imagery — existing water bodies and buildable land |
| `app/pipeline.py` | Ties it together: a contour sheet or a map area, same answer out |
| `app/jobs.py` | Async jobs with per-stage progress, a result cache, and dispatch across workers |
| `static/index.html` | The whole front end — one file, no build step |
| `docs/` | API reference, methodology, validation against a real survey, and the load-test results |

Full endpoint reference: [docs/API.md](docs/API.md), or `/docs` on any running instance.

## How accurate is it?

The free SRTM elevation was checked against a real 1 m contour survey of the same
ground ([docs/VALIDATION.md](docs/VALIDATION.md)): the two DEMs agree to 0.26 m RMSE,
and both recommend a pond in the same valley, 400 m apart. Good enough to find the
site — for construction, still survey it.

## Tests

```bash
pytest
```

588 tests, no network required — elevation and rainfall are stubbed with known answers,
so a run takes about three minutes and reproduces the same result every time.

## More

- [docs/REPORT_PHASE3.md](docs/REPORT_PHASE3.md) — the written report for this system, or
  [docs/report/](docs/report/) for the full technical report as LaTeX source and PDF
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — the hydrology and siting method
- [docs/SCALING.md](docs/SCALING.md) — throughput and memory under load
