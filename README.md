# Pond Catchment Analysis API

Send a contour map as KML or KMZ. Get back a village pond site, the ground that drains
into it, and how much water that ground delivers in an average year.

Phase 2 of the Village Pond Planning System. The Phase 1 high-level design is in
[references/](references/). The full implementation plan, with the validated catchment
methodology, is in [PLAN.md](PLAN.md), and the evidence is in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## How it works

1. **Contours to points.** Every vertex of every contour line is a known `(x, y, z)`.
2. **Points to a grid.** Delaunay triangulation, then linear interpolation onto a square
   grid whose cell size comes from the mean contour spacing.
3. **Smooth.** Raw interpolation leaves flat stair steps instead of a hillside. A
   NaN-aware Gaussian takes them out. Worth up to 12.8% of the catchment area against a
   valley whose answer can be worked out on paper.
4. **Fill the pits,** so water is never trapped in a hole the data invented.
5. **Route the water.** Each cell drains to its steepest neighbour. Add up what passes
   through.
6. **Trace the catchment** by walking the flow arrows backwards from the outlet.

## Where the pond goes

Siting starts from the drainage network. Find the streams, keep buildable low-slope
ground, rank by how much drains into each spot, and remove each pick's whole catchment so
the alternatives are separate basins.

One rule is worth reading twice. Ranking by catchment area alone asks for the cell that
the most water passes through, and on a sheet with a river across it the answer is the
river. So any channel already draining more than 150 ha counts as a watercourse, and a
site has to stand 3 m above the one it drains into.

Checked against the OpenStreetMap water layer over the sample sheet, that takes candidate
ground standing in the river from 310 cells of 2,413 down to none. The old top site sat in
the middle of the Shivnath. See [docs/METHODOLOGY.md §3](docs/METHODOLOGY.md).

## Rainfall

Leave `rainfall_mm` out of the request and the service reads ten years of daily rainfall
for the chosen site from [Open-Meteo](https://open-meteo.com/), which is free and needs no
key. Send a figure of your own and it is used instead. If the weather service cannot be
reached, a documented regional climatology answers and the response says so, so an
analysis is never blocked by the weather.

`GET /api/v1/rainfall?lat=&lon=` exposes the same feed on its own, which is what the demo
page calls to fill in the rainfall box.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000` for the demo page. Drop the sample map on it and the
catchment is drawn over satellite imagery. Click any candidate to highlight the ground it
covers. Interactive API documentation is at `http://localhost:8000/docs`.

```bash
curl -F file=@data/contours_1m.kml http://localhost:8000/api/v1/analyzeContour
```

`POST /api/v1/analyzeContour`, also reachable as `findCatchment`, takes the contour file
plus optional `grid_resolution`, `top_n`, `lat`, `lon`, `curve_number`, `rainfall_mm`,
`rain_days`, `target_depth_m` and `ensemble`. It returns the recommended site, its
catchment with an error bar, the stage-storage curve, the yearly runoff, and a GeoJSON
`FeatureCollection` that loads straight into geojson.io. Errors always come back as
`{"status": "error", "code", "detail", "hint"}`.

## Layout

| Path | What lives there |
|---|---|
| `app/core/` | The analysis: parsing, projection, grid, flow routing, catchment, siting, hydrology |
| `app/pipeline.py` | Wires the stages together. The only orchestration point |
| `app/providers/` | Swappable data sources. Rainfall lives here |
| `app/routers/` | HTTP surface only: validation and error mapping |
| `app/config.py` | Every tunable, documented and overridable from the environment |
| `tests/` | Analytic validation, mass balance, structural variants |
| `static/` | `index.html`, the demo page. One file, no build step, no CDN |
| `data/` | `contours_1m.kml`, the provided sample sheet |
| `docs/` | Methodology and the evidence behind the numbers |

## Configuration

Every setting is overridable from the environment with a `POND_` prefix. The ones worth
knowing about:

```bash
POND_SITING_TRUNK_DRAINAGE_AREA_HA=150     # a channel over this is a watercourse
POND_SITING_MIN_HEIGHT_ABOVE_TRUNK_M=3     # freeboard a site must keep above it
POND_RAINFALL_ENABLED=false                # skip the live rainfall fetch entirely
POND_RAINFALL_YEARS=10                     # years of daily records to average
POND_API_DEFAULT_ENSEMBLE=false            # one grid instead of four, roughly 3x faster
```

## Tests

```bash
pytest
```

388 tests, about 105 seconds. No test needs the network: `tests/conftest.py` switches the
live rainfall fetch off, and the Open-Meteo provider is exercised against a payload built
in the test.

## Status

Under construction, one commit per phase (see PLAN.md §7).

- [x] Phases 0 to 10, scaffold through the demo page
- [ ] Phases 11 and 12, deployment and report
