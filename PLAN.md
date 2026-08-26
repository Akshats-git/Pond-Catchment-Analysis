# Pond Catchment Analysis API — Implementation Plan

**Assignment:** Phase 2 — backend API that accepts a contour map (KML/KMZ), analyses the
terrain, identifies a suitable pond location, and returns catchment information as JSON.

**Parent design:** `HLD_Village_Pond_Planning_System_v1_basic.pdf` (Phase 1, submitted).
This plan implements the `POST /api/catchment` slice of that HLD and keeps its module
boundaries so Phases 3+ bolt on without rework.

**Sample input:** `contours_1m.kml` — 1,355 contour lines, 159,113 vertices, 32 levels,
267–298 m, 1 m interval, 3.24 km × 2.63 km ≈ **830.9 ha**, near Raipur (Chhattisgarh).

---

## 1. Rubric mapping

| Rubric item | Marks | Where it is earned |
|---|---|---|
| Working API URL | 5 | Phase 9 (route) + Phase 11 (deploy) |
| Catchment Analysis / Estimation | 10 | Phases 2–6, validated in Phase 5 |
| Report | 3 | Phase 12 |
| Code Reusability | 2 | `core/` + `providers/` seams, present from Phase 1 |

---

## 2. The catchment calculation (validated)

This is the core of the assignment. Six steps, each explainable in one sentence.

### Step 1 — Contours to elevation points
Every vertex of every contour line becomes `(x, y, z)`, where `z` is the contour's label.
> *"Each contour line is a line of constant height, so every point on it has a known height."*

### Step 2 — Points to a grid (DEM)
Delaunay-triangulate the points; linearly interpolate onto a square grid.
Grid resolution is **derived from the data**, never hard-coded:

```
mean contour spacing = mapped area / total contour length
                     = 8.309e6 m² / 663,914 m = 12.52 m
grid resolution      = spacing / 4 ≈ 3.1 m
```
> *"Flow algorithms need a grid of heights, so I fill in the gaps between contour lines."*

### Step 3 — Remove the stair-steps (do not skip)
Contour interpolation produces flat bands between contour lines — the surface is a
staircase, not a hillside. Gaussian smoothing with `σ = spacing / 8 = 1.56 m` removes them.
Moves the surface by at most **0.9 m**, i.e. less than one contour interval.
> *"Interpolating between contours makes flat steps; a light smooth turns the staircase back into a slope."*

**This step is worth 12% accuracy — see §3, Test A.**

### Step 4 — Fill the pits
Priority-flood (Barnes et al. 2014): flood inward from the map edges so no cell is a dead
end, adding a tiny `ε` slope so water keeps moving across flats.
> *"Water shouldn't get stuck in a hole that only exists because the data is imperfect."*

### Step 5 — Flow direction and accumulation (D8)
Each cell sends all its water to the steepest of its 8 neighbours:

```
S_i = (z_c − z_i) / d_i        d_i = res (4 sides), res·√2 (4 diagonals)
receiver(c) = argmax_i S_i
```

Then process cells from highest to lowest, adding each cell's count to its receiver.
Result = *flow accumulation* = how many cells drain through each cell.
> *"Every cell gives its water to its steepest neighbour; counting how much arrives tells me where the streams are."*

### Step 6 — Catchment = walk upstream
From the pond outlet, follow the receiver pointers **backwards** and collect every cell
that drains into it.

```
A_cell = res² · cos(φ_cell) / cos(φ₀)          ← latitude weighting
A_catchment = Σ A_cell   over the upstream mask
```
> *"The catchment is every cell whose water eventually flows through the pond."*

### Siting sits on top, and is catchment-first
1. Stream network = cells with upstream area ≥ 0.5% of the map (4.2 ha here).
2. Buildable = slope < 3%, not within 30 m of the map edge.
3. Rank by **catchment area**.
4. After each pick, remove that entire catchment from the pool → alternatives are
   independent sub-basins, not five points on one stream.

---

## 3. Validation evidence (already run — reproduce in Phase 5)

### Test A — synthetic valley with an exact analytic answer
Surface `z = 0.05·|x| + 0.01·y` over a 1 km × 1 km domain. Steepest descent is pure −x,
so the catchment of a channel point at `y = Y` is provably `1000 × (1000 − Y)` m².
Written out as a contour KML and read back through the identical pipeline:

| Grid | Y=250 | Y=500 | Y=750 |
|---|---|---|---|
| No smoothing (step 3 off) | −4.30% | 0.00% | **−12.79%** |
| **σ = 2.5 m, 5 m grid** | **0.00%** | **0.00%** | **0.00%** |
| σ = 2.5 m, 10 m grid | 0.00% | 0.00% | 0.00% |

Root cause of the −12.79%: the 7 grid rows above the outlet contributed **1 cell each
instead of 200** — hillslope water ran *along* a flat stair-step band and entered the
stream *below* the outlet. Smoothing fixes it exactly.

### Test B — mass balance on the real map
Every cell drains to exactly one outlet, so basin areas must sum to the mapped area:

```
sum of all 594 basin areas = 8.3091 km²
mapped area                = 8.3091 km²      difference: 0.000000%
```

### Test C — resolution ensemble
Delineate each site on three grids (5.0 / 3.5 / 2.5 m). Agreement ⇒ trustworthy;
disagreement ⇒ flag as low-confidence. This is what turns a bare number into a number
with an error bar.

### Results on the provided contour map

| # | Outlet (lon, lat) | Catchment | Ensemble | Edge contact | Relief | Verdict |
|---|---|---|---|---|---|---|
| 1 | 81.286465, 21.240094 | 395.4 ha | 423.5 ± 17.6 ha | 1.3% | 27.0 m | best — 48% of map |
| 2 | 81.293549, 21.263343 | 101.8 ha | 102.8 ± 1.5 ha | 11.6% | 14.4 m | good, partly clipped |
| 3 | 81.284248, 21.262484 | 37.1 ha | 37.3 ± 0.1 ha | 3.9% | 17.7 m | 7,757 m³ natural storage |
| 4 | 81.297453, 21.240094 | 35.7 ha | **14.1 ± 15.4 ha** | 0.9% | 10.0 m | **rejected — unstable** |
| 5 | 81.312393, 21.259544 | 30.4 ha | 31.6 ± 0.8 ha | 4.4% | 11.0 m | good |

**Edge contact** = fraction of the catchment's perimeter lying on the edge of *valid data*
(no-data or map border). >15% ⇒ the true catchment extends off the map and the reported
area is a **lower bound**.

---

## 4. Runoff: SCS-CN is an event model

Applying SCS-CN to an *annual* rainfall total gives a 92% runoff coefficient — not physical.
It must be applied per rain day and summed.

```
S  = 25400/CN − 254  (mm)          Ia = 0.2·S
Q  = (P − Ia)² / (P − Ia + S)      for P > Ia, else 0
V  = (Q / 1000) · A_catchment      (m³)
```

| Method | Runoff | Coefficient | Top-site volume |
|---|---|---|---|
| Annual total as one event ❌ | 1104 mm | 92% | 4,365,000 m³/yr |
| Per-day over 55 rain days ✅ | 177 mm | 15% | 694,000 m³/yr |

6× difference. 11–19% is the realistic range for this terrain. Phase 2 uses a documented
daily distribution; Phase 3 swaps in real daily rainfall from Open-Meteo behind the same
`RainfallProvider` interface.

---

## 5. Technology stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async, auto Swagger at `/docs`, Pydantic validation. Matches HLD. |
| Numerics | numpy + scipy **only** | No GDAL/pysheds/rasterio ⇒ small image, deploys on a free tier. |
| Parsing | stdlib `xml.etree`, `zipfile` | KML is XML; KMZ is a zip. No extra dependency. |
| Frontend demo | Leaflet (CDN-free, single file) | Makes the live URL self-demonstrating for the grader. |
| Deploy | Docker → Render free tier | Public HTTPS URL, no card required. |

`requirements.txt`: `fastapi`, `uvicorn[standard]`, `python-multipart`, `numpy`, `scipy`, `pydantic`.

**On pysheds:** the HLD names it. We implement the same D8 methodology directly so the
service stays deployable. `TerrainEngine` is an interface — swapping in pysheds later
touches one file. Note this in the report.

---

## 6. Repository layout

```
pond-catchment-api/
├── app/
│   ├── main.py                  FastAPI app, CORS, error handlers
│   ├── config.py                every tunable, env-overridable — no magic numbers in code
│   ├── pipeline.py              wires the stages; the only orchestration point
│   ├── routers/
│   │   └── analyze.py           the route — validation + orchestration only
│   ├── schemas/
│   │   ├── requests.py
│   │   └── responses.py
│   ├── core/
│   │   ├── kml_parser.py        ← generalisation surface
│   │   ├── projection.py        lon/lat ↔ local metric ENU
│   │   ├── dem_builder.py       interpolation, auto-resolution, smoothing
│   │   ├── terrain.py           fill / D8 / accumulation / slope
│   │   ├── catchment.py         upstream mask, area, edge contact, ensemble
│   │   ├── pond_siting.py       stream threshold → rank → NMS
│   │   ├── hydrology.py         SCS-CN (event-based), stage-storage, Kirpich
│   │   └── geojson.py           mask → polygon, simplification
│   └── providers/
│       ├── rainfall.py          constant now; Open-Meteo in Phase 3
│       └── elevation.py         KML now; DEM raster/API in Phase 3
├── tests/
│   ├── fixtures/                synthetic + structural-variant KMLs
│   ├── test_parser.py
│   ├── test_dem.py
│   ├── test_catchment_analytic.py   ← the marks-winning test
│   ├── test_massbalance.py
│   └── test_api.py
├── static/index.html            Leaflet demo page
├── data/contours_1m.kml
├── docs/
│   ├── API.md
│   └── METHODOLOGY.md
├── requirements.txt
├── Dockerfile
├── render.yaml
├── README.md
└── REPORT.md
```

---

## 7. Build phases

One commit per phase. Each phase is independently testable and leaves the repo working.

| # | Phase | Commit message | Est. |
|---|---|---|---|
| 0 | Scaffold | `chore: scaffold repo, deps, sample data` | 20 m |
| 1 | KML/KMZ parser | `feat: KML/KMZ contour parser with elevation strategies` | 1 h |
| 2 | Projection + DEM | `feat: local metric projection and contour-interpolated DEM` | 1 h |
| 3 | Terrain engine | `feat: pit filling, D8 flow direction, flow accumulation` | 1.5 h |
| 4 | Catchment | `feat: upstream catchment delineation with area and edge contact` | 1 h |
| 5 | **Validation** | `test: analytic catchment validation and mass balance` | 1.5 h |
| 6 | Pond siting | `feat: catchment-ranked pond site selection` | 1 h |
| 7 | Hydrology | `feat: event-based SCS-CN runoff and stage-storage capacity` | 1 h |
| 8 | GeoJSON | `feat: GeoJSON export of catchment, outlet and flow path` | 45 m |
| 9 | API route | `feat: POST /analyzeContour endpoint with structured errors` | 1.5 h |
| 10 | Demo page | `feat: Leaflet demo page for uploading and viewing results` | 45 m |
| 11 | Deploy | `chore: Dockerfile and Render deployment config` | 45 m |
| 12 | Docs | `docs: report, API reference and methodology` | 1.5 h |

---

### Phase 0 — Scaffold
`git init` (not currently a repo). Create the tree, `requirements.txt`, `.gitignore`
(`__pycache__`, `.venv`, `*.npy`), `README.md` stub. Copy `contours_1m.kml` into `data/`.
Create `config.py` with every constant from this plan as a named, documented default.

**Accept:** `pip install -r requirements.txt` succeeds in a fresh venv.

---

### Phase 1 — KML/KMZ parser
`app/core/kml_parser.py`. This is where generalisation lives.

Elevation resolved by a **cascade**, first strategy that yields a consistent numeric field wins:
1. Z-coordinate of vertices (3D KML)
2. `<ExtendedData>/<SimpleData name=…>` matching `elev|elevation|level|contour|height|z|alt|value`
3. `<Placemark><name>` parsed as a number ← **the sample uses this**
4. Enclosing `<Folder><name>` (e.g. `contours_1.0m`)

Geometry: `LineString`, `LinearRing`, `Polygon` outer/inner rings, `MultiGeometry`.
KMZ: `zipfile`, first `.kml` member. Namespace-agnostic tag matching (strip `{ns}`).

Returns `ContourSet(points: ndarray[N,2], elevations: ndarray[N], metadata)` with
`elevation_source`, `interval`, `bbox`, `line_count`, `vertex_count`.

> ⚠ **Ignore the `labels` folder.** The sample has 1,355 duplicate `Point` placemarks
> carrying the same elevations. They add no surface information; use them only as a
> consistency check.

> ⚠ Detect feet-vs-metres from the contour interval and **flag it — do not silently convert**.

**Accept:** parses `contours_1m.kml` → 1,355 lines, 159,113 vertices, 32 levels, 267–298 m.
Structural-variant fixtures (Z-coords, ExtendedData, folder-name, KMZ, Polygon) all parse.

---

### Phase 2 — Projection + DEM builder
`app/core/projection.py` — equirectangular ENU about the dataset centroid:

```
x = (λ − λ₀)·111320·cos(φ₀)        y = (φ − φ₀)·110540
```
Sub-metre over 3 km, zero dependencies, exactly invertible. Behind a `Projection`
interface so Phase 3 can drop in pyproj/UTM.

`app/core/dem_builder.py` — `LinearNDInterpolator` over all vertices; auto-resolution from
mean contour spacing (§2 Step 2), clamped to [2, 20] m, request-overridable; then the
smoothing of §2 Step 3.

> ⚠ **Normalised (NaN-aware) Gaussian.** Fill invalid cells with **0**, smooth, and divide
> by the smoothed validity mask. Filling with the mean and dividing by the valid-cell
> weight inflates edge cells — it produced a 357 m peak on a map whose true maximum is
> 298 m. Assert the smoothed range stays within the raw range.

```python
valid = (~nodata).astype(float)
num   = gaussian_filter(np.where(nodata, 0.0, dem), sigma)
den   = gaussian_filter(valid, sigma)
out   = np.where(den > 1e-6, num / den, np.nan)
```

**Accept:** 8.309 km² mapped area, ~3% no-data, elevation range exactly 267–298 m,
`max|smoothed − raw| < 1 contour interval`.

---

### Phase 3 — Terrain engine
`app/core/terrain.py` — three functions, all validated in the spikes:

- `fill_depressions(dem, nodata, eps)` — priority-flood with heapq, seeded from the array
  border **and** every cell adjacent to no-data.
- `d8_receivers(filled, nodata, res)` — vectorised over the 8 neighbour shifts,
  distance-weighted slope, returns a flat receiver index per cell.
- `flow_accumulation(rec, filled, nodata)` — sort cells by descending filled elevation,
  single pass adding each cell's total to its receiver.

**Accept:** on the real map at 5 m — fill 1.9 s, ~157,766 max accumulation, and the
mass-balance check of Phase 5 passes.

---

### Phase 4 — Catchment delineation
`app/core/catchment.py`:

- `upstream_mask(rec, outlet, shape)` — reverse-D8 traversal via a CSR-style donor index
  (`argsort` on receivers + `searchsorted`), then an explicit stack. No recursion.
- `catchment_area(mask, cell_area)` — latitude-weighted sum (§2 Step 6).
- `edge_contact_ratio(mask, nodata)` — fraction of the one-cell perimeter collar lying on
  no-data **or** the array border.
- `snap_outlet(lon, lat, acc, radius)` — snap to maximum accumulation within `3 × contour
  spacing`; return the snap distance.
- `ensemble_area(lon, lat, grids)` — delineate on 5.0 / 3.5 / 2.5 m; return mean, std,
  and a `high/medium/low` confidence from the spread.

> ⚠ **Edge contact must test `nodata | border`, not just the array border.** 3% of the grid
> is outside the contour hull, so a catchment can run off the mapped area without ever
> touching row 0. Testing only the border wrongly labelled the 395 ha basin "complete".

> ⚠ **Clamp every window slice** (`max(0, y-r)`, `min(ny, y+r+1)`) — sites near the south
> edge crash an unclamped `argmax`.

> ⚠ Index rounding is `int(round((v - origin) / res))` — keep the division **inside**
> `round`. Getting this wrong silently returns a catchment from the wrong cell.

**Accept:** site 1 → 395.4 ha at 5 m, 1.3% edge contact; ensemble 423.5 ± 17.6 ha.

---

### Phase 5 — Validation suite (the marks-winning phase)
`tests/test_catchment_analytic.py` + `tests/fixtures/make_synthetic.py`.

1. **Analytic test** — generate the `z = 0.05|x| + 0.01y` valley, write it as a contour
   KML, run the full pipeline, assert error < 1% at Y = 250 / 500 / 750 on 5 m and 10 m
   grids. Also assert the *unsmoothed* run is worse — this is the evidence that justifies
   the smoothing step in the report.
2. **Mass balance** — sum of all basin areas equals mapped area to < 0.01%.
3. **Sub-tile test** — clip the sample KML to a random quadrant, re-run, confirm a valid
   site is found in *that* quadrant (proves nothing is hard-coded).
4. **Structural variants** — the Phase 1 fixtures, end to end.

**Accept:** `pytest` green. Save the analytic table as `docs/METHODOLOGY.md` Table 1.

---

### Phase 6 — Pond siting
`app/core/pond_siting.py` — the four steps of §2. Return top-N with score components
(`upstream_area`, `slope`, `depression_depth`, `relative_elevation`) so the response can
explain *why* each site was chosen.

> ⚠ **Do not rank by percentile-ranked flow accumulation.** Accumulation is heavily skewed;
> percentile ranking let 0.7 ha hollows score 0.98 alongside a 320 ha valley. Use an
> absolute stream threshold (0.5% of mapped area), then rank by catchment area.

> ⚠ **Suppression must remove the whole upstream catchment**, not a square window.
> A square window returned five points strung along the same stream (391, 361, 215, 202,
> 179 ha — all nested).

**Accept:** reproduces the §3 results table, including site 4 flagged low-confidence.

---

### Phase 7 — Hydrology
`app/core/hydrology.py`:

- `scs_cn_runoff(daily_rainfall, CN)` — **per day, then summed** (§4).
- `stage_storage(dem, mask, outlet)` — fill the depression to its spill elevation and
  integrate depth → real `(depth, area, volume)` triples, not an assumed shape.
  Report the frustum formula `V ≈ (d/3)(A_top + A_bot + √(A_top·A_bot))` as a cross-check.
- `time_of_concentration(L, H)` — Kirpich.
- `fill_ratio` = annual runoff ÷ pond capacity, with a plain-English assessment.

`app/providers/rainfall.py` — `RainfallProvider` returning a documented default daily
series (1200 mm over 55 rain days, gamma-distributed). Phase 3 swaps in Open-Meteo.

> ⚠ Sites on a channel have **zero natural depression storage** — the pond is excavated.
> Compute excavated storage from the target depth and local terrain, not from fill depth.

**Accept:** top site → ~177 mm runoff, ~694,000 m³/yr, coefficient in 11–19%.

---

### Phase 8 — GeoJSON export
`app/core/geojson.py` — marching-squares boundary trace of the catchment mask (matplotlib
`contour` at level 0.5, or a hand-rolled tracer), grid → lon/lat, Douglas–Peucker
simplification. Emit a `FeatureCollection`: catchment polygon, pond footprint, outlet
point, longest flow path, alternate sites.

**Accept:** output loads in geojson.io and overlays correctly on satellite imagery.

---

### Phase 9 — API route
`app/routers/analyze.py`, `app/main.py`.

```
POST /api/v1/analyzeContour        (alias: POST /api/v1/findCatchment)
Content-Type: multipart/form-data

file             required   .kml or .kmz
grid_resolution  optional   metres; default auto (spacing/4)
top_n            optional   default 3
lat, lon         optional   explicit pour point, overrides auto-siting
curve_number     optional   default 75
rainfall_mm      optional   default 1200
rain_days        optional   default 55
target_depth_m   optional   default 3.0
ensemble         optional   default true (3 grids; ~12 s vs ~4 s)
```

Response (abridged — values shown as `...` are computed at runtime; the concrete
figures below are the measured results from §3):

```jsonc
{
  "status": "ok",
  "input": { "filename": "...", "contour_count": 1355, "vertex_count": 159113,
             "elevation_source": "placemark_name", "interval_m": 1.0,
             "elevation_range_m": [267, 298], "mapped_area_ha": 830.9, "bbox": [...] },
  "dem": { "resolution_m": 3.1, "smoothing_sigma_m": 1.56, "shape": [...],
           "nodata_fraction": 0.03 },
  "recommended_site": {
    "rank": 1,
    "location": { "lat": 21.240094, "lon": 81.286465 },
    "snap_distance_m": ...,
    "why": ["largest upstream area (48% of map)", "low slope", "27 m relief"],
    "catchment": {
      "area_ha": 395.4,
      "area_uncertainty_ha": 17.6,
      "confidence": "high",
      "edge_contact_pct": 1.3,
      "is_lower_bound": false,
      "relief_m": 27.0,
      "longest_flow_path_m": ...,
      "time_of_concentration_min": ...,
      "method": "D8 steepest-descent on contour-interpolated DEM",
      "grid_resolutions_m": [5.0, 3.5, 2.5]
    },
    "storage": { "capacity_m3": ..., "surface_area_m2": ..., "max_depth_m": 3.0,
                 "stage_storage": [...] },
    "runoff": { "method": "SCS-CN, event-based", "curve_number": 75,
                "rainfall_mm": 1200, "rain_days": 55,
                "runoff_depth_mm": 177, "runoff_coefficient": 0.15,
                "annual_runoff_m3": 694000, "fill_ratio": ...,
                "assessment": "catchment yields far more runoff than the pond can hold" }
  },
  "alternative_sites": [ ... ],
  "geojson": { "type": "FeatureCollection", "features": [ ... ] },
  "warnings": ["Depression depths quantised to the 1.0 m contour interval"],
  "timing_ms": { "parse": ..., "dem": ..., "flow": ..., "total": ... }
}
```

Errors are structured (`code`, `detail`, `hint`): `400` unparseable / no contours /
no elevations resolvable, `413` too large, `422` bad parameters.
Also `GET /health`, `GET /docs` (auto Swagger — satisfies the API-documentation rubric line).

**Accept:** `curl -F file=@data/contours_1m.kml localhost:8000/api/v1/analyzeContour`
returns the §3 results.

---

### Phase 10 — Demo page
`static/index.html` — drag-and-drop upload, Leaflet map, draws returned GeoJSON over
OSM/Esri satellite tiles, side panel with the numbers. Single file, no build step.

**Accept:** upload the sample, see the catchment polygon over satellite imagery.

---

### Phase 11 — Deployment
`Dockerfile` (python:3.12-slim, `pip install -r requirements.txt`, uvicorn on `$PORT`),
`render.yaml`. Deploy to Render free tier.

> ⚠ Free tier sleeps — cold start ~30 s. Note it in the report and keep a `curl`
> transcript + screenshots so the demo does not depend on the tier being awake.
> Cap `grid_resolution` server-side so a request cannot exhaust 512 MB RAM.

**Accept:** public HTTPS URL responds to `/health` and a real upload.

---

### Phase 12 — Documentation
- `README.md` — what it is, quickstart, live URL, one-command Docker run.
- `docs/API.md` — endpoints, every parameter, every response field, error codes, `curl`
  examples.
- `docs/METHODOLOGY.md` — §2 and §3 of this plan, with the analytic table as Table 1.
- `REPORT.md` (→ PDF) — GitHub link, live API URL, catchment approach, demonstration on
  the provided map with figures, API documentation, extensibility to Phase 3.

**Figures to generate:** DEM hillshade, flow accumulation (log scale), catchment overlay
for the top site, stage–storage curve, analytic-validation table.

---

## 8. Extensibility to Phase 3 (the "Code Reusability" marks)

| Phase 3 requirement | Seam already in place |
|---|---|
| DEM from an elevation API instead of KML | `providers/elevation.py` — pipeline takes a DEM, not a KML |
| Real rainfall | `providers/rainfall.py` — `RainfallProvider` interface |
| Larger regions / different CRS | `Projection` interface — swap ENU for UTM |
| Faster or GPU flow routing | `TerrainEngine` interface — swap in pysheds |
| Persisting results | `pipeline.py` returns a plain dataclass, trivially serialisable |
| Land-availability masks (OpenCV service) | siting takes an optional `exclusion_mask` argument |

---

## 9. Anti-hard-coding guarantees

Enforced, not asserted:
- Grid resolution, smoothing σ and stream threshold are all **derived from the input**.
- No latitude, longitude or elevation literal exists outside `data/` and `tests/`.
- The sub-tile test (Phase 5.3) proves results follow the input, not the sample.
- Every remaining constant lives in `config.py` with a documented rationale.

---

## 10. Viva cheat sheet

| Question | Answer |
|---|---|
| How do you get a surface from contour lines? | Triangulate the contour vertices and interpolate linearly between them. |
| Why smooth the DEM? | Interpolating between contours creates flat steps; without smoothing, catchment area was off by up to 12.8% on a case with a known exact answer. |
| Why fill pits? | So water isn't trapped in hollows that are artefacts of the data rather than real. |
| What is D8? | Each cell sends all its water to whichever of its 8 neighbours is steepest downhill. |
| How do you compute the catchment? | Follow those flow arrows backwards from the pond and collect every cell that drains into it. |
| How do you know the area is right? | A synthetic valley with a provable answer reproduces it to 0.00%, and all basin areas sum to the mapped area to 0.000000%. |
| Why an error bar? | I run it at three grid resolutions; if they disagree, the site is flagged low-confidence. Site 4 was rejected this way. |
| Why is your runoff coefficient 15%? | SCS-CN is an event model — applied per rain day and summed. Applying it to an annual total gives an impossible 92%. |
| What are the limitations? | Depression depth is quantised to the 1 m contour interval; catchments touching the map edge are lower bounds; DEM accuracy is bounded by contour spacing (12.5 m here). |

---

## 11. Pitfalls already found (do not re-discover)

1. The `labels` folder duplicates all 1,355 elevations as points — ignore it.
2. NaN-aware Gaussian must fill with **0** and divide by the smoothed mask, else edge
   cells inflate (produced a 357 m peak on a 298 m map).
3. Edge-contact test must use `nodata | border`, not the array border alone.
4. Outlet snap radius must scale with contour spacing (~3×), not a fixed 30 m — the
   channel shifts ~90 m between grid resolutions.
5. SCS-CN applied to an annual total overestimates runoff ~6×.
6. Percentile-ranked accumulation is degenerate for siting — use an absolute threshold.
7. Suppression must remove the whole catchment, not a square window.
8. Keep the division inside `round`: `int(round((v - origin) / res))`.
9. Clamp all window slices near map edges.

---

## 12. Estimated total

~13 hours across 13 commits. Phases 1–5 (the catchment calculation and its proof) are
~6 hours and carry 10 of the 20 marks — build and validate those before anything else.
