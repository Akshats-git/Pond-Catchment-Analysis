# API reference

Base URL of the running service: **`http://10.1.75.53:5229`**
Interactive documentation, generated from the same models that serialise the responses:
**`http://10.1.75.53:5229/docs`**

Three endpoints, plus an alias. Everything is `multipart/form-data` in and JSON out.

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/v1/analyzeContour` | Contour map in, pond site + catchment + yield out |
| `POST` | `/api/v1/findCatchment` | Alias of the above, identical signature |
| `GET` | `/api/v1/rainfall` | Ten years of daily rainfall for one point |
| `GET` | `/health` | Liveness. Does no work on purpose |

---

## POST /api/v1/analyzeContour

Send a contour map. Get back where the pond goes, what drains into it, and how much water
that ground delivers in an average year.

### Request

`multipart/form-data`. Only `file` is required; every other field has a derived or
configured default, and the defaults are what the sample run in
[REPORT.md](../REPORT.md) used.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `file` | file | — | The contour map, `.kml` or `.kmz`. Contour **lines**, not point labels. Max 64 MB |
| `grid_resolution` | float | derived | Grid cell size in metres. Left out, it is the mean contour spacing ÷ 4, clamped to 2–20 m |
| `top_n` | int | 3 | How many separate basins to return |
| `lat` | float | — | Latitude of a pour point you have chosen yourself. Send with `lon` |
| `lon` | float | — | Longitude of that point. Sending both skips the site search entirely |
| `curve_number` | float | 75.0 | SCS curve number, 30–98. How readily the ground sheds rain |
| `rainfall_mm` | float | live | Yearly rainfall in mm. Left out, ten years of Open-Meteo records for the site fill it in |
| `rain_days` | int | live | Days a year that rain falls on. Comes from the same records |
| `target_depth_m` | float | 3.0 | How deep the pond is to be built |
| `ensemble` | bool | see note | Re-trace each site on three more grids for an error bar |

> **`ensemble` on this deployment.** The container is capped at 512 MB and the four-grid
> ensemble peaks at 581 MB, so it is switched off here — `POND_API_DEFAULT_ENSEMBLE=false`
> for the default, and `POND_API_ALLOW_ENSEMBLE=false` so that an explicit `ensemble=true`
> comes back as a `422 ensemble_unavailable` rather than being attempted. Off the
> container both default to on. See [REPORT.md §7](../REPORT.md).

### Response, 200

The full field list is in `/docs`. The shape:

```jsonc
{
  "status": "ok",
  "input":   { "contour_count": 1355, "vertex_count": 159113, "mapped_area_ha": 830.8, ... },
  "dem":     { "resolution_m": 3.65, "shape": [719, 888], "nodata_fraction": 0.0254, ... },
  "parameters": { ... },                  // every parameter as it was actually applied
  "recommended_site": {
    "rank": 1,
    "is_recommended": true,               // false = returned but the evidence does not support it
    "location": { "lat": 21.243922, "lon": 81.289998 },
    "why": [ "largest upstream area: 66.3 ha, 8% of the mapped sheet", ... ],
    "slope": 0.0176,
    "height_above_trunk_m": 4.0,          // freeboard over the nearest watercourse
    "catchment": {
      "area_ha": 66.3,
      "area_uncertainty_ha": 17.9,        // null when the ensemble is off
      "confidence": "medium",             // high | medium | low | unassessed
      "edge_contact_pct": 3.0,            // >15% ⇒ area is a lower bound
      "is_lower_bound": false,
      "relief_m": 18.0,
      "time_of_concentration_min": 49.1,
      "method": "D8 steepest-descent on a contour-interpolated, smoothed DEM"
    },
    "storage": {
      "capacity_m3": 20993.0,
      "surface_area_m2": 12699.0,
      "natural_storage_m3": 1557.0,
      "stage_storage": [[0.0, 2617.0, 1557.0], ...]   // [depth_m, area_m2, volume_m3]
    },
    "runoff": {
      "method": "SCS-CN, applied per rain day and summed",
      "rainfall_mm": 1397.3, "rain_days": 118,
      "rainfall_source": "Open-Meteo ERA5 daily records, 10 years",
      "runoff_depth_mm": 111.0, "runoff_coefficient": 0.079,
      "annual_runoff_m3": 73619.0,
      "single_event_coefficient": 0.931,  // what the year-as-one-storm mistake would give
      "overestimate_factor": 11.72,
      "fill_ratio": 3.51,
      "assessment": "Far more water arrives than the pond can hold. ..."
    }
  },
  "alternative_sites": [ /* same shape, ranks 2..top_n, each a separate basin */ ],
  "search":  { "stream_threshold_ha": 4.2, "candidate_cells": 566, ... },
  "geojson": { "type": "FeatureCollection", "features": [ /* 8 features */ ] },
  "warnings": [ "..." ],                  // never fatal; always worth showing the user
  "timing_ms": { "parse": 221.7, "dem": 1677.8, "flow": 894.8, "total": 10122.9 }
}
```

The `geojson` block is a complete `FeatureCollection` — catchment polygons, pond
footprints, outlets and the longest flow path — that pastes straight into
[geojson.io](https://geojson.io). Each feature carries `role` and `rank` properties so a
client can style them without guessing.

### curl

```bash
# Simplest possible call: the file, and nothing else.
curl -F file=@data/contours_1m.kml \
     http://10.1.75.53:5229/api/v1/analyzeContour

# Your own rain gauge, a coarser grid, and five sites instead of three.
curl -F file=@data/contours_1m.kml \
     -F rainfall_mm=1200 -F rain_days=55 \
     -F grid_resolution=5 -F top_n=5 \
     http://10.1.75.53:5229/api/v1/analyzeContour

# A pour point you have already chosen: no site search, just that catchment.
curl -F file=@data/contours_1m.kml \
     -F lat=21.243922 -F lon=81.289998 \
     http://10.1.75.53:5229/api/v1/analyzeContour

# Just the headline numbers.
curl -sF file=@data/contours_1m.kml \
     http://10.1.75.53:5229/api/v1/analyzeContour \
  | python3 -c 'import json,sys; s=json.load(sys.stdin)["recommended_site"]; \
print(s["location"], s["catchment"]["area_ha"], "ha", s["runoff"]["annual_runoff_m3"], "m3/yr")'
```

---

## GET /api/v1/rainfall

Ten years of daily rainfall for one point, averaged to a year. The same feed
`/analyzeContour` uses when no rainfall figure is given, exposed on its own so a client
can show the number before committing to a 15-second analysis. The demo page calls it as
soon as a pour point is dropped.

| Query | Type | Required | Range |
|---|---|---|---|
| `lat` | float | yes | −90 to 90 |
| `lon` | float | yes | −180 to 180 |

```bash
curl "http://10.1.75.53:5229/api/v1/rainfall?lat=21.25&lon=81.63"
```

```json
{
  "status": "ok",
  "lat": 21.25, "lon": 81.63,
  "annual_rainfall_mm": 1475.5,
  "rain_days": 123,
  "wettest_day_mm": 173.9,
  "years": 10.0,
  "source": "Open-Meteo ERA5 daily records, 10 years",
  "is_measured": true,
  "description": "1476 mm a year over 123 rain days, averaged across 10 years ...",
  "warnings": ["..."]
}
```

This endpoint does not fail on a weather service that is down. It answers with the
documented regional climatology instead, `is_measured` false, and the reason in
`warnings`. An analysis is never blocked by the weather.

---

## GET /health

```bash
curl http://10.1.75.53:5229/health
# {"status":"ok","service":"Pond Catchment Analysis API","version":"1.0.0"}
```

Deliberately does no work, so it answers before the first analysis rather than after one.

---

## Errors

Every error, from any endpoint, comes back in one shape:

```json
{
  "status": "error",
  "code": "no_buildable_ground",
  "detail": "What went wrong, in a sentence.",
  "hint": "What to change about the request."
}
```

`code` is stable and machine-readable; `detail` and `hint` are for a person. The status
code says which of three things happened.

**400 — the file cannot yield an answer.** Unparseable XML, no contour lines, no
resolvable elevations, degenerate geometry. Codes include `invalid_kml`,
`no_contours_found`, `no_elevations_found`. Nothing about the request can fix it; the
file has to change.

**413 — more data than the service will take on.**

| Code | Cause |
|---|---|
| `file_too_large` | Upload over 64 MB |
| `sheet_too_large` | The sheet needs more cells than the 12,000,000 ceiling even at the coarsest grid |

**422 — understood, but impossible as asked.**

| Code | Cause |
|---|---|
| `invalid_request` | A field missing or of the wrong type (FastAPI validation) |
| `invalid_resolution` | `grid_resolution` outside the 2–20 m band |
| `curve_number_out_of_range` | `curve_number` outside 30–98 |
| `bad_target_depth` | `target_depth_m` not positive |
| `bad_rainfall_series` | `rainfall_mm` / `rain_days` inconsistent or non-physical |
| `invalid_parameters` | A combination that cannot be honoured |
| `pour_point_outside_map` | `lat`/`lon` off the sheet |
| `pour_point_unusable` | On the sheet but on no-data ground |
| `no_stream_network` | No channel crosses the derived stream threshold |
| `no_buildable_ground` | Nothing on the sheet is flat enough to build on |
| `no_ground_clear_of_watercourse` | Every candidate stands in or too near a watercourse |
| `no_site_found` | The search finished with nothing that satisfies every rule |
| `ensemble_unavailable` | `ensemble=true` on a host without the memory for it (this deployment) |

The last few are not malformed requests, but to a client they are the same thing: nothing
about the file changes, and only a different ask can help.

**504 — `analysis_timeout`.** The analysis passed the 120-second limit. The hint asks for
a coarser `grid_resolution` or `ensemble=false`.

An unrecognised code from a future failure mode falls back to 400, which is the honest
default: the request could not be analysed.

---

## Notes for a client

- **Timing.** A full analysis of the 831 ha sample takes about **15 s** on the container
  with the ensemble off, about 10 s locally with it on. `/health` and `/rainfall` are
  immediate. Set a client timeout above 120 s, which is the server's own limit.
- **Concurrency.** The analysis runs in a worker thread, so a long request never blocks
  `/health`. Analyses themselves run **one at a time** and further ones queue: two at once
  need more than a gigabyte, which a 512 MB container does not survive. Three concurrent
  uploads of the sample returned 200 in 15 s, 27 s and 39 s. A queue that cannot clear
  inside the 120 s limit gets the usual 504, so waiting is bounded, never indefinite.
- **CORS** is open for `GET`, `POST` and `OPTIONS`, so a browser page on another origin
  can call this directly.
- **Warnings are not errors.** A 200 response with a `warnings` array is a complete
  answer with caveats attached — a skipped contour line, a reanalysis rainfall grid, the
  1 m contour interval bounding how precise any depth can be. Show them.
