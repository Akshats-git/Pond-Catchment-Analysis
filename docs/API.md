# API reference

Base URL of the running service: **`http://10.1.75.53:5229`**
Interactive documentation, generated from the same models that serialise the responses:
**`http://10.1.75.53:5229/docs`**

Five endpoints, plus an alias. Everything is `multipart/form-data` in, and JSON out
except `/renderMap`, which returns a PNG.

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/v1/analyzeContour` | Contour map in, pond site + catchment + yield out |
| `POST` | `/api/v1/findCatchment` | Alias of the above, identical signature |
| `POST` | `/api/v1/renderMap` | The same analysis, drawn as a PNG map |
| `POST` | `/api/v1/contours` | The same map back as drawable lines, without analysing it |
| `GET` | `/api/v1/rainfall` | Ten years of daily rainfall for one point |
| `GET` | `/health` | Liveness. Does no work on purpose |

---

## POST /api/v1/analyzeContour

Send a contour map. Get back where the pond goes, what drains into it, and how much water
that ground delivers in an average year.

### Request

`multipart/form-data`. Only `contour_map` is required; every other field has a derived
or configured default, and the defaults are what the sample run in
[REPORT.md](../REPORT.md) used.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `contour_map` | file | — | The contour map, `.kml` or `.kmz`. Contour **lines**, not point labels. Max 64 MB |
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
  "geojson": { "type": "FeatureCollection", "features": [ /* 10 for three sites */ ] },
  "warnings": [ "..." ],                  // never fatal; always worth showing the user
  "timing_ms": { "parse": 221.7, "dem": 1677.8, "flow": 894.8, "total": 10122.9 }
}
```

The `geojson` block is a complete `FeatureCollection` that pastes straight into
[geojson.io](https://geojson.io). Every site contributes a catchment polygon, the pond
footprint at capacity and an outlet point; the longest flow path is drawn for the
recommended site alone. Each feature carries `role` and `rank` properties so a client can
style them without guessing, and the alternates' ponds and markers come in a paler blue.

### curl

```bash
# Simplest possible call: the file, and nothing else.
curl -F contour_map=@data/contours_1m.kml \
     http://10.1.75.53:5229/api/v1/analyzeContour

# Your own rain gauge, a coarser grid, and five sites instead of three.
curl -F contour_map=@data/contours_1m.kml \
     -F rainfall_mm=1200 -F rain_days=55 \
     -F grid_resolution=5 -F top_n=5 \
     http://10.1.75.53:5229/api/v1/analyzeContour

# A pour point you have already chosen: no site search, just that catchment.
curl -F contour_map=@data/contours_1m.kml \
     -F lat=21.243922 -F lon=81.289998 \
     http://10.1.75.53:5229/api/v1/analyzeContour

# Just the headline numbers.
curl -sF contour_map=@data/contours_1m.kml \
     http://10.1.75.53:5229/api/v1/analyzeContour \
  | python3 -c 'import json,sys; s=json.load(sys.stdin)["recommended_site"]; \
print(s["location"], s["catchment"]["area_ha"], "ha", s["runoff"]["annual_runoff_m3"], "m3/yr")'
```

---

## POST /api/v1/renderMap

The answer as a picture: the catchment, the pond and the ranked sites drawn over
satellite imagery and the contour lines they were derived from. Same input as
`/analyzeContour`, same analysis, same colours. What comes back is `image/png`.

It exists for the check that no number can make. A catchment boundary is right when it
runs along the ridges, and that is legible at a glance with the contours underneath it
and invisible without them. `/analyzeContour` hands back the GeoJSON to draw that
yourself; this draws it for a reader who has no map client in front of them, and for a
report that needs a figure.

Costs a full analysis, so it is as slow as `/analyzeContour` (several seconds on the
sample sheet) and shares the same one-at-a-time queue.

### Request

Every field `/analyzeContour` takes, plus six that decide what the picture looks like.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `contour_map` | file | yes | — | `.kml` or `.kmz` |
| `grid_resolution`, `top_n`, `lat`, `lon`, `curve_number`, `rainfall_mm`, `rain_days`, `target_depth_m`, `ensemble` | | no | | Exactly as `/analyzeContour`; same ranges, same errors |
| `width` | int | no | `1200` | 240 to 1600 |
| `height` | int | no | `900` | 240 to 1600 |
| `basemap` | string | no | `satellite` | `satellite`, `street`, `hillshade` or `none` |
| `contours` | bool | no | `true` | Draw the uploaded contour lines under the answer |
| `frame` | string | no | `sheet` | `sheet` for the whole uploaded map, `sites` to zoom to the answer |
| `legend` | bool | no | `true` | Draw the recommended site's numbers on the image |

`width` and `height` are capped because the vector overlay is drawn supersampled in RGBA
before being scaled down: the pixel count is a memory bound, not a matter of taste.

### Response, 200

`image/png`, at exactly the size asked for, with

```
Content-Type: image/png
Content-Disposition: inline; filename="contours_1m-catchment.png"
X-Pond-Warnings: Open-Meteo returned HTTP 429. The documented regional climatology ...
```

**Read the `X-Pond-Warnings` header.** A picture has nowhere to carry a caveat, and
dropping "this rainfall is a climatology, not an observation" because the client asked for
an image would be the service deciding what the client is allowed to know. Every warning
the JSON response would have carried is in that header, joined with ` | `, truncated at
900 characters. Ask `/analyzeContour` for the same file to see the full list.

What is drawn, in this order: the contour lines, each catchment as a translucent blue
polygon, each pond as the water surface at the stage the site holds, the recommended
site's longest flow path in red, a numbered marker per site, the recommended catchment's
area on a pill, the legend, a scale bar and the basemap's attribution.

Two things adapt to the image rather than being fixed:

* **Contour density.** Below seven pixels between neighbouring lines only the index
  contours are drawn, and the warning header says so and names the new interval. One-pixel
  lines closer than that stop reading as lines and become a haze over the imagery, hiding
  the ground the contours were there to let you check. A printed sheet drops to every
  fifth line for the same reason.
* **The basemap.** If the tile server cannot be reached, or more than 40% of tiles fail,
  the image falls back to a hillshade of the uploaded sheet and says so in the header.
  Somebody else's outage should not become a 502 on a catchment this service had already
  computed. `basemap=hillshade` asks for that directly and needs no network at all.

### curl

```bash
# The default picture: satellite, contours, legend, whole sheet.
curl -X POST -F "contour_map=@data/contours_1m.kml" -F "ensemble=false" \
     http://10.1.75.53:5229/api/v1/renderMap -o catchment.png

# Zoomed to the answer, no imagery, larger.
curl -X POST -F "contour_map=@data/contours_1m.kml" -F "ensemble=false" \
     -F "frame=sites" -F "basemap=hillshade" -F "width=1600" -F "height=1200" \
     http://10.1.75.53:5229/api/v1/renderMap -o catchment.png

# The picture and the caveats that go with it.
curl -sD - -o catchment.png -X POST -F "contour_map=@data/contours_1m.kml" \
     -F "ensemble=false" http://10.1.75.53:5229/api/v1/renderMap \
  | grep -i x-pond-warnings
```

---

## POST /api/v1/contours

The contour lines in an uploaded sheet, styled and thinned for a map. No analysis: this
is the parser and nothing else, so it answers in about half a second on the sample sheet
where `/analyzeContour` takes several.

It exists because a catchment boundary drawn on satellite imagery cannot be checked by
eye. Imagery does not show where the ridges are, and a boundary is right when it runs
along them. Ask for the contours, lay the catchment over them, and the answer can be read
rather than taken on trust. The demo page calls it on upload and draws the result under
the analysis; its **Contours** button toggles that layer.

Both endpoints run the same parser, so the lines drawn are the lines analysed: on one
file, `contour_count`, `bbox`, `interval_m` and `elevation_range_m` match
`/analyzeContour`'s `input` block exactly.

### Request

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `contour_map` | file | yes | — | `.kml` or `.kmz`, the same file `/analyzeContour` takes |
| `simplify_m` | float | no | `1.5` | How far a drawn line may depart from the one in the file. `0` sends every vertex. 0 to 1000 |

The default is three quarters of the finest grid the service will ever build, so the
overlay cannot disagree with a catchment drawn over it by anything the analysis could
resolve. On the sample sheet it turns 159,113 vertices into 30,538: 0.9 MB of JSON, 160 kB
gzipped, which is what the service sends.

### Response, 200

```json
{
  "status": "ok",
  "filename": "contours_1m.kml",
  "contour_count": 1355,
  "vertex_count": 30538,
  "source_vertex_count": 159113,
  "simplify_tolerance_m": 1.5,
  "elevation_source": "placemark_name",
  "interval_m": 1.0,
  "level_count": 32,
  "elevation_range_m": [267.0, 298.0],
  "index_interval_m": 5.0,
  "bbox": [81.281404, 21.239822, 81.312647, 21.263581],
  "geojson": { "type": "FeatureCollection", "bbox": [...], "features": [...] },
  "warnings": [],
  "timing_ms": {"parse": 316.0, "draw": 350.9, "total": 667.1}
}
```

Every feature is a `LineString`:

```json
{
  "type": "Feature",
  "geometry": {"type": "LineString", "coordinates": [[81.2903, 21.2511], ...]},
  "properties": {
    "role": "contour",
    "elevation_m": 280.0,
    "index": true,
    "stroke": "#e08b1e",
    "stroke-width": 1.6,
    "stroke-opacity": 0.95
  }
}
```

`index` marks the heavy lines a topographic sheet prints every fifth level, counted off a
round multiple of the interval rather than off the lowest line in the file, so a sheet
starting at 267 m still makes 270 and 275 the heavy ones. `stroke` comes off an elevation
ramp, dark at the bottom of the sheet and pale at the top: without it a reader sees nested
loops with no way to tell a hill from a hollow. They are simplestyle properties, so the
collection draws the same in the demo page and in geojson.io.

`vertex_count` is what was drawn; `source_vertex_count` is what was in the file. If a
sheet is large enough that even the requested tolerance would blow the vertex budget, the
service thins further, reports the tolerance it settled on, and says so in `warnings`.

### curl

```bash
# The overlay, at the default thinning.
curl -X POST -F "contour_map=@data/contours_1m.kml" \
     http://10.1.75.53:5229/api/v1/contours -o contours.geojson

# Every vertex in the file, nothing dropped.
curl -X POST -F "contour_map=@data/contours_1m.kml" -F "simplify_m=0" \
     http://10.1.75.53:5229/api/v1/contours

# Just the shape of the answer.
curl -s -X POST -F "contour_map=@data/contours_1m.kml" \
     http://10.1.75.53:5229/api/v1/contours \
  | python3 -c 'import json,sys; b=json.load(sys.stdin); \
print(b["contour_count"], "lines,", b["level_count"], "levels,", b["vertex_count"], "points")'
```

---

## GET /api/v1/rainfall

Ten years of daily rainfall for one point, averaged to a year. The same feed
`/analyzeContour` uses when no rainfall figure is given, exposed on its own so a client
can show the number before committing to a 15-second analysis. The demo page calls it for
the middle of the sheet as soon as a file is read.

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
# {"status":"ok","service":"Pond Catchment Analysis API","version":"1.0.0",
#  "ensemble_available":false,"ensemble_default":false}
```

Deliberately does no work, so it answers before the first analysis rather than after one.

`ensemble_available` is whether this host can run the three-grid cross-check at all, and
`ensemble_default` whether an ordinary request runs it. Both are false on this deployment
(see the note under [POST /api/v1/analyzeContour](#post-apiv1analyzecontour)). A client
that reads them before its first upload never earns a `422 ensemble_unavailable`; the demo
page reads them here to set its own switch.

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
| `missing_file` | No contour map in the request. It goes in the `contour_map` field |
| `invalid_request` | A field of the wrong type (FastAPI validation) |
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
| `no_available_ground` | An `exclusion_mask` ruled out every site the terrain allows. Library callers only; see [REPORT.md §8](../REPORT.md) |
| `exclusion_mask_shape` | An `exclusion_mask` was not built on the analysis grid. Library callers only |
| `ensemble_unavailable` | `ensemble=true` on a host without the memory for it (this deployment) |
| `invalid_simplify` | `/contours` asked for a tolerance outside 0–1000 m |
| `invalid_image_size` | `/renderMap` asked for a width or height outside 240–1600 px |
| `invalid_basemap` | `/renderMap` asked for a basemap that is not one of the four |
| `invalid_frame` | `/renderMap` asked for a frame that is not `sheet` or `sites` |
| `render_too_large` | The view needs more than 256 map tiles. Ask for a smaller image |

The last few are not malformed requests, but to a client they are the same thing: nothing
about the file changes, and only a different ask can help.

**504 — `analysis_timeout`.** The analysis passed the 120-second limit. The hint asks for
a coarser `grid_resolution` or `ensemble=false`.

An unrecognised code from a future failure mode falls back to 400, which is the honest
default: the request could not be analysed.

---

## Notes for a client

- **Timing.** A full analysis of the 831 ha sample takes about **15 s** on the container
  with the ensemble off, about 10 s locally with it on. `/contours` on the same file takes
  well under a second; `/health` and `/rainfall` are immediate. Set a client timeout above
  120 s, which is the server's own limit.
- **Concurrency.** The analysis runs in a worker thread, so a long request never blocks
  `/health`. Analyses themselves run **one at a time** and further ones queue: two at once
  need more than a gigabyte, which a 512 MB container does not survive. Three concurrent
  uploads of the sample returned 200 in 15 s, 27 s and 39 s. A queue that cannot clear
  inside the 120 s limit gets the usual 504, so waiting is bounded, never indefinite.
  `/contours` is not in that queue: it holds the parsed file and nothing else, so it can
  answer while an analysis is running, which is what lets a page draw the sheet first and
  the catchment when it arrives.
- **Responses are gzipped** above 2 kB when the client sends `Accept-Encoding: gzip`. It
  matters for one of them: the sample sheet's contour overlay is 0.9 MB of coordinates and
  160 kB compressed.
- **CORS** is open for `GET`, `POST` and `OPTIONS`, so a browser page on another origin
  can call this directly.
- **Warnings are not errors.** A 200 response with a `warnings` array is a complete
  answer with caveats attached — a skipped contour line, a reanalysis rainfall grid, the
  1 m contour interval bounding how precise any depth can be. Show them.
