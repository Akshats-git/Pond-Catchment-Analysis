# Pond Catchment Analysis API

A backend API that takes a contour map (KML/KMZ), analyses the terrain, identifies a
suitable village pond location, and returns the catchment information as JSON.

Phase 2 of the Village Pond Planning System. The Phase 1 high-level design is in
[references/](references/); the full implementation plan, including the validated
catchment methodology, is in [PLAN.md](PLAN.md).

## How it works

1. **Contours → points** — every contour vertex is a known `(x, y, z)`.
2. **Points → DEM** — Delaunay triangulation, linear interpolation onto a square grid
   whose resolution is derived from the mean contour spacing.
3. **Smooth** — a NaN-aware Gaussian removes the stair-step artefact of contour
   interpolation. Worth up to 12.8% catchment accuracy; see PLAN §3.
4. **Fill pits** — priority-flood, so water is never trapped in a data artefact.
5. **D8 flow** — each cell drains to its steepest neighbour; accumulate downstream.
6. **Catchment** — walk the flow arrows backwards from the pond outlet.

Siting is catchment-first: find the stream network, keep buildable low-slope ground, rank
by catchment area, and suppress each pick's whole catchment so alternatives are
independent sub-basins.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload           # (from Phase 9)
```

Interactive API documentation at `http://localhost:8000/docs`.

```bash
curl -F file=@data/contours_1m.kml http://localhost:8000/api/v1/analyzeContour
```

## Layout

| Path | What lives there |
|---|---|
| `app/core/` | The analysis: parsing, projection, DEM, flow routing, catchment, siting, hydrology |
| `app/providers/` | Swappable data sources (rainfall, elevation) — the Phase 3 seam |
| `app/routers/` | HTTP surface only: validation and orchestration |
| `app/config.py` | Every tunable, documented and environment-overridable |
| `tests/` | Analytic validation, mass balance, structural variants |
| `data/` | `contours_1m.kml`, the provided sample sheet |
| `docs/` | API reference and methodology |

## Status

Under construction, one commit per phase (see PLAN.md §7).

- [x] Phase 0 — scaffold
- [ ] Phases 1–12
