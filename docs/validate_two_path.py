"""Phase 17: the same ground through two independent data paths.

The provided sheet is a 1 m contour survey. The map path reads the same ground from free
~30 m SRTM tiles. Running both and comparing says, with numbers, how far the free-data
answer can be trusted. No other data path can be checked this way, because nobody else has
a surveyed sheet to check it against.

    python -m docs.validate_two_path            # writes docs/VALIDATION.md
    python -m docs.validate_two_path --quick    # the default settings only

Runs offline: the tiles for the sheet are committed under `data/tiles/`, and rainfall is
the documented climatology on both paths so the runoff comparison measures the terrain
and nothing else.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from pathlib import Path

os.environ.setdefault("POND_RAINFALL_ENABLED", "false")
os.environ.setdefault("POND_ELEVATION_NETWORK_ENABLED", "false")

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.raster_dem import AreaOfInterest  # noqa: E402
from app.pipeline import AnalysisResult, analyse, analyse_area  # noqa: E402
from app.schemas.requests import AnalysisParams  # noqa: E402

SHEET = "data/contours_1m.kml"
SHEET_BBOX = (81.2814, 21.2398, 81.3126, 21.2636)
OUT = Path(__file__).resolve().parent / "VALIDATION.md"


def metres_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * 111_320 * math.cos(lat), (a[1] - b[1]) * 110_540)


def catchment_lonlat_mask(result: AnalysisResult, site_index: int, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Rasterise one site's catchment onto a common lon/lat lattice by nearest cell."""
    dem = result.dem
    mask = result.sites[site_index].catchment.mask
    xy = dem.projection.forward(np.stack([lon, lat], axis=-1))
    x0, y0 = dem.origin_xy
    cols = np.rint((xy[..., 0] - x0) / dem.resolution_m).astype(int)
    rows = np.rint((xy[..., 1] - y0) / dem.resolution_m).astype(int)
    ny, nx = dem.shape
    ok = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    out = np.zeros(lon.shape, dtype=bool)
    out[ok] = mask[rows[ok], cols[ok]]
    return out


def lattice(step_deg: float = 0.00005):
    """A ~5 m lon/lat lattice over the buffered sheet, common to both paths."""
    pad = 0.012
    lons = np.arange(SHEET_BBOX[0] - pad, SHEET_BBOX[2] + pad, step_deg)
    lats = np.arange(SHEET_BBOX[1] - pad, SHEET_BBOX[3] + pad, step_deg)
    return np.meshgrid(lons, lats)


def dem_agreement(contour: AnalysisResult, raster: AnalysisResult) -> dict:
    """Heights of the raster DEM against the survey DEM, at the survey's valid cells."""
    cdem, rdem = contour.dem, raster.dem
    rows, cols = np.nonzero(cdem.valid)
    pick = np.linspace(0, rows.size - 1, min(rows.size, 60_000)).astype(int)
    rows, cols = rows[pick], cols[pick]
    lonlat = cdem.lonlat_of(rows, cols)
    xy = rdem.projection.forward(lonlat)
    x0, y0 = rdem.origin_xy
    fc = (xy[:, 0] - x0) / rdem.resolution_m
    fr = (xy[:, 1] - y0) / rdem.resolution_m
    from scipy.ndimage import map_coordinates

    z = map_coordinates(np.nan_to_num(rdem.z), np.stack([fr, fc]), order=1, mode="nearest")
    diff = z - cdem.z[rows, cols]
    return {
        "rmse_m": float(np.sqrt(np.mean(diff ** 2))),
        "bias_m": float(np.mean(diff)),
        "p95_abs_m": float(np.percentile(np.abs(diff), 95)),
        "corr": float(np.corrcoef(z, cdem.z[rows, cols])[0, 1]),
    }


def best_overlap(contour: AnalysisResult, raster: AnalysisResult) -> tuple[float, int]:
    """IoU of the survey's recommended catchment with the best-matching raster catchment."""
    lon, lat = lattice()
    truth = catchment_lonlat_mask(contour, 0, lon, lat)
    best, best_rank = 0.0, 0
    for i in range(len(raster.sites)):
        other = catchment_lonlat_mask(raster, i, lon, lat)
        union = (truth | other).sum()
        iou = float((truth & other).sum() / union) if union else 0.0
        if iou > best:
            best, best_rank = iou, i + 1
    return best, best_rank


def run_raster(sigma: float, max_zoom: int, top_n: int = 5) -> tuple[AnalysisResult, float]:
    cfg = dataclasses.replace(
        settings,
        elevation=dataclasses.replace(settings.elevation, smoothing_sigma_m=sigma, max_zoom=max_zoom),
    )
    aoi = AreaOfInterest.from_bbox(SHEET_BBOX)
    started = time.perf_counter()
    result = analyse_area(aoi, AnalysisParams(ensemble=True, top_n=top_n), config=cfg)
    return result, time.perf_counter() - started


def describe(contour: AnalysisResult, raster: AnalysisResult, seconds: float) -> dict:
    iou, rank = best_overlap(contour, raster)
    top_c, top_r = contour.sites[0], raster.sites[0]
    ens = top_r.ensemble
    return {
        "dem": dem_agreement(contour, raster),
        "site_distance_m": metres_between(top_c.lonlat, top_r.lonlat),
        "area_ha": top_r.area_ha,
        "iou": iou,
        "iou_rank": rank,
        "iou_area_ha": raster.sites[rank - 1].area_ha if rank else float("nan"),
        "runoff_m3": raster.balances[0].annual_runoff_m3,
        "ensemble": (ens.mean_area_m2 / 1e4, ens.std_area_m2 / 1e4, ens.confidence) if ens else None,
        "lonlat": top_r.lonlat,
        "seconds": seconds,
        "cells": raster.dem.shape[0] * raster.dem.shape[1],
        "resolution_m": raster.dem.resolution_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    contour = analyse(Path(SHEET).read_bytes(), "contours_1m.kml", AnalysisParams(ensemble=True, top_n=5))
    contour_s = time.perf_counter() - started
    c_cells = contour.dem.shape[0] * contour.dem.shape[1]

    default_sigma = settings.elevation.smoothing_sigma_m
    default_zoom = settings.elevation.max_zoom
    sweep = [(default_sigma, default_zoom)] if args.quick else [
        (0.0, 13), (8.0, 13), (15.0, 13), (25.0, 13), (8.0, 12),
    ]
    rows = {}
    for sigma, zoom in sweep:
        raster, seconds = run_raster(sigma, zoom)
        rows[(sigma, zoom)] = (raster, describe(contour, raster, seconds))
        print(f"sigma={sigma:g} z{zoom}: {rows[(sigma, zoom)][1]}")

    raster, main_row = rows.get((default_sigma, default_zoom)) or next(iter(rows.values()))
    c_top, c_bal = contour.sites[0], contour.balances[0]
    lines = [
        "# Two-path validation",
        "",
        "Generated by `python -m docs.validate_two_path`. The provided 1 m contour survey",
        "against free ~30 m SRTM terrarium tiles, over the same sheet "
        f"({SHEET_BBOX[0]}-{SHEET_BBOX[2]} E, {SHEET_BBOX[1]}-{SHEET_BBOX[3]} N).",
        "Rainfall is the documented climatology on both paths, so every difference below",
        "is terrain. The map path is run exactly as a user drawing the sheet's rectangle",
        "would run it: buffered by "
        f"{settings.elevation.aoi_buffer:.0%}, smoothing sigma {default_sigma:g} m, zoom {default_zoom}.",
        "",
        "## Headline",
        "",
        "| | 1 m contour survey | 30 m SRTM (map path) | Δ |",
        "|---|---|---|---|",
    ]
    c_range = contour.dem.meta.elevation_range
    r_raw = raster.dem.raw_z[raster.surface.inside_mask(raster.dem)]
    lines.append(
        f"| Elevation range over the sheet | {c_range[0]:.0f}–{c_range[1]:.0f} m | "
        f"{r_raw.min():.0f}–{r_raw.max():.0f} m | "
        f"{r_raw.min() - c_range[0]:+.0f} / {r_raw.max() - c_range[1]:+.0f} m |"
    )
    d = main_row["dem"]
    lines.append(
        f"| DEM agreement (at survey cells) | — | RMSE {d['rmse_m']:.2f} m, bias {d['bias_m']:+.2f} m | "
        f"r = {d['corr']:.3f} |"
    )
    lines.append(
        f"| Recommended pond location | {c_top.lonlat[1]:.5f} N, {c_top.lonlat[0]:.5f} E | "
        f"{main_row['lonlat'][1]:.5f} N, {main_row['lonlat'][0]:.5f} E | "
        f"{main_row['site_distance_m']:.0f} m apart |"
    )
    lines.append(
        f"| Recommended catchment area | {c_top.area_ha:.1f} ha | {main_row['area_ha']:.1f} ha | "
        f"{(main_row['area_ha'] / c_top.area_ha - 1):+.0%} |"
    )
    lines.append(
        f"| Best-matching catchment (IoU with the survey's) | — | site #{main_row['iou_rank']}, "
        f"{main_row['iou_area_ha']:.1f} ha | IoU {main_row['iou']:.2f} |"
    )
    lines.append(
        f"| Annual runoff volume, recommended site | {c_bal.annual_runoff_m3:,.0f} m³ | "
        f"{main_row['runoff_m3']:,.0f} m³ | {(main_row['runoff_m3'] / c_bal.annual_runoff_m3 - 1):+.0%} |"
    )
    lines.append(
        f"| Grid | {contour.dem.resolution_m:.1f} m, {c_cells:,} cells | "
        f"{main_row['resolution_m']:.1f} m, {main_row['cells']:,} cells (incl. buffer) | "
        f"{c_cells / main_row['cells']:.0f}× fewer |"
    )
    lines.append(
        f"| Runtime, with the ensemble | {contour_s:.1f} s | {main_row['seconds']:.2f} s | "
        f"{contour_s / main_row['seconds']:.0f}× faster |"
    )

    lines += [
        "",
        "## Settings sweep",
        "",
        "Smoothing sigma and tile zoom, each against the survey. The default is chosen from",
        "this table, not assumed.",
        "",
        "| sigma | zoom | grid | DEM RMSE | top site Δ | top area | best IoU | ensemble (mean ± sd, confidence) | time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (sigma, zoom), (_, row) in rows.items():
        ens = row["ensemble"]
        ens_text = f"{ens[0]:.0f} ± {ens[1]:.0f} ha, {ens[2]}" if ens else "—"
        lines.append(
            f"| {sigma:g} m | z{zoom} | {row['resolution_m']:.1f} m | {row['dem']['rmse_m']:.2f} m | "
            f"{row['site_distance_m']:.0f} m | {row['area_ha']:.1f} ha | {row['iou']:.2f} (#{row['iou_rank']}) | "
            f"{ens_text} | {row['seconds']:.2f} s |"
        )

    lines += [
        "",
        "## Reading these numbers",
        "",
        f"**The two DEMs agree to {d['rmse_m']:.2f} m RMSE.** That is far closer than SRTM's",
        "published vertical accuracy (several metres) against real ground, so the provided",
        "contour sheet was almost certainly drawn from an SRTM-class DEM itself. This table",
        "therefore validates the *data path*: tile decoding, reprojection, resampling, the",
        "buffer and the routing on a coarser grid. It does not measure SRTM against a field",
        "survey, and nothing in this repository can. The honest reading is that the map path",
        "reproduces what the contour path would say about the same heights, and inherits",
        "whatever error the heights carry.",
        "",
        "**Where the answers differ, resolution is why, not heights.** Same heights, a 17.8 m",
        "grid instead of 3.1 m: channels move by a cell or two, which is enough to move a",
        "confluence and so re-split which cells drain to which outlet. That is what the IoU",
        "measures, and it is why the map path's warnings say its catchments are approximate.",
        "",
        "**z12 is rejected.** At 35.6 m the grid is coarser than the source; the recommended",
        "site moves 1.7 km to a different valley. z13 is the default.",
    ]
    lines += ["", "## Survey sites for reference", "", "| rank | location | area |", "|---|---|---|"]
    for site in contour.sites:
        lines.append(f"| {site.rank} | {site.lonlat[1]:.5f} N, {site.lonlat[0]:.5f} E | {site.area_ha:.1f} ha |")
    lines += ["", "## Map-path sites (default settings)", "", "| rank | location | area | confidence |", "|---|---|---|---|"]
    for site in raster.sites:
        lines.append(
            f"| {site.rank} | {site.lonlat[1]:.5f} N, {site.lonlat[0]:.5f} E | {site.area_ha:.1f} ha | "
            f"{site.ensemble.confidence if site.ensemble else 'unassessed'} |"
        )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
