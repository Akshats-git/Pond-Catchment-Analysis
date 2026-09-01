"""Regenerate the figures in REPORT.md from one run of the real pipeline.

Not part of the service. matplotlib is a documentation dependency only, deliberately
kept out of `requirements.txt` so the deployed image stays small:

    pip install matplotlib && python docs/make_figures.py

Every figure is drawn from the same `AnalysisResult` the API would have returned for
`data/contours_1m.kml`, so a figure cannot drift from what the endpoint reports.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.pipeline import analyse
from app.schemas.requests import AnalysisParams

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
DPI = 150


def hillshade(z: np.ndarray, res: float, azimuth: float = 315.0, altitude: float = 45.0):
    """Standard illumination model. NaN stays NaN so no-data reads as blank, not black."""
    dy, dx = np.gradient(z, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    return np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)


def fig_dem(result) -> None:
    """The DEM the contours interpolate to, shaded so the valley is legible."""
    dem = result.dem
    z = dem.z
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.imshow(hillshade(z, dem.resolution_m), cmap="gray", origin="lower", alpha=0.55)
    im = ax.imshow(z, cmap="terrain", origin="lower", alpha=0.65)
    fig.colorbar(im, ax=ax, label="elevation (m)", shrink=0.82)
    ax.set_title(
        f"DEM from {result.contours.metadata.line_count} contour lines, "
        f"{z.shape[1]}x{z.shape[0]} cells at {dem.resolution_m:.2f} m",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "dem_hillshade.png", dpi=DPI)
    plt.close(fig)


def fig_accumulation(result) -> None:
    """Flow accumulation on a log scale: the drainage network the siting search reads."""
    acc = result.flow.accumulation
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(np.log10(np.where(np.isfinite(acc), acc, np.nan) + 1), cmap="Blues",
                   origin="lower")
    fig.colorbar(im, ax=ax, label="log10(cells draining through)", shrink=0.82)
    ax.set_title("Flow accumulation: the channels emerge, unlabelled, from the routing",
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "flow_accumulation.png", dpi=DPI)
    plt.close(fig)


def fig_catchment(result) -> None:
    """The recommended site's catchment over the hillshade, with its outlet."""
    dem, site = result.dem, result.recommended
    mask = site.catchment.mask
    r, c = site.catchment.outlet_rc
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.imshow(hillshade(dem.z, dem.resolution_m), cmap="gray", origin="lower")
    ax.imshow(np.where(mask, 1.0, np.nan), cmap="autumn", origin="lower", alpha=0.45)
    ax.plot(c, r, "o", ms=9, mfc="#1b6ac9", mec="white", mew=1.6, label="outlet / pond site")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(
        f"Catchment of site 1: {site.catchment.area_ha:.1f} ha, "
        f"{site.catchment.edge_contact * 100:.1f}% edge contact, confidence {site.confidence}",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "catchment_site1.png", dpi=DPI)
    plt.close(fig)


def fig_stage_storage(result) -> None:
    """Depth against volume and water-spread area for the recommended site."""
    balance = result.recommended_balance
    stage = balance.storage
    depth = np.asarray(stage.stages_m)
    area = np.asarray(stage.areas_m2)
    volume = np.asarray(stage.volumes_m3)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(depth, volume, "o-", color="#1b6ac9", label="storage (m³)")
    ax.set_xlabel("depth above the outlet (m)")
    ax.set_ylabel("storage (m³)", color="#1b6ac9")
    ax.tick_params(axis="y", labelcolor="#1b6ac9")
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    twin.plot(depth, area, "s--", color="#c2570a", label="water spread (m²)")
    twin.set_ylabel("water spread (m²)", color="#c2570a")
    twin.tick_params(axis="y", labelcolor="#c2570a")
    ax.set_title("Stage-storage at site 1, integrated cell by cell from the DEM", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "stage_storage.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = (ROOT / "data" / "contours_1m.kml").read_bytes()
    result = analyse(data, "contours_1m.kml", AnalysisParams(ensemble=True))
    for fn in (fig_dem, fig_accumulation, fig_catchment, fig_stage_storage):
        fn(result)
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
