"""Fetch elevation tiles for a region into the committed seed directory.

The demo must not depend on the network (PLAN3 §10). Run this once for the demo region and
commit `data/tiles/`; the provider then reads those tiles before it ever tries the network.

    python -m tools.prewarm_tiles                       # the sample sheet, z10-z13
    python -m tools.prewarm_tiles --bbox 81.2,21.2,81.4,21.3 --zooms 12 13
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings
from app.providers.elevation import REPO_ROOT, TerrariumProvider, tile_range

SAMPLE_REGION = (81.262, 21.225, 81.332, 21.278)
"""The provided survey sheet (81.2814-81.3126 E, 21.2398-21.2636 N) with room for the
25% analysis buffer and some panning around it."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", default=",".join(map(str, SAMPLE_REGION)))
    parser.add_argument("--zooms", type=int, nargs="+", default=[10, 11, 12, 13])
    parser.add_argument("--out", default=settings.elevation.seed_dir)
    args = parser.parse_args()

    bbox = tuple(float(v) for v in args.bbox.split(","))
    provider = TerrariumProvider()
    root = Path(args.out)
    root = root if root.is_absolute() else REPO_ROOT / root
    total = fetched = 0
    for zoom in args.zooms:
        x0, y0, x1, y1 = tile_range(bbox, zoom)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                total += 1
                path = root / provider.encoding / str(zoom) / str(x) / f"{y}.png"
                if path.is_file():
                    continue
                data = provider.fetch(zoom, x, y)
                if data is None:
                    print(f"missing z{zoom}/{x}/{y}")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                fetched += 1
    print(f"{total} tiles in range, {fetched} fetched, stored under {root}")


if __name__ == "__main__":
    main()
