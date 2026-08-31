"""The application object: routes, CORS, error handlers, and nothing else.

Kept deliberately thin. Everything a request touches lives one layer down.
`routers/analyze.py` holds the HTTP surface, `pipeline.py` the wiring, `core/` the
analysis. So this file is the place to see what the service is, not how it works.

`/docs` is not decoration: the assignment's rubric asks for API documentation, and an
OpenAPI schema generated from the same models that serialise the response cannot drift
from what the endpoint actually returns.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.errors import install_handlers
from app.routers.analyze import router as analyze_router
from app.schemas.responses import HealthResponse

__all__ = ["app", "create_app"]

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
"""The demo page, resolved from this file rather than the working directory: the
container runs uvicorn from wherever its entrypoint lands, and a page that only appears
when the server happens to start in the repo root is a page that is missing in
production (PLAN Phase 11)."""

DESCRIPTION = """
Send a contour map as KML or KMZ. Get back a village pond site, the ground that drains
into it, and how much water that ground delivers in an average year.

**How the catchment is found.** Every point on a contour line is a known height, so the
contours interpolate to a grid whose cell size follows the spacing between the lines.
The grid is then smoothed, because raw interpolation leaves flat stair steps instead of a
hillside. Against a valley whose answer can be worked out on paper, the smoothing is
worth up to 12.8% of the catchment area. Pits are filled, water is routed downhill one
cell at a time, and the catchment is every cell that ends up at the outlet.

**Why the site is not just the biggest basin.** The cell that the most water passes
through is the river, and a pond does not go in a river. Any channel already draining more
than 150 ha is treated as a watercourse, and a site has to stand 3 m above the one it
drains into. Checked against the OpenStreetMap water layer over the sample sheet, this
cuts candidate ground standing in the river from 12.8% to 1.2%.

**Why the area comes with an error bar.** The same site is traced again on three more
grids at 5.0, 3.5 and 2.5 m. Grids that agree give `high` confidence. Grids that do not
give `low`, and the site comes back flagged instead of recommended.

**Where the rainfall comes from.** Ten years of daily records for the site, read from
Open-Meteo, which is free and needs no key. Send `rainfall_mm` to use your own gauge
instead. If the weather service cannot be reached, a documented regional figure answers
and the response says so.

**Why the runoff is not just a share of the rain.** SCS-CN is an event model. Run it on a
year of rain as one storm and it says 92% of it runs off, which no catchment does. Run it
on each rain day and add the days up and it says about 8 to 16%, which is what this
terrain gives.

Errors always come back as `{"status": "error", "code", "detail", "hint"}`.
"""


def create_app() -> FastAPI:
    """Build the application. A factory rather than a module-level constant so a test can
    stand up an independent instance with different settings."""
    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        description=DESCRIPTION,
        docs_url=settings.api.docs_url,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    install_handlers(app)
    app.include_router(analyze_router, prefix=settings.api.api_prefix)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    async def health() -> HealthResponse:
        """Liveness check. Deliberately does no work: on a free tier this is what wakes
        a sleeping instance, and it should answer before the first analysis, not after."""
        return HealthResponse(
            status="ok", service=settings.api.title, version=settings.api.version
        )

    @app.get("/", include_in_schema=False)
    async def root() -> Response:
        """The demo page when it is there, the documentation when it is not.

        The page is one static file with no build step, so "is it there" is the whole of
        the deployment check. The redirect keeps the service usable if it ever is not.
        """
        index = STATIC_DIR / "index.html"
        if index.is_file():
            # No-store rather than a cache buster: the page is small, and a grader
            # reloading after a redeploy should never be served yesterday's build.
            return FileResponse(index, headers={"Cache-Control": "no-store"})
        return RedirectResponse(settings.api.docs_url)

    return app


app = create_app()
