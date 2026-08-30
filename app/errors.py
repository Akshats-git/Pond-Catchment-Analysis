"""One exception and one handler, so every failure leaves the service the same shape.

`app/core/` raises five different exception types. The route maps each to a status code
and re-raises it as an `APIError`; `app.main` renders that -- and FastAPI's own request
validation failures -- into the `{status, code, detail, hint}` body documented as
`ErrorResponse`. A client therefore parses one thing, whether the file was unreadable,
the curve number was 200, or the framework rejected the multipart body before any of this
code ran.
"""

from __future__ import annotations

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

__all__ = ["APIError", "error_body", "install_handlers"]


class APIError(Exception):
    """A failure with a status code attached. The only exception the route raises."""

    def __init__(self, status_code: int, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.hint = hint


def error_body(code: str, detail: str, hint: str = "") -> dict:
    return {"status": "error", "code": code, "detail": detail, "hint": hint}


def install_handlers(app) -> None:
    """Attach the handlers. Called by `app.main`; kept here so the shape of an error and
    the rendering of one live in the same file."""

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.detail, exc.hint),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI rejects a missing or mistyped form field before the route body runs.
        # Without this the response would be FastAPI's `{"detail": [...]}`, which is a
        # second error format for clients to learn.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'][1:]) or 'request'}: "
            f"{error['msg']}"
            for error in exc.errors()
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_body(
                "invalid_request",
                problems or "The request could not be validated.",
                "See /docs for the required fields and their types.",
            ),
        )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        # 404s and 405s from Starlette's router, put into the same envelope.
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(f"http_{exc.status_code}", detail),
            headers=getattr(exc, "headers", None),
        )
