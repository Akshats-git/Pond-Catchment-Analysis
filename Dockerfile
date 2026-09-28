# One image, four roles: gateway, worker, CV service, and (behind nginx) the page.
# Which role a container plays is decided by its environment and command in
# docker-compose.yml, exactly as on the lab containers it is decided by .env.role.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/pond
COPY requirements.txt .
RUN pip install --retries 10 -r requirements.txt \
 && useradd --create-home --uid 1000 pond \
 && mkdir -p /data /srv/pond/.cache && chown pond /data /srv/pond/.cache

COPY app ./app
COPY static ./static
COPY data ./data

USER pond
EXPOSE 5000
# The same settings run.sh applies on a 512 MB lab container: the contour path's
# four-grid ensemble does not fit, and asking for it gets a 422 rather than an OOM.
ENV POND_API_DEFAULT_ENSEMBLE=false \
    POND_API_ALLOW_ENSEMBLE=false \
    POND_STORE_PATH=/data/ponds.sqlite3

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '5000') + '/health', timeout=2)" || exit 1

# Shell form so $PORT expands: platforms like Render assign the port at runtime and
# route to whatever the container listens on. Local `docker run`/compose leave PORT
# unset and get the same 5000 as before.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-5000} --workers 1 --timeout-keep-alive 65
