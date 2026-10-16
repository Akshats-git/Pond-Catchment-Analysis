"""OpenTelemetry: traces, metrics and logs, switched on by one environment variable.

The rest of the code only ever touches the OpenTelemetry *API* (a tracer, a meter, the
standard `logging` module). With nothing configured those calls do nothing, so the tests,
the Render demo and a bare `uvicorn` run behave exactly as before. Set
`OTEL_EXPORTER_OTLP_ENDPOINT` and `setup()` installs the SDK, which sends all three signals
over OTLP/HTTP to whatever listens there (the Collector in docker-compose.yml).

What gets recorded:

* **Traces** - one per request. FastAPI and httpx are auto-instrumented, and httpx carries
  the `traceparent` header, so a job that goes gateway -> worker -> cv is a single trace.
  The pipeline adds a span per stage (elevation, flow, siting, ...). urllib calls
  (elevation tiles, Open-Meteo) show up as child spans, so a slow external call is visible.
* **Metrics** - analyses by outcome, analysis and stage duration, result-cache hits, tile
  sources, and the queue and per-worker load.
* **Logs** - the `pond.*` loggers and uvicorn's, with the active trace id attached, so a
  log line links back to the request that wrote it.
"""

from __future__ import annotations

import logging
import os
import socket

from opentelemetry import metrics, trace

__all__ = ["tracer", "meter", "setup", "enabled"]

tracer = trace.get_tracer("pond")
meter = metrics.get_meter("pond")

# Instruments are created at import time. Before `setup()` they are no-ops, after it they
# report, with no change at the call sites.
analyses = meter.create_counter(
    "pond.analyses", unit="{analysis}", description="Analyses finished, by outcome."
)
analysis_duration = meter.create_histogram(
    "pond.analysis.duration", unit="s", description="Wall time of one whole analysis."
)
stage_duration = meter.create_histogram(
    "pond.stage.duration", unit="s", description="Wall time of one pipeline stage."
)
job_cache = meter.create_counter(
    "pond.job.cache", unit="{lookup}", description="Result-cache lookups, hit or miss."
)
tile_sources = meter.create_counter(
    "pond.tiles", unit="{tile}", description="Elevation tiles by where they came from."
)

_BUCKETS_S = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60]


def enabled() -> bool:
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def setup(app=None, default_service: str = "pond") -> bool:
    """Install the SDK and instrument the app. Returns False (and does nothing) when no
    endpoint is configured. Safe to call once per process."""
    if not enabled():
        return False

    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.urllib import URLLibInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": os.environ.get("OTEL_SERVICE_NAME", default_service),
            "service.instance.id": socket.gethostname(),
        }
    )

    # Spans
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    # Metrics: a 5 s export interval so a demo sees numbers move without a long wait.
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=5000)
    seconds = ExplicitBucketHistogramAggregation(_BUCKETS_S)
    metrics.set_meter_provider(
        MeterProvider(
            resource=resource,
            metric_readers=[reader],
            views=[
                View(instrument_name="pond.analysis.duration", aggregation=seconds),
                View(instrument_name="pond.stage.duration", aggregation=seconds),
            ],
        )
    )

    # Logs: the standard logging module, bridged. Records pick up the current trace id.
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    set_logger_provider(logger_provider)
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    for name in ("", "uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(name)
        target.addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("pond").setLevel(logging.INFO)

    # Automatic instrumentation. /health is hit by the container healthcheck every 15 s
    # and would bury the real requests; the per-chunk ASGI send/receive spans are noise too.
    if app is not None:
        FastAPIInstrumentor.instrument_app(app, excluded_urls="health", exclude_spans=["receive", "send"])
    HTTPXClientInstrumentor().instrument()
    URLLibInstrumentor().instrument()

    logging.getLogger("pond").info(
        "telemetry on: service=%s endpoint=%s",
        resource.attributes["service.name"],
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"],
    )
    return True
