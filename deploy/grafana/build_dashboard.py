"""Writes pond-dashboard.json, the Grafana dashboard for the OpenTelemetry metrics.

    python deploy/grafana/build_dashboard.py
    curl -u admin:admin -H 'Content-Type: application/json' \
         -d @deploy/grafana/pond-dashboard.json http://localhost:3000/api/dashboards/db
"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}


def target(expr, legend="", instant=True):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "instant": instant,
            "range": not instant, "refId": "A"}


def panel(i, title, kind, expr, x, y, w, h, legend="", unit="short", instant=True, desc=""):
    return {
        "id": i, "title": title, "type": kind, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr, legend, instant)],
        "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "palette-classic"}}, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"], "values": False}, "orientation": "horizontal",
                    "displayMode": "gradient", "showUnfilled": True},
    }


panels = [
    panel(1, "Analyses completed", "stat", 'sum(pond_analyses_total{outcome="ok"})', 0, 0, 4, 4,
          desc="Finished without error, summed over every worker."),
    panel(2, "Analyses failed", "stat", 'sum(pond_analyses_total{outcome="error"}) or vector(0)', 4, 0, 4, 4,
          desc="Mostly 'no pond site' on ground that cannot hold one."),
    panel(3, "Result-cache hit ratio", "stat", 'sum(pond_job_cache_total{result="hit"}) / sum(pond_job_cache_total)',
          8, 0, 4, 4, unit="percentunit", desc="Repeated areas are answered without any worker."),
    panel(4, "Analysis time, p95", "stat",
          'histogram_quantile(0.95, sum by (le) (pond_analysis_duration_seconds_bucket))', 12, 0, 4, 4, unit="s"),
    panel(5, "Analysis time, p50", "stat",
          'histogram_quantile(0.50, sum by (le) (pond_analysis_duration_seconds_bucket))', 16, 0, 4, 4, unit="s"),
    panel(6, "Jobs waiting", "stat", 'sum(pond_jobs_queued)', 20, 0, 4, 4),
    panel(7, "Average time per stage", "bargauge",
          'sum by (stage) (pond_stage_duration_seconds_sum) / sum by (stage) (pond_stage_duration_seconds_count)',
          0, 4, 12, 8, legend="{{stage}}", unit="s",
          desc="Where an analysis spends its time. Hydrology includes the Open-Meteo call."),
    panel(8, "Analyses per worker", "bargauge",
          'sum by (service_instance_id) (pond_analyses_total)', 12, 4, 6, 8, legend="{{service_instance_id}}",
          desc="How evenly the least_busy dispatcher spread the work."),
    panel(9, "Elevation tiles by source", "piechart", 'sum by (source) (pond_tiles_total)', 18, 4, 6, 8,
          legend="{{source}}", desc="seed = committed tiles, disk/memory = caches, network = fetched."),
    panel(10, "Requests by route (gateway)", "bargauge",
          'sum by (http_method, http_target, http_status_code) (http_server_duration_milliseconds_count{service_name="pond-gateway"})',
          0, 12, 12, 8, legend="{{http_method}} {{http_target}} {{http_status_code}}",
          desc="From the FastAPI auto-instrumentation."),
    panel(11, "Jobs running on each worker", "timeseries", 'sum by (worker) (pond_worker_in_flight)', 12, 12, 12, 8,
          legend="{{worker}}", instant=False, desc="Each worker takes one analysis at a time."),
]

dashboard = {
    "dashboard": {
        "uid": "pond-otel", "title": "Pond Catchment - OpenTelemetry", "tags": ["pond"],
        "timezone": "browser", "schemaVersion": 39, "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"}, "panels": panels,
    },
    "overwrite": True,
}
out = Path(__file__).with_name("pond-dashboard.json")
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print("wrote", out)
