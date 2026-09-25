"""Every endpoint of a running service, once, with the answer checked: a deployment check.

    python -m tools.smoke --url http://localhost:5000        # on sys1, against the gateway
    python -m tools.smoke --url http://localhost:5229        # the compose stack

Uses the real network where the endpoint does (Open-Meteo, Nominatim, Esri imagery), so a
failure there is reported as that endpoint's failure, not hidden. Prints one line per check
and exits non-zero if any failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

SAMPLE = [81.2814, 21.2398, 81.3126, 21.2636]
SHEET = Path(__file__).resolve().parent.parent / "data" / "contours_1m.kml"


class Smoke:
    def __init__(self, url: str) -> None:
        self.client = httpx.Client(base_url=url, timeout=300)
        self.failures = 0

    def check(self, name: str, fn) -> object:
        started = time.perf_counter()
        try:
            detail = fn()
            ok = True
        except Exception as exc:  # noqa: BLE001, a smoke test reports, it does not stop
            detail, ok = f"{exc.__class__.__name__}: {exc}", False
            self.failures += 1
        ms = (time.perf_counter() - started) * 1e3
        print(f"{'PASS' if ok else 'FAIL'}  {name:<34} {ms:8.0f} ms  {detail}", flush=True)
        return detail if ok else None

    def get(self, path: str, **kw) -> httpx.Response:
        return self.client.get(path, **kw)

    def post(self, path: str, **kw) -> httpx.Response:
        return self.client.post(path, **kw)

    def delete(self, path: str) -> httpx.Response:
        return self.client.delete(path)


def expect(response: httpx.Response, code: int = 200) -> httpx.Response:
    if response.status_code != code:
        raise AssertionError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response


def where(site: dict) -> str:
    return (f"site {site['location']['lat']:.5f},{site['location']['lon']:.5f}  "
            f"catchment {site['catchment']['area_ha']:.1f} ha")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:5000")
    args = parser.parse_args()
    s = Smoke(args.url)
    api = "/api/v1"

    def health():
        body = expect(s.get("/health")).json()
        assert body["status"] == "ok", body
        return f"role={body.get('role')} workers={body.get('workers')} version={body.get('version')}"

    def cluster():
        body = expect(s.get(f"{api}/cluster")).json()
        healthy = [w["url"] for w in body["workers"] if w["healthy"]]
        assert len(healthy) == len(body["workers"]), body["workers"]
        return f"{body['strategy']}, healthy: {', '.join(healthy) or 'local'}"

    def page():
        text = expect(s.get("/")).text
        assert "<html" in text.lower() and "Try the sample" in text, "not the Phase 3 page"
        return f"{len(text) // 1024} KB"

    def area():
        body = expect(s.post(f"{api}/analyzeArea", json={"bbox": SAMPLE, "rainfall_mm": 1200, "rain_days": 55,
                                                          "curve_number": round(70 + time.time() % 10, 3)})).json()
        site = body["recommended_site"]
        return (f"{where(site)}  grid {body['dem']['resolution_m']} m  "
                f"worker total {body['timing_ms']['total']:.0f} ms")

    def area_cached():
        body = {"bbox": SAMPLE, "rainfall_mm": 1200, "rain_days": 55, "curve_number": 71.5}
        expect(s.post(f"{api}/analyzeArea", json=body))
        started = time.perf_counter()
        expect(s.post(f"{api}/analyzeArea", json=body))
        return f"repeat answered in {(time.perf_counter() - started) * 1e3:.0f} ms"

    def polygon():
        ring = [[81.285, 21.242], [81.305, 21.240], [81.310, 21.258], [81.290, 21.262]]
        body = expect(s.post(f"{api}/analyzeArea", json={"polygon": ring, "rainfall_mm": 1200, "rain_days": 55})).json()
        return f"{body['area']['selection_area_ha']:.0f} ha selected, site inside"

    def job():
        created = expect(s.post(f"{api}/jobs", json={"bbox": [81.30, 21.20, 81.33, 21.225], "rainfall_mm": 1100,
                                                      "rain_days": 50, "curve_number": round(60 + time.time() % 10, 3)}),
                         202).json()
        stages = []
        deadline = time.time() + 120
        while time.time() < deadline:
            body = expect(s.get(f"{api}/jobs/{created['job_id']}")).json()
            if body["stage"] not in stages:
                stages.append(body["stage"])
            if body["status"] in ("done", "failed"):
                break
            time.sleep(0.2)
        assert body["status"] == "done", body.get("error")
        return f"worker {body.get('worker')}; stages seen: {' > '.join(stages)}"

    def too_large():
        body = expect(s.post(f"{api}/analyzeArea", json={"bbox": [80.0, 20.0, 81.5, 21.5]}), 422).json()
        assert body["code"] == "aoi_too_large", body
        return body["code"]

    def contours_area():
        body = expect(s.post(f"{api}/contours", data={"bbox": ",".join(map(str, SAMPLE))})).json()
        return f"{body['contour_count']} lines at {body['interval_m']} m, {len(body['geojson']['features'])} features"

    def render_area():
        response = expect(s.post(f"{api}/renderMap", data={"bbox": ",".join(map(str, SAMPLE))}))
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
        return f"PNG {len(response.content) // 1024} KB"

    def sheet():
        with SHEET.open("rb") as handle:
            body = expect(s.post(f"{api}/analyzeContour", files={"contour_map": ("contours_1m.kml", handle)},
                                 data={"rainfall_mm": "1200", "rain_days": "55"})).json()
        return where(body["recommended_site"])

    def rainfall():
        body = expect(s.get(f"{api}/rainfall", params={"lat": 21.25, "lon": 81.29})).json()
        return (f"{body['annual_rainfall_mm']} mm/yr, measured={body['is_measured']}, "
                f"{len(body.get('annual_totals') or [])} years")

    def places():
        body = expect(s.get(f"{api}/places", params={"q": "Raipur"})).json()
        results = body.get("results", body if isinstance(body, list) else [])
        assert results, body
        return f"{len(results)} results, first: {results[0].get('name', results[0].get('display_name', ''))[:40]}"

    def ponds_detect():
        body = expect(s.post(f"{api}/imagery/detectPonds", json={"bbox": SAMPLE})).json()
        return f"{body['count']} water bodies, {body['total_area_ha']} ha"

    def land():
        body = expect(s.post(f"{api}/land/available", json={"bbox": SAMPLE})).json()
        return f"{body['available_share']:.0%} free, built {body['classes']['built']['share']:.1%}"

    def saved():
        created = expect(s.post(f"{api}/jobs", json={"bbox": SAMPLE, "rainfall_mm": 1200, "rain_days": 55}), 202).json()
        for _ in range(600):
            if expect(s.get(f"{api}/jobs/{created['job_id']}")).json()["status"] in ("done", "failed"):
                break
            time.sleep(0.2)
        pond = expect(s.post(f"{api}/ponds", json={"name": "smoke test", "job_id": created["job_id"]}), 201).json()
        pid = pond["id"]
        listed = expect(s.get(f"{api}/ponds")).json()
        expect(s.get(f"{api}/ponds/{pid}"))
        geo = expect(s.get(f"{api}/ponds.geojson")).json()
        expect(s.delete(f"{api}/ponds/{pid}"), 204)
        expect(s.get(f"{api}/ponds/{pid}"), 404)
        return f"saved, listed ({listed.get('total', '?')}), fetched, geojson ({len(geo['features'])}), deleted"

    for name, fn in [
        ("health", health), ("cluster", cluster), ("page", page),
        ("analyzeArea bbox", area), ("analyzeArea cache", area_cached), ("analyzeArea polygon", polygon),
        ("jobs with progress", job), ("aoi_too_large", too_large),
        ("contours from DEM", contours_area), ("renderMap for an area", render_area),
        ("analyzeContour (sheet)", sheet), ("rainfall (Open-Meteo)", rainfall), ("places (Nominatim)", places),
        ("imagery/detectPonds (Esri)", ponds_detect), ("land/available", land), ("saved analyses", saved),
    ]:
        s.check(name, fn)
    print(f"{s.failures} failed")
    return 1 if s.failures else 0


if __name__ == "__main__":
    sys.exit(main())
