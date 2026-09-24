"""Load generator for the map path (Phase 21): throughput, latency, errors and memory.

    python -m tools.loadtest --url http://localhost:5229 --requests 60 --concurrency 6
    python -m tools.loadtest --url http://10.1.75.53:5229 --docker ""        # the lab

A closed loop: `concurrency` clients, each sending its next request as soon as the last one
answers, until `requests` have been sent. Every request is `POST /analyzeArea`, a mix of
small selections (~1-2 km2, the common case: one village) and large ones (~20 km2), at
seeded random positions inside the committed tile region so the network is never what is
being measured. Each request carries a distinct curve number, so the result cache never
answers for a request that is meant to be computed; `--repeat` does the opposite, to
measure the cache.

Memory is sampled from `docker stats` for the named containers while the load runs; on
the lab it is read from each container's cgroup by `deploy/status.sh` instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import threading
import time
from collections import Counter

import httpx

REGION = (81.20, 21.15, 81.40, 21.35)
"""Inside the committed seed tiles (data/tiles), with room for the 25% buffer."""


def selection(rng: random.Random, large: bool) -> list[float]:
    side_km = rng.uniform(4.0, 5.0) if large else rng.uniform(0.9, 1.5)
    dlon = side_km / (111.32 * 0.9317)
    dlat = side_km / 110.54
    pad_lon, pad_lat = dlon * 0.6, dlat * 0.6
    lon = rng.uniform(REGION[0] + pad_lon, REGION[2] - pad_lon - dlon)
    lat = rng.uniform(REGION[1] + pad_lat, REGION[3] - pad_lat - dlat)
    return [round(lon, 5), round(lat, 5), round(lon + dlon, 5), round(lat + dlat, 5)]


def workload(n: int, large_share: float, seed: int, repeat: bool) -> list[dict]:
    rng = random.Random(seed)
    bodies = []
    for i in range(n):
        if repeat:
            bodies.append({"bbox": [81.2814, 21.2398, 81.3126, 21.2636]})
            continue
        large = rng.random() < large_share
        bodies.append({
            "bbox": selection(rng, large),
            # Distinct per request and per run, so no request meant to be computed is
            # answered from the result cache.
            "curve_number": round(rng.uniform(55.0, 95.0), 4),
            # Stated rather than fetched: a load test should measure this system, not
            # Open-Meteo's latency, and must not hammer a free service with it.
            "rainfall_mm": 1200.0,
            "rain_days": 55,
            "_large": large,
        })
    return bodies


class MemorySampler(threading.Thread):
    """Peak `docker stats` memory per container while the load runs."""

    def __init__(self, containers: list[str]) -> None:
        super().__init__(daemon=True)
        self.containers = containers
        self.peak: dict[str, float] = {}
        self._halt = threading.Event()

    @staticmethod
    def _mib(text: str) -> float:
        value = text.split("/")[0].strip()
        for unit, scale in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024), ("B", 1 / 1048576)):
            if value.endswith(unit):
                return float(value[: -len(unit)]) * scale
        return 0.0

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}", *self.containers],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                for line in out.splitlines():
                    name, usage = line.split("\t")
                    self.peak[name] = max(self.peak.get(name, 0.0), self._mib(usage))
            except Exception:  # noqa: BLE001, sampling is best-effort
                pass
            self._halt.wait(0.5)

    def stop(self) -> dict[str, float]:
        self._halt.set()
        self.join(timeout=15)
        return {k: round(v, 1) for k, v in sorted(self.peak.items())}


async def run(url: str, bodies: list[dict], concurrency: int, timeout: float) -> list[dict]:
    queue: asyncio.Queue = asyncio.Queue()
    for index, body in enumerate(bodies):
        queue.put_nowait((index, body))
    results: list[dict] = []

    async def client(worker: int) -> None:
        async with httpx.AsyncClient(timeout=timeout) as http:
            while True:
                try:
                    index, body = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                payload = {k: v for k, v in body.items() if not k.startswith("_")}
                started = time.perf_counter()
                try:
                    response = await http.post(f"{url}/api/v1/analyzeArea", json=payload)
                    status = response.status_code
                    code = None if status == 200 else response.json().get("code")
                    server_ms = response.json().get("timing_ms", {}).get("total") if status == 200 else None
                except (httpx.HTTPError, ValueError) as exc:
                    status, code, server_ms = 0, type(exc).__name__, None
                results.append({
                    "index": index, "large": body.get("_large", False), "status": status, "code": code,
                    "latency_s": time.perf_counter() - started, "server_ms": server_ms,
                })

    await asyncio.gather(*(client(i) for i in range(concurrency)))
    return results


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(results: list[dict], wall_s: float) -> dict:
    ok = [r for r in results if r["status"] == 200]
    lat = [r["latency_s"] for r in ok]
    small = [r["latency_s"] for r in ok if not r["large"]]
    large = [r["latency_s"] for r in ok if r["large"]]
    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": dict(Counter(f"{r['status']} {r['code']}" for r in results if r["status"] != 200)),
        "wall_s": round(wall_s, 2),
        "throughput_rps": round(len(ok) / wall_s, 3) if wall_s else 0,
        "p50_s": round(percentile(lat, 0.50), 2),
        "p95_s": round(percentile(lat, 0.95), 2),
        "p99_s": round(percentile(lat, 0.99), 2),
        "mean_small_s": round(statistics.mean(small), 2) if small else None,
        "mean_large_s": round(statistics.mean(large), 2) if large else None,
        "server_ms_mean": round(statistics.mean([r["server_ms"] for r in ok if r["server_ms"]]), 1) if ok else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:5229")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--large-share", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=None, help="default: a fresh one per run")
    parser.add_argument("--repeat", action="store_true", help="one AOI every time: measures the result cache")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--docker", default="pondcatchmentanalysis-worker1-1,pondcatchmentanalysis-worker2-1,"
                        "pondcatchmentanalysis-worker3-1,pondcatchmentanalysis-api-1")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % 2**31
    bodies = workload(args.requests, args.large_share, seed, args.repeat)
    sampler = MemorySampler([c for c in args.docker.split(",") if c]) if args.docker else None
    if sampler:
        sampler.start()
    started = time.perf_counter()
    results = asyncio.run(run(args.url, bodies, args.concurrency, args.timeout))
    wall = time.perf_counter() - started
    summary = {"label": args.label, "concurrency": args.concurrency, **summarise(results, wall)}
    if sampler:
        summary["peak_mem_mib"] = sampler.stop()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
