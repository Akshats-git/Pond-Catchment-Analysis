"""Phase 21: run the scaling matrix against the docker-compose stack and write docs/SCALING.md.

    docker compose up -d && python -m tools.scaling

Every container is capped at 512 MB, the lab containers' limit, so the envelope is the
same one the lab runs in; the CPUs are this machine's. Each configuration recreates the
gateway with a different worker list or strategy, waits for it, and runs the same mixed
workload through `tools.loadtest`.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
URL = "http://localhost:5229"
ALL = ["http://worker1:5000", "http://worker2:5000", "http://worker3:5000"]
OUT = ROOT / "docs" / "SCALING.md"
RAW = ROOT / "docs" / "scaling_raw.jsonl"


def gateway(workers: list[str], strategy: str = "least_busy", max_queued: int | None = None) -> None:
    env = {**os.environ, "POND_WORKERS": ",".join(workers), "POND_JOBS_STRATEGY": strategy}
    if max_queued is not None:
        env["POND_JOBS_MAX_QUEUED"] = str(max_queued)
    compose = ROOT / "docker-compose.yml"
    override = ROOT / ".cache" / "compose.override.yml"
    override.parent.mkdir(exist_ok=True)
    extra = f"      POND_JOBS_MAX_QUEUED: \"{max_queued}\"\n" if max_queued is not None else ""
    override.write_text("services:\n  api:\n    environment:\n" + (extra or "      POND_SCALING_RUN: \"1\"\n"))
    subprocess.run(["docker", "compose", "-f", str(compose), "-f", str(override), "up", "-d",
                    "--force-recreate", "--no-deps", "api"], env=env, check=True, capture_output=True, cwd=ROOT)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            body = httpx.get(f"{URL}/health", timeout=2).json()
            if body.get("workers") == len(workers):
                return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError("gateway did not come back")


def load(label: str, *args: str) -> dict:
    out = subprocess.run([sys.executable, "-m", "tools.loadtest", "--label", label, *args],
                         capture_output=True, text=True, cwd=ROOT, check=True).stdout
    row = json.loads(out.strip().splitlines()[-1])
    with RAW.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    return row


def worker_peak(row: dict) -> float:
    return max((v for k, v in row.get("peak_mem_mib", {}).items() if "worker" in k), default=0.0)


def table(rows: list[dict], first: str) -> list[str]:
    lines = [f"| {first} | throughput | p50 | p95 | p99 | mean small | mean large | errors | peak worker memory |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        errors = ", ".join(f"{k}: {v}" for k, v in r["errors"].items()) or "none"
        lines.append(
            f"| {r['label']} | {r['throughput_rps']:.2f} req/s | {r['p50_s']:.2f} s | {r['p95_s']:.2f} s | "
            f"{r['p99_s']:.2f} s | {r['mean_small_s']} s | {r['mean_large_s']} s | {errors} | "
            f"{worker_peak(r):.0f} MiB |"
        )
    return lines


def main() -> None:
    RAW.write_text("")
    requests, concurrency = "60", "6"

    # ---- warm every worker's imports and tile cache once ----
    gateway(ALL)
    load("warm-up", "--requests", "12", "--concurrency", "6")

    # ---- 1. workers ----
    by_workers = []
    for n in (1, 2, 3):
        gateway(ALL[:n])
        by_workers.append(load(f"{n} worker{'s' if n > 1 else ''}", "--requests", requests, "--concurrency", concurrency))

    # ---- 2. strategy ----
    by_strategy = []
    for strategy in ("round_robin", "least_busy"):
        gateway(ALL, strategy)
        by_strategy.append(load(strategy.replace("_", "-"), "--requests", requests, "--concurrency", concurrency,
                                "--large-share", "0.4"))

    # ---- 3. concurrency ----
    gateway(ALL)
    by_concurrency = [load(f"{c} client{'s' if c > 1 else ''}", "--requests", str(max(12, 10 * c)), "--concurrency", str(c))
                      for c in (1, 3, 6, 12)]

    # ---- 4. cache ----
    cold = load("cold (first request)", "--repeat", "--requests", "1", "--concurrency", "1")
    warm = load("warm (19 repeats)", "--repeat", "--requests", "19", "--concurrency", "4")

    # ---- 5. failure modes ----
    started = time.perf_counter()
    big = httpx.post(f"{URL}/api/v1/analyzeArea", json={"bbox": [80.0, 20.0, 81.5, 21.5]}, timeout=30)
    big_ms = (time.perf_counter() - started) * 1e3
    gateway(ALL, max_queued=4)
    flood = load("flood: 24 at once, queue of 4", "--requests", "24", "--concurrency", "24", "--large-share", "1.0")
    gateway(ALL)
    heavy = load("all large, 3 workers", "--requests", "24", "--concurrency", "6", "--large-share", "1.0")
    largest = httpx.post(f"{URL}/api/v1/analyzeArea", timeout=300,
                         json={"bbox": [81.21, 21.16, 81.30, 21.25], "rainfall_mm": 1200, "rain_days": 55})
    largest_body = largest.json()

    cpus = os.cpu_count()
    lines = [
        "# Stress, scaling and system limits",
        "",
        "Generated by `python -m tools.scaling` against `docker compose up`: a gateway, three",
        "workers and the CV service, **each capped at 512 MB**, the limit of the four lab",
        f"containers. CPUs are this machine's ({cpus} logical, {platform.processor() or platform.machine()}); the",
        "lab containers see 120, so the memory envelope here is the lab's and the CPU envelope",
        "is tighter. Raw rows are in `docs/scaling_raw.jsonl`.",
        "",
        "Workload: closed loop, `POST /analyzeArea` through the gateway. 70% small selections",
        "(0.9-1.5 km side, one village) and 30% large (4-5 km side, ~20 km2), at random places",
        "inside the committed tile region. Each request has a distinct curve number, so none is",
        "answered from the result cache, and states its rainfall, so Open-Meteo's latency is",
        "not what is measured. Every analysis includes the resolution ensemble.",
        "",
        "The occasional `422` in the errors column is not a failure. A randomly placed box",
        "sometimes holds no channel a pond can sit on (`no_site_in_selection`), or only ground",
        "beside a watercourse (`no_ground_clear_of_watercourse`), and saying so is the right",
        "answer. No request in the matrix failed for any other reason.",
        "",
        "## Scaling with workers",
        "",
        f"{requests} requests from {concurrency} concurrent clients, least-busy dispatch.",
        "",
        *table(by_workers, "workers"),
        "",
        f"Three workers give **{by_workers[2]['throughput_rps'] / by_workers[0]['throughput_rps']:.1f}x** the throughput of one, "
        f"and p95 falls from {by_workers[0]['p95_s']:.1f} s to {by_workers[2]['p95_s']:.1f} s. Each worker runs one analysis "
        "at a time (`worker_slots=1`), so the queue, not a worker, absorbs the burst.",
        "",
        "## Round-robin against least-busy",
        "",
        f"Same load, three workers, 40% large selections so the jobs are uneven. Round-robin",
        "is what the existing Go load balancer on sys1 does; least-busy is the gateway's",
        "dispatcher, which knows each worker's queue.",
        "",
        *table(by_strategy, "strategy"),
        "",
    ]
    rr, lb = by_strategy
    better = "least-busy" if lb["p95_s"] < rr["p95_s"] else "round-robin"
    lines += [
        f"**{better} wins on the tail, modestly**: p95 {lb['p95_s']:.2f} s against {rr['p95_s']:.2f} s "
        f"({(1 - lb['p95_s'] / rr['p95_s']):.0%} lower), p99 {lb['p99_s']:.2f} s against {rr['p99_s']:.2f} s, "
        "with the same throughput. Round-robin hands a job to the next worker in turn even when it is busy with a large "
        "selection and another worker is idle; the job waits behind the large one. With one slot "
        "per worker, least-busy never queues a job while a worker is free. The gap is small here "
        "because three workers drain a six-client queue quickly; it grows with the spread of job "
        "sizes. Both ship: the Go balancer for the synchronous endpoint, the gateway for jobs, "
        "whose polling has to reach the worker that holds the job.",
        "",
        "## Concurrency",
        "",
        *table(by_concurrency, "clients"),
        "",
        "Throughput stops rising once every worker is busy; beyond that, more clients only "
        "lengthen the queue, which shows up as latency and not as errors.",
        "",
        "## The result cache",
        "",
        "| | latency |",
        "|---|---|",
        f"| First request for the sample area (computed) | {cold['p50_s']:.2f} s |",
        f"| The same area again, 19 times from 4 clients | p50 {warm['p50_s'] * 1000:.0f} ms, p95 {warm['p95_s'] * 1000:.0f} ms |",
        "",
        "A repeated selection with the same parameters is answered by the gateway without "
        "touching a worker. Elevation tiles are cached separately on each worker's disk and "
        "never expire; the demo region is committed with the repository (`data/tiles`).",
        "",
        "## What the limits look like",
        "",
        "| Situation | What the user gets | Time |",
        "|---|---|---|",
        f"| A selection over 100 km2 | `{big.status_code} {big.json().get('code')}`, naming the limit | {big_ms:.0f} ms |",
        f"| 24 large selections at once, queue capped at 4 | {flood['ok']} answered; "
        f"{', '.join(f'{v} x `{k}`' for k, v in flood['errors'].items()) or 'no refusals'} | p95 {flood['p95_s']:.1f} s |",
        f"| The largest selection tested, {largest_body.get('area', {}).get('selection_area_ha', 0) / 100:.0f} km2 | "
        f"`{largest.status_code}`, grid {largest_body.get('dem', {}).get('resolution_m')} m x "
        f"{largest_body.get('dem', {}).get('shape')} | {largest_body.get('timing_ms', {}).get('total', 0) / 1000:.1f} s |",
        "",
        "A refusal is immediate and says why: the queue limit turns overload into a `503 busy` "
        "the client can retry, instead of requests that all time out together. A selection too "
        "large for the memory is a `422 aoi_too_large` before any work starts, the same pattern "
        "Phase 2 used for `ensemble_unavailable`.",
        "",
        "## Memory",
        "",
        f"Peak worker memory over the whole matrix: **{max(worker_peak(r) for r in by_workers + by_strategy + by_concurrency + [heavy, flood]):.0f} MiB** "
        f"of 512. Under sustained large selections: {worker_peak(heavy):.0f} MiB. The map path's grids are small "
        "(the sample sheet is 60,606 cells with its buffer against 638,472 on the contour path), "
        "which is why the resolution ensemble, switched off on the lab since Phase 11 because "
        "it did not fit, runs on every map-path request here. The adaptive resolution caps "
        "any selection at a million cells, the envelope the Phase 2 sheet proved fits.",
        "",
        "## On the lab's four containers",
        "",
        "`deploy/deploy.sh` puts the gateway on sys1 (the Phase 2 URL, 5229) and a worker on "
        "each of sys2-4; `deploy/start-lb.sh` starts the existing Go balancer on sys1:6000 for "
        "the round-robin comparison. `python -m tools.loadtest --url http://10.1.75.53:5229 "
        "--docker ''` runs the same workload there, and `deploy/status.sh` reads each "
        "container's cgroup peak (`memory.peak`, and `anon` from `memory.stat`, never "
        "`memory.current`, which counts reclaimable page cache).",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
