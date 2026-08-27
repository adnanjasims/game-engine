#!/usr/bin/env python3
"""run eoc_bench, optionally sweeping worker count and arena size."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run eoc_bench and parse metrics")
    parser.add_argument("--bin", default="build/eoc_bench", help="path to eoc_bench")
    parser.add_argument("--workers", default="", help="single worker count")
    parser.add_argument("--arena-bytes", default="", help="single arena size in bytes")
    parser.add_argument("--frames", type=int, default=-1, help="tick frames, -1 keeps binary default")
    parser.add_argument("--assets", default="", help="optional packed asset directory")
    parser.add_argument("--trace", default="", help="chrome trace output path")
    parser.add_argument(
        "--sweep-workers",
        default="",
        help="comma list of worker counts, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--sweep-arena",
        default="",
        help="comma list of arena sizes, e.g. 65536,262144,1M",
    )
    parser.add_argument("--csv", default="", help="optional csv output path")
    return parser.parse_args()


def parse_size(text: str) -> int:
    raw = text.strip().lower().replace("_", "")
    if not raw:
        raise ValueError("empty size")
    mul = 1
    if raw.endswith("k"):
        mul = 1024
        raw = raw[:-1]
    elif raw.endswith("m"):
        mul = 1024 * 1024
        raw = raw[:-1]
    return int(float(raw) * mul)


def parse_list(text: str, sizes: bool) -> list[str]:
    if not text.strip():
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(str(parse_size(part) if sizes else int(part)))
    return out


def parse_metrics(stdout: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def run_once(
    binary: Path,
    workers: str,
    arena: str,
    frames: int,
    assets: str,
    trace: str,
) -> tuple[int, dict[str, str], str, str]:
    cmd = [str(binary)]
    if workers:
        cmd.extend(["--workers", workers])
    if arena:
        cmd.extend(["--arena-bytes", arena])
    if frames >= 0:
        cmd.extend(["--frames", str(frames)])
    if assets:
        cmd.extend(["--assets", assets])
    if trace:
        cmd.extend(["--trace", trace])
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return 124, parse_metrics(stdout), stdout, stderr + "timeout\n"
    metrics = parse_metrics(proc.stdout)
    return proc.returncode, metrics, proc.stdout, proc.stderr


def required_ok(metrics: dict[str, str]) -> list[str]:
    required = (
        "arena_ns_per_alloc",
        "pool_ns_per_alloc",
        "taskgraph_independent_us",
        "taskgraph_dag_us",
    )
    return [name for name in required if name not in metrics]


def print_row(metrics: dict[str, str]) -> None:
    workers = metrics.get("taskgraph_workers", "?")
    arena = metrics.get("arena_bytes", "?")
    indep = metrics.get("taskgraph_independent_us", "?")
    dag = metrics.get("taskgraph_dag_us", "?")
    tick = metrics.get("tick_us_avg", "?")
    hits = metrics.get("cache_hits", "?")
    misses = metrics.get("cache_misses", "?")
    print(
        f"sweep workers={workers} arena={arena} independent_us={indep} "
        f"dag_us={dag} tick_us_avg={tick} cache_hits={hits} cache_misses={misses}"
    )


def main() -> int:
    args = parse_args()
    binary = Path(args.bin)
    if not binary.exists():
        print(f"missing binary: {binary}", file=sys.stderr)
        return 1

    workers = parse_list(args.sweep_workers, sizes=False)
    arenas = parse_list(args.sweep_arena, sizes=True)
    if args.workers:
        workers = [str(int(args.workers))]
    if args.arena_bytes:
        arenas = [str(parse_size(args.arena_bytes))]
    if not workers:
        workers = [""]
    if not arenas:
        arenas = [""]

    rows: list[dict[str, str]] = []
    worst = 0
    for w in workers:
        for a in arenas:
            rc, metrics, stdout, stderr = run_once(
                binary, w, a, args.frames, args.assets, args.trace
            )
            missing = required_ok(metrics)
            if missing:
                print("missing metrics: " + ", ".join(missing), file=sys.stderr)
                if stderr:
                    sys.stderr.write(stderr)
                return 1
            if len(workers) == 1 and len(arenas) == 1:
                sys.stdout.write(stdout)
                if stderr:
                    sys.stderr.write(stderr)
            else:
                print_row(metrics)
            metrics["_rc"] = str(rc)
            rows.append(metrics)
            if rc != 0:
                worst = rc

    if args.csv:
        keys = [
            "taskgraph_workers",
            "arena_bytes",
            "taskgraph_independent_us",
            "taskgraph_dag_us",
            "tick_us_avg",
            "cache_hits",
            "cache_misses",
            "tick_ok",
            "_rc",
        ]
        dest = Path(args.csv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in keys})
        print(f"csv: {dest}")

    print("parsed_ok: 1")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
