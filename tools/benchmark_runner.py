#!/usr/bin/env python3
"""run eoc_bench and print parsed timings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run eoc_bench and parse metrics")
    parser.add_argument("--bin", default="build/eoc_bench", help="path to eoc_bench")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = Path(args.bin)
    if not binary.exists():
        print(f"missing binary: {binary}", file=sys.stderr)
        return 1
    proc = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    metrics = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = value.strip()
    required = (
        "arena_ns_per_alloc",
        "pool_ns_per_alloc",
        "taskgraph_independent_us",
        "taskgraph_dag_us",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        print("missing metrics: " + ", ".join(missing), file=sys.stderr)
        return 1
    print("parsed_ok: 1")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
