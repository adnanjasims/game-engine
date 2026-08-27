#!/usr/bin/env python3
"""validate chrome tracing json and print cache / allocator counters."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="validate chrome tracing json")
    parser.add_argument("--input", default="trace.json", help="input trace path")
    parser.add_argument("--output", default="", help="optional copy destination")
    return parser.parse_args()


def counter_value(event: dict) -> float | None:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    if "value" in args:
        return float(args["value"])
    if "bytes" in args:
        return float(args["bytes"])
    return None


def main() -> int:
    args = parse_args()
    src = Path(args.input)
    if not src.exists():
        print(f"missing trace: {src}", file=sys.stderr)
        return 1
    with src.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    events = data.get("traceEvents")
    if not isinstance(events, list):
        print("invalid trace: missing traceEvents list", file=sys.stderr)
        return 1
    if data.get("displayTimeUnit") != "us":
        print("invalid trace: displayTimeUnit must be us", file=sys.stderr)
        return 1

    names = {e.get("name") for e in events if isinstance(e, dict)}
    if "process_name" not in names:
        print("invalid trace: missing process_name metadata", file=sys.stderr)
        return 1
    if "cache_hits" not in names or "cache_misses" not in names:
        print("invalid trace: missing cache counters", file=sys.stderr)
        return 1

    print(f"events: {len(events)}")
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "C":
            continue
        value = counter_value(event)
        if value is None:
            continue
        print(f"{event.get('name')}: {value:g}")

    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        print(f"copied: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
