#!/usr/bin/env python3
"""validate and copy chrome tracing json."""

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
    print(f"events: {len(events)}")
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        print(f"copied: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
