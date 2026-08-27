#!/usr/bin/env python3
"""generate packed pcm + i420 assets for eoc_bench --assets."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pack demo audio and i420 frames")
    parser.add_argument("--out", default="assets", help="output directory")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2048)
    return parser.parse_args()


def i420_size(width: int, height: int) -> int:
    return width * height + (width * height) // 2


def write_pcm(path: Path, rate: int, channels: int, samples: int) -> None:
    frames = []
    for i in range(samples):
        t = i / float(rate)
        sample = int(16000.0 * math.sin(2.0 * math.pi * 440.0 * t))
        sample = max(-32767, min(32767, sample))
        for _ in range(channels):
            frames.append(struct.pack("<h", sample))
    path.write_bytes(b"".join(frames))


def write_i420(path: Path, width: int, height: int, frames: int) -> None:
    y_size = width * height
    uv_size = y_size // 4
    blob = bytearray()
    for f in range(frames):
        t = (f * 3) & 255
        y = bytes(((x + y + t) & 255) for y in range(height) for x in range(width))
        u = bytes([128] * uv_size)
        v = bytes((64 + ((x + t) & 63)) for y in range(height // 2) for x in range(width // 2))
        if len(v) != uv_size:
            v = (v + bytes(uv_size))[:uv_size]
        blob.extend(y)
        blob.extend(u)
        blob.extend(v)
    path.write_bytes(bytes(blob))


def main() -> int:
    args = parse_args()
    if args.width < 2 or args.height < 2 or (args.width & 1) or (args.height & 1):
        print("width and height must be even and >= 2", flush=True)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audio = out / "tone.s16le"
    video = out / "clip.i420"
    write_pcm(audio, args.rate, args.channels, args.samples)
    write_i420(video, args.width, args.height, args.frames)
    manifest = out / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"sample_rate={args.rate}",
                f"channels={args.channels}",
                f"width={args.width}",
                f"height={args.height}",
                f"frames={args.frames}",
                "audio=tone.s16le",
                "video=clip.i420",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {audio}")
    print(f"wrote {video} ({i420_size(args.width, args.height) * args.frames} bytes)")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
