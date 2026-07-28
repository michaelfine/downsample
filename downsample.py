#!/usr/bin/env python3
"""Reduce the bit depth of audio files (e.g. 32-bit or 64-bit float) to 16-bit PCM.

Despite the project name, this performs *bit-depth* reduction (requantization),
not sample-rate conversion. The sample rate and channel layout are preserved.

Reads anything libsndfile understands (WAV, AIFF, FLAC, ...) and writes 16-bit
PCM. Optional TPDF dithering trades a small amount of low-level noise for the
elimination of harmonic quantization distortion, which is usually the better
choice when going to 16-bit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

INT16_MAX = 32767
INT16_MIN = -32768


def to_int16(samples: np.ndarray, dither: bool, rng: np.random.Generator) -> np.ndarray:
    """Quantize float samples in [-1.0, 1.0] to int16 with optional TPDF dither.

    ``samples`` may be any float dtype and any shape (mono or (frames, channels)).
    """
    x = samples.astype(np.float64, copy=True)

    if dither:
        # Triangular PDF dither, ±1 LSB peak-to-peak, as the sum of two
        # independent uniform variables scaled to one quantization step.
        lsb = 1.0 / INT16_MAX
        noise = (rng.random(x.shape) - rng.random(x.shape)) * lsb
        x += noise

    scaled = np.rint(x * INT16_MAX)
    np.clip(scaled, INT16_MIN, INT16_MAX, out=scaled)
    return scaled.astype(np.int16)


def downsample_file(
    src: Path,
    dst: Path,
    dither: bool = True,
    seed: int | None = None,
    overwrite: bool = False,
) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {dst} (use --overwrite)")

    info = sf.info(str(src))
    # Read as float64 regardless of source subtype so int and float sources are
    # both normalized into [-1.0, 1.0] before requantizing.
    data, samplerate = sf.read(str(src), dtype="float64", always_2d=False)

    rng = np.random.default_rng(seed)
    quantized = to_int16(data, dither=dither, rng=rng)

    sf.write(str(dst), quantized, samplerate, subtype="PCM_16")

    print(
        f"{src.name}: {info.subtype} @ {info.samplerate} Hz, {info.channels} ch "
        f"-> {dst.name}: PCM_16 (dither={'on' if dither else 'off'})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="input audio file(s)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (single input only); default: alongside input with a -16bit suffix",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="directory to write outputs into (for one or more inputs)",
    )
    parser.add_argument(
        "--no-dither",
        action="store_true",
        help="disable TPDF dithering (produces truncation/rounding distortion)",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible dither")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting outputs")
    args = parser.parse_args(argv)

    if args.output and len(args.inputs) > 1:
        parser.error("--output takes a single input; use --outdir for multiple files")

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)

    rc = 0
    for src in args.inputs:
        if not src.exists():
            print(f"error: no such file: {src}", file=sys.stderr)
            rc = 1
            continue

        if args.output:
            dst = args.output
        else:
            out_name = f"{src.stem}-16bit{src.suffix}"
            dst = (args.outdir / out_name) if args.outdir else src.with_name(out_name)

        try:
            downsample_file(
                src,
                dst,
                dither=not args.no_dither,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - surface a clean message per file
            print(f"error processing {src}: {exc}", file=sys.stderr)
            rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
