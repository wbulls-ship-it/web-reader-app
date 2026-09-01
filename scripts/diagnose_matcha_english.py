#!/usr/bin/env python3
"""Compare reference, fixed-provider, and legacy-FST English Matcha output."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.matcha_provider import (  # noqa: E402
    MatchaPaths, MatchaTTSProvider, _create_engine, _wav_bytes,
)
from tts.provider import SynthesisRequest  # noqa: E402

DEFAULT_TEXT = "In 2026, the Web Reader processed 123 articles in 4 hours."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", help="matcha-icefall-zh-en directory")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", default="matcha-diagnostic-output")
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def report(label: str, elapsed: float, duration: float, payload: bytes, path: Path) -> None:
    path.write_bytes(payload)
    print(
        f"{label}: synthesis={elapsed:.3f}s audio={duration:.3f}s "
        f"RTF={elapsed / duration:.3f} sha256={hashlib.sha256(payload).hexdigest()} path={path}"
    )


def main() -> None:
    args = parse_args()
    paths = MatchaPaths.from_environment(args.model_dir)
    missing = paths.missing()
    if missing:
        raise SystemExit("Missing model assets: " + ", ".join(map(str, missing)))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # A: the minimal English reference: identical model assets, sid, speed,
    # vocoder, and sample rate, with no language-inappropriate TN rules.
    started = time.perf_counter()
    reference = _create_engine(paths, args.threads, "en").generate(args.text, sid=0, speed=1.0)
    elapsed = time.perf_counter() - started
    reference_wav = _wav_bytes(reference.samples, int(reference.sample_rate))
    report(
        "A reference (no FSTs)", elapsed, len(reference.samples) / reference.sample_rate,
        reference_wav, output / "a-reference-english.wav",
    )

    # B: exercise the application API with the exact same sentence.
    provider = MatchaTTSProvider(paths, num_threads=args.threads)
    started = time.perf_counter()
    result = provider.synthesize(SynthesisRequest(text=args.text, language="en"))
    elapsed = time.perf_counter() - started
    report("B application provider", elapsed, result.duration_seconds, result.audio,
           output / "b-provider-english.wav")

    # Reproduce the pre-fix production configuration for a direct listening
    # comparison: an English sentence through all three Chinese TN grammars.
    started = time.perf_counter()
    legacy = _create_engine(paths, args.threads, "zh").generate(args.text, sid=0, speed=1.0)
    elapsed = time.perf_counter() - started
    legacy_wav = _wav_bytes(legacy.samples, int(legacy.sample_rate))
    report(
        "Legacy regression (Chinese FSTs)", elapsed, len(legacy.samples) / legacy.sample_rate,
        legacy_wav, output / "legacy-chinese-fsts-english.wav",
    )


if __name__ == "__main__":
    main()
