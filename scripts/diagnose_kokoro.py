#!/usr/bin/env python3
"""Synthesize one Chinese sentence with both configured Kokoro voices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permit direct execution from a checkout without installing the application.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts import KokoroTTSProvider  # noqa: E402
from tts.provider import SynthesisRequest  # noqa: E402


VOICES = ("zf_xiaoxiao", "zf_xiaoyi")
DEFAULT_TEXT = "你好，欢迎使用网页朗读功能。今天天气很好。"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostic-audio"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    provider = KokoroTTSProvider()
    reports = []
    for voice in VOICES:
        result = provider.synthesize(
            SynthesisRequest(text=args.text, language="zh", voice_id=voice)
        )
        output = args.output_dir / f"{voice}.wav"
        output.write_bytes(result.audio)
        report = dict(result.metadata)
        report.update({"requested_voice": voice, "result_voice": result.voice_id, "output": str(output)})
        reports.append(report)
        print(json.dumps(report, indent=2, ensure_ascii=False))

    voices_honored = all(item["requested_voice"] == item["selected_voice"] for item in reports)
    audio_differs = reports[0]["audio_sha256"] != reports[1]["audio_sha256"]
    print(json.dumps({"voices_honored": voices_honored, "audio_differs": audio_differs}, indent=2))
    return 0 if voices_honored and audio_differs else 1


if __name__ == "__main__":
    raise SystemExit(main())
