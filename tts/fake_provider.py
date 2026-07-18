"""Deterministic fake TTS provider for tests and local development."""

from __future__ import annotations

from .provider import SynthesisRequest, SynthesisResult, Voice
from .text_utils import normalize_text


class FakeTTSProvider:
    """A provider implementation that returns deterministic bytes, not real speech."""

    name = "fake"

    def __init__(self, voices: list[Voice] | None = None):
        self._voices = voices or [Voice(id="fake-default", name="Fake Default", language="en-US", provider=self.name)]
        self.requests: list[SynthesisRequest] = []

    def list_voices(self) -> list[Voice]:
        return list(self._voices)

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        normalized = normalize_text(request.text)
        voice_id = request.voice_id or self._voices[0].id
        payload = f"FAKE_TTS\nvoice={voice_id}\nformat={request.audio_format}\ntext={normalized}".encode("utf-8")
        return SynthesisResult(
            audio=payload,
            audio_format=request.audio_format,
            provider=self.name,
            voice_id=voice_id,
            metadata={"characters": str(len(normalized))},
        )
