"""Provider-neutral interfaces for text-to-speech backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Voice:
    """A provider-neutral description of a synthetic voice."""

    id: str
    name: str
    language: str
    provider: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SynthesisRequest:
    """A provider-neutral request to synthesize speech from text."""

    text: str
    voice_id: str | None = None
    language: str | None = None
    speaking_rate: float = 1.0
    audio_format: str = "wav"


@dataclass(frozen=True)
class SynthesisResult:
    """Provider-neutral synthesized audio and descriptive metadata."""

    audio: bytes
    audio_format: str
    provider: str
    voice_id: str | None = None
    duration_seconds: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class TTSProvider(Protocol):
    """Interface every TTS backend adapter must implement."""

    name: str

    def list_voices(self) -> Sequence[Voice]:
        """Return voices supported by this provider."""

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize speech for a provider-neutral request."""
