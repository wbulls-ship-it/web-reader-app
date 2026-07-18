"""Provider-neutral orchestration service for text-to-speech."""

from __future__ import annotations

from collections.abc import Sequence

from .provider import SynthesisRequest, SynthesisResult, TTSProvider, Voice
from .text_utils import normalize_text, split_text


class TTSService:
    """Coordinates provider lookup, text preparation, and synthesis calls."""

    def __init__(self, providers: Sequence[TTSProvider], default_provider: str | None = None):
        self._providers = {provider.name: provider for provider in providers}
        if not self._providers:
            raise ValueError("at least one TTS provider is required")
        self._default_provider = default_provider or next(iter(self._providers))
        if self._default_provider not in self._providers:
            raise ValueError(f"unknown default provider: {self._default_provider}")

    def list_providers(self) -> list[str]:
        return list(self._providers)

    def list_voices(self, provider_name: str | None = None) -> list[Voice]:
        return list(self._provider(provider_name).list_voices())

    def synthesize(
        self,
        text: str,
        *,
        provider_name: str | None = None,
        voice_id: str | None = None,
        language: str | None = None,
        speaking_rate: float = 1.0,
        audio_format: str = "wav",
    ) -> SynthesisResult:
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("text is required")
        request = SynthesisRequest(
            text=normalized,
            voice_id=voice_id,
            language=language,
            speaking_rate=speaking_rate,
            audio_format=audio_format,
        )
        return self._provider(provider_name).synthesize(request)

    def split_for_provider(self, text: str, max_chars: int = 1_000) -> list[str]:
        return split_text(text, max_chars=max_chars)

    def _provider(self, provider_name: str | None) -> TTSProvider:
        name = provider_name or self._default_provider
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ValueError(f"unknown TTS provider: {name}") from exc
