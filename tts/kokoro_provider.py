"""Kokoro implementation of the provider-neutral TTS interface."""

from __future__ import annotations

import importlib
import io
import sys
import wave
from collections.abc import Callable, Iterable

from .provider import SynthesisRequest, SynthesisResult, Voice
from .text_utils import normalize_text

KOKORO_SAMPLE_RATE = 24_000
DEFAULT_VOICES = {
    "zh": "zf_xiaoxiao",
    "en": "af_heart",
}


class KokoroNotInstalledError(RuntimeError):
    """Raised when Kokoro cannot be used in the current Python environment."""


class KokoroSynthesisError(RuntimeError):
    """Raised when Kokoro does not produce usable audio."""


class KokoroTTSProvider:
    """Local Kokoro adapter. Kokoro is imported only when synthesis starts."""

    name = "kokoro"

    def __init__(self, pipeline_factory: Callable[..., object] | None = None):
        self._pipeline_factory = pipeline_factory
        self._pipelines: dict[str, object] = {}

    def list_voices(self) -> list[Voice]:
        return [
            Voice("zf_xiaoxiao", "Xiaoxiao", "zh-CN", self.name),
            Voice("zf_xiaoyi", "Xiaoyi", "zh-CN", self.name),
            Voice("af_heart", "Heart", "en-US", self.name),
        ]

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if request.audio_format.lower() != "wav":
            raise ValueError("Kokoro only supports WAV output")
        text = normalize_text(request.text)
        if not text:
            raise ValueError("text is required")
        if not 0.5 <= request.speaking_rate <= 2.0:
            raise ValueError("speaking_rate must be between 0.5 and 2.0")

        language = _language_code(request.language)
        voice = request.voice_id or DEFAULT_VOICES[language]
        if voice.startswith("zf_"):
            language = "zh"
        elif voice.startswith("af_"):
            language = "en"

        pipeline = self._pipeline(language)
        try:
            generated = pipeline(text, voice=voice, speed=request.speaking_rate)
            samples = _collect_samples(item[2] for item in generated)
        except (KokoroNotInstalledError, ValueError):
            raise
        except Exception as exc:
            raise KokoroSynthesisError(f"Kokoro speech synthesis failed: {exc}") from exc
        if not samples:
            raise KokoroSynthesisError("Kokoro speech synthesis produced no audio")

        return SynthesisResult(
            audio=_wav_bytes(samples),
            audio_format="wav",
            provider=self.name,
            voice_id=voice,
            metadata={"characters": str(len(text)), "language": language},
        )

    def _pipeline(self, language: str):
        if language not in self._pipelines:
            factory = self._pipeline_factory or self._load_pipeline_factory()
            self._pipelines[language] = factory(lang_code="z" if language == "zh" else "a")
        return self._pipelines[language]

    @staticmethod
    def _load_pipeline_factory():
        if sys.version_info >= (3, 13):
            raise KokoroNotInstalledError(
                "Kokoro is supported by this app on Python 3.11/3.12, not Python 3.13. "
                "Run the TTS app in a Python 3.11 environment."
            )
        try:
            return importlib.import_module("kokoro").KPipeline
        except (ImportError, AttributeError) as exc:
            raise KokoroNotInstalledError(
                "Kokoro is not installed. Install the Python 3.11 TTS dependencies from requirements.txt."
            ) from exc


def _language_code(language: str | None) -> str:
    return "zh" if (language or "").lower().startswith("zh") else "en"


def _collect_samples(parts: Iterable[object]) -> list[float]:
    samples: list[float] = []
    for audio in parts:
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        if hasattr(audio, "reshape"):
            audio = audio.reshape(-1)
        if hasattr(audio, "tolist"):
            audio = audio.tolist()
        samples.extend(float(value) for value in audio)
    return samples


def _wav_bytes(samples: Iterable[float]) -> bytes:
    pcm = bytearray()
    for sample in samples:
        value = round(max(-1.0, min(1.0, sample)) * 32767)
        pcm.extend(int(value).to_bytes(2, "little", signed=True))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(KOKORO_SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()
