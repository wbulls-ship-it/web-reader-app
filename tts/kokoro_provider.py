"""Kokoro implementation of the provider-neutral TTS interface."""

from __future__ import annotations

import hashlib
import importlib
import io
import logging
import sys
import threading
import time
import wave
from collections.abc import Callable, Iterable

import numpy as np

from .provider import SynthesisRequest, SynthesisResult, Voice
from .text_utils import detect_language, normalize_text

KOKORO_SAMPLE_RATE = 24_000
DEFAULT_VOICES = {
    "zh": "zf_xiaoxiao",
    "en": "af_heart",
}
VOICE_LANGUAGES = {
    "zf_xiaoxiao": "zh",
    "zf_xiaoyi": "zh",
    "af_heart": "en",
}
logger = logging.getLogger(__name__)


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
        # Gradio may dispatch multiple requests concurrently. Avoid loading the
        # same (large) model twice while still allowing cached lookups to be fast.
        self._pipeline_lock = threading.Lock()

    def list_voices(self) -> list[Voice]:
        return [
            Voice("zf_xiaoxiao", "Xiaoxiao", "zh-CN", self.name),
            Voice("zf_xiaoyi", "Xiaoyi", "zh-CN", self.name),
            Voice("af_heart", "Heart", "en-US", self.name),
        ]

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        request_started = time.perf_counter()
        if request.audio_format.lower() != "wav":
            raise ValueError("Kokoro only supports WAV output")
        preprocessing_started = time.perf_counter()
        text = normalize_text(request.text)
        detected_language = detect_language(text) if text else "en"
        language = _language_code(request.language) if request.language else detected_language
        voice = request.voice_id or DEFAULT_VOICES[language]
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        if not text:
            raise ValueError("text is required")
        if not 0.5 <= request.speaking_rate <= 2.0:
            raise ValueError("speaking_rate must be between 0.5 and 2.0")

        voice_language = VOICE_LANGUAGES.get(voice)
        if voice_language is None:
            raise ValueError(f"unsupported Kokoro voice: {voice}")
        if voice_language != language:
            raise ValueError(
                f"voice {voice} is not available for {language}; "
                f"choose a {language} voice or Auto"
            )

        logger.info(
            "Kokoro synthesis routing: detected_language=%s selected_voice=%s "
            "pipeline_language=%s characters=%d speed=%.2f",
            detected_language,
            voice,
            language,
            len(text),
            request.speaking_rate,
        )
        pipeline, model_load_seconds, pipeline_cached = self._pipeline(language)
        device = _pipeline_device(pipeline)
        logger.info(
            "Kokoro runtime: torch_device=%s cuda_available=%s model_load=%.3fs "
            "pipeline_cached=%s preprocessing=%.3fs",
            device["torch_device"],
            device["cuda_available"],
            model_load_seconds,
            pipeline_cached,
            preprocessing_seconds,
        )
        synthesis_started = time.perf_counter()
        try:
            generated = pipeline(text, voice=voice, speed=request.speaking_rate)
            samples = _collect_samples(item[2] for item in generated)
        except (KokoroNotInstalledError, ValueError):
            raise
        except Exception as exc:
            logger.exception(
                "Kokoro synthesis failed after %.3fs: language=%s voice=%s",
                time.perf_counter() - synthesis_started,
                language,
                voice,
            )
            raise KokoroSynthesisError(f"Kokoro speech synthesis failed: {exc}") from exc
        synthesis_seconds = time.perf_counter() - synthesis_started
        if samples.size == 0:
            raise KokoroSynthesisError("Kokoro speech synthesis produced no audio")

        serialization_started = time.perf_counter()
        audio = _wav_bytes(samples)
        serialization_seconds = time.perf_counter() - serialization_started
        total_seconds = time.perf_counter() - request_started
        logger.info(
            "Finished Kokoro synthesis: voice=%s samples=%d model_load=%.3fs "
            "preprocessing=%.3fs inference=%.3fs wav=%.3fs total=%.3fs",
            voice,
            samples.size,
            model_load_seconds,
            preprocessing_seconds,
            synthesis_seconds,
            serialization_seconds,
            total_seconds,
        )

        return SynthesisResult(
            audio=audio,
            audio_format="wav",
            provider=self.name,
            voice_id=voice,
            metadata={
                "characters": str(len(text)),
                "language": language,
                "selected_voice": voice,
                "model_load_seconds": f"{model_load_seconds:.3f}",
                "pipeline_cached": str(pipeline_cached).lower(),
                "text_preprocessing_seconds": f"{preprocessing_seconds:.3f}",
                "kokoro_inference_seconds": f"{synthesis_seconds:.3f}",
                "wav_serialization_seconds": f"{serialization_seconds:.3f}",
                "total_request_seconds": f"{total_seconds:.3f}",
                "torch_device": device["torch_device"],
                "cuda_available": device["cuda_available"],
                "torch_version": device["torch_version"],
                "cuda_version": device["cuda_version"],
                "audio_sha256": hashlib.sha256(audio).hexdigest(),
                "synthesis_seconds": f"{synthesis_seconds:.3f}",
                "wav_seconds": f"{serialization_seconds:.3f}",
            },
        )

    def _pipeline(self, language: str):
        pipeline = self._pipelines.get(language)
        if pipeline is not None:
            return pipeline, 0.0, True

        with self._pipeline_lock:
            pipeline = self._pipelines.get(language)
            if pipeline is not None:
                return pipeline, 0.0, True
            if pipeline is None:
                started = time.perf_counter()
                logger.info("Loading Kokoro pipeline: language=%s", language)
                factory = self._pipeline_factory or self._load_pipeline_factory()
                try:
                    pipeline = factory(lang_code="z" if language == "zh" else "a")
                except Exception:
                    logger.exception(
                        "Kokoro pipeline load failed: language=%s elapsed=%.3fs",
                        language,
                        time.perf_counter() - started,
                    )
                    raise
                self._pipelines[language] = pipeline
                logger.info(
                    "Loaded Kokoro pipeline: language=%s elapsed=%.3fs",
                    language,
                    time.perf_counter() - started,
                )
                load_seconds = time.perf_counter() - started
        return pipeline, load_seconds, False

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


def _pipeline_device(pipeline: object) -> dict[str, str]:
    """Report both CUDA capability and the device actually holding the model."""

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {
            "torch_device": "unavailable",
            "cuda_available": "false",
            "torch_version": "unavailable",
            "cuda_version": "unavailable",
        }

    model = getattr(pipeline, "model", None)
    try:
        actual_device = str(next(model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "torch_device": actual_device,
        "cuda_available": str(torch.cuda.is_available()).lower(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or "none"),
    }


def _collect_samples(parts: Iterable[object]) -> np.ndarray:
    """Collect generated chunks without converting every sample to Python."""

    chunks: list[np.ndarray] = []
    for audio in parts:
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size:
            chunks.append(chunk)
    if not chunks:
        return np.empty(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def _wav_bytes(samples: Iterable[float]) -> bytes:
    sample_array = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = np.rint(np.clip(sample_array, -1.0, 1.0) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(KOKORO_SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()
