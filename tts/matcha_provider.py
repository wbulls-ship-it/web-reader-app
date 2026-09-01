"""CPU sherpa-onnx Matcha provider for bilingual Chinese/English speech."""

from __future__ import annotations

import importlib
import io
import os
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .provider import SynthesisRequest, SynthesisResult, Voice
from .text_utils import detect_language, normalize_text

MATCHA_SAMPLE_RATE = 16_000


class MatchaNotInstalledError(RuntimeError):
    """Raised when the sherpa-onnx Python package is unavailable."""


class MatchaModelNotFoundError(RuntimeError):
    """Raised when one or more required Matcha assets are unavailable."""


class MatchaSynthesisError(RuntimeError):
    """Raised when sherpa-onnx cannot generate usable audio."""


@dataclass(frozen=True)
class MatchaPaths:
    """All filesystem assets needed by the validated bilingual Matcha model."""

    acoustic_model: Path
    vocoder: Path
    lexicon: Path
    tokens: Path
    data_dir: Path
    phone_fst: Path
    date_fst: Path
    number_fst: Path

    @classmethod
    def from_environment(cls, model_dir: str | os.PathLike[str] | None = None) -> "MatchaPaths":
        root = Path(model_dir or os.getenv("MATCHA_MODEL_DIR", "models/matcha-icefall-zh-en")).expanduser()

        def asset(env: str, relative: str) -> Path:
            return Path(os.getenv(env, str(root / relative))).expanduser()

        return cls(
            acoustic_model=asset("MATCHA_ACOUSTIC_MODEL", "model-steps-3.onnx"),
            vocoder=asset("MATCHA_VOCODER", "vocos-16khz-univ.onnx"),
            lexicon=asset("MATCHA_LEXICON", "lexicon.txt"),
            tokens=asset("MATCHA_TOKENS", "tokens.txt"),
            data_dir=asset("MATCHA_ESPEAK_DATA", "espeak-ng-data"),
            phone_fst=asset("MATCHA_PHONE_FST", "phone-zh.fst"),
            date_fst=asset("MATCHA_DATE_FST", "date-zh.fst"),
            number_fst=asset("MATCHA_NUMBER_FST", "number-zh.fst"),
        )

    def missing(self) -> list[Path]:
        files = [
            self.acoustic_model, self.vocoder, self.lexicon, self.tokens,
            self.phone_fst, self.date_fst, self.number_fst,
        ]
        missing = [path for path in files if not path.is_file()]
        if not self.data_dir.is_dir():
            missing.append(self.data_dir)
        return missing


class MatchaTTSProvider:
    """Lazy, process-cached sherpa-onnx OfflineTts adapter running on CPU."""

    name = "matcha"

    def __init__(
        self,
        paths: MatchaPaths | None = None,
        *,
        num_threads: int | None = None,
        engine_factory: Callable[[MatchaPaths, int, str], object] | None = None,
    ):
        self.paths = paths or MatchaPaths.from_environment()
        self.num_threads = num_threads or int(os.getenv("MATCHA_NUM_THREADS", "2"))
        if self.num_threads < 1:
            raise ValueError("MATCHA_NUM_THREADS must be at least 1")
        self._engine_factory = engine_factory
        # The Chinese TN FSTs are global OfflineTts configuration, not a
        # per-generate language option.  Keep an FST-free reference engine for
        # English and a Chinese-normalizing engine for Chinese.
        self._engines: dict[str, object] = {}
        self._lock = threading.Lock()

    def list_voices(self) -> list[Voice]:
        return [Voice("matcha-zh-en", "Matcha bilingual ZH+EN", "zh-CN/en-US", self.name)]

    def readiness_error(self) -> str | None:
        missing = self.paths.missing()
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            return f"Matcha is not ready; missing required model assets: {rendered}"
        try:
            importlib.import_module("sherpa_onnx")
        except ImportError:
            return "Matcha is not ready; install sherpa-onnx==1.13.6 from requirements.txt."
        return None

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if request.audio_format.lower() != "wav":
            raise ValueError("Matcha only supports WAV output")
        text = normalize_text(request.text)
        if not text:
            raise ValueError("text is required")
        if not 0.5 <= request.speaking_rate <= 2.0:
            raise ValueError("speaking_rate must be between 0.5 and 2.0")

        missing = self.paths.missing()
        if missing:
            raise MatchaModelNotFoundError(
                "Matcha cannot start because required model assets are missing: "
                + ", ".join(str(path) for path in missing)
                + ". Set MATCHA_MODEL_DIR or the individual MATCHA_* path variables."
            )
        language = _language_code(request.language) if request.language else detect_language(text)
        engine = self._get_engine(language)
        try:
            generated = engine.generate(text, sid=0, speed=request.speaking_rate)
            samples = np.asarray(generated.samples, dtype=np.float32).reshape(-1)
            sample_rate = int(generated.sample_rate)
        except Exception as exc:
            raise MatchaSynthesisError(f"Matcha speech synthesis failed: {exc}") from exc
        if not samples.size:
            raise MatchaSynthesisError("Matcha speech synthesis produced no audio")
        audio = _wav_bytes(samples, sample_rate)
        return SynthesisResult(
            audio=audio,
            audio_format="wav",
            provider=self.name,
            voice_id=request.voice_id or "matcha-zh-en",
            duration_seconds=samples.size / sample_rate,
            metadata={
                "characters": str(len(text)),
                "language": language,
                "sample_rate": str(sample_rate),
                "speaking_rate": str(request.speaking_rate),
                "device": "cpu",
            },
        )

    def _get_engine(self, language: str) -> object:
        if language in self._engines:
            return self._engines[language]
        with self._lock:
            if language not in self._engines:
                self._engines[language] = (self._engine_factory or _create_engine)(
                    self.paths, self.num_threads, language
                )
        return self._engines[language]


def _create_engine(paths: MatchaPaths, num_threads: int, language: str) -> object:
    try:
        sherpa = importlib.import_module("sherpa_onnx")
    except ImportError as exc:
        raise MatchaNotInstalledError(
            "sherpa-onnx 1.13.6 is not installed; install requirements.txt."
        ) from exc
    matcha = sherpa.OfflineTtsMatchaModelConfig(
        acoustic_model=str(paths.acoustic_model), vocoder=str(paths.vocoder),
        lexicon=str(paths.lexicon), tokens=str(paths.tokens), data_dir=str(paths.data_dir),
    )
    model = sherpa.OfflineTtsModelConfig(
        matcha=matcha, num_threads=num_threads, debug=False, provider="cpu"
    )
    config = sherpa.OfflineTtsConfig(
        model=model,
        # These are Chinese text-normalization grammars. Applying them to an
        # English request makes e.g. English digits expand as Chinese words.
        rule_fsts=(
            ",".join(map(str, (paths.phone_fst, paths.date_fst, paths.number_fst)))
            if language == "zh"
            else ""
        ),
        max_num_sentences=1,
    )
    if not config.validate():
        raise MatchaModelNotFoundError("sherpa-onnx rejected the configured Matcha model assets")
    return sherpa.OfflineTts(config)


def _language_code(language: str) -> str:
    return "zh" if language.lower().startswith("zh") else "en"


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()
