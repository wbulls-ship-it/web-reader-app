"""Provider-neutral text-to-speech architecture skeleton."""

from .fake_provider import FakeTTSProvider
from .kokoro_provider import (
    DEFAULT_VOICES,
    VOICE_LANGUAGES,
    KokoroNotInstalledError,
    KokoroSynthesisError,
    KokoroTTSProvider,
)
from .piper_provider import (
    PiperNotInstalledError,
    PiperSynthesisError,
    PiperTTSProvider,
    PiperVoiceNotFoundError,
)
from .provider import SynthesisRequest, SynthesisResult, TTSProvider, Voice
from .service import TTSService
from .text_utils import detect_language, join_chunks, normalize_text, split_text

__all__ = [
    "FakeTTSProvider",
    "DEFAULT_VOICES",
    "KokoroNotInstalledError",
    "KokoroSynthesisError",
    "KokoroTTSProvider",
    "PiperNotInstalledError",
    "PiperSynthesisError",
    "PiperTTSProvider",
    "PiperVoiceNotFoundError",
    "SynthesisRequest",
    "SynthesisResult",
    "TTSProvider",
    "TTSService",
    "Voice",
    "VOICE_LANGUAGES",
    "detect_language",
    "join_chunks",
    "normalize_text",
    "split_text",
]
