"""Provider-neutral text-to-speech architecture skeleton."""

from .fake_provider import FakeTTSProvider
from .provider import SynthesisRequest, SynthesisResult, TTSProvider, Voice
from .service import TTSService
from .text_utils import join_chunks, normalize_text, split_text

__all__ = [
    "FakeTTSProvider",
    "SynthesisRequest",
    "SynthesisResult",
    "TTSProvider",
    "TTSService",
    "Voice",
    "join_chunks",
    "normalize_text",
    "split_text",
]
