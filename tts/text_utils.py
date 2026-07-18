"""Text normalization and chunking helpers for TTS providers."""

from __future__ import annotations

import re
from collections.abc import Iterable

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?])\s+")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")


def normalize_text(text: str | None) -> str:
    """Normalize user/article text before it is sent to a TTS provider."""

    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _WHITESPACE.sub(" ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_text(text: str, max_chars: int = 1_000) -> list[str]:
    """Split text into provider-sized chunks without dropping content."""

    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    for paragraph in normalized.split("\n\n"):
        chunks.extend(_split_paragraph(paragraph, max_chars))
    return chunks


def join_chunks(chunks: Iterable[str]) -> str:
    """Join chunks back into normalized text for deterministic providers/tests."""

    return "\n\n".join(chunk for chunk in (normalize_text(item) for item in chunks) if chunk)


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    pieces = _SENTENCE_BOUNDARY.split(paragraph)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(piece[index : index + max_chars] for index in range(0, len(piece), max_chars))
            continue
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks
