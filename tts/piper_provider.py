"""Piper command-line text-to-speech provider."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .provider import SynthesisRequest, SynthesisResult, Voice
from .text_utils import normalize_text


COLAB_PIPER_EXECUTABLE = Path("./piper/piper")
COLAB_PIPER_MODEL = Path("./piper_models/zh_CN-huayan-medium.onnx")
COLAB_PIPER_CONFIG = Path("./piper_models/zh_CN-huayan-medium.onnx.json")


class PiperNotInstalledError(RuntimeError):
    """Raised when the Piper executable cannot be found."""


class PiperVoiceNotFoundError(RuntimeError):
    """Raised when the configured Piper voice model cannot be found."""


class PiperSynthesisError(RuntimeError):
    """Raised when Piper fails to synthesize audio."""


class PiperTTSProvider:
    """Text-to-speech provider backed by the local Piper CLI."""

    name = "piper"

    def __init__(
        self,
        *,
        executable: str | None = None,
        voice_model: str | os.PathLike[str] | None = None,
        voice_config: str | os.PathLike[str] | None = None,
    ):
        env_voice_model = os.environ.get("PIPER_VOICE_MODEL")
        using_default_colab_model = voice_model is None and not env_voice_model

        self.executable = executable or os.environ.get("PIPER_EXECUTABLE") or str(COLAB_PIPER_EXECUTABLE)
        self.voice_model = Path(voice_model or env_voice_model or COLAB_PIPER_MODEL).expanduser()
        config = voice_config or os.environ.get("PIPER_VOICE_CONFIG")
        if config is None and using_default_colab_model:
            config = COLAB_PIPER_CONFIG
        self.voice_config = Path(config).expanduser() if config else None

    def list_voices(self) -> list[Voice]:
        voice_id = str(self.voice_model) if str(self.voice_model) != "." else "piper-default"
        return [Voice(id=voice_id, name=self.voice_model.stem or "Piper Voice", language="", provider=self.name)]

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if request.audio_format.lower() != "wav":
            raise ValueError("Piper only supports WAV output")

        text = normalize_text(request.text)
        if not text:
            raise ValueError("text is required")

        self._validate_ready()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            output_path = Path(audio_file.name)

        command = [self.executable, "--model", str(self.voice_model), "--output_file", str(output_path)]
        if self.voice_config:
            command.extend(["--config", str(self.voice_config)])

        try:
            completed = subprocess.run(
                command,
                input=text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout or "unknown Piper error").strip()
                raise PiperSynthesisError(f"Piper speech synthesis failed: {details}")
            audio = output_path.read_bytes()
            if not audio:
                raise PiperSynthesisError("Piper speech synthesis failed: generated audio file is empty")
        finally:
            output_path.unlink(missing_ok=True)

        return SynthesisResult(
            audio=audio,
            audio_format="wav",
            provider=self.name,
            voice_id=request.voice_id or str(self.voice_model),
            metadata={"characters": str(len(text))},
        )

    def _validate_ready(self) -> None:
        missing_installation = self._missing_installation_message()
        missing_voice = self._missing_voice_messages()

        if missing_installation and missing_voice:
            raise PiperNotInstalledError(" ".join([missing_installation, *missing_voice]))
        if missing_installation:
            raise PiperNotInstalledError(missing_installation)
        if missing_voice:
            raise PiperVoiceNotFoundError(" ".join(missing_voice))

    def _validate_installation(self) -> None:
        message = self._missing_installation_message()
        if message:
            raise PiperNotInstalledError(message)

    def _missing_installation_message(self) -> str | None:
        if shutil.which(self.executable) is None:
            return "Piper executable was not found. Expected Colab path ./piper/piper, or set PIPER_EXECUTABLE to the Piper binary."
        return None

    def _validate_voice_model(self) -> None:
        missing = self._missing_voice_messages()
        if missing:
            raise PiperVoiceNotFoundError(" ".join(missing))

    def _missing_voice_messages(self) -> list[str]:
        missing = []
        if not str(self.voice_model) or str(self.voice_model) == ".":
            missing.append("Piper voice model is not configured; expected ./piper_models/zh_CN-huayan-medium.onnx or set PIPER_VOICE_MODEL.")
        elif not self.voice_model.is_file():
            missing.append(f"Piper voice model was not found: {self.voice_model}")

        if self.voice_config and not self.voice_config.is_file():
            missing.append(f"Piper voice config was not found: {self.voice_config}")

        return missing
