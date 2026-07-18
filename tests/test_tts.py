import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tts import (
    FakeTTSProvider,
    PiperNotInstalledError,
    PiperTTSProvider,
    PiperVoiceNotFoundError,
    TTSService,
    Voice,
    join_chunks,
    normalize_text,
    split_text,
)
from tts.provider import SynthesisRequest


class TextUtilitiesTests(unittest.TestCase):
    def test_normalize_text_collapses_spaces_and_blank_lines(self):
        self.assertEqual(normalize_text("  Hello\t world\r\n\r\n\r\nNext  line  "), "Hello world\n\nNext line")

    def test_split_text_preserves_content_in_chunks(self):
        text = "First sentence. Second sentence is longer. Third sentence."
        chunks = split_text(text, max_chars=32)

        self.assertTrue(all(len(chunk) <= 32 for chunk in chunks))
        self.assertEqual(join_chunks(chunks).replace("\n\n", " "), text)

    def test_split_text_rejects_invalid_max_chars(self):
        with self.assertRaises(ValueError):
            split_text("hello", max_chars=0)


class TTSServiceTests(unittest.TestCase):
    def test_service_lists_providers_and_voices(self):
        provider = FakeTTSProvider([Voice(id="voice-1", name="Test Voice", language="en-US", provider="fake")])
        service = TTSService([provider])

        self.assertEqual(service.list_providers(), ["fake"])
        self.assertEqual(service.list_voices()[0].id, "voice-1")

    def test_service_synthesizes_with_fake_provider(self):
        provider = FakeTTSProvider()
        service = TTSService([provider])

        result = service.synthesize(" Hello   world ", voice_id="fake-default", audio_format="mp3")

        self.assertEqual(result.provider, "fake")
        self.assertEqual(result.audio_format, "mp3")
        self.assertIn(b"text=Hello world", result.audio)
        self.assertEqual(provider.requests[0].text, "Hello world")

    def test_service_rejects_empty_text_and_unknown_provider(self):
        service = TTSService([FakeTTSProvider()])

        with self.assertRaises(ValueError):
            service.synthesize("   ")
        with self.assertRaises(ValueError):
            service.synthesize("hello", provider_name="missing")


class PiperTTSProviderTests(unittest.TestCase):
    def test_piper_reports_missing_executable(self):
        with TemporaryDirectory() as tmpdir, patch("tts.piper_provider.shutil.which", return_value=None):
            model = Path(tmpdir) / "voice.onnx"
            model.write_text("model")
            provider = PiperTTSProvider(executable="missing-piper", voice_model=model)

            with self.assertRaises(PiperNotInstalledError):
                provider.synthesize(SynthesisRequest(text="hello"))

    def test_piper_reports_missing_voice_model(self):
        with patch("tts.piper_provider.shutil.which", return_value="/usr/bin/piper"):
            provider = PiperTTSProvider(voice_model="/missing/voice.onnx")

            with self.assertRaises(PiperVoiceNotFoundError):
                provider.synthesize(SynthesisRequest(text="hello"))

    def test_piper_invokes_cli_and_returns_wav_bytes(self):
        with TemporaryDirectory() as tmpdir, patch("tts.piper_provider.shutil.which", return_value="/usr/bin/piper"):
            model = Path(tmpdir) / "voice.onnx"
            model.write_text("model")

            def fake_run(command, input, text, stdout, stderr, check):
                output_path = Path(command[command.index("--output_file") + 1])
                output_path.write_bytes(b"RIFF fake wav")

                class Completed:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Completed()

            provider = PiperTTSProvider(executable="piper", voice_model=model)
            with patch("tts.piper_provider.subprocess.run", side_effect=fake_run) as run:
                result = provider.synthesize(SynthesisRequest(text=" hello "))

        self.assertEqual(result.audio, b"RIFF fake wav")
        self.assertEqual(result.audio_format, "wav")
        self.assertEqual(result.provider, "piper")
        self.assertEqual(run.call_args.kwargs["input"], "hello")


if __name__ == "__main__":
    unittest.main()
