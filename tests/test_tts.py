import os
import unittest
import wave
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from tts import (
    FakeTTSProvider,
    KokoroTTSProvider,
    MatchaModelNotFoundError,
    MatchaPaths,
    MatchaTTSProvider,
    PiperNotInstalledError,
    PiperTTSProvider,
    PiperVoiceNotFoundError,
    TTSService,
    Voice,
    detect_language,
    join_chunks,
    normalize_text,
    split_text,
)
from tts.provider import SynthesisRequest


class TextUtilitiesTests(unittest.TestCase):
    def test_detect_language_selects_primary_script(self):
        self.assertEqual(detect_language("这是以中文为主的 article。"), "zh")
        self.assertEqual(detect_language("This article has one 中文 term."), "en")

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

    def test_explicit_provider_selection_overrides_default(self):
        first = FakeTTSProvider()
        fallback = FakeTTSProvider()
        first.name = "matcha"
        fallback.name = "kokoro"
        service = TTSService([first, fallback], default_provider="matcha")

        self.assertEqual(service.synthesize("default").provider, "matcha")
        self.assertEqual(service.synthesize("fallback", provider_name="kokoro").provider, "kokoro")


class MatchaTTSProviderTests(unittest.TestCase):
    def _paths(self, root: Path, *, complete: bool = True) -> MatchaPaths:
        paths = MatchaPaths.from_environment(root)
        if complete:
            paths.data_dir.mkdir(parents=True)
            for path in (
                paths.acoustic_model, paths.vocoder, paths.lexicon, paths.tokens,
                paths.phone_fst, paths.date_fst, paths.number_fst,
            ):
                path.write_text("test")
        return paths

    def test_routes_chinese_and_english_through_bilingual_engine(self):
        calls = []

        class Audio:
            samples = np.array([0.1, -0.1], dtype=np.float32)
            sample_rate = 16_000

        class Engine:
            def generate(self, text, sid, speed):
                calls.append((text, sid, speed))
                return Audio()

        with TemporaryDirectory() as tmpdir:
            provider = MatchaTTSProvider(self._paths(Path(tmpdir)), engine_factory=lambda paths, threads: Engine())
            chinese = provider.synthesize(SynthesisRequest(text="这是中文。"))
            english = provider.synthesize(SynthesisRequest(text="This is English."))

        self.assertEqual(chinese.metadata["language"], "zh")
        self.assertEqual(english.metadata["language"], "en")
        self.assertEqual(calls, [("这是中文。", 0, 1.0), ("This is English.", 0, 1.0)])

    def test_passes_reading_speed_and_returns_valid_wav(self):
        speeds = []

        class Audio:
            samples = [0.0, 0.5, -0.5]
            sample_rate = 16_000

        class Engine:
            def generate(self, text, sid, speed):
                speeds.append(speed)
                return Audio()

        with TemporaryDirectory() as tmpdir:
            provider = MatchaTTSProvider(self._paths(Path(tmpdir)), engine_factory=lambda paths, threads: Engine())
            result = provider.synthesize(SynthesisRequest(text="Speed test", speaking_rate=1.4))

        self.assertEqual(speeds, [1.4])
        self.assertEqual(result.provider, "matcha")
        with wave.open(BytesIO(result.audio), "rb") as audio:
            self.assertEqual(audio.getparams()[:4], (1, 2, 16_000, 3))

    def test_missing_models_fail_with_every_missing_path(self):
        with TemporaryDirectory() as tmpdir:
            provider = MatchaTTSProvider(self._paths(Path(tmpdir), complete=False))
            with self.assertRaises(MatchaModelNotFoundError) as error:
                provider.synthesize(SynthesisRequest(text="hello"))

        self.assertIn("model-steps-3.onnx", str(error.exception))
        self.assertIn("vocos-16khz-univ.onnx", str(error.exception))
        self.assertIn("espeak-ng-data", str(error.exception))

    def test_rejects_invalid_speed_and_non_wav_output(self):
        with TemporaryDirectory() as tmpdir:
            provider = MatchaTTSProvider(self._paths(Path(tmpdir)))
            with self.assertRaisesRegex(ValueError, "speaking_rate"):
                provider.synthesize(SynthesisRequest(text="hello", speaking_rate=2.1))
            with self.assertRaisesRegex(ValueError, "WAV"):
                provider.synthesize(SynthesisRequest(text="hello", audio_format="mp3"))


class KokoroTTSProviderTests(unittest.TestCase):
    def test_lists_required_chinese_and_english_voices(self):
        voices = KokoroTTSProvider(lambda **kwargs: None).list_voices()

        self.assertEqual([voice.id for voice in voices], ["zf_xiaoxiao", "zf_xiaoyi", "af_heart"])

    def test_uses_language_pipeline_default_voice_and_speed(self):
        calls = []

        class Pipeline:
            def __init__(self, lang_code):
                self.lang_code = lang_code

            def __call__(self, text, voice, speed):
                calls.append((self.lang_code, text, voice, speed))
                return [(None, None, [0.0, 0.5, -0.5])]

        provider = KokoroTTSProvider(lambda lang_code: Pipeline(lang_code))
        result = provider.synthesize(SynthesisRequest(text=" 你好 ", language="zh-CN", speaking_rate=1.2))

        self.assertEqual(calls, [("z", "你好", "zf_xiaoxiao", 1.2)])
        self.assertEqual(result.voice_id, "zf_xiaoxiao")
        self.assertEqual(result.metadata["language"], "zh")
        with wave.open(BytesIO(result.audio), "rb") as audio:
            self.assertEqual(audio.getframerate(), 24_000)
            self.assertEqual(audio.getnframes(), 3)

    def test_english_voice_choice_selects_english_pipeline(self):
        language_codes = []

        def factory(lang_code):
            language_codes.append(lang_code)
            return lambda text, voice, speed: [(None, None, [0.1])]

        provider = KokoroTTSProvider(factory)
        provider.synthesize(SynthesisRequest(text="English", language="en", voice_id="af_heart"))

        self.assertEqual(language_codes, ["a"])

    def test_auto_detects_english_and_uses_heart(self):
        calls = []

        def factory(lang_code):
            return lambda text, voice, speed: calls.append((lang_code, voice, speed)) or [(None, None, [0.1])]

        provider = KokoroTTSProvider(factory)
        result = provider.synthesize(SynthesisRequest(text="This is an English article."))

        self.assertEqual(calls, [("a", "af_heart", 1.0)])
        self.assertEqual(result.voice_id, "af_heart")

    def test_rejects_cross_language_voice_choices(self):
        provider = KokoroTTSProvider(lambda **kwargs: None)

        with self.assertRaisesRegex(ValueError, "not available for en"):
            provider.synthesize(SynthesisRequest(text="English", language="en", voice_id="zf_xiaoyi"))
        with self.assertRaisesRegex(ValueError, "not available for zh"):
            provider.synthesize(SynthesisRequest(text="中文", language="zh", voice_id="af_heart"))

    def test_switching_chinese_voices_reaches_pipeline(self):
        voices = []
        provider = KokoroTTSProvider(
            lambda **kwargs: lambda text, voice, speed: voices.append(voice) or [(None, None, [0.1])]
        )

        provider.synthesize(SynthesisRequest(text="你好", language="zh", voice_id="zf_xiaoxiao"))
        provider.synthesize(SynthesisRequest(text="你好", language="zh", voice_id="zf_xiaoyi"))

        self.assertEqual(voices, ["zf_xiaoxiao", "zf_xiaoyi"])

    def test_diagnostics_report_timings_device_and_distinct_voice_audio(self):
        def pipeline(text, voice, speed):
            sample = 0.1 if voice == "zf_xiaoxiao" else 0.2
            return [(None, None, np.array([sample], dtype=np.float32))]

        provider = KokoroTTSProvider(lambda **kwargs: pipeline)
        first = provider.synthesize(
            SynthesisRequest(text="你好", language="zh", voice_id="zf_xiaoxiao")
        )
        second = provider.synthesize(
            SynthesisRequest(text="你好", language="zh", voice_id="zf_xiaoyi")
        )

        for result, voice in ((first, "zf_xiaoxiao"), (second, "zf_xiaoyi")):
            self.assertEqual(result.metadata["selected_voice"], voice)
            self.assertIn(result.metadata["torch_device"], ("cpu", "cuda", "unavailable"))
            for key in (
                "model_load_seconds",
                "text_preprocessing_seconds",
                "kokoro_inference_seconds",
                "wav_serialization_seconds",
                "total_request_seconds",
            ):
                self.assertGreaterEqual(float(result.metadata[key]), 0.0)
        self.assertEqual(first.metadata["pipeline_cached"], "false")
        self.assertEqual(second.metadata["pipeline_cached"], "true")
        self.assertNotEqual(first.metadata["audio_sha256"], second.metadata["audio_sha256"])

    def test_reuses_each_loaded_language_pipeline(self):
        language_codes = []

        def factory(lang_code):
            language_codes.append(lang_code)
            return lambda text, voice, speed: [(None, None, np.array([0.1], dtype=np.float32))]

        provider = KokoroTTSProvider(factory)
        provider.synthesize(SynthesisRequest(text="First", language="en"))
        provider.synthesize(SynthesisRequest(text="Second", language="en"))
        provider.synthesize(SynthesisRequest(text="你好", language="zh"))
        provider.synthesize(SynthesisRequest(text="再见", language="zh"))

        self.assertEqual(language_codes, ["a", "z"])

    def test_wav_output_combines_numpy_and_tensor_like_chunks(self):
        class TensorLike:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([[1.5, -1.5]], dtype=np.float32)

        def pipeline(text, voice, speed):
            return [
                (None, None, np.array([0.0, 0.5], dtype=np.float32)),
                (None, None, TensorLike()),
            ]

        provider = KokoroTTSProvider(lambda **kwargs: pipeline)
        result = provider.synthesize(SynthesisRequest(text="audio", language="en"))

        with wave.open(BytesIO(result.audio), "rb") as audio:
            self.assertEqual(audio.getparams()[:4], (1, 2, 24_000, 4))
            pcm = np.frombuffer(audio.readframes(4), dtype="<i2")
        np.testing.assert_array_equal(pcm, np.array([0, 16384, 32767, -32767], dtype=np.int16))

    def test_rejects_unsupported_speed_and_format(self):
        provider = KokoroTTSProvider(lambda **kwargs: None)
        with self.assertRaises(ValueError):
            provider.synthesize(SynthesisRequest(text="hello", speaking_rate=2.1))
        with self.assertRaises(ValueError):
            provider.synthesize(SynthesisRequest(text="hello", audio_format="mp3"))


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
            config = Path(tmpdir) / "voice.onnx.json"
            config.write_text("config")

            def fake_run(command, input, text, stdout, stderr, check):
                output_path = Path(command[command.index("--output_file") + 1])
                output_path.write_bytes(b"RIFF fake wav")

                class Completed:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Completed()

            provider = PiperTTSProvider(executable="piper", voice_model=model, voice_config=config)
            with patch("tts.piper_provider.subprocess.run", side_effect=fake_run) as run:
                result = provider.synthesize(SynthesisRequest(text=" hello "))

        self.assertEqual(result.audio, b"RIFF fake wav")
        self.assertEqual(result.audio_format, "wav")
        self.assertEqual(result.provider, "piper")
        command = run.call_args.args[0]
        self.assertIn("--config", command)
        self.assertEqual(command[command.index("--config") + 1], str(config))
        self.assertEqual(run.call_args.kwargs["input"], "hello")

    def test_piper_auto_detects_colab_layout(self):
        with TemporaryDirectory() as tmpdir, patch("tts.piper_provider.shutil.which", return_value="./piper/piper"):
            previous_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                executable = Path("piper/piper")
                executable.parent.mkdir()
                executable.write_text("#!/bin/sh\n")
                model = Path("piper_models/zh_CN-huayan-medium.onnx")
                model.parent.mkdir()
                model.write_text("model")
                config = Path("piper_models/zh_CN-huayan-medium.onnx.json")
                config.write_text("config")

                provider = PiperTTSProvider()

                self.assertEqual(provider.executable, "piper/piper")
                self.assertEqual(provider.voice_model, model)
                self.assertEqual(provider.voice_config, config)
                provider._validate_installation()
                provider._validate_voice_model()
            finally:
                os.chdir(previous_cwd)

    def test_piper_reports_missing_colab_defaults(self):
        with TemporaryDirectory() as tmpdir, patch("tts.piper_provider.shutil.which", return_value=None):
            previous_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                provider = PiperTTSProvider()

                with self.assertRaises(PiperNotInstalledError) as executable_error:
                    provider._validate_installation()
                with self.assertRaises(PiperVoiceNotFoundError) as voice_error:
                    provider._validate_voice_model()

                self.assertIn("./piper/piper", str(executable_error.exception))
                self.assertIn("piper_models/zh_CN-huayan-medium.onnx", str(voice_error.exception))
                self.assertIn("piper_models/zh_CN-huayan-medium.onnx.json", str(voice_error.exception))

                with self.assertRaises(PiperNotInstalledError) as ready_error:
                    provider.synthesize(SynthesisRequest(text="hello"))
                self.assertIn("./piper/piper", str(ready_error.exception))
                self.assertIn("piper_models/zh_CN-huayan-medium.onnx", str(ready_error.exception))
                self.assertIn("piper_models/zh_CN-huayan-medium.onnx.json", str(ready_error.exception))
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
