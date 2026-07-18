import unittest

from tts import FakeTTSProvider, TTSService, Voice, join_chunks, normalize_text, split_text


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


if __name__ == "__main__":
    unittest.main()
