import logging
import tempfile
from pathlib import Path
from uuid import uuid4

import gradio as gr
from core import extract_article
from tts import (
    KokoroNotInstalledError,
    KokoroSynthesisError,
    KokoroTTSProvider,
    MatchaModelNotFoundError,
    MatchaNotInstalledError,
    MatchaSynthesisError,
    MatchaTTSProvider,
    TTSService,
    detect_language,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

AUDIO_DIR = Path(tempfile.gettempdir()) / "web-reader-audio"
# Providers are process-scoped so their expensive engines are loaded once. Matcha
# is the production default; Kokoro remains an explicit optional fallback.
MATCHA_PROVIDER = MatchaTTSProvider()
TTS_SERVICE = TTSService([MATCHA_PROVIDER, KokoroTTSProvider()], default_provider="matcha")


def read_article(url):
    url = (url or "").strip()

    if not url:
        return "请输入网页网址", "", "", "", None

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    article = extract_article(url)

    if not article["success"]:
        return article["message"], "", "", "", None

    article_info = (
        f"字数：{article['characters']} 字\n"
        f"预计阅读时间：约 {article['reading_minutes']} 分钟"
    )

    return (
        article["title"],
        article_info,
        article["text"],
        "文章提取成功。点击“Read Aloud”生成语音。",
        None,
    )


def read_aloud(article_text, provider_name="matcha", reading_speed=1.0):
    article_text = (article_text or "").strip()
    if not article_text:
        return "请先成功提取一篇文章，再点击 Read Aloud。", None

    try:
        language = detect_language(article_text)
        result = TTS_SERVICE.synthesize(
            article_text,
            provider_name=provider_name,
            language=language,
            speaking_rate=float(reading_speed),
            audio_format="wav",
        )
    except (MatchaNotInstalledError, MatchaModelNotFoundError) as exc:
        return f"Matcha 尚未就绪：{exc}", None
    except KokoroNotInstalledError as exc:
        return f"Kokoro fallback 尚未就绪：{exc}", None
    except (MatchaSynthesisError, KokoroSynthesisError, ValueError, OSError) as exc:
        return f"语音合成失败：{exc}", None

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = AUDIO_DIR / f"article-{uuid4().hex}.wav"
    audio_path.write_bytes(result.audio)
    return "语音生成成功。请使用下方播放器收听。", str(audio_path)


with gr.Blocks(title="Web Reader") as demo:
    gr.Markdown("# Web Reader")
    gr.Markdown("粘贴文章网址，提取正文，然后用 CPU 生成中英文朗读音频。")

    readiness = MATCHA_PROVIDER.readiness_error()
    gr.Markdown(
        "**TTS 状态：** " + (f"⚠️ {readiness}" if readiness else "✅ Matcha 已就绪（CPU）")
    )

    url_input = gr.Textbox(label="网页网址", placeholder="请输入文章网址……")
    extract_button = gr.Button("1. Extract Article", variant="primary")

    title_output = gr.Textbox(label="文章标题")
    info_output = gr.Textbox(label="文章信息")
    text_output = gr.Textbox(label="文章正文", lines=25)
    status_output = gr.Textbox(label="状态")

    read_button = gr.Button("2. Read Aloud", variant="primary")
    provider_input = gr.Dropdown(
        [("Matcha ZH+EN (production default)", "matcha"), ("Kokoro (optional fallback)", "kokoro")],
        value="matcha",
        label="TTS provider",
        info="Matcha automatically handles Chinese and English. Kokoro is optional.",
    )
    speed_input = gr.Slider(
        0.5,
        2.0,
        value=1.0,
        step=0.1,
        label="Reading speed",
        info="0.5× is slower, 1.0× is normal, and 2.0× is faster.",
    )
    audio_output = gr.Audio(label="Audio player (playback and volume)", type="filepath")

    extract_button.click(
        fn=read_article,
        inputs=url_input,
        outputs=[title_output, info_output, text_output, status_output, audio_output],
    )
    url_input.submit(
        fn=read_article,
        inputs=url_input,
        outputs=[title_output, info_output, text_output, status_output, audio_output],
    )
    read_button.click(
        fn=read_aloud,
        inputs=[text_output, provider_input, speed_input],
        outputs=[status_output, audio_output],
    )


if __name__ == "__main__":
    demo.launch()
