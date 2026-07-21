import tempfile
from pathlib import Path
from uuid import uuid4

import gradio as gr
from core import extract_article
from tts import PiperNotInstalledError, PiperSynthesisError, PiperTTSProvider, PiperVoiceNotFoundError, TTSService


AUDIO_DIR = Path(tempfile.gettempdir()) / "web-reader-audio"


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


def read_aloud(article_text):
    article_text = (article_text or "").strip()
    if not article_text:
        return "请先成功提取一篇文章，再点击 Read Aloud。", None

    try:
        service = TTSService([PiperTTSProvider()], default_provider="piper")
        result = service.synthesize(article_text, provider_name="piper", audio_format="wav")
    except PiperNotInstalledError as exc:
        return f"Piper 未安装或不可用：{exc}", None
    except PiperVoiceNotFoundError as exc:
        return f"Piper 语音模型缺失：{exc}", None
    except (PiperSynthesisError, ValueError, OSError) as exc:
        return f"语音合成失败：{exc}", None

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = AUDIO_DIR / f"article-{uuid4().hex}.wav"
    audio_path.write_bytes(result.audio)
    return "语音生成成功。请使用下方播放器收听。", str(audio_path)


with gr.Blocks(title="Web Reader") as demo:
    gr.Markdown("# Web Reader")
    gr.Markdown("输入网页网址，自动提取文章标题和正文。")

    url_input = gr.Textbox(label="网页网址", placeholder="请输入文章网址……")
    extract_button = gr.Button("Extract Article")

    title_output = gr.Textbox(label="文章标题")
    info_output = gr.Textbox(label="文章信息")
    text_output = gr.Textbox(label="文章正文", lines=25)
    status_output = gr.Textbox(label="状态")

    read_button = gr.Button("Read Aloud")
    audio_output = gr.Audio(label="朗读音频", type="filepath")

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
    read_button.click(fn=read_aloud, inputs=text_output, outputs=[status_output, audio_output])


if __name__ == "__main__":
    demo.launch(debug=True, share=True)
