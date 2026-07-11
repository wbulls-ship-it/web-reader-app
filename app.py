
import gradio as gr
from core import extract_article


def read_article(url):
    url = (url or "").strip()

    if not url:
        return "请输入网页网址", "", ""

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    article = extract_article(url)

    if not article["success"]:
        return article["message"], "", ""

    article_info = (
        f"字数：{article['characters']} 字\n"
        f"预计阅读时间：约 {article['reading_minutes']} 分钟"
    )

    return (
        article["title"],
        article_info,
        article["text"],
    )


demo = gr.Interface(
    fn=read_article,
    inputs=gr.Textbox(
        label="网页网址",
        placeholder="请输入文章网址……"
    ),
    outputs=[
        gr.Textbox(label="文章标题"),
        gr.Textbox(label="文章信息"),
        gr.Textbox(label="文章正文", lines=25),
    ],
    title="Web Reader",
    description="输入网页网址，自动提取文章标题和正文。",
)

if __name__ == "__main__":
    demo.launch(debug=True)
