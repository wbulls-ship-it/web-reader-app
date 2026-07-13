import unittest
from unittest.mock import patch

from core import extract_article, extract_html_text, extract_html_title


class ArticleExtractionTests(unittest.TestCase):
    def test_extracts_clean_chinese_news_title_and_text(self):
        html = """
        <html>
          <head>
            <title>习近平将出席大会开幕式并发表主旨讲话_新浪新闻</title>
            <meta property="og:title" content="习近平将出席2026世界人工智能大会暨人工智能全球治理高级别会议开幕式并发表主旨讲话">
          </head>
          <body>
            <div class="article">
              <p>外交部发言人宣布：</p>
              <p>2026世界人工智能大会暨人工智能全球治理高级别会议将于7月17日至20日在上海举行。国家主席习近平将出席大会开幕式并发表主旨讲话。</p>
              <p>责任编辑：张玉</p>
            </div>
          </body>
        </html>
        """

        self.assertEqual(
            extract_html_title(html),
            "习近平将出席2026世界人工智能大会暨人工智能全球治理高级别会议开幕式并发表主旨讲话",
        )
        text = extract_html_text(html)
        self.assertIn("外交部发言人宣布", text)
        self.assertNotIn("责任编辑", text)

    def test_extracts_json_ld_graph_title_and_removes_ifeng_boilerplate(self):
        html = """
        <html>
          <head>
            <script type="application/ld+json">
              {
                "@graph": [
                  {"@type": "WebPage", "name": "凤凰网"},
                  {"@type": "NewsArticle", "headline": "国新办举行新闻发布会 介绍2026年一季度工业和信息化发展情况"}
                ]
              }
            </script>
          </head>
          <body>
            <article>
              <p>新华社照片，北京，2026年4月21日</p>
              <p>4月21日，国务院新闻办公室在北京举行新闻发布会，工业和信息化部副部长张云明介绍情况，并答记者问。</p>
              <p>下载客户端</p>
              <p>独家抢先看</p>
            </article>
            <div>亲爱的凤凰网用户:</div>
            <div>您当前使用的浏览器版本过低，导致网站不能正常访问，建议升级浏览器</div>
          </body>
        </html>
        """

        self.assertEqual(
            extract_html_title(html),
            "国新办举行新闻发布会 介绍2026年一季度工业和信息化发展情况",
        )
        text = extract_html_text(html)
        self.assertIn("国务院新闻办公室", text)
        self.assertNotIn("下载客户端", text)
        self.assertNotIn("浏览器版本过低", text)

    def test_cleans_english_news_title_and_advertisements(self):
        html = """
        <html>
          <head><title>Climate change: World leaders meet - BBC News</title></head>
          <body>
            <main>
              <h1>Climate change: World leaders meet</h1>
              <p>World leaders have met to discuss a new climate agreement after months of negotiations.</p>
              <p>Advertisement</p>
              <p>The talks focused on energy security, rising temperatures and funding for vulnerable countries.</p>
            </main>
          </body>
        </html>
        """

        self.assertEqual(extract_html_title(html), "Climate change: World leaders meet")
        text = extract_html_text(html)
        self.assertIn("World leaders have met", text)
        self.assertIn("The talks focused", text)
        self.assertNotIn("Advertisement", text)

    def test_extract_article_uses_html_fallback_when_trafilatura_is_incomplete(self):
        html = """
        <html>
          <head><meta name="twitter:title" content="Fallback title - CNN"></head>
          <body>
            <article>
              <p>This is the complete first paragraph from the page body.</p>
              <p>This is the complete second paragraph that should beat the short extraction.</p>
            </article>
          </body>
        </html>
        """

        class ShortDoc:
            title = "Fallback title - CNN"
            text = "Short extraction."

        with patch("core.trafilatura.fetch_url", return_value=html), patch(
            "core.trafilatura.bare_extraction", return_value=ShortDoc()
        ):
            article = extract_article("https://www.cnn.com/example")

        self.assertTrue(article["success"])
        self.assertEqual(article["title"], "Fallback title")
        self.assertIn("complete second paragraph", article["text"])


if __name__ == "__main__":
    unittest.main()
