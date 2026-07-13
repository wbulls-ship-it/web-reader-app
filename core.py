
import json
import re
from collections import deque
from html import unescape

import trafilatura
from bs4 import BeautifulSoup


UNWANTED_SELECTORS = [
    "script",
    "style",
    "noscript",
    "iframe",
    "form",
    "nav",
    "footer",
    "header",
    "aside",
    ".share",
    ".shares",
    ".social",
    ".related",
    ".recommend",
    ".advert",
    ".advertisement",
    ".ad",
    ".ads",
    "#comment",
    ".comment",
]


def clean_title(title):
    if not title:
        return ""

    if isinstance(title, list):
        title = " ".join(str(item) for item in title if item)

    title = unescape(str(title))
    title = re.sub(r"\s+", " ", title).strip()

    patterns = [
        r"\s*[-|｜_]\s*搜狐.*$",
        r"\s*[-|｜_]\s*腾讯.*$",
        r"\s*[-|｜_]\s*网易.*$",
        r"\s*[-|｜_]\s*新浪.*$",
        r"\s*[-|｜_]\s*凤凰.*$",
        r"\s*[-|｜_]\s*澎湃新闻.*$",
        r"\s*[-|｜_]\s*人民网.*$",
        r"\s*[-|｜_]\s*新华网.*$",
        r"\s*[-|｜_]\s*央视网.*$",
        r"\s*[-|｜_]\s*BBC News.*$",
        r"\s*[-|｜_]\s*CNN.*$",
        r"\s*[-|｜_]\s*The Guardian.*$",
        r"\s*[-|｜_]\s*Reuters.*$",
        r"\s*[-|｜_]\s*AP News.*$",
    ]

    for pattern in patterns:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)

    return title.strip()


def clean_text(text):
    if not text:
        return ""

    text = unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    unwanted_line_patterns = [
        r"^\s*返回搜狐，查看更多\s*$",
        r"^\s*责任编辑[:：].*$",
        r"^\s*责编[:：].*$",
        r"^\s*编辑[:：].{0,20}$",
        r"^\s*更多精彩内容.*$",
        r"^\s*下载客户端\s*$",
        r"^\s*独家抢先看\s*$",
        r"^\s*缩小字体\s+放大字体.*$",
        r"^\s*微博\s+微信.*$",
        r"^\s*腾讯QQ\s+QQ空间\s*$",
        r"^\s*关闭\s*$",
        r"^\s*Image(?::.*)?\s*$",
        r"^\s*图片来源[:：].*$",
        r"^\s*Advertisement\s*$",
        r"^\s*ADVERTISEMENT\s*$",
        r"^\s*Sign up for.*$",
        r"^\s*Subscribe to.*$",
        r"^\s*Read more:.*$",
        r"^\s*Follow .* on .*$",
        r"^\s*Share this.*$",
        r"^\s*All rights reserved\.?\s*$",
        r"^\s*Copyright .*$",
        r"^\s*亲爱的.*用户[:：]?.*$",
        r"^\s*您当前使用的浏览器版本过低.*$",
        r"^\s*第三方浏览器推荐[:：]?.*$",
    ]

    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in unwanted_line_patterns):
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_json_ld_items(data):
    queue = deque([data])

    while queue:
        item = queue.popleft()
        if isinstance(item, dict):
            yield item
            graph = item.get("@graph")
            if isinstance(graph, list):
                queue.extend(graph)
        elif isinstance(item, list):
            queue.extend(item)


def title_score(title):
    cleaned = clean_title(title)
    if not cleaned:
        return 0
    score = min(len(cleaned), 120)
    if re.search(r"\b(article|news|world|china|business)\b", cleaned, flags=re.IGNORECASE):
        score += 5
    return score


def extract_html_title(html):
    soup = BeautifulSoup(html, "lxml")
    candidates = []

    for attrs in (
        {"property": "og:title"},
        {"name": "twitter:title"},
        {"name": "title"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            candidates.append(tag["content"])

    for h1 in soup.find_all("h1", limit=3):
        candidates.append(h1.get_text(" ", strip=True))

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            for item in iter_json_ld_items(data):
                item_type = item.get("@type", "")
                if isinstance(item_type, list):
                    item_type = " ".join(item_type)
                if re.search(r"Article|NewsArticle|BlogPosting|Reportage", str(item_type), re.I):
                    candidates.extend([item.get("headline"), item.get("name")])
                else:
                    candidates.append(item.get("headline"))

        except Exception:
            pass

    if soup.title:
        candidates.append(soup.title.get_text(" ", strip=True))

    cleaned_candidates = [clean_title(candidate) for candidate in candidates if clean_title(candidate)]
    if cleaned_candidates:
        return max(cleaned_candidates, key=title_score)

    return "未识别到标题"


def remove_unwanted_nodes(soup):
    for selector in UNWANTED_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()


def node_text_score(tag):
    text = tag.get_text("\n", strip=True)
    if not text:
        return 0
    paragraph_count = len(tag.find_all(["p", "section"]))
    link_text_len = sum(len(link.get_text(" ", strip=True)) for link in tag.find_all("a"))
    text_len = len(text)
    link_density = link_text_len / max(text_len, 1)
    return text_len + paragraph_count * 80 - link_density * 500


def is_meaningful_paragraph(text):
    text = text.strip()
    if not text:
        return False
    if re.search(r"[\u4e00-\u9fff]", text):
        return len(text) >= 6
    return len(text) >= 20


def extract_html_text(html):
    soup = BeautifulSoup(html, "lxml")
    remove_unwanted_nodes(soup)

    candidates = []
    for selector in [
        "article",
        "main",
        "[role='main']",
        ".article",
        ".article-content",
        ".article-body",
        ".post-content",
        ".content",
        "#article",
        "#content",
    ]:
        candidates.extend(soup.select(selector))

    if not candidates and soup.body:
        candidates = soup.body.find_all(["article", "main", "section", "div"], recursive=True)

    if not candidates:
        return ""

    best = max(candidates, key=node_text_score)
    paragraphs = [
        part.get_text(" ", strip=True)
        for part in best.find_all(["p", "li", "blockquote"])
        if is_meaningful_paragraph(part.get_text(" ", strip=True))
    ]
    if not paragraphs:
        paragraphs = [best.get_text("\n", strip=True)]

    return clean_text("\n\n".join(paragraphs))


def article_statistics(text):
    if not text:
        return {
            "characters": 0,
            "estimated_reading_minutes": 0,
        }

    characters = len(text)
    reading_minutes = max(1, round(characters / 300))

    return {
        "characters": characters,
        "estimated_reading_minutes": reading_minutes,
    }


def extract_article(url):
    try:
        html = trafilatura.fetch_url(url)
    except Exception:
        html = None

    if html is None:
        return {
            "success": False,
            "title": "",
            "text": "",
            "message": "无法下载网页",
        }

    docs = []
    for options in ({"favor_precision": True}, {"favor_recall": True}):
        try:
            docs.append(
                trafilatura.bare_extraction(
                    html,
                    url=url,
                    include_comments=False,
                    include_tables=False,
                    **options,
                )
            )
        except Exception:
            pass
    docs = [doc for doc in docs if doc is not None and doc.text]
    fallback_text = extract_html_text(html)

    if not docs and not fallback_text:
        return {
            "success": False,
            "title": "",
            "text": "",
            "message": "正文提取失败",
        }

    doc = max(docs, key=lambda item: len(clean_text(item.text)), default=None)
    title = clean_title(doc.title) if doc and doc.title else extract_html_title(html)
    text = clean_text(doc.text) if doc else ""
    if len(fallback_text) > len(text) * 1.25:
        text = fallback_text
    stats = article_statistics(text)

    return {
        "success": True,
        "title": title,
        "text": text,
        "characters": stats["characters"],
        "reading_minutes": stats["estimated_reading_minutes"],
        "message": "成功",
    }
