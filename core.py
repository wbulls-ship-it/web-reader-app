
import json
import re

import trafilatura
from bs4 import BeautifulSoup


def clean_title(title):
    if not title:
        return ""

    title = re.sub(r"\s+", " ", title).strip()

    patterns = [
        r"\s*[-|｜_]\s*搜狐.*$",
        r"\s*[-|｜_]\s*腾讯.*$",
        r"\s*[-|｜_]\s*网易.*$",
        r"\s*[-|｜_]\s*新浪.*$",
    ]

    for pattern in patterns:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)

    return title.strip()


def clean_text(text):
    if not text:
        return ""

    patterns = [
        r"\n返回搜狐，查看更多.*$",
        r"\n责任编辑[:：].*$",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    return text.strip()


def extract_html_title(html):
    soup = BeautifulSoup(html, "lxml")

    tag = soup.find("meta", attrs={"property": "og:title"})
    if tag and tag.get("content"):
        return clean_title(tag["content"])

    tag = soup.find("meta", attrs={"name": "twitter:title"})
    if tag and tag.get("content"):
        return clean_title(tag["content"])

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)

            if isinstance(data, dict):
                if "headline" in data:
                    return clean_title(data["headline"])

                if "name" in data:
                    return clean_title(data["name"])

        except Exception:
            pass

    if soup.title:
        return clean_title(soup.title.get_text())

    h1 = soup.find("h1")
    if h1:
        return clean_title(h1.get_text())

    return "未识别到标题"


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
    html = trafilatura.fetch_url(url)

    if html is None:
        return {
            "success": False,
            "title": "",
            "text": "",
            "message": "无法下载网页",
        }

    doc = trafilatura.bare_extraction(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    if doc is None or not doc.text:
        return {
            "success": False,
            "title": "",
            "text": "",
            "message": "正文提取失败",
        }

    title = doc.title if doc.title else extract_html_title(html)
    text = clean_text(doc.text)
    stats = article_statistics(text)

    return {
        "success": True,
        "title": title,
        "text": text,
        "characters": stats["characters"],
        "reading_minutes": stats["estimated_reading_minutes"],
        "message": "成功",
    }
