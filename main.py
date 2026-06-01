from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import textwrap
from pathlib import Path
from typing import Any

import arxiv
import feedparser
import requests
import yaml
from google import genai
from google.genai import errors as genai_errors


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
SEEN_PATH = DATA_DIR / "seen.json"
SOURCES_PATH = ROOT / "sources.yaml"
DISCORD_LIMIT = 1900
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_HEADERS = {
    "User-Agent": "ai-daily-digest/0.1 (+https://github.com/) Python requests"
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and send a daily AI digest.")
    parser.add_argument("--sources", default=str(SOURCES_PATH), help="Path to sources.yaml")
    parser.add_argument("--lookback-hours", type=int, default=30, help="How far back to collect items")
    parser.add_argument("--max-items", type=int, default=12, help="Maximum items sent to the summarizer")
    parser.add_argument("--dry-run", action="store_true", help="Generate report without Discord delivery")
    return parser.parse_args()


def load_sources(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    with open(SEEN_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return set(data.get("ids", []))


def save_seen(ids: set[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEEN_PATH, "w", encoding="utf-8") as file:
        json.dump({"ids": sorted(ids)}, file, ensure_ascii=False, indent=2)


def item_id(source: str, url: str, title: str) -> str:
    raw = f"{source}|{url}|{title}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def parse_feed_time(entry: Any) -> dt.datetime | None:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not parsed:
        return None
    return dt.datetime(*parsed[:6], tzinfo=dt.timezone.utc)


def fetch_rss(sources: list[dict[str, str]], since: dt.datetime) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in sources:
        name = source["name"]
        print(f"Fetching RSS: {name}")
        try:
            response = requests.get(
                source["url"],
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except requests.RequestException as exc:
            print(f"Skipped RSS source {name}: {exc}")
            continue

        for entry in feed.entries:
            published_at = parse_feed_time(entry)
            if published_at and published_at < since:
                continue

            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
            if not title or not link:
                continue

            items.append(
                {
                    "id": item_id(name, link, title),
                    "kind": "rss",
                    "source": name,
                    "title": title,
                    "url": link,
                    "published_at": published_at.isoformat() if published_at else None,
                    "summary": clean_text(summary),
                }
            )
    return items


def fetch_arxiv(config: dict[str, Any], since: dt.datetime) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    client = arxiv.Client(page_size=20, delay_seconds=1, num_retries=2)
    max_results = int(config.get("max_results_per_category", 8))

    for category in config.get("categories", []):
        print(f"Fetching arXiv: {category}")
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        try:
            papers = list(client.results(search))
        except Exception as exc:
            print(f"Skipped arXiv category {category}: {exc}")
            continue

        for paper in papers:
            published_at = paper.published
            if published_at and published_at < since:
                continue
            url = paper.entry_id
            title = paper.title.strip()
            authors = ", ".join(author.name for author in paper.authors[:5])

            items.append(
                {
                    "id": item_id(f"arXiv {category}", url, title),
                    "kind": "paper",
                    "source": f"arXiv {category}",
                    "title": title,
                    "url": url,
                    "published_at": published_at.isoformat() if published_at else None,
                    "summary": clean_text(paper.summary),
                    "authors": authors,
                }
            )
    return items


def clean_text(value: str, limit: int = 900) -> str:
    text = " ".join(value.replace("\n", " ").split())
    return text[:limit].rstrip()


def dedupe_new_items(items: list[dict[str, Any]], seen: set[str], max_items: int) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        if item["id"] in seen:
            continue
        unique.setdefault(item["id"], item)

    sorted_items = sorted(
        unique.values(),
        key=lambda item: item.get("published_at") or "",
        reverse=True,
    )
    return sorted_items[:max_items]


def summarize_with_gemini(items: list[dict[str, Any]], report_date: str) -> str:
    if not items:
        return f"**AI Daily Digest - {report_date}**\n\n今天沒有抓到新的 AI 更新。"

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback_report(items, report_date, "未設定 GEMINI_API_KEY。")

    model = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite"
    fallback_models = [
        name.strip()
        for name in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash").split(",")
        if name.strip()
    ]
    models = [model] + [fallback for fallback in fallback_models if fallback != model]
    client = genai.Client(api_key=api_key)
    payload = json.dumps(items, ensure_ascii=False, indent=2)

    prompt = f"""
你是一位 AI 產業與研究情報編輯。請把以下資料整理成「易讀、精煉、繁體中文」的每日 AI Digest，供 Discord 頻道閱讀。

硬性規則：
- 日期：{report_date}
- 所有英文標題、摘要與論文內容都要翻譯成自然的繁體中文。
- 不要逐字硬翻；請用台灣讀者容易理解的科技新聞語氣改寫。
- 不要誇大。若只是推測或尚未證實，請明確標示。
- 每則都要保留來源名稱與原始連結。
- 控制在 1800 字以內。
- 不要使用表格。
- 如果某個分類沒有重要內容，可以省略該分類。

請使用這個固定格式：

**AI 每日快報 - {report_date}**

**今日重點**
用 2 到 3 句話總結今天最值得注意的 AI 動態。

**模型與產品更新**
- **中文標題**：一句話說明發生什麼事。
  影響：用一句話說明對開發者、研究者或產品團隊的意義。
  來源：來源名稱 - 連結

**研究與論文**
- **中文論文標題**：一句話說明研究在做什麼。
  重點：用一句話說明技術亮點或限制。
  來源：來源名稱 - 連結

**開源與工具**
- **中文標題**：一句話說明工具或專案更新。
  適合誰看：指出開發者、研究者、產品團隊或一般使用者。
  來源：來源名稱 - 連結

**趨勢觀察**
用 1 到 2 句話指出今天資料中呈現的共同方向。

資料：
{payload}
""".strip()

    failures: list[str] = []
    for model_name in models:
        print(f"Summarizing {len(items)} items with Gemini model: {model_name}")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = (response.text or "").strip()
            if text:
                return text
            print(f"Gemini model {model_name} returned an empty response.")
        except genai_errors.APIError as exc:
            print(f"Gemini model {model_name} failed: {exc}")
            failures.append(f"{model_name}: {exc}")

    print("All Gemini models failed; writing fallback report.")
    reason = "Gemini 模型都呼叫失敗：" + " | ".join(failures[:3])
    return fallback_report(items, report_date, reason)


def fallback_report(items: list[dict[str, Any]], report_date: str, reason: str) -> str:
    lines = [
        f"**AI 每日快報 - {report_date}**",
        "",
        "**原始抓取結果**",
        f"Gemini 暫時無法產生中文摘要，原因：{reason}",
        "以下先列出抓到的來源：",
        "",
    ]
    for item in items[:15]:
        lines.append(f"- **{item['title']}**")
        lines.append(f"  來源：{item['source']} - {item['url']}")
        if item.get("summary"):
            lines.append(f"  摘要：{item['summary'][:160].rstrip()}")
        lines.append("")
    return "\n".join(lines)


def save_report(report: str, report_date: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{report_date}.md"
    with open(path, "w", encoding="utf-8") as file:
        file.write(report)
        file.write("\n")
    return path


def split_for_discord(message: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for paragraph in message.split("\n\n"):
        addition = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(addition) <= DISCORD_LIMIT:
            current = addition
            continue
        if current:
            chunks.append(current)
        current = paragraph

    if current:
        chunks.append(current)

    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= DISCORD_LIMIT:
            final_chunks.append(chunk)
            continue
        final_chunks.extend(textwrap.wrap(chunk, width=DISCORD_LIMIT, replace_whitespace=False))
    return final_chunks


def send_discord(report: str) -> None:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL is not set.")

    for chunk in split_for_discord(report):
        print("Sending report chunk to Discord")
        response = requests.post(webhook_url, json={"content": chunk}, timeout=20)
        response.raise_for_status()


def main() -> None:
    args = parse_args()
    sources = load_sources(args.sources)
    now = utc_now()
    since = now - dt.timedelta(hours=args.lookback_hours)
    report_date = now.astimezone(dt.timezone(dt.timedelta(hours=8))).date().isoformat()

    seen = load_seen()
    items = []
    print(f"Collecting items since {since.isoformat()}")
    items.extend(fetch_rss(sources.get("rss", []), since))
    items.extend(fetch_arxiv(sources.get("arxiv", {}), since))

    new_items = dedupe_new_items(items, seen, args.max_items)
    print(f"Fetched {len(items)} items, {len(new_items)} are new.")
    report = summarize_with_gemini(new_items, report_date)
    report_path = save_report(report, report_date)

    if not args.dry_run:
        send_discord(report)
        seen.update(item["id"] for item in new_items)
        save_seen(seen)
    else:
        print("Dry run enabled; Discord delivery and seen-state update were skipped.")

    print(f"Fetched {len(items)} items, summarized {len(new_items)} new items.")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
