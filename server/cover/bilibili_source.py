from __future__ import annotations

import hashlib
import html
import re
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse

import requests


_BV_RE = re.compile(r"\b(BV[0-9A-Za-z]+)\b", re.IGNORECASE)
_BILIBILI_HOME_URL = "https://www.bilibili.com/"
_BILIBILI_NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
_BILIBILI_SEARCH_URL = "https://api.bilibili.com/x/web-interface/wbi/search/type"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_WBI_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]


def is_bilibili_source(value: str) -> bool:
    text = str(value or "").strip()
    host = urlparse(text).netloc.casefold()
    return (
        bool(_BV_RE.search(text))
        or host == "b23.tv"
        or host.endswith(".b23.tv")
        or "bilibili.com" in host
    )


def is_bilibili_search_keyword(value: str) -> bool:
    text = str(value or "").strip()
    if not text or is_bilibili_source(text) or re.fullmatch(r"\d{1,20}", text):
        return False
    parsed = urlparse(text)
    if parsed.scheme or parsed.netloc:
        return False
    local = Path(text).expanduser()
    return not local.drive and not text.startswith((".", "~", "\\", "/"))


def resolve_bilibili_video_url(
    value: str,
    search: Callable[[str], str] | None = None,
) -> str:
    text = str(value or "").strip()
    match = _BV_RE.search(text)
    if match:
        return f"https://www.bilibili.com/video/{match.group(1)}"

    parsed = urlparse(text)
    host = parsed.netloc.casefold()
    if host == "b23.tv" or host.endswith(".b23.tv"):
        return text
    if "bilibili.com" in host and "/video/" in parsed.path:
        return text
    if search is not None and is_bilibili_search_keyword(text):
        return search(text)
    raise ValueError("B站来源必须是 BV 号、b23.tv 短链或 bilibili.com/video 链接")


def _wbi_key_from_url(url: object) -> str:
    filename = urlparse(str(url or "")).path.rsplit("/", maxsplit=1)[-1]
    key = filename.split(".", maxsplit=1)[0]
    if not key:
        raise RuntimeError("B站 WBI 密钥响应无效")
    return key


def _wbi_signed_params(params: dict[str, object], img_key: str, sub_key: str) -> dict[str, object]:
    source_key = img_key + sub_key
    try:
        mixin_key = "".join(source_key[index] for index in _WBI_MIXIN_KEY_ENC_TAB)[:32]
    except IndexError as exc:
        raise RuntimeError("B站 WBI 密钥长度无效") from exc

    signed_params = {
        key: re.sub(r"[!'()*]", "", str(value))
        for key, value in params.items()
    }
    signed_params["wts"] = int(time.time())
    query = urlencode(sorted(signed_params.items()), quote_via=quote, safe="~")
    signed_params["w_rid"] = hashlib.md5(f"{query}{mixin_key}".encode("utf-8")).hexdigest()
    return signed_params


def _clean_title(value: object, fallback: str) -> str:
    title = html.unescape(re.sub(r"<[^>]+>", "", str(value or ""))).strip()
    return title or fallback


def _cover_url(value: object) -> str:
    cover = str(value or "").strip()
    if cover.startswith("//"):
        return f"https:{cover}"
    parsed = urlparse(cover)
    return cover if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def search_bilibili_videos(
    keyword: str,
    session: requests.Session | None = None,
    limit: int = 10,
    page: int = 1,
) -> list[dict[str, str]]:
    text = str(keyword or "").strip()
    if not text:
        raise ValueError("B站搜索关键词不能为空")
    limit = max(1, int(limit))
    page = max(1, int(page))
    client = session or requests.Session()
    close_client = session is None
    headers = {
        "User-Agent": _USER_AGENT,
        "Referer": _BILIBILI_HOME_URL,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        home = client.get(_BILIBILI_HOME_URL, headers=headers, timeout=20)
        home.raise_for_status()
        nav = client.get(_BILIBILI_NAV_URL, headers=headers, timeout=20)
        nav.raise_for_status()
        payload = nav.json()
        wbi_img = payload.get("data", {}).get("wbi_img", {}) if isinstance(payload, dict) else {}
        if not isinstance(wbi_img, dict):
            raise RuntimeError("B站 WBI 密钥响应无效")
        img_key = _wbi_key_from_url(wbi_img.get("img_url"))
        sub_key = _wbi_key_from_url(wbi_img.get("sub_url"))

        for parameters in (
            {"search_type": "video", "keyword": text, "page": page, "tids": 3},
            {"search_type": "video", "keyword": text, "page": page},
        ):
            response = client.get(
                _BILIBILI_SEARCH_URL,
                params=_wbi_signed_params(parameters, img_key, sub_key),
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            result_payload = response.json()
            if not isinstance(result_payload, dict) or result_payload.get("code") != 0:
                message = result_payload.get("message", "未知错误") if isinstance(result_payload, dict) else "响应无效"
                raise RuntimeError(f"B站搜索失败: {message}")
            raw_results = result_payload.get("data", {}).get("result", [])
            if not isinstance(raw_results, list):
                continue
            results: list[dict[str, str]] = []
            seen_bvids: set[str] = set()
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                bvid = str(item.get("bvid") or "").strip()
                if not _BV_RE.fullmatch(bvid) or bvid in seen_bvids:
                    continue
                seen_bvids.add(bvid)
                results.append(
                    {
                        "title": _clean_title(item.get("title"), bvid),
                        "artist": str(item.get("author") or "").strip(),
                        "cover": _cover_url(item.get("pic")),
                        "url": f"https://www.bilibili.com/video/{bvid}",
                    }
                )
                if len(results) >= limit:
                    return results
            if results:
                return results
        return []
    except requests.RequestException as exc:
        raise RuntimeError(f"B站搜索失败: {exc}") from exc
    finally:
        if close_client:
            client.close()


def search_bilibili_first_video(
    keyword: str,
    session: requests.Session | None = None,
) -> str:
    videos = search_bilibili_videos(keyword, session=session, limit=1)
    if videos:
        return videos[0]["url"]
    raise ValueError("B站搜索没有找到视频")


def download_bilibili_audio(
    video_url: str,
    destination: Path,
    ffmpeg_location: Path,
    max_bytes: int,
) -> Path:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("缺少 yt-dlp，请重新运行安装便携Python.bat") from exc

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "noplaylist": True,
        "format": "bestaudio/best",
        "max_filesize": max(1, int(max_bytes)),
        "outtmpl": str(destination.with_suffix(".%(ext)s")),
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": str(ffmpeg_location),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }
    with YoutubeDL(options) as downloader:
        downloader.download([video_url])

    result = destination.with_suffix(".wav")
    if not result.is_file() or result.stat().st_size <= 0:
        raise RuntimeError("B站音频下载后没有生成 WAV 文件")
    return result
