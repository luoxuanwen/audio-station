from __future__ import annotations

import base64
import hashlib
import html
import ipaddress
import json
import os
import random
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import gradio as gr
import requests
from bilibili_source import (
    download_bilibili_audio,
    is_bilibili_search_keyword,
    is_bilibili_source,
    resolve_bilibili_video_url,
    search_bilibili_first_video,
    search_bilibili_videos,
)
from gradio_client import Client, handle_file
from msst_bridge import MSSTBridgeError, describe_msst, list_msst_models, separate_target


APP_NAME = "Seed-VC Singing Gateway"
APP_VERSION = "2.0.0"
PIPELINE_VERSION = "seedvc-44k-v1-msst"
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("SVCVC_CONFIG", BASE_DIR / "config.json"))
VOICE_DIR = BASE_DIR / "voice_profiles"
CACHE_DIR = BASE_DIR / "cache"
DOWNLOAD_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "outputs"
MSST_STEM_DIR = CACHE_DIR / "_msst_stems"
FFMPEG_PATH = BASE_DIR / "bin" / "ffmpeg.exe"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
GPU_JOB_LOCK = threading.Lock()
DOWNLOAD_CACHE_LOCK = threading.Lock()
CONFIG_WRITE_LOCK = threading.Lock()
MAX_PROFILE_ID_LENGTH = 80
WINDOWS_DANGEROUS_CHARS = set('<>:"/\\|?*')
MAX_HTTP_REDIRECTS = 8


DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 6767,
        "queue_max_size": 8,
    },
    "msst": {
        "root": ".",
        "model": "bs_roformer_ep_317_sdr_12.9755",
        "batch_size": 1,
        "num_overlap": 4,
        "normalize": False,
        "use_tta": False,
        "stem_cache_max_files": 20,
    },
    "download": {
        "timeout_seconds": 60,
        "max_size_mb": 500,
        "max_files": 100,
    },
    "cache": {
        "enabled": True,
        "random_seed_persistent": False,
        "max_files": 100,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return DEFAULT_CONFIG
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取配置文件 {CONFIG_PATH}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("config.json 顶层必须是 JSON 对象")
    return _deep_merge(DEFAULT_CONFIG, data)


CONFIG = _load_config()
for directory in (
    VOICE_DIR,
    CACHE_DIR,
    DOWNLOAD_DIR,
    OUTPUT_DIR,
    MSST_STEM_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


def _save_config() -> None:
    """Persist the merged configuration atomically."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    serialized = json.dumps(CONFIG, ensure_ascii=False, indent=2) + "\n"
    with CONFIG_WRITE_LOCK:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, CONFIG_PATH)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_voice_file(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(VOICE_DIR.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return resolved.parent == VOICE_DIR.resolve()


def _validate_profile_id(value: Any) -> str:
    profile_id = str(value or "").strip()
    if not profile_id:
        raise ValueError("profile_id 不能为空")
    if len(profile_id) > MAX_PROFILE_ID_LENGTH:
        raise ValueError(f"profile_id 不能超过 {MAX_PROFILE_ID_LENGTH} 个字符")
    if "|||" in profile_id:
        raise ValueError("profile_id 不能包含模型别名分隔符 |||")
    if profile_id in {".", ".."} or profile_id.endswith((" ", ".")):
        raise ValueError("profile_id 不能是相对路径标记，也不能以空格或点结尾")
    if any(ch in WINDOWS_DANGEROUS_CHARS for ch in profile_id):
        raise ValueError("profile_id 包含 Windows 文件名危险字符")
    if any(unicodedata.category(ch) == "Cc" for ch in profile_id):
        raise ValueError("profile_id 不能包含控制字符")
    return profile_id


def _read_profile_metadata(audio_path: Path) -> dict[str, Any]:
    metadata_path = audio_path.with_suffix(".json")
    if not metadata_path.is_file() or metadata_path.is_symlink():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] 忽略无效音色元数据 {metadata_path.name}: {exc}", flush=True)
        return {}


def _scan_voice_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for audio_path in sorted(VOICE_DIR.iterdir(), key=lambda item: item.name.casefold()):
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        if not _safe_voice_file(audio_path):
            print(f"[WARN] 跳过不安全的音色文件: {audio_path}", flush=True)
            continue
        metadata = _read_profile_metadata(audio_path)
        try:
            profile_id = _validate_profile_id(metadata.get("profile_id") or audio_path.stem)
        except ValueError as exc:
            print(f"[WARN] 跳过 ID 不安全的音色文件 {audio_path.name}: {exc}", flush=True)
            continue
        if profile_id in used_ids:
            print(f"[WARN] 音色 ID 重复，跳过 {audio_path.name}: {profile_id}", flush=True)
            continue
        used_ids.add(profile_id)
        profiles.append(
            {
                "profile_id": profile_id,
                "display_name": str(metadata.get("display_name") or audio_path.stem),
                "description": str(metadata.get("description") or ""),
                "audio_file": audio_path.name,
                "audio_path": str(audio_path.resolve()),
                "audio_sha256": _sha256_file(audio_path),
                "prompt_vocal_sep": bool(metadata.get("prompt_vocal_sep", False)),
            }
        )
    return profiles


def show_model() -> list[str]:
    """返回兼容 Matsuko Cover 模型刷新逻辑的音色 ID 列表。"""
    models = [profile["profile_id"] for profile in _scan_voice_profiles()]
    print(f"[音色列表Debug] 返回 {len(models)} 个 SVCVC 音色: {', '.join(models) or '(空)'}", flush=True)
    return models


def list_voice_profiles() -> list[dict[str, Any]]:
    """返回音色库的详细信息。"""
    profiles = _scan_voice_profiles()
    print(f"[音色详情Debug] 返回 {len(profiles)} 个音色配置", flush=True)
    return profiles


def _cache_areas() -> dict[str, Path]:
    return {"results": CACHE_DIR, "downloads": DOWNLOAD_DIR, "outputs": OUTPUT_DIR}


def _managed_cache_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root.resolve(strict=True))
            if path.is_file() and not path.is_symlink():
                files.append(resolved)
        except (OSError, ValueError):
            continue
    return files


def cache_info() -> dict[str, Any]:
    areas: dict[str, dict[str, int]] = {}
    for name, root in _cache_areas().items():
        files = _managed_cache_files(root)
        areas[name] = {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }
    return {
        "service": APP_NAME,
        "areas": areas,
        "total_files": sum(item["files"] for item in areas.values()),
        "total_bytes": sum(item["bytes"] for item in areas.values()),
    }


def clear_cache(scope: str = "all") -> dict[str, Any]:
    requested = str(scope or "all").strip().lower()
    all_areas = _cache_areas()
    selected = all_areas if requested == "all" else {key: value for key, value in all_areas.items() if key == requested}
    if not selected:
        raise gr.Error(f"未知缓存区域: {scope}；可用值: all, {', '.join(all_areas)}")
    deleted = freed = 0
    with GPU_JOB_LOCK:
        for root in selected.values():
            for path in _managed_cache_files(root):
                try:
                    size = path.stat().st_size
                    path.unlink(missing_ok=True)
                    deleted += 1
                    freed += size
                except OSError as exc:
                    print(f"[WARN] 无法清理缓存 {path}: {exc}", flush=True)
            for directory in sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda p: len(p.parts), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        # results/all 会移除 CACHE_DIR 下的 _msst_stems。缓存清理完成后必须
        # 恢复运行期目录，否则下一次 tempfile.mkdtemp(dir=...) 会触发 WinError 3。
        for directory in (
            CACHE_DIR,
            DOWNLOAD_DIR,
            OUTPUT_DIR,
            MSST_STEM_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)
    print(f"[缓存清理Debug] scope={requested} files={deleted} bytes={freed}", flush=True)
    return {"scope": requested, "deleted_files": deleted, "freed_bytes": freed}


def _extract_netease_id(source: str) -> str | None:
    value = source.strip()
    if re.fullmatch(r"\d{1,20}", value):
        return value
    parsed = urlparse(value)
    if parsed.hostname and (parsed.hostname.endswith("music.163.com") or parsed.hostname.endswith("163cn.tv")):
        query_id = parse_qs(parsed.query).get("id", [None])[0]
        if query_id and re.fullmatch(r"\d{1,20}", query_id):
            return query_id
        match = re.search(r"(?:song/|id=)(\d{1,20})", value)
        if match:
            return match.group(1)
    return None


def _download_max_bytes() -> int:
    return max(1, int(float(CONFIG["download"]["max_size_mb"]) * 1024 * 1024))


def _touch_and_trim_downloads(current_path: Path) -> None:
    """Update LRU time and bound the managed source-audio cache."""
    try:
        current = current_path.resolve(strict=True)
        download_root = DOWNLOAD_DIR.resolve(strict=True)
        current.relative_to(download_root)
    except (OSError, ValueError):
        return
    if current.parent != download_root or current.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        return

    max_files = max(1, int(CONFIG["download"].get("max_files", 100)))
    with DOWNLOAD_CACHE_LOCK:
        try:
            os.utime(current, None)
        except OSError as exc:
            print(f"[WARN] 无法更新下载缓存使用时间 {current.name}: {exc}", flush=True)
        files: list[Path] = []
        for path in DOWNLOAD_DIR.iterdir():
            try:
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                ):
                    files.append(path.resolve(strict=True))
            except OSError:
                continue
        other_files = sorted(
            (path for path in files if path != current),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in other_files[max_files - 1 :]:
            try:
                stale.unlink(missing_ok=True)
            except OSError as exc:
                print(f"[WARN] 无法清理下载缓存 {stale.name}: {exc}", flush=True)


def _validate_target_file(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            "目标文件必须是受支持的音频格式: "
            + ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        )
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError("目标音频为空文件")
    max_bytes = _download_max_bytes()
    if size > max_bytes:
        raise ValueError(f"目标音频超过 {CONFIG['download']['max_size_mb']} MB 限制")
    return resolved


def _validate_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("只允许使用 HTTP/HTTPS 音频 URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("远程音频 URL 不能包含用户名或密码")
    hostname = parsed.hostname.rstrip(".")
    if hostname.casefold() == "localhost" or hostname.casefold().endswith(".localhost"):
        raise ValueError("拒绝访问本机地址")
    try:
        literal_addresses = [ipaddress.ip_address(hostname.split("%", 1)[0])]
    except ValueError:
        try:
            records = socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise ValueError(f"无法解析远程音频主机 {hostname}: {exc}") from exc
        literal_addresses = []
        for record in records:
            address = record[4][0].split("%", 1)[0]
            ip = ipaddress.ip_address(address)
            if ip not in literal_addresses:
                literal_addresses.append(ip)
    if not literal_addresses:
        raise ValueError(f"远程音频主机 {hostname} 没有可用地址")
    blocked = [str(ip) for ip in literal_addresses if not ip.is_global]
    if blocked:
        raise ValueError(f"拒绝访问非公网地址: {', '.join(blocked)}")


def _looks_like_audio_payload(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
    except OSError:
        return False
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return True
    if header.startswith((b"fLaC", b"OggS", b"ID3")):
        return True
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return True
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return bool(info.frames > 0 and info.samplerate > 0)
    except Exception:
        return False


def _validate_downloaded_audio(path: Path, content_type: str = "") -> None:
    _validate_target_file(path)
    mime = content_type.split(";", 1)[0].strip().casefold()
    plausible_mime = (
        not mime
        or mime.startswith("audio/")
        or mime in {"application/octet-stream", "application/ogg", "video/mp4"}
    )
    if not plausible_mime:
        raise ValueError(f"远程服务器返回的内容类型不是音频: {content_type}")
    if not _looks_like_audio_payload(path):
        raise ValueError("远程服务器返回的内容无法识别为有效音频")


def _emit_status(
    callback: Callable[[float, str], None] | None,
    value: float,
    description: str,
) -> None:
    print(f"[下载Debug] {description}", flush=True)
    if callback is None:
        return
    try:
        callback(max(0.0, min(1.0, float(value))), description)
    except Exception:
        pass


def _download_stream(
    url: str,
    destination: Path,
    status_callback: Callable[[float, str], None] | None = None,
    source_label: str = "远程音频",
) -> Path:
    timeout = float(CONFIG["download"]["timeout_seconds"])
    max_bytes = _download_max_bytes()
    temporary = destination.with_suffix(destination.suffix + ".part")
    written = 0
    response: requests.Response | None = None
    session: requests.Session | None = None
    try:
        _emit_status(status_callback, 0.02, f"正在连接{source_label}下载源")
        current_url = url
        session = requests.Session()
        for redirect_count in range(MAX_HTTP_REDIRECTS + 1):
            _validate_public_http_url(current_url)
            response = session.get(
                current_url,
                headers={"User-Agent": "Mozilla/5.0"},
                stream=True,
                timeout=timeout,
                allow_redirects=False,
            )
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    response.close()
                    raise ValueError("远程服务器返回了没有 Location 的重定向")
                next_url = urljoin(str(response.url), location)
                _validate_public_http_url(next_url)
                response.close()
                response = None
                if redirect_count >= MAX_HTTP_REDIRECTS:
                    raise ValueError("远程音频重定向次数过多")
                current_url = next_url
                continue
            break
        if response is None:
            raise ValueError("未能取得远程音频响应")
        _validate_public_http_url(str(response.url))
        with response:
            response.raise_for_status()
            length = int(response.headers.get("Content-Length") or 0)
            if length > max_bytes:
                raise ValueError(f"远程音频超过 {CONFIG['download']['max_size_mb']} MB 限制")
            with temporary.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    if not block:
                        continue
                    written += len(block)
                    if written > max_bytes:
                        raise ValueError(f"远程音频超过 {CONFIG['download']['max_size_mb']} MB 限制")
                    output.write(block)
                    downloaded_mb = written / (1024 * 1024)
                    if length > 0:
                        ratio = min(1.0, written / length)
                        total_mb = length / (1024 * 1024)
                        _emit_status(
                            status_callback,
                            0.08 + ratio * 0.84,
                            f"正在下载{source_label}：{downloaded_mb:.1f}/{total_mb:.1f} MiB（{ratio * 100:.0f}%）",
                        )
                    else:
                        _emit_status(
                            status_callback,
                            min(0.90, 0.08 + written / max_bytes * 0.82),
                            f"正在下载{source_label}：已接收 {downloaded_mb:.1f} MiB",
                        )
        if written == 0:
            raise ValueError("远程服务器返回了空音频")
        temporary.replace(destination)
        try:
            _validate_downloaded_audio(destination, response.headers.get("Content-Type", ""))
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        _touch_and_trim_downloads(destination)
        _emit_status(status_callback, 1.0, f"{source_label}下载完成：{destination.name}")
        return destination
    finally:
        if response is not None:
            response.close()
        if session is not None:
            session.close()
        temporary.unlink(missing_ok=True)


def _get_public_json(url: str, timeout: float) -> dict[str, Any]:
    current_url = url
    with requests.Session() as session:
        for redirect_count in range(MAX_HTTP_REDIRECTS + 1):
            _validate_public_http_url(current_url)
            with session.get(current_url, timeout=timeout, allow_redirects=False) as response:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("JSON 服务返回了没有 Location 的重定向")
                    next_url = urljoin(str(response.url), location)
                    _validate_public_http_url(next_url)
                    if redirect_count >= MAX_HTTP_REDIRECTS:
                        raise ValueError("JSON 服务重定向次数过多")
                    current_url = next_url
                    continue
                _validate_public_http_url(str(response.url))
                response.raise_for_status()
                if len(response.content) > 2 * 1024 * 1024:
                    raise ValueError("JSON 服务响应过大")
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("JSON 服务未返回对象")
                return data
    raise ValueError("未能取得 JSON 服务响应")


_NETEASE_IMAGE_KEY = b"3go8&$8*3*3h0k(2)2"
_NETEASE_WEB_SEARCH_URL = "https://music.163.com/api/search/get/web"


def _netease_provider_page(page: int) -> int:
    return (max(1, int(page)) - 1) * 10 + 1


def _netease_cover_url(pic_id: object) -> str:
    text = str(pic_id or "").strip()
    if not text:
        return ""
    encrypted = bytearray(text.encode("utf-8"))
    for index in range(len(encrypted)):
        encrypted[index] ^= _NETEASE_IMAGE_KEY[index % len(_NETEASE_IMAGE_KEY)]
    hash_part = base64.b64encode(hashlib.md5(encrypted).digest()).decode("ascii")
    hash_part = hash_part.replace("/", "_").replace("+", "-")
    return f"https://p1.music.126.net/{hash_part}/{text}.jpg"


def _search_netease_songs_vkeys(keyword: str, page: int) -> list[dict[str, str]]:
    provider_page = _netease_provider_page(page)
    timeout = float(CONFIG["download"]["timeout_seconds"])
    payload = _get_public_json(
        f"https://api.vkeys.cn/v2/music/netease?{urlencode({'word': keyword, 'page': provider_page})}",
        timeout=timeout,
    )
    raw_results = payload.get("data", [])
    if isinstance(raw_results, dict):
        raw_results = [raw_results]
    if not isinstance(raw_results, list):
        return []

    songs: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        song_id = str(item.get("id") or "").strip()
        if not song_id or song_id in seen_ids:
            continue
        seen_ids.add(song_id)
        songs.append(
            {
                "title": str(item.get("song") or song_id).strip(),
                "artist": str(item.get("singer") or "").strip(),
                "cover": str(item.get("cover") or "").strip(),
                "url": f"https://music.163.com/#/song?id={song_id}",
            }
        )
        if len(songs) >= 10:
            break
    return songs


def _search_netease_songs_official(keyword: str, page: int) -> list[dict[str, str]]:
    timeout = float(CONFIG["download"]["timeout_seconds"])
    response = requests.post(
        _NETEASE_WEB_SEARCH_URL,
        data={
            "s": keyword,
            "limit": 10,
            "type": 1,
            "offset": (max(1, int(page)) - 1) * 10,
        },
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://music.163.com/",
        },
        cookies={"appver": "2.0.2"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("code") != 200:
        raise ValueError("网易云备用搜索接口返回异常")
    result = payload.get("result", {})
    raw_results = result.get("songs", []) if isinstance(result, dict) else []
    if not isinstance(raw_results, list):
        return []

    songs: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        song_id = str(item.get("id") or "").strip()
        if not song_id:
            continue
        artists = item.get("artists", [])
        artist = " / ".join(
            str(entry.get("name") or "").strip()
            for entry in artists
            if isinstance(entry, dict) and str(entry.get("name") or "").strip()
        )
        album = item.get("album", {})
        cover = _netease_cover_url(album.get("picId")) if isinstance(album, dict) else ""
        songs.append(
            {
                "title": str(item.get("name") or song_id).strip(),
                "artist": artist,
                "cover": cover,
                "url": f"https://music.163.com/#/song?id={song_id}",
            }
        )
    return songs


def _search_netease_songs(keyword: str, page: int = 1) -> list[dict[str, str]]:
    text = str(keyword or "").strip()
    if not text:
        raise ValueError("网易云搜索关键词不能为空")
    try:
        primary_results = _search_netease_songs_vkeys(text, page)
    except (requests.RequestException, ValueError) as primary_error:
        primary_results = []
    else:
        primary_error = None
    if primary_results:
        return primary_results
    try:
        return _search_netease_songs_official(text, page)
    except (requests.RequestException, ValueError) as backup_error:
        if primary_error is not None:
            raise RuntimeError(f"网易云搜索失败：主接口和备用接口均不可用（{backup_error}）") from backup_error
        raise RuntimeError(f"网易云搜索失败：备用接口不可用（{backup_error}）") from backup_error


def search_song_catalog(platform: str, keyword: str, page: int = 1) -> list[dict[str, str]]:
    if platform == "B站":
        return search_bilibili_videos(keyword, page=page)
    if platform == "网易云":
        return _search_netease_songs(keyword, page=page)
    raise ValueError("请选择 B站 或 网易云")


def _catalog_url(value: object) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return url
    return ""


def _catalog_label(item: dict[str, str]) -> str:
    title = str(item.get("title") or "未命名结果").strip()
    artist = str(item.get("artist") or "").strip()
    return f"{title} — {artist}" if artist else title


def _catalog_cover_html(item: dict[str, str]) -> str:
    title = html.escape(str(item.get("title") or "歌曲"), quote=True)
    cover = _catalog_url(item.get("cover"))
    if not cover:
        return '<div style="padding: 1rem; color: #666;">暂无封面</div>'
    return (
        f'<img src="{html.escape(cover, quote=True)}" alt="{title} 封面" '
        'referrerpolicy="no-referrer" '
        'style="width: min(100%, 320px); aspect-ratio: 1; object-fit: cover; border-radius: 10px;">'
    )


def _catalog_cards_html(results: list[dict[str, str]]) -> str:
    cards: list[str] = []
    for item in results:
        title = html.escape(str(item.get("title") or "未命名结果"), quote=True)
        artist = html.escape(str(item.get("artist") or "未知作者"), quote=True)
        url = _catalog_url(item.get("url"))
        cover = _catalog_url(item.get("cover"))
        cover_html = (
            f'<img src="{html.escape(cover, quote=True)}" alt="" '
            'referrerpolicy="no-referrer" '
            'style="width: 72px; height: 72px; object-fit: cover; border-radius: 8px; flex: 0 0 auto;">'
            if cover
            else '<div style="width: 72px; height: 72px; border-radius: 8px; background: #eee; flex: 0 0 auto;"></div>'
        )
        link_html = (
            f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">打开链接</a>'
            if url
            else "链接不可用"
        )
        copy_html = (
            f'<button type="button" data-copy-url="{html.escape(url, quote=True)}" '
            'onclick="navigator.clipboard.writeText(this.dataset.copyUrl)" '
            'style="margin-left: auto; white-space: nowrap;">复制链接</button>'
            if url
            else ""
        )
        cards.append(
            '<div style="display: flex; gap: 12px; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee;">'
            f'{cover_html}<div style="min-width: 0;"><div style="font-weight: 600;">{title}</div>'
            f'<div style="color: #666; margin: 4px 0 8px;">{artist}</div>{link_html}</div>{copy_html}</div>'
        )
    return "".join(cards) or '<div style="padding: 1rem; color: #666;">没有结果</div>'


def _selected_catalog_result(selected_url: str, results: list[dict[str, str]]) -> tuple[str, str]:
    for item in results or []:
        if isinstance(item, dict) and item.get("url") == selected_url:
            return _catalog_cover_html(item), selected_url
    return '<div style="padding: 1rem; color: #666;">请选择结果</div>', ""


def search_song_catalog_ui(
    platform: str,
    keyword: str,
    page: int = 1,
) -> tuple[list[dict[str, str]], str, dict[str, Any], str, str, dict[str, Any], dict[str, Any]]:
    current_page = max(1, int(page))
    results = search_song_catalog(platform, keyword, page=current_page)
    if not results:
        raise ValueError(f"{platform} 没有找到与“{str(keyword).strip()}”匹配的结果")
    first = results[0]
    selected_url = str(first["url"])
    choices = [(_catalog_label(item), str(item["url"])) for item in results]
    cover_html, link = _selected_catalog_result(selected_url, results)
    return (
        results,
        _catalog_cards_html(results),
        gr.update(choices=choices, value=selected_url),
        cover_html,
        link,
        gr.update(value=current_page),
        gr.update(interactive=current_page > 1),
    )


def fetch_source_gate_api(platform: str, keyword: str, index: int = 0):
    """AstrBot 备用音源接口：按平台搜索并下载到网关缓存，返回文件路径。"""
    raw_platform = str(platform or "").strip().lower()
    platform_norm = ""
    if raw_platform in {"bilibili", "b站", "bili"}:
        platform_norm = "B站"
    elif raw_platform in {"netease", "网易云"}:
        platform_norm = "网易云"
    if not platform_norm:
        raise gr.Error("platform 必须是 bilibili 或 netease")
    kw = str(keyword or "").strip()
    if not kw:
        raise gr.Error("keyword 不能为空")
    results = search_song_catalog(platform_norm, kw, page=1)
    if not results:
        raise gr.Error(f"{platform_norm} 没有搜到《{kw}》")
    idx = max(0, int(index))
    if idx >= len(results):
        idx = 0
    item = results[idx]
    url = str(item.get("url") or "").strip()
    if not url:
        raise gr.Error("搜索结果链接不可用")

    def _status_cb(value: float, description: str) -> None:
        try:
            print(f"[GATEWAY_SRC] {int(max(0.0, min(1.0, value)) * 100)}% {description}", flush=True)
        except Exception:
            pass

    if platform_norm == "网易云":
        song_id = _extract_netease_id(url)
        if not song_id:
            raise gr.Error(f"无法从结果链接解析网易云歌曲 ID: {url}")
        path = _download_netease(str(song_id), _status_cb)
    else:
        path = _resolve_bilibili_audio(url, _status_cb)
    p = Path(path)
    if not p.is_file() or p.stat().st_size <= 0:
        raise gr.Error("备用音源下载后无有效文件")
    return str(p.resolve())


def _change_catalog_page(page: int, delta: int) -> int:
    return max(1, int(page) + delta)


def _reset_catalog_page() -> tuple[int, dict[str, Any]]:
    return 1, gr.update(interactive=False)


def _download_netease(
    song_id: str,
    status_callback: Callable[[float, str], None] | None = None,
) -> Path:
    destination = DOWNLOAD_DIR / f"netease_{song_id}.mp3"
    if destination.is_file() and destination.stat().st_size > 0:
        try:
            _validate_downloaded_audio(destination)
            _touch_and_trim_downloads(destination)
            _emit_status(status_callback, 1.0, f"网易云歌曲下载缓存命中：ID {song_id}")
            return destination
        except ValueError:
            destination.unlink(missing_ok=True)
    primary = f"https://biliplayer.91vrchat.com/player/?url=https://music.163.com/song?id={song_id}"
    _emit_status(status_callback, 0.0, f"正在下载网易云歌曲：ID {song_id}")
    try:
        return _download_stream(primary, destination, status_callback, "网易云歌曲")
    except Exception as primary_error:
        print(f"[WARN] 网易云主下载源失败: {primary_error}", flush=True)
        _emit_status(status_callback, 0.10, "网易云主下载源失败，正在切换备用源")
    timeout = float(CONFIG["download"]["timeout_seconds"])
    metadata = _get_public_json(
        f"https://api.vkeys.cn/v2/music/netease?id={song_id}", timeout=timeout
    )
    media_url = metadata.get("data", {}).get("url")
    if not media_url:
        raise ValueError("网易云备用下载源没有返回音频地址")
    return _download_stream(str(media_url), destination, status_callback, "网易云歌曲（备用源）")


def _download_direct(
    url: str,
    status_callback: Callable[[float, str], None] | None = None,
) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        suffix = ".mp3"
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    destination = DOWNLOAD_DIR / f"url_{key}{suffix}"
    if destination.is_file() and destination.stat().st_size > 0:
        try:
            _validate_downloaded_audio(destination)
            _touch_and_trim_downloads(destination)
            _emit_status(status_callback, 1.0, f"远程音频下载缓存命中：{destination.name}")
            return destination
        except ValueError:
            destination.unlink(missing_ok=True)
    return _download_stream(url, destination, status_callback, "远程/QQ音乐音频")


def _resolve_target_audio(
    source: str,
    status_callback: Callable[[float, str], None] | None = None,
) -> Path:
    if not source or not source.strip():
        raise ValueError("song_name_src 不能为空")
    source = source.strip().strip('"')
    local = Path(source).expanduser()
    if local.is_file():
        if local.is_symlink():
            raise ValueError("不接受符号链接目标音频")
        resolved = _validate_target_file(local)
        _emit_status(
            status_callback,
            1.0,
            f"已收到本地音频：{resolved.name}（QQ音乐由 AstrBot 插件预下载）",
        )
        return resolved
    netease_id = _extract_netease_id(source)
    if netease_id:
        return _download_netease(netease_id, status_callback)
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return _download_direct(source, status_callback)
    raise FileNotFoundError(f"找不到目标音频，也无法识别为 HTTP/网易云来源: {source}")


def _ffmpeg_paths() -> tuple[Path, Path]:
    bundled_ffmpeg = FFMPEG_PATH
    bundled_ffprobe = bundled_ffmpeg.with_name("ffprobe.exe")
    if bundled_ffmpeg.is_file() and bundled_ffprobe.is_file():
        return bundled_ffmpeg, bundled_ffprobe

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        system_ffmpeg = Path(ffmpeg).resolve()
        system_ffprobe = Path(ffprobe).resolve()
        if system_ffmpeg.parent == system_ffprobe.parent:
            return system_ffmpeg, system_ffprobe
    raise FileNotFoundError(
        "找不到可用的 FFmpeg 和 FFprobe；请将它们放到 "
        f"{FFMPEG_PATH.parent}，或加入系统 PATH"
    )


def _ffmpeg_location() -> Path:
    return _ffmpeg_paths()[0].parent


def _ffmpeg_executable() -> Path:
    return _ffmpeg_paths()[0]


def _resolve_bilibili_audio(
    source: str,
    status_callback: Callable[[float, str], None] | None = None,
) -> Path:
    video_url = resolve_bilibili_video_url(source, search=search_bilibili_first_video)
    key = hashlib.sha256(video_url.encode("utf-8")).hexdigest()[:24]
    destination = DOWNLOAD_DIR / f"bilibili_{key}.wav"
    if not destination.is_file() or destination.stat().st_size <= 0:
        _emit_status(status_callback, 0.0, "正在下载 B 站音频...")
        downloaded = download_bilibili_audio(
            video_url,
            destination,
            _ffmpeg_location(),
            _download_max_bytes(),
        )
        if downloaded.resolve() != destination.resolve():
            downloaded.replace(destination)
    _validate_downloaded_audio(destination)
    _touch_and_trim_downloads(destination)
    _emit_status(status_callback, 1.0, f"B站音频已就绪：{destination.name}")
    return destination


def _resolve_audio_source(
    source: str,
    status_callback: Callable[[float, str], None] | None = None,
) -> Path:
    if is_bilibili_source(source) or is_bilibili_search_keyword(source):
        return _resolve_bilibili_audio(source, status_callback)
    return _resolve_target_audio(source, status_callback)


def _resolve_input_audio(
    source: str | None,
    upload: str | None,
    role: str,
    status_callback: Callable[[float, str], None] | None = None,
) -> Path:
    if upload:
        return _validate_target_file(Path(upload))
    if str(source or "").strip():
        return _resolve_audio_source(str(source), status_callback)
    raise ValueError(f"请提供{role}来源或上传{role}文件")


def _validate_parameters(
    pitch_shift: int,
    n_step: int,
    cfg: float,
    seed: int,
) -> tuple[int, int, float, int]:
    pitch_shift = int(pitch_shift)
    n_step = int(n_step)
    cfg = float(cfg)
    seed = int(seed)
    if not -36 <= pitch_shift <= 36:
        raise ValueError("pitch_shift 必须在 -36 到 36 之间")
    if not 1 <= n_step <= 200:
        raise ValueError("n_step 必须在 1 到 200 之间")
    if not 0.0 <= cfg <= 10.0:
        raise ValueError("cfg 必须在 0 到 10 之间")
    if seed != -1 and not 0 <= seed <= 4294967295:
        raise ValueError("seed 必须是 -1（随机）或 0 到 4294967295")
    return pitch_shift, n_step, cfg, seed


def _find_profile(profile_id: str) -> dict[str, Any]:
    profile_id = _validate_profile_id(profile_id)
    for profile in _scan_voice_profiles():
        if profile["profile_id"] == profile_id:
            return profile
    raise ValueError(f"找不到参考音色: {profile_id}；可用音色: {', '.join(show_model()) or '(空)'}")


def _normalize_target_separation(value: Any, legacy_target_vocal_sep: bool = True) -> str:
    aliases = {
        "msst": "msst",
        "msst人声分离": "msst",
        "msst 人声分离": "msst",
        "none": "none",
        "off": "none",
        "关闭": "none",
        "不分离": "none",
    }
    raw = str(value or "").strip().casefold()
    if not raw:
        return "msst" if bool(legacy_target_vocal_sep) else "none"
    normalized = aliases.get(raw)
    if normalized is None:
        raise ValueError("目标歌曲人声分离方式只能是 msst 或 none")
    return normalized


def _msst_description() -> dict[str, Any]:
    return describe_msst(BASE_DIR, dict(CONFIG.get("msst") or {}))


def show_msst_models() -> list[dict[str, Any]]:
    """Return installed SVCVC MSST models and mark the persisted default."""
    models = list_msst_models(BASE_DIR, dict(CONFIG.get("msst") or {}))
    safe_models = [
        {
            "id": item["id"],
            "name": item["name"],
            "current": bool(item.get("current")),
        }
        for item in models
    ]
    print(
        "[SVCVC MSST模型列表Debug] "
        + ", ".join(
            f"{item['id']}{' [current]' if item['current'] else ''}" for item in safe_models
        ),
        flush=True,
    )
    return safe_models


def select_msst_model(model_name: str) -> dict[str, Any]:
    """Select an installed bundled MSST model and persist it for later tasks."""
    requested = str(model_name or "").strip()
    models = show_msst_models()
    selected: dict[str, Any] | None = None
    if requested.isdigit():
        index = int(requested) - 1
        if 0 <= index < len(models):
            selected = models[index]
    else:
        lowered = requested.casefold()
        exact = [
            item
            for item in models
            if lowered in {str(item["id"]).casefold(), str(item["name"]).casefold()}
        ]
        partial = [
            item
            for item in models
            if lowered and (
                lowered in str(item["id"]).casefold()
                or lowered in str(item["name"]).casefold()
            )
        ]
        matches = exact or partial
        if len(matches) == 1:
            selected = matches[0]
    if selected is None:
        raise gr.Error(
            f"没有唯一匹配的 SVCVC MSST 模型: {requested or '(空)'}；"
            f"可用模型: {', '.join(item['id'] for item in models)}"
        )

    with GPU_JOB_LOCK:
        CONFIG.setdefault("msst", {})["model"] = selected["id"]
        _save_config()
    refreshed = show_msst_models()
    print(
        f"[SVCVC MSST模型切换Debug] 已切换并保存: {selected['id']} ({selected['name']})",
        flush=True,
    )
    return {
        "success": True,
        "id": selected["id"],
        "name": selected["name"],
        "current": selected["id"],
        "models": refreshed,
    }


def _trim_msst_stems() -> None:
    max_entries = max(1, int(CONFIG["msst"].get("stem_cache_max_files", 20)))
    entries = sorted(
        (
            path
            for path in MSST_STEM_DIR.iterdir()
            if path.is_dir() and not path.name.startswith(".work_")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in entries[max_entries:]:
        shutil.rmtree(stale, ignore_errors=True)


def _separate_target_with_msst(
    target_path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[Path, Path, dict[str, Any], bool]:
    # clear_cache("results"/"all") may have removed this nested cache directory.
    # convert() and clear_cache() share GPU_JOB_LOCK, so recreating it here is
    # sufficient and also repairs directories removed manually while idle.
    MSST_STEM_DIR.mkdir(parents=True, exist_ok=True)
    info = _msst_description()
    payload = {
        "target_sha256": _sha256_file(target_path),
        "msst_digest": info["digest"],
        "model_id": info["model_id"],
        "settings": info["settings"],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stem_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    stem_dir = MSST_STEM_DIR / stem_key
    vocal_path = stem_dir / "vocals.wav"
    instrumental_path = stem_dir / "other.wav"
    if (
        vocal_path.is_file()
        and instrumental_path.is_file()
        and vocal_path.stat().st_size > 0
        and instrumental_path.stat().st_size > 0
        and _looks_like_audio_payload(vocal_path)
        and _looks_like_audio_payload(instrumental_path)
    ):
        os.utime(stem_dir, None)
        print(f"[MSST缓存命中] {stem_key[:12]} | model={info['model_id']}", flush=True)
        if progress_callback is not None:
            progress_callback(1.0, "MSST 分离缓存命中，人声与伴奏已就绪")
        return vocal_path.resolve(), instrumental_path.resolve(), info, True

    work_dir = Path(tempfile.mkdtemp(prefix=".work_", dir=MSST_STEM_DIR))
    try:
        if progress_callback is not None:
            progress_callback(0.02, f"MSST 正在加载 {info['model_id']} 并分离目标歌曲")
        print(
            "[MSST分离Debug] "
            f"model={info['model_id']} target={target_path.name} settings={info['settings']}",
            flush=True,
        )
        vocal_raw, instrumental_raw, info = separate_target(
            target_path,
            work_dir,
            BASE_DIR,
            dict(CONFIG.get("msst") or {}),
            progress_callback=progress_callback,
        )
        if not _looks_like_audio_payload(vocal_raw) or not _looks_like_audio_payload(instrumental_raw):
            raise MSSTBridgeError("MSST 输出不是有效音频")

        shutil.rmtree(stem_dir, ignore_errors=True)
        stem_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vocal_raw, vocal_path)
        shutil.copy2(instrumental_raw, instrumental_path)
        os.utime(stem_dir, None)
        _trim_msst_stems()
        print(
            f"[MSST分离完成Debug] vocals={vocal_path.name} other={instrumental_path.name}",
            flush=True,
        )
        if progress_callback is not None:
            progress_callback(1.0, "MSST 人声与伴奏分离完成")
        return vocal_path.resolve(), instrumental_path.resolve(), info, False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _export_mp3(source: Path, destination: Path) -> Path:
    """Export a QQ-friendly high-quality MP3 atomically."""
    ffmpeg = _ffmpeg_executable()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.mp3")
    temporary.unlink(missing_ok=True)
    command = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "320k",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "unknown FFmpeg error").strip()
        raise RuntimeError(f"导出 MP3 失败: {detail}")
    temporary.replace(destination)
    return destination


def _trim_cache() -> None:
    max_files = max(1, int(CONFIG["cache"]["max_files"]))
    files = sorted(
        (
            path
            for path in CACHE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp3", ".wav"}
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in files[max_files:]:
        stale.unlink(missing_ok=True)


def _trim_outputs() -> None:
    """Limit non-cached random outputs so random mode cannot grow forever."""
    max_files = max(1, int(CONFIG["cache"]["max_files"]))
    files = sorted(
        (
            path
            for path in OUTPUT_DIR.iterdir()
            if path.is_file()
            and path.name.startswith("svcvc_")
            and path.suffix.lower() in {".mp3", ".wav"}
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in files[max_files:]:
        stale.unlink(missing_ok=True)
