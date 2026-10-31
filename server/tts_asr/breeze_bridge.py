# -*- coding: utf-8 -*-
"""
Breeze TTS 2 桥接模块（audio-station 统一网关组件）
==================================================
对外: 提供 OpenAI /v1/audio/speech 兼容的处理函数, 由 gateway.py 按 model 分流挂载。
对内: 管理 Breeze 推理子进程(7860) 的启动/就绪等待/退出清理。

解决 Breeze 原生接口的三个问题:
    1. 只返回裸 PCM(24k s16le) -> lameenc/ffmpeg 转码成 mp3/wav/pcm
    2. 没有 voice 参数        -> voice 映射到 persona(ref_audio+ref_text 克隆 / instruction 设计)
    3. 单并发(忙时 409)       -> 异步锁排队 + 409 重试

环境变量:
    BREEZE_URL    Breeze 服务地址 (默认 http://127.0.0.1:7860)
    BREEZE_PORT   Breeze 子进程端口 (默认 7860)
"""
import asyncio
import atexit
import json
import os
import shutil
import struct
import subprocess
import sys

import httpx
from fastapi import Request, Response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BREEZE_DIR = os.path.join(BASE_DIR, "breeze")

BREEZE_URL = os.environ.get("BREEZE_URL", "http://127.0.0.1:7860").rstrip("/")
BREEZE_PORT = int(os.environ.get("BREEZE_PORT", "7860"))
VOICES_DIR = os.environ.get("VOICES_DIR", os.path.join(BASE_DIR, "voices"))
CONFIG_PATH = os.environ.get("CONFIG_JSON", os.path.join(BREEZE_DIR, "breeze_personas.json"))

# embedded python 忽略 PYTHONPATH, 需显式把隔离依赖目录(lameenc 等)加进 sys.path
if os.path.isdir(os.path.join(BREEZE_DIR, "pylibs")):
    sys.path.insert(0, os.path.join(BREEZE_DIR, "pylibs"))

DEFAULT_INSTRUCTION = "一位温柔清晰、自然的年轻女性，语气亲切。"
SAMPLE_RATE = 24000
START_TIMEOUT = 300  # Breeze 子进程就绪等待(秒), 首次加载模型较慢

_proc: subprocess.Popen | None = None
_ready = False
_BREEZE_LOG = os.path.join(BASE_DIR, "logs", "breeze_tts.log")
_log_fh = None


def _log(*a):
    print("[BreezeTTS]", *a, flush=True)


# ---------- 子进程管理 ----------
def ensure_breeze_started() -> None:
    """启动 Breeze 推理子进程(若未运行)。"""
    global _proc, _log_fh
    # 已有别的 Breeze 在监听 -> 不重复拉起
    try:
        if httpx.get(f"{BREEZE_URL}/docs", timeout=2.0).status_code < 500:
            _log("检测到 Breeze 已在运行:", BREEZE_URL)
            return
    except Exception:
        pass

    py = os.path.join(BASE_DIR, "runtime", "python.exe")
    model = os.path.join(BREEZE_DIR, "models", "breeze-tts-2")
    launcher = os.path.join(BREEZE_DIR, "run_breeze_api.py")
    if not (os.path.isfile(py) and os.path.isfile(launcher)):
        _log("!! 未找到 runtime/python.exe 或 run_breeze_api.py, TTS 将不可用")
        return

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_OFFLINE"] = "1"
    # 子进程输出落文件: 崩溃(尤其 CUDA OOM)时才有据可查, 否则随网关控制台一起丢失
    try:
        os.makedirs(os.path.dirname(_BREEZE_LOG), exist_ok=True)
        if _log_fh is None:
            _log_fh = open(_BREEZE_LOG, "ab", buffering=0)
        out = _log_fh
    except Exception as e:
        _log("子进程日志文件不可用, 输出并入网关控制台:", e)
        out = None
    _proc = subprocess.Popen(
        [py, launcher, model, "--host", "127.0.0.1", "--port", str(BREEZE_PORT)],
        cwd=BREEZE_DIR, env=env,
        stdout=out, stderr=subprocess.STDOUT if out is not None else None,
    )
    atexit.register(_shutdown)
    _log(f"Breeze 子进程已启动 pid={_proc.pid}, 模型加载中(最长 {START_TIMEOUT}s) ... 日志: {_BREEZE_LOG}")


def _shutdown():
    global _proc, _log_fh
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.terminate()
            _proc.wait(timeout=10)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
    _proc = None
    if _log_fh is not None:
        try:
            _log_fh.close()
        except Exception:
            pass
        _log_fh = None


async def wait_breeze_ready(timeout: float = START_TIMEOUT) -> bool:
    """轮询 /docs 直到 Breeze 就绪(FastAPI 默认提供 /docs)。"""
    global _ready
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=3.0) as c:
        while asyncio.get_event_loop().time() < deadline:
            if _proc is not None and _proc.poll() is not None:
                _ready = False
                raise RuntimeError(f"Breeze 子进程异常退出 code={_proc.returncode}")
            try:
                r = await c.get(f"{BREEZE_URL}/docs")
                if r.status_code == 200:
                    _ready = True
                    return True
            except Exception:
                pass
            await asyncio.sleep(2.0)
    _ready = False
    return False


async def _health(timeout: float = 2.0) -> bool:
    """轻量探活: /docs 可访问即认为进程还在跑。"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            return (await c.get(f"{BREEZE_URL}/docs")).status_code == 200
    except Exception:
        return False


async def ensure_alive() -> bool:
    """确保 Breeze 可服务: 进程退出/端口不可达时自动重启并等待就绪。

    只靠模块级 _ready 标记是不够的 —— 子进程可能在任意时刻崩溃(CUDA OOM 等),
    标记不会自动复位, 之后每个请求都会打到死端口并返回 500。
    """
    global _ready
    if _proc is not None and _proc.poll() is not None:
        _log(f"!! Breeze 子进程已退出 code={_proc.returncode}, 准备重启")
        _ready = False
    elif await _health():
        _ready = True
        return True
    else:
        _ready = False
        _log("Breeze 端口不可达, 准备重启子进程 ...")

    _shutdown()
    ensure_breeze_started()
    try:
        return await wait_breeze_ready()
    except RuntimeError as e:
        _log("Breeze 重启失败:", e)
        return False


# ---------- persona 配置 ----------
def _load_personas() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _log("读取 persona 配置失败, 用默认:", e)
        return {}


PERSONAS = _load_personas()


def _resolve_persona(voice: str) -> dict:
    """把 AstrBot 传来的 voice 名解析成 persona 配置。"""
    if not voice:
        return {}
    voice = str(voice).strip().lower()
    if voice in PERSONAS:
        return PERSONAS[voice]
    stem = os.path.splitext(os.path.basename(voice))[0].lower()
    if stem in PERSONAS:
        return PERSONAS[stem]
    return {}


def _ref_text_for(wav_path: str) -> str:
    """优先读 persona 里的 ref_text, 否则读同名的 .txt 文字稿(tools/persona_reftext.py 生成)。"""
    p = _resolve_persona(os.path.splitext(os.path.basename(wav_path))[0])
    if p.get("ref_text"):
        return p["ref_text"]
    txt = os.path.splitext(wav_path)[0] + ".txt"
    if os.path.isfile(txt):
        try:
            return open(txt, "r", encoding="utf-8").read().strip()
        except Exception:
            pass
    return ""


# ---------- 转码 ----------
def _pcm_to_wav(pcm: bytes, sr: int = SAMPLE_RATE) -> bytes:
    hdr = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
        + b"data" + struct.pack("<I", len(pcm))
    )
    return hdr + pcm


def _transcode(pcm: bytes, fmt: str):
    """把 24k s16le PCM 转成目标格式, 返回 (media_type, bytes)。"""
    fmt = (fmt or "mp3").lower()
    if fmt == "pcm":
        return "audio/pcm", pcm
    if fmt == "wav":
        return "audio/wav", _pcm_to_wav(pcm)
    if shutil.which("ffmpeg"):
        acodec = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac", "flac": "flac"}.get(fmt, "libmp3lame")
        ext = fmt if fmt in ("mp3", "opus", "aac", "flac") else "mp3"
        try:
            p = subprocess.run(
                ["ffmpeg", "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
                 "-i", "pipe:0", "-f", ext, "-acodec", acodec, "-b:a", "128k", "pipe:1"],
                input=pcm, capture_output=True,
            )
            if p.returncode == 0 and p.stdout:
                mt = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac"}.get(fmt, "audio/mpeg")
                return mt, p.stdout
        except Exception as e:
            _log("ffmpeg 转码失败:", e)
    if fmt == "mp3":
        # 无 ffmpeg 时用 lameenc(纯 python 绑定)
        try:
            import lameenc

            enc = lameenc.Encoder()
            enc.set_bit_rate(128)
            enc.set_in_sample_rate(SAMPLE_RATE)
            enc.set_channels(1)
            enc.set_quality(2)
            data = enc.encode(pcm) + enc.flush()
            if data:
                return "audio/mpeg", bytes(data)
        except Exception as e:
            _log("lameenc 转 mp3 失败, 回退 wav:", e)
    return "audio/wav", _pcm_to_wav(pcm)


# ---------- 调用 Breeze ----------
_lock = asyncio.Lock()


async def _call_breeze(text: str, wav_path: str = "", ref_text: str = "", instruction: str = "", seed: int = 42) -> bytes:
    use_clone = bool(wav_path and ref_text)
    data = {"text": text, "seed": seed}
    if use_clone:
        data["ref_text"] = ref_text
        data["cfg_scale"] = 1.0
    else:
        data["instruction"] = instruction or DEFAULT_INSTRUCTION
        data["cfg_scale"] = 4.0

    async with httpx.AsyncClient(timeout=300.0) as c:
        for attempt in range(4):
            files = None
            fh = None
            try:
                if use_clone:
                    fh = open(wav_path, "rb")  # 每次重试重新打开
                    files = {"ref_audio": fh}
                try:
                    r = await c.post(f"{BREEZE_URL}/v1/audio/speech", data=data, files=files)
                except httpx.TransportError as e:
                    # 连接被拒/中断: 多半是子进程崩了, 先尝试拉起来再重试
                    _log(f"!! 请求 Breeze 失败({type(e).__name__}), 尝试恢复子进程 (第 {attempt + 1} 次)")
                    if attempt < 3 and await ensure_alive():
                        continue
                    raise RuntimeError(f"无法连接 Breeze 服务({type(e).__name__}): {e}") from e
            finally:
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
            if r.status_code == 409:
                wait = 2 + attempt * 2
                _log("Breeze 忙(409), 等待", wait, "秒后重试")
                await asyncio.sleep(wait)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Breeze 返回 {r.status_code}: {r.text[:300]}")
            return r.content  # 裸 PCM s16le 24k
    raise RuntimeError("Breeze 多次 409, 放弃")


# ---------- OpenAI 兼容端点(由 gateway 挂载) ----------
async def breeze_tts_endpoint(request: Request) -> Response:
    """处理 /v1/audio/speech 中 model=breeze-tts-2 的请求。"""
    # 每次请求都确认子进程还活着(本地 /docs 探活开销极小), 死了就自动拉起
    if not await ensure_alive():
        return Response("Breeze TTS 服务未就绪(子进程启动失败, 详见 logs/breeze_tts.log)",
                        status_code=503)
    try:
        j = await request.json()
    except Exception:
        j = {}
    text = str(j.get("input") or j.get("text") or "").strip()
    voice = j.get("voice", "")
    fmt = j.get("response_format", "mp3")
    if not text:
        return Response("missing input", status_code=400)

    persona = _resolve_persona(voice)
    wav_path = persona.get("wav", "")
    if wav_path and not os.path.isabs(wav_path):
        wav_path = os.path.join(VOICES_DIR, wav_path)
    ref_text = _ref_text_for(wav_path) if wav_path else ""
    instruction = persona.get("instruction", "")

    mode = "克隆" if (wav_path and ref_text) else "设计"
    _log(f">>> [TTS] 文本: {text[:60]}{'...' if len(text) > 60 else ''} | voice={voice or '(默认)'} | 模式={mode}")

    async with _lock:
        try:
            pcm = await _call_breeze(text, wav_path=wav_path, ref_text=ref_text, instruction=instruction)
        except RuntimeError as e:
            _log("!! 合成失败:", e)
            return Response(f"Breeze 合成失败: {e}", status_code=503)

    media_type, audio = _transcode(pcm, fmt)
    return Response(content=audio, media_type=media_type)
