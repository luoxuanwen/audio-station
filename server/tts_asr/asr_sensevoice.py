# -*- coding: utf-8 -*-
"""
SenseVoice ASR 模块（audio-station 统一网关组件）
================================================
- OpenAI Whisper API 兼容: POST /v1/audio/transcriptions (multipart file)
- 模型: iic/SenseVoiceSmall (ModelScope, ~500MB, 中文质量优秀/带标点)
- 进程内懒加载, 首次请求时初始化
"""
import asyncio
import os
import tempfile
import threading

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_ID = os.environ.get("ASR_MODEL_ID", "iic/SenseVoiceSmall")
DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")

_model = None
_model_error = ""
_loading = threading.Event()


def _log(*a):
    print("[ASR]", *a, flush=True)


def _load_worker():
    global _model, _model_error
    try:
        from funasr import AutoModel

        t0 = asyncio.get_event_loop().time() if False else None
        import time

        t0 = time.monotonic()
        _model = AutoModel(
            model=MODEL_ID,
            device=DEVICE,
            disable_update=True,
            disable_pbar=True,
        )
        _log(f"SenseVoice 就绪 ({time.monotonic() - t0:.1f}s, device={DEVICE})")
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        _model_error = f"{type(exc).__name__}: {exc}"
        _log("模型加载失败:", _model_error)
    finally:
        _loading.clear()


def _get_model():
    global _model
    if _model is not None:
        return _model
    if _model_error:
        raise RuntimeError(f"ASR 模型加载失败: {_model_error}")
    if not _loading.is_set():
        _loading.set()
        threading.Thread(target=_load_worker, name="asr-loader", daemon=True).start()
    raise RuntimeError("ASR 模型加载中, 请稍后重试(约 30-60s)")


def _postprocess(text: str) -> str:
    """SenseVoice 输出带 <|zh|><|NEUTRAL|> 等标签, 用官方后处理清理。"""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return rich_transcription_postprocess(text)
    except Exception:
        return text


def _load_wav16k(src: str) -> "np.ndarray":
    """任意输入音频 -> 16k 单声道 float32 ndarray。

    不落文件、直接喂 ndarray 给 funasr: funasr 1.3 对文件路径会先试
    torchaudio(其 ffmpeg backend 缺失)再落外部 ffmpeg 子进程, 本机均不可用;
    ndarray 输入可完全绕开文件加载。
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(src, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16000:
        import soxr

        mono = soxr.resample(mono, sr, 16000).astype(np.float32)
    return np.ascontiguousarray(mono)


@router.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str = "", language: str = "auto"):
    """OpenAI Whisper API 兼容的语音识别。"""
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()
        m = _get_model()
        mono16k = await asyncio.to_thread(_load_wav16k, tmp.name)
        res = await asyncio.to_thread(
            m.generate,
            input=mono16k,
            fs=16000,
            cache={},
            language=language or "auto",
            use_itn=True,
            batch_size_s=60,
        )
        text = _postprocess(res[0]["text"]) if res and res[0].get("text") else ""
        _log(f">>> [ASR] {file.filename}: {text[:80]}{'...' if len(text) > 80 else ''}")
        return JSONResponse({"text": text})
    except RuntimeError as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=503)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@router.get("/v1/asr/status")
async def status():
    return {"loaded": _model is not None, "error": _model_error, "model": MODEL_ID}
