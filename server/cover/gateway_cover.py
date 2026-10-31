# -*- coding: utf-8 -*-
"""Seed-VC V1 翻唱网关。

对外提供两套协议：
1. Gradio 兼容接口（AstrBot 插件 astrbot_plugin_ace_singing 专用，零改动切换）：
   - /convert            15 参数，与旧 SVCVC-API 完全一致
   - /fetch_source_gate  B站/网易云 备用音源
   - show_model / list_voice_profiles / health / cache_info / clear_cache ...
2. OpenAI TTS 协议：
   - GET  /v1/models          音色列表
   - POST /v1/audio/speech    翻唱（model="seed-svc", input="源音频|半音", voice=音色ID）

内部管线：目标歌曲 -> MSST(BS-Roformer) 人声/伴奏分离 -> Seed-VC 44.1kHz 唱歌换声
        -> (可选)自动变调 -> 与伴奏混音 -> 320kbps MP3。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import soundfile as sf

# 管线模块（音源解析/下载/MSST 分离/缓存/导出）。
# cover_core.py 与本文件同目录，其 BASE_DIR 等路径常量自动指向本项目。
import cover_core as legacy

BASE_DIR = Path(__file__).resolve().parent
APP_NAME = "Seed-VC Singing Gateway"
APP_VERSION = "2.0.0"
PIPELINE_VERSION = "seedvc-44k-v1-msst"

# ffmpeg 指向本项目自带 bin/ffmpeg.exe
legacy.FFMPEG_PATH = BASE_DIR / "bin" / "ffmpeg.exe"
FFMPEG = legacy.FFMPEG_PATH
VOICE_DIR = legacy.VOICE_DIR
CACHE_DIR = legacy.CACHE_DIR
OUTPUT_DIR = legacy.OUTPUT_DIR

CONFIG = legacy.CONFIG
MIX_VOCAL_GAIN = float(CONFIG.get("mix", {}).get("vocal_gain", 1.15))
MIX_INSTR_GAIN = float(CONFIG.get("mix", {}).get("instr_gain", 0.85))
SMART_F0_DEFAULT = bool(CONFIG.get("seedvc", {}).get("smart_f0", True))

for _d in (CACHE_DIR, OUTPUT_DIR, legacy.DOWNLOAD_DIR, legacy.MSST_STEM_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ================= 引擎懒加载（后台线程预热，转换时等待就绪） =================
_ENGINE = None
_ENGINE_READY = threading.Event()
_ENGINE_ERROR: str = ""
ENGINE_LOAD_TIMEOUT = 900  # 秒


def _load_engine_worker():
    global _ENGINE, _ENGINE_ERROR
    try:
        from seed_vc_engine import SeedVCEngine

        t0 = time.monotonic()
        fp16 = bool(CONFIG.get("seedvc", {}).get("fp16", True))
        _ENGINE = SeedVCEngine(fp16=fp16)
        print(f"[SeedVC] engine ready in {time.monotonic() - t0:.1f}s", flush=True)
        _ENGINE_READY.set()
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        _ENGINE_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[SeedVC] 引擎加载失败: {_ENGINE_ERROR}", flush=True)


def _wait_engine() -> Any:
    if _ENGINE_READY.is_set():
        return _ENGINE
    if _ENGINE_ERROR:
        raise RuntimeError(f"Seed-VC 引擎加载失败: {_ENGINE_ERROR}")
    print("[SeedVC] 引擎仍在加载，等待中 ...", flush=True)
    if not _ENGINE_READY.wait(timeout=ENGINE_LOAD_TIMEOUT):
        raise RuntimeError(f"Seed-VC 引擎加载超时（>{ENGINE_LOAD_TIMEOUT}s）" + (f": {_ENGINE_ERROR}" if _ENGINE_ERROR else ""))
    return _ENGINE


threading.Thread(target=_load_engine_worker, name="seedvc-loader", daemon=True).start()


# ================= 工具 =================
def _validate_cfg_rate(cfg: float) -> float:
    """校验 Seed-VC inference_cfg_rate（0.05~1.0）。

    Seed-VC CFM 引导公式 dphi = (1+cfg)*cond - cfg*uncond：cfg 越大越往参考音色外推，
    音色遵循度越高。0.95~1.0 为安全上限（1.0 即完全跟随参考），超过 1.0 进入外推区易出 artifacts。
    """
    cfg = float(cfg)
    if not 0.05 <= cfg <= 1.0:
        raise ValueError(f"cfg 必须在 0.05 到 1.0 之间（Seed-VC inference_cfg_rate），收到 {cfg}")
    return cfg


def _separate_reference_if_needed(
    prompt_path: Path, prompt_vocal_sep: bool,
    progress_cb: Callable[[float, str], None] | None = None,
) -> Path:
    """参考音色先做一次人声分离再提音色，避免伴奏被当成音色特征。"""
    if not prompt_vocal_sep:
        return prompt_path
    vocal, _inst, _info, _hit = legacy._separate_target_with_msst(prompt_path, progress_cb)
    return vocal


def _mix_vocal_instrumental(vocal: Path, instrumental: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.stem + ".mix.wav")
    tmp.unlink(missing_ok=True)
    graph = (
        f"[0:a]volume={MIX_VOCAL_GAIN}[v];"
        f"[1:a]volume={MIX_INSTR_GAIN}[a];"
        f"[v][a]amix=inputs=2:duration=first:normalize=0[aout];"
        f"[aout]alimiter=limit=0.95:level=false[lout]"
    )
    cmd = [
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(vocal), "-i", str(instrumental),
        "-filter_complex", graph, "-map", "[lout]",
        "-ac", "2", "-ar", "44100",
        str(tmp),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if completed.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"混音失败: {detail}")
    tmp.replace(destination)
    return destination


def _cache_key(target_path: Path, prompt_path: Path, parameters: dict[str, Any]) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "target_sha256": legacy._sha256_file(target_path),
        "prompt_sha256": legacy._sha256_file(prompt_path),
        **parameters,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ================= 核心翻唱（Gradio /convert 与 OpenAI 协议共用） =================
def run_cover_pipeline(
    target_path: Path,
    prompt_path: Path,
    prompt_vocal_sep: bool,
    separation_mode: str,
    auto_shift: bool,
    auto_mix_acc: bool,
    pitch_shift: int,
    n_step: int,
    cfg_rate: float,
    actual_seed: int,
    requested_random: bool,
    smart_f0: bool = True,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[Path, bool]:
    """执行完整翻唱管线，返回 (输出文件路径, 是否命中缓存)。"""
    started = time.monotonic()

    def step(v: float, desc: str):
        print(f"[进度] {int(v * 100)}% {desc}", flush=True)
        if progress_cb:
            progress_cb(v, desc)

    parameters = {
        "prompt_vocal_sep": bool(prompt_vocal_sep),
        "target_separation": separation_mode,
        "auto_shift": bool(auto_shift),
        "auto_mix_acc": bool(auto_mix_acc),
        "pitch_shift": int(pitch_shift),
        "n_step": int(n_step),
        "cfg_rate": round(float(cfg_rate), 3),
        "seed": int(actual_seed),
    }
    if separation_mode == "msst":
        msst_info = legacy._msst_description()
        parameters["msst"] = {
            "digest": msst_info["digest"],
            "model_id": msst_info["model_id"],
            "settings": msst_info["settings"],
        }
    key = _cache_key(target_path, prompt_path, parameters)
    cache_path = CACHE_DIR / f"{key}.mp3"
    persistent_cache = bool(CONFIG["cache"]["enabled"]) and not requested_random

    if persistent_cache and cache_path.is_file() and cache_path.stat().st_size > 0:
        os.utime(cache_path, None)
        print(f"[缓存命中] {cache_path.name}", flush=True)
        step(1.0, "缓存命中")
        return cache_path.resolve(), True

    step(0.08, "等待 GPU 队列 ...")
    with legacy.GPU_JOB_LOCK:
        if persistent_cache and cache_path.is_file() and cache_path.stat().st_size > 0:
            os.utime(cache_path, None)
            step(1.0, "缓存命中")
            return cache_path.resolve(), True

        engine = _wait_engine()

        instrumental_path: Path | None = None
        # 1) 目标歌曲人声/伴奏分离
        if separation_mode == "msst":
            step(0.12, "MSST 正在分离目标歌曲人声与伴奏 ...")
            vocal_path, instrumental_path, _info, _hit = legacy._separate_target_with_msst(
                target_path, lambda v, d: step(0.12 + 0.20 * v, d)
            )
        else:
            vocal_path = target_path

        # 2) 参考音色（可选先分离人声）
        step(0.35, "准备参考音色 ...")
        ref_path = _separate_reference_if_needed(
            prompt_path, prompt_vocal_sep,
            lambda v, d: step(0.35 + 0.10 * v, d),
        )

        # 3) Seed-VC 44.1kHz 唱歌换声（smart_f0 同时是智能音域决策总开关）
        step(0.48, f"Seed-VC 推理中（步数={n_step}, cfg_rate={cfg_rate:.2f}, 变调={pitch_shift}, 智能音域={smart_f0}）...")
        sr, wav = engine.convert(
            source=str(vocal_path),
            target=str(ref_path),
            diffusion_steps=int(n_step),
            inference_cfg_rate=float(cfg_rate),
            auto_f0_adjust=bool(auto_shift),
            pitch_shift=int(pitch_shift),
            seed=int(actual_seed),
            smart_f0=bool(smart_f0),
            auto_tone=bool(smart_f0),
        )

        # 4) 落盘 WAV（与伴奏等长混音）
        step(0.92, "导出音频 ...")
        _fd, vocal_wav_path = tempfile.mkstemp(prefix="svc_vocal_", suffix=".wav", dir=OUTPUT_DIR)
        os.close(_fd)
        vocal_wav = Path(vocal_wav_path)
        sf.write(str(vocal_wav), wav, sr, format="WAV", subtype="FLOAT")

        final_wav: Path
        if auto_mix_acc and instrumental_path is not None:
            final_wav = vocal_wav.with_name(vocal_wav.stem + ".mixed.wav")
            _mix_vocal_instrumental(vocal_wav, instrumental_path, final_wav)
            vocal_wav.unlink(missing_ok=True)
        else:
            final_wav = vocal_wav

        # 5) 导出 320kbps MP3
        if persistent_cache:
            out_path = legacy._export_mp3(final_wav, cache_path)
            legacy._trim_cache()
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"svcvc_{stamp}_{actual_seed}_{key[:10]}.mp3"
            legacy._export_mp3(final_wav, out_path)
            legacy._trim_outputs()
        try:
            final_wav.unlink(missing_ok=True)
        except OSError:
            pass

    elapsed = time.monotonic() - started
    print(f"[OK] 翻唱完成 {elapsed:.1f}s -> {out_path.name}", flush=True)
    step(1.0, "转换完成")
    return out_path.resolve(), False


# ================= Gradio 兼容入口（AstrBot 插件专用） =================
def convert(
    song_name_src: str,
    model_dropdown: str,
    prompt_vocal_sep: bool = False,
    target_vocal_sep: bool = True,
    auto_shift: bool = True,
    auto_mix_acc: bool = True,
    pitch_shift: int = 0,
    n_step: int = 32,
    cfg: float = 0.7,
    seed: int = 42,
    random_seed: bool = False,
    target_upload: str | None = None,
    reference_source: str = "",
    reference_upload: str | None = None,
    target_separation: str = "",
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    """与旧 SVCVC-API /convert 完全同签名，返回 (输出 MP3 路径, 是否命中缓存)。"""
    started = time.monotonic()
    try:
        progress(0.02, desc="正在检查参数与参考音色 ...")
        pitch_shift, n_step, cfg, seed = legacy._validate_parameters(pitch_shift, n_step, cfg, seed)
        separation_mode = legacy._normalize_target_separation(target_separation, target_vocal_sep)
        if separation_mode not in {"msst", "none"}:
            raise gr.Error(f"不支持的目标歌曲人声分离方式: {separation_mode}；只能是 msst 或 none")
        requested_random = bool(random_seed) or seed == -1
        actual_seed = random.SystemRandom().randint(0, 4294967295) if seed == -1 else seed
        cfg_rate = _validate_cfg_rate(cfg)

        def source_progress(value: float, description: str) -> None:
            progress(0.02 + 0.06 * value, desc=description)

        target_path = legacy._resolve_input_audio(song_name_src, target_upload, "目标歌曲", source_progress)
        print(f"\n{'=' * 64}", flush=True)
        print(
            f"[{time.strftime('%X')}] >>> [Seed-VC 翻唱] 歌曲/来源: {song_name_src or '(本地上传)'} | 音色: {model_dropdown} | 文件: {target_path.name}",
            flush=True,
        )
        if reference_upload or str(reference_source or "").strip():
            ref_path = legacy._resolve_input_audio(
                reference_source, reference_upload, "参考音色", source_progress,
            )
            prompt = {
                "profile_id": "external-reference",
                "audio_path": str(ref_path.resolve()),
                "audio_sha256": legacy._sha256_file(ref_path),
            }
        else:
            prompt = legacy._find_profile(str(model_dropdown).strip())
        prompt_path = Path(prompt["audio_path"])

        print(
            "[SVCVC任务] "
            f"音色={prompt['profile_id']} target={target_path.name} "
            f"prompt_sep={bool(prompt_vocal_sep)} separation={separation_mode} "
            f"auto_shift={bool(auto_shift)} auto_mix={bool(auto_mix_acc)} "
            f"pitch={pitch_shift} steps={n_step} cfg={cfg:.2f} "
            f"seed={actual_seed} random={requested_random}",
            flush=True,
        )

        def pipeline_progress(value: float, desc: str) -> None:
            progress(max(0.08, min(0.99, value)), desc=desc)

        final_path, cache_hit = run_cover_pipeline(
            target_path=target_path,
            prompt_path=prompt_path,
            prompt_vocal_sep=bool(prompt_vocal_sep),
            separation_mode=separation_mode,
            auto_shift=bool(auto_shift),
            auto_mix_acc=bool(auto_mix_acc),
            pitch_shift=pitch_shift,
            n_step=n_step,
            cfg_rate=cfg_rate,
            actual_seed=actual_seed,
            requested_random=requested_random,
            smart_f0=SMART_F0_DEFAULT,
            progress_cb=pipeline_progress,
        )
        progress(1.0, desc=f"转换完成，实际种子 {actual_seed}")
        print(f"[OK] 总耗时 {time.monotonic() - started:.1f}s", flush=True)
        return str(final_path), "true" if cache_hit else "false"
    except gr.Error:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] {type(exc).__name__}: {exc}", flush=True)
        raise gr.Error(str(exc)) from exc


# ================= OpenAI TTS 协议入口 =================
from fastapi import FastAPI, Request, Response  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
import uvicorn  # noqa: E402

api_app = FastAPI(title=APP_NAME, version=APP_VERSION)

PROBE_KEYWORDS = {"hello", "test", "ping", "你好", "hi", "testing", "this is a test"}


@api_app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "seed-vc-44k-v1",
        "engine_ready": _ENGINE_READY.is_set(),
        "profiles": legacy.show_model(),
        "msst": legacy._msst_description().get("model_id"),
    }


@api_app.get("/v1/models")
async def openai_models():
    """OpenAI 标准模型列表：voices 字段供 AstrBot 插件拉取音色 ID。"""
    return {
        "object": "list",
        "data": [
            {
                "id": "seed-svc",
                "object": "model",
                "created": 1700000000,
                "owned_by": "seed-vc",
                "permission": [],
                "root": "seed-svc",
                "parent": None,
                "voices": legacy.show_model(),
            }
        ],
    }


def _parse_speech_input(raw: str) -> tuple[str, int, int, float]:
    """解析 input 字段：支持 JSON 或 '路径|半音' 或纯来源。返回 (来源, pitch, steps, cfg)。"""
    raw = str(raw or "").strip()
    pitch, steps, cfg = 0, 32, 0.7
    if raw.startswith("{"):
        data = json.loads(raw)
        src = str(data.get("source", "")).strip()
        pitch = int(data.get("pitch", 0))
        steps = int(data.get("steps", 32))
        cfg = float(data.get("cfg", 0.7))
    elif "|" in raw:
        src, pitch_str = raw.rsplit("|", 1)
        src = src.strip()
        pitch = int(pitch_str.strip())
    else:
        src = raw
    if not src:
        raise ValueError("input 不能为空")
    return src, pitch, steps, cfg


@api_app.post("/v1/audio/speech")
async def openai_audio_speech(request: Request):
    """OpenAI TTS 协议翻唱入口。

    - model: "seed-svc"（忽略大小写，含 svc 即视为翻唱）
    - voice: 音色 ID（voice_profiles 目录下的参考音色）
    - input:  三种格式
        "D:/songs/song.wav|12"                        本地路径 + 升 12 半音
        "https://.../vocal.wav|0"                     URL
        "{\"source\":\"...\",\"pitch\":12,...}"        JSON（pitch/steps/cfg）
        也可以直接给 B站关键词 / BV号 / 网易云链接。
    - response_format: mp3（默认）/ wav
    - acapella: true 时输出清唱（只有换声后人声，不混伴奏）
    """
    try:
        data = await request.json()
    except Exception:
        data = {}
    model = str(data.get("model", "seed-svc"))
    raw_input = str(data.get("input", "")).strip()
    voice = str(data.get("voice", "") or "").strip() or (legacy.show_model() or [""])[0]
    response_format = str(data.get("response_format", "mp3") or "mp3").lower()

    print(
        f"[OpenAI-API] /v1/audio/speech model={model} input='{raw_input[:60]}' voice={voice}",
        flush=True,
    )

    # AstrBot 握手探测：返回参考音色采样
    if not raw_input or raw_input.lower() in PROBE_KEYWORDS or len(raw_input) <= 3:
        test_wav = VOICE_DIR / f"{voice}.wav"
        if not test_wav.is_file():
            for f in sorted(VOICE_DIR.glob("*.wav")):
                test_wav = f
                break
        if test_wav.is_file():
            return FileResponse(path=str(test_wav), media_type="audio/wav")
        return Response(content=b"", media_type="audio/wav")

    if model.lower() not in {"seed-svc", "svc-1"}:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"未知模型 {model}；本服务只提供 seed-svc 翻唱", "type": "invalid_request_error"}},
        )

    try:
        src, pitch, steps, cfg = _parse_speech_input(raw_input)
        # body 顶层的 steps/cfg 可覆盖 "|" 简写模式的默认值
        if data.get("steps") is not None:
            steps = int(data["steps"])
        if data.get("cfg") is not None:
            cfg = float(data["cfg"])
        pitch, steps, cfg, _seed = legacy._validate_parameters(pitch, steps, cfg, -1)
        target_path = legacy._resolve_input_audio(src, None, "目标歌曲", None)
        prompt = legacy._find_profile(voice)
        final_path, _hit = run_cover_pipeline(
            target_path=target_path,
            prompt_path=Path(prompt["audio_path"]),
            prompt_vocal_sep=True,
            separation_mode="msst",
            auto_shift=True,
            auto_mix_acc=not bool(data.get("acapella", False)),
            pitch_shift=pitch,
            n_step=steps,
            cfg_rate=_validate_cfg_rate(cfg),
            actual_seed=random.SystemRandom().randint(0, 4294967295),
            requested_random=True,  # OpenAI 协议调用不走持久缓存
            smart_f0=bool(data.get("smart_f0", SMART_F0_DEFAULT)),
            progress_cb=None,
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error"}})
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "server_error"}})

    if response_format == "wav":
        wav_path = final_path.with_suffix(".wav")
        subprocess.run(
            [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(final_path), "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True, text=True, timeout=300,
        )
        if wav_path.is_file():
            return FileResponse(path=str(wav_path), media_type="audio/wav")
    return FileResponse(path=str(final_path), media_type="audio/mpeg")


# ================= Gradio UI（接口兼容层） =================
def _gateway_health() -> dict[str, Any]:
    """Gradio health 端点：仅检查本网关自身，不探测任何外部服务。"""
    return {
        "status": "ok",
        "engine": "seed-vc-44k-v1",
        "engine_ready": _ENGINE_READY.is_set(),
        "profiles": legacy.show_model(),
        "msst": legacy._msst_description().get("model_id"),
        "cache": legacy.cache_info(),
    }


def build_app() -> gr.Blocks:
    profiles = legacy._scan_voice_profiles()
    choices = [item["profile_id"] for item in profiles]
    default_profile = choices[0] if choices else None
    with gr.Blocks(title="Seed-VC Singing Gateway") as app:
        gr.Markdown(
            f"# {APP_NAME}\n"
            "Seed-VC V1 (44.1kHz) 翻唱网关。MSST 分离 + Seed-VC 换声 + 伴奏混音，单进程完成。"
        )
        with gr.Tabs():
            with gr.Tab("音色转换"):
                with gr.Row():
                    target_source = gr.Textbox(label="目标歌曲来源（B站关键词/BV/链接、网易云、HTTP 或本地路径）")
                    target_upload = gr.Audio(label="目标歌曲本地上传（优先）", type="filepath")
                with gr.Row():
                    reference_source = gr.Textbox(
                        value="",
                        label="参考音色来源（B站关键词/BV/链接、网易云、HTTP 或本地路径）",
                    )
                    reference_upload = gr.Audio(label="参考音色本地上传（优先）", type="filepath")
                    model_dropdown = gr.Dropdown(
                        choices=choices,
                        value=default_profile,
                        label="已保存参考音色（未填写参考来源时使用）",
                        allow_custom_value=True,
                    )
                    refresh_profiles = gr.Button("刷新参考音色", size="sm")
                with gr.Row():
                    prompt_vocal_sep = gr.Checkbox(False, label="参考人声分离")
                    target_separation = gr.Dropdown(
                        choices=[
                            ("MSST 人声分离（BS-Roformer）", "msst"),
                            ("不分离（输入已是纯人声）", "none"),
                        ],
                        value="msst",
                        label="目标歌曲人声分离方式",
                    )
                    auto_shift = gr.Checkbox(True, label="自动变调")
                    auto_mix_acc = gr.Checkbox(True, label="自动混合伴奏")
                target_vocal_sep = gr.Checkbox(True, visible=False)
                with gr.Row():
                    pitch_shift = gr.Slider(-36, 36, value=0, step=1, label="指定变调（半音）")
                    n_step = gr.Slider(1, 200, value=50, step=1, label="扩散步数")
                    cfg = gr.Slider(0.05, 1.0, value=0.95, step=0.05, label="CFG 系数（Seed-VC inference_cfg_rate，越大越贴参考音色）")
                    seed = gr.Slider(-1, 4294967295, value=42, step=1, label="种子（-1 为随机）")
                random_seed = gr.Checkbox(False, label="随机任务（不读取/写入持久缓存）")
                run_button = gr.Button("开始翻唱（Seed-VC）", variant="primary")
                output_audio = gr.Audio(label="转换结果", type="filepath")
                cache_hit = gr.Textbox(label="Cache Hit", interactive=False)
                run_button.click(
                    fn=convert,
                    inputs=[
                        target_source, model_dropdown, prompt_vocal_sep, target_vocal_sep,
                        auto_shift, auto_mix_acc, pitch_shift, n_step, cfg, seed, random_seed,
                        target_upload, reference_source, reference_upload, target_separation,
                    ],
                    outputs=[output_audio, cache_hit],
                    api_name="convert",
                    concurrency_limit=1,
                )

                def refresh_profile_choices():
                    refreshed = legacy.show_model()
                    return gr.Dropdown(
                        choices=refreshed,
                        value=refreshed[0] if refreshed else None,
                        allow_custom_value=True,
                    )

                refresh_profiles.click(refresh_profile_choices, outputs=model_dropdown, api_name="refresh_profiles")

            with gr.Tab("歌曲搜索"):
                gr.Markdown("搜索 B站音乐区或网易云歌曲；复制链接后可粘贴到目标歌曲或参考音色来源。")
                with gr.Row():
                    search_platform = gr.Radio(choices=["B站", "网易云"], value="B站", label="平台")
                    search_keyword = gr.Textbox(label="关键词", placeholder="例如：七里香 周杰伦", scale=3)
                    search_button = gr.Button("搜索", variant="primary")
                with gr.Row():
                    search_state = gr.State([])
                    search_cards = gr.HTML('<div style="padding: 1rem; color: #666;">请输入关键词后搜索</div>')
                with gr.Row():
                    previous_page = gr.Button("上一页", interactive=False, scale=1)
                    search_page = gr.Number(value=1, minimum=1, precision=0, label="页码", interactive=False, scale=1)
                    next_page = gr.Button("下一页", scale=1)
                with gr.Row():
                    search_choice = gr.Dropdown(choices=[], label="选择结果", interactive=True, scale=3)
                    search_link = gr.Textbox(label="歌曲链接", interactive=False, scale=4)
                search_cover = gr.HTML('<div style="padding: 1rem; color: #666;">选择结果后显示封面</div>')
                search_button.click(
                    fn=legacy._reset_catalog_page,
                    outputs=[search_page, previous_page],
                    api_name=False,
                ).then(
                    fn=legacy.search_song_catalog_ui,
                    inputs=[search_platform, search_keyword, search_page],
                    outputs=[search_state, search_cards, search_choice, search_cover, search_link, search_page, previous_page],
                    api_name="search_song_catalog",
                )
                previous_page.click(
                    fn=lambda page: legacy._change_catalog_page(page, -1),
                    inputs=search_page,
                    outputs=search_page,
                ).then(
                    fn=legacy.search_song_catalog_ui,
                    inputs=[search_platform, search_keyword, search_page],
                    outputs=[search_state, search_cards, search_choice, search_cover, search_link, search_page, previous_page],
                )
                next_page.click(
                    fn=lambda page: legacy._change_catalog_page(page, 1),
                    inputs=search_page,
                    outputs=search_page,
                ).then(
                    fn=legacy.search_song_catalog_ui,
                    inputs=[search_platform, search_keyword, search_page],
                    outputs=[search_state, search_cards, search_choice, search_cover, search_link, search_page, previous_page],
                )
                search_choice.change(
                    fn=legacy._selected_catalog_result,
                    inputs=[search_choice, search_state],
                    outputs=[search_cover, search_link],
                )

        # ---- 隐藏 API 兼容端点（与旧网关一致） ----
        api_models = gr.JSON(visible=False)
        api_profiles = gr.JSON(visible=False)
        api_health = gr.JSON(visible=False)
        api_cache = gr.JSON(visible=False)
        api_msst_models = gr.JSON(visible=False)
        api_msst_model_name = gr.Textbox(visible=False)
        api_msst_selection = gr.JSON(visible=False)
        api_cache_scope = gr.Textbox(value="all", visible=False)
        app.load(fn=legacy.show_model, inputs=[], outputs=api_models, api_name="show_model")
        app.load(fn=legacy.list_voice_profiles, inputs=[], outputs=api_profiles, api_name="list_voice_profiles")
        app.load(fn=_gateway_health, inputs=[], outputs=api_health, api_name="health")
        app.load(fn=legacy.cache_info, inputs=[], outputs=api_cache, api_name="cache_info")
        app.load(fn=legacy.show_msst_models, inputs=[], outputs=api_msst_models, api_name="show_msst_models")
        gr.Button("API Select MSST Model", visible=False).click(
            fn=legacy.select_msst_model,
            inputs=[api_msst_model_name],
            outputs=[api_msst_selection],
            api_name="select_msst_model",
        )
        gr.Button("API Clear Cache", visible=False).click(
            fn=legacy.clear_cache,
            inputs=[api_cache_scope],
            outputs=[api_cache],
            api_name="clear_cache",
        )
        api_fetch_platform = gr.Textbox(visible=False)
        api_fetch_keyword = gr.Textbox(visible=False)
        api_fetch_index = gr.Number(value=0, precision=0, visible=False)
        api_fetch_out = gr.File(label="backup source audio", visible=False)
        gr.Button("API Fetch Backup Source", visible=False).click(
            fn=legacy.fetch_source_gate_api,
            inputs=[api_fetch_platform, api_fetch_keyword, api_fetch_index],
            outputs=[api_fetch_out],
            api_name="fetch_source_gate",
        )
    return app


gr_blocks = build_app()
queue_max_size = int(CONFIG["server"]["queue_max_size"])
gr_blocks.queue(api_open=True, max_size=queue_max_size, default_concurrency_limit=1)

APP = gr.mount_gradio_app(
    api_app,
    gr_blocks,
    path="/",
    allowed_paths=[str(CACHE_DIR), str(OUTPUT_DIR)],
)

if __name__ == "__main__":
    host = str(CONFIG["server"]["host"])
    port = int(CONFIG["server"]["port"])
    print(f"[INFO] {APP_NAME} {APP_VERSION} -> http://{host}:{port}", flush=True)
    print(f"[INFO] OpenAI 协议: http://{host}:{port}/v1/models , /v1/audio/speech", flush=True)
    print(f"[INFO] Gradio 兼容接口: /convert , /fetch_source_gate（AstrBot 插件用）", flush=True)
    print(f"[INFO] 已加载 {len(legacy._scan_voice_profiles())} 个参考音色；引擎后台加载中 ...", flush=True)
    uvicorn.run(APP, host=host, port=port, log_level="warning")
