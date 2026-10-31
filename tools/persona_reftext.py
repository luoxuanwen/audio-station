#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为参考音色生成「准确文字稿」(ref_text)
=====================================
Breeze 的声音克隆要求 --ref-text 与参考音频的 spoken content 完全对应。
本脚本用本地已有的 Qwen3-ASR (audiocpp, 端口 18080 代理 / 18081 直连) 对
voices/*.wav 做转写, 在同目录写出 <同名>.txt 文字稿。

生成的 .txt 会被 breeze_tts_bridge.py 在「克隆」模式下读取。

用法:
    python persona_reftext.py
可选环境变量:
    ASR_URL   转写服务地址 (默认 http://127.0.0.1:18080/v1/audio/transcriptions)
    VOICES_DIR 参考音色目录 (默认 脚本同级的 ../voices)
"""
import os
import sys

import httpx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.environ.get("VOICES_DIR", os.path.join(SCRIPT_DIR, "..", "voices"))
ASR_URL = os.environ.get("ASR_URL", "http://127.0.0.1:18080/v1/audio/transcriptions")
# audiocpp 的 ASR 模型 id 是 qwen3-asr(不是 whisper-1)
ASR_MODEL = os.environ.get("ASR_MODEL", "qwen3-asr")

SUPPORTED = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def transcribe(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        r = httpx.post(
            ASR_URL,
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            data={"model": ASR_MODEL, "language": "zh"},
            timeout=120.0,
        )
    r.raise_for_status()
    # OpenAI Whisper 返回 {"text": "..."}
    try:
        return r.json().get("text", "").strip()
    except Exception:
        return r.text.strip()


def ensure_loaded():
    """audiocpp 的 ASR 模型默认 loaded=false, 需先调 /v1/models/load 加载。"""
    base = ASR_URL.split("/v1/")[0]
    try:
        # 注意: audiocpp 的 /v1/models/load 要求的键是 "id"(不是 "model")
        r = httpx.post(f"{base}/v1/models/load", json={"id": ASR_MODEL}, timeout=600.0)
        print(f"加载模型 {ASR_MODEL}: HTTP {r.status_code} {r.text[:150]}")
    except Exception as e:
        print("加载模型请求失败(若已加载可忽略):", e)


def main():
    ensure_loaded()
    if not os.path.isdir(VOICES_DIR):
        print("voices 目录不存在:", VOICES_DIR)
        sys.exit(1)
    wavs = [f for f in os.listdir(VOICES_DIR) if f.lower().endswith(SUPPORTED)]
    if not wavs:
        print("voices 目录下没有音频文件")
        sys.exit(1)
    print(f"ASR 服务: {ASR_URL}")
    print(f"扫描到 {len(wavs)} 个音色文件:\n")
    for fn in sorted(wavs):
        wav_path = os.path.join(VOICES_DIR, fn)
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if os.path.exists(txt_path):
            print(f"  [跳过] {fn} 已有文字稿")
            continue
        try:
            text = transcribe(wav_path)
        except Exception as e:
            print(f"  [失败] {fn}: {e}")
            continue
        if not text:
            print(f"  [空结果] {fn}: ASR 未返回文本, 跳过")
            continue
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  [完成] {fn} -> {os.path.basename(txt_path)}: {text[:50]}{'...' if len(text) > 50 else ''}")


if __name__ == "__main__":
    main()
