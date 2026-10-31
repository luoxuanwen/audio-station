# -*- coding: utf-8 -*-
"""
audio-station 统一网关
======================
单进程单端口, 聚合全部音频能力:
    - 翻唱  : Seed-VC 44.1kHz + MSST 人声分离(进程内, Gradio /convert + OpenAI 协议)
    - TTS   : Breeze TTS 2 (子进程 7860, OpenAI /v1/audio/speech, voice=音色克隆)
    - ASR   : SenseVoiceSmall (进程内, OpenAI /v1/audio/transcriptions)

路由分发: /v1/audio/speech 按 model 字段分流
    model 含 "svc"  -> Seed-VC 翻唱
    其他/缺省       -> Breeze TTS
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "breeze", "pylibs"))
sys.path.insert(0, os.path.join(BASE, "server", "cover"))
sys.path.insert(0, os.path.join(BASE, "server", "tts_asr"))

import uvicorn  # noqa: E402
from starlette.routing import Mount  # noqa: E402

import gateway_cover  # noqa: E402  # 构建翻唱 APP(FastAPI+Gradio), Seed-VC 引擎后台预热
import breeze_bridge  # noqa: E402
import asr_sensevoice  # noqa: E402
from fastapi import Request, Response  # noqa: E402

APP = gateway_cover.APP

# ---- 1) 摘掉翻唱自己的 /v1/audio/speech, 换成统一分流端点 ----
cover_route = next(
    r for r in list(APP.routes)
    if getattr(r, "path", "") == "/v1/audio/speech" and "POST" in getattr(r, "methods", set())
)
APP.routes.remove(cover_route)
_cover_endpoint = cover_route.endpoint


@APP.post("/v1/audio/speech")
async def speech_dispatch(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    model = str(body.get("model", "")).lower()
    if "svc" in model:
        # Request body 已被 Starlette 缓存, 翻唱端点可重复读取
        return await _cover_endpoint(request)
    return await breeze_bridge.breeze_tts_endpoint(request)


# ---- 2) 挂载 ASR ----
APP.include_router(asr_sensevoice.router)

# ---- 3) 路由顺序修正: Gradio mount("/") 在表尾会拦截之后注册的 API 路由,
#         统一把 Mount 挪到最后, 保证 /v1/* 全部优先命中 FastAPI 路由 ----
for _m in [r for r in list(APP.routes) if isinstance(r, Mount)]:
    APP.routes.remove(_m)
    APP.routes.append(_m)

# ---- 4) 启动 Breeze 推理子进程(模型后台加载, 首次 TTS 请求会等待就绪) ----
breeze_bridge.ensure_breeze_started()

if __name__ == "__main__":
    cfg = gateway_cover.CONFIG.get("server", {})
    host = str(cfg.get("host", "0.0.0.0"))
    port = int(os.environ.get("GATEWAY_PORT", cfg.get("port", 18000)))
    print("=" * 60, flush=True)
    print("audio-station 统一网关", flush=True)
    print(f"  监听: http://{host}:{port}", flush=True)
    print("  TTS  : POST /v1/audio/speech  (model=breeze-tts-2)", flush=True)
    print("  ASR  : POST /v1/audio/transcriptions (SenseVoice)", flush=True)
    print("  翻唱 : POST /v1/audio/speech  (model=seed-svc) + Gradio /convert", flush=True)
    print("  关闭本窗口 = 停止全部服务(含 Breeze 子进程)", flush=True)
    print("=" * 60, flush=True)
    uvicorn.run(APP, host=host, port=port, log_level="warning")
