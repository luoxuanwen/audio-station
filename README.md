# audio-station

Windows 本机一体化音频服务：**TTS（Breeze TTS 2 声音克隆）+ ASR（SenseVoice）+ 神经翻唱（Seed-VC 44.1kHz + MSST 人声分离）**，单进程单端口、OpenAI 兼容协议，为 AstrBot 等 bot 框架提供语音能力后端。

- GPU 实测：RTX 5070 Ti Laptop 12GB（sm_120 / Blackwell，CUDA 12.8），TTS+ASR+翻唱引擎全常驻
- 协议：`POST /v1/audio/speech`（TTS/翻唱，按 model 分流）、`POST /v1/audio/transcriptions`（ASR）、Gradio 管理台（`/`）与 `/convert`（带进度流）

## 架构

```
AstrBot / 任意 OpenAI 客户端
        │  frp / 局域网
        ▼
gateway.py (FastAPI :18000)
├── /v1/audio/speech   model=breeze-tts-2 → Breeze 子进程(:7860) → lameenc 转 MP3
├── /v1/audio/speech   model=seed-svc     → 翻唱管线(MSST 分离 → Seed-VC 换声 → 伴奏混音)
├── /v1/audio/transcriptions                → SenseVoiceSmall (funasr, 进程内)
└── Gradio 管理台 + /convert (gr.Progress 进度流, 供 gradio_client 订阅)
```

`/v1/audio/speech` 按 `model` 字段分流：含 `svc` 走翻唱，否则走 TTS。

## 目录结构

```
audio-station/
├── gateway.py              # 统一网关入口（唯一需要启动的进程）
├── server/cover/           # 翻唱: gateway_cover.py + cover_core.py + seed_vc_engine.py ...
├── server/tts_asr/         # breeze_bridge.py (TTS 桥接+子进程管理) + asr_sensevoice.py
├── breeze/                 # Breeze 推理: run_breeze_api.py + pylibs/ + breeze-tts/(仓库) + models/
├── tools/                  # download_models.py / persona_reftext.py
├── scripts/                # 部署辅助
├── 启动audio-station.bat   # 双击启动, 前台窗口, 关窗即停
└── config.example.json
```

## 显存参考（12GB, eager 模式）

| 引擎 | 显存 | 说明 |
|---|---|---|
| Breeze TTS 2 | ~7.7 GB | 勿用 `--fast-all`（需 14.4GB）；eager 路径无需 flash-attn |
| Seed-VC DiT-f0 44k + BigVGAN + Whisper-small + RMVPE | ~2 GB | 进程内常驻，启动 3.8s |
| SenseVoiceSmall | ~1 GB | 进程内懒加载，启动 3.2s |
| MSST BS-Roformer | 按需 | 翻唱时以独立 python 子进程运行 |

## 安装（Windows）

1. **Python 运行时**：需 Python 3.10+ 且带 CUDA 12.8 的 PyTorch（Blackwell 显卡必须 `cu128`）。若已有可复用环境，把目录 junction 为 `runtime/` 即可。
2. **依赖**：`runtime/python.exe -m pip install -r requirements.txt`
   - embedded Python 无 venv 模块、忽略 `PYTHONPATH`，推荐 `--no-deps --target breeze/pylibs` 隔离安装 qwen-tts/transformers 等推理依赖（`breeze/pylibs` 已含验证过的版本组合）
   - **numpy 必须 `<2.4`**（numba 上限 2.3）；transformers 4.57 需补 tokenizers>=0.22
3. **Breeze 推理仓库**：`git clone https://github.com/breezeblue-ai/breeze-tts breeze/breeze-tts`
4. **模型权重**（全部走 ModelScope，HF 直连/hf-mirror 均不可用或极慢）：

   | 模型 | 来源 | 落位 |
   |---|---|---|
   | breeze-tts-2 (~5GB) | `modelscope: BreezeBlue/breeze-tts-2` | `breeze/models/breeze-tts-2/` |
   | Seed-VC DiT f0 44k | `modelscope: AI-ModelScope/SeedVC` | `checkpoints/` |
   | bigvgan_v2_44k / whisper-small / rmvpe / campplus | 同上脚本 | `models/` |
   | MSST BS-Roformer | 见 tools/download_models.py | `msst/` |
   | SenseVoiceSmall (~500MB) | funasr 首次运行自动从 ModelScope 下载 | HF/MS 缓存 |

5. **音色**：`voice_profiles/*.wav`（翻唱参考音色）、`voices/*.wav + *.txt`（TTS 克隆参考+文字稿，`tools/persona_reftext.py` 生成）
6. **启动**：双击 `启动audio-station.bat` → http://127.0.0.1:18000

## 对接 AstrBot（经 frp）

`frpc.toml` 三条代理全部指向网关端口 18000（remotePort 按需 10090~10093）：

| AstrBot 提供商 | Base URL | model | voice |
|---|---|---|---|
| OpenAI TTS(API) | `http://<frp>:<port>/v1` | `breeze-tts-2` | `112` / `galgame2` / `wr`（克隆） |
| OpenAI ASR(API) | 同上 | — | — |
| 翻唱插件 | 同上 | `seed-svc` | 同 voice_profiles |

## 常见坑（实测）

- **embedded Python**：无 `venv`、`sys.path` 不含 cwd、忽略 `PYTHONPATH` → 一律用启动器脚本显式 `sys.path.insert`
- **funasr 1.3 音频加载**：对文件输入会试 torchcodec/外部 ffmpeg，未装则崩 → ASR 端点内用 `soundfile+soxr` 预处理成 16k ndarray 直喂
- **Starlette 路由顺序**：`gr.mount_gradio_app(path="/")` 会拦截其后注册的 API 路由 → 网关统一把 Mount 挪到路由表末尾
- **HF 下载**：直连超时、hf-mirror 失效/极慢 → 一律 ModelScope
- **权重许可**：breeze-tts-2 权重非商用许可，个人使用无碍

## License

MIT（本仓库代码）。所含各模型权重遵循其各自许可。
