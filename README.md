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
├── runtime/                # Python 嵌入式运行时(自含 torch cu128, 不入库)
├── voices/                 # TTS 克隆参考音频 + 文字稿
├── server/cover/           # 翻唱: gateway_cover.py + cover_core.py + seed_vc_engine.py ...
│   ├── seed-vc/ checkpoints/ models/ msst/ bin/ runtime-rocm/   # 引擎与权重(数据, 不入库)
│   └── voice_profiles/ cache/ outputs/                           # 音色与运行产物
├── tools/                  # download_models.py (权重下载) / persona_reftext.py (克隆文字稿)
├── 启动audio-station.bat   # 双击启动, 前台窗口, 关窗即停
└── config.example.json
```

## 显存参考（12GB, eager 模式）

| 引擎 | 显存 | 说明 |
|---|---|---|
| Breeze TTS 2（**int8 量化**，默认） | ~6.3 GB | 权重 4.9GB；勿用 `--fast-all`（需 14.4GB）；eager 路径无需 flash-attn |
| Breeze TTS 2（bf16 原版） | ~7.7 GB | 与 Seed-VC/ASR 共存时翻唱峰值会顶穿 12GB，故默认改用 int8 |
| Seed-VC DiT-f0 44k + BigVGAN + Whisper-small + RMVPE | ~2 GB | 进程内常驻，启动 3.8s |
| SenseVoiceSmall | ~1 GB | 进程内懒加载，启动 3.2s |
| MSST BS-Roformer | 按需 | 翻唱时以独立 python 子进程运行，峰值 +2.3GB |

实测（RTX 5070 Ti Laptop 12GB，三服务齐开 + int8）：常驻 **10.9 GB**，翻唱峰值约 **11.9 GB**，余量 ~0.3–1.0 GB。bf16 原版在同一时机余量更小，Breeze 会因 CUDA OOM 退出（网关已做存活检测+自动重启兜底）。

## 安装（Windows）

1. **Python 运行时**：需 Python 3.10+ 且带 CUDA 12.8 的 PyTorch（Blackwell 显卡必须 `cu128`），安装到本目录 `runtime/`（自包含嵌入式运行时即可，仓库发布版已含完整 `runtime/`）。
2. **依赖**：`runtime/python.exe -m pip install -r requirements.txt`
   - embedded Python 无 venv 模块、忽略 `PYTHONPATH`，推荐 `--no-deps --target breeze/pylibs` 隔离安装 qwen-tts/transformers 等推理依赖（`breeze/pylibs` 已含验证过的版本组合）
   - **numpy 必须 `<2.4`**（numba 上限 2.3）；transformers 4.57 需补 tokenizers>=0.22
   - **int8 量化需 `torchao==0.17.0`**：`runtime/python.exe -m pip install --no-deps --target breeze/pylibs torchao==0.17.0`（纯 Python 轮，Windows 可直接装）
3. **Breeze 推理仓库**：`git clone https://github.com/breezeblue-ai/breeze-tts breeze/breeze-tts`
4. **模型权重**（实测 HF 直连超时，一律走 ModelScope 或 hf-mirror）：

   | 模型 | 来源 | 落位 |
   |---|---|---|
   | breeze-tts-2 (~5GB) | `modelscope: BreezeBlue/breeze-tts-2` | `breeze/models/breeze-tts-2/` |
   | **int8 量化版 (~5.5GB)** | 用上一步的 bf16 权重本地转换：`runtime/python.exe tools/quantize_breeze_int8.py breeze/models/breeze-tts-2 breeze/models/breeze-tts-2-int8` | `breeze/models/breeze-tts-2-int8/` |
   | Seed-VC DiT f0 44k + bigvgan_v2_44k + whisper-small + rmvpe + campplus | `tools/download_models.py`（hf-mirror 一键拉取） | `server/cover/checkpoints/` 与 `server/cover/models/` |
   | MSST BS-Roformer (bs_roformer_ep_317_sdr_12.9755) | [ZFTurbo/MSS-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) Releases 直链 | `server/cover/msst/pretrain/vocal_models/` |
   | Seed-VC 源码包(引擎依赖) | `git clone https://github.com/Plachtaa/seed-vc server/cover/seed-vc` | GPL-3.0 |
   | SenseVoiceSmall (~500MB) | funasr 首次运行自动从 ModelScope 下载 | HF/MS 缓存 |

5. **音色**：`server/cover/voice_profiles/*.wav`（翻唱参考音色）、`voices/*.wav + *.txt`（TTS 克隆参考+文字稿，`tools/persona_reftext.py` 生成）
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
- **Breeze 子进程会崩**：长期运行可能因显存/推理异常退出（曾观测到退出码 1），旧实现只认模块级 `_ready` 标记，进程死了仍继续转发 7860 → 每个 TTS 请求都返 500。现在每次请求前做 `/docs` 探活，检出进程退出即自动重启并等就绪；子进程标准输出（含 CUDA OOM 堆栈）落在 `logs/breeze_tts.log`，排查崩溃先看这个文件
- **int8 权重的两个坑**：① 只能存 `.bin`（pickle）分片，不能 `safe_serialization=True`——transformers 4.57.3 的 safetensors 读路径只认 torchao≤0.14 的扁平 key；② 上游 `api.py` 计算模型指纹时硬编码 `model.safetensors.index.json`，量化目录里是 `pytorch_model.bin.index.json`，会直接 FileNotFoundError。已在 `breeze/run_breeze_api.py` 里做兼容（按目录实际索引名取），不改上游代码
- **torchao 版本**：必须 `0.17.0`（`Int8WeightOnlyConfig(version=2)`）。0.17 移除了 `int8_weight_only()` 等函数式 API，所以量化脚本用 `TorchAoConfig(Int8WeightOnlyConfig(...))` 传配置对象——传字符串 `"int8_weight_only"` 会因 transformers 内部导入失败而报 `cannot import name 'autoquant'`

## 核心引擎说明

### TTS — Breeze TTS 2（BreezeBlue）

- **是什么**：Breeze TTS 2 是 [BreezeBlue](https://breezeblue.ai/breeze-tts-2) 发布的开源权重实时 TTS 模型，支持中英文、零样本音色克隆（参考音频 + 逐字文字稿）、自然语言音色设计与指令控 prosody，在 Artificial Analysis TTS 榜单开源模型中排名第 1。
- **出处引用**：
  - 推理代码仓库：<https://github.com/breezeblue-ai/breeze-tts>（Apache-2.0）
  - 模型权重：<https://huggingface.co/BreezeBlue/breeze-tts-2>（ModelScope 镜像：`BreezeBlue/breeze-tts-2`）
  - 官方博客：<https://breezeblue.ai/breeze-tts-2>
- **⚠️ 许可性质（重要）**：**源码为 Apache-2.0，但模型权重、衍生模型及自托管输出仅限研究与非商用用途**（上游 README 原文：“Breeze TTS 2 model weights, derivative models, and self-hosted outputs are for research and non-commercial use only”）。本仓库及其服务面向个人自托管场景；商业部署请先阅读并遵守上游模型许可条款。
- **显存**：eager 推理约 7.7 GiB，12GB 卡为最低推荐配置（`--fast-all` 快路径需 14.4GB，12GB 卡勿用）。
- **int8 量化（本仓库默认）**：用 `tools/quantize_breeze_int8.py` 在本地把 bf16 权重转成 torchao 8bit weight-only（每行一个 scale，只压主干与深度解码器，文本编码器/codec/embedding/lm_head 全部保持 bf16）。权重 7.16GB → 5.53GB，**实测常驻显存 7.7GB → 6.3GB**，音色与吐字主观无差别（用 ASR 回读校验文本一致）。代价：量化不提速（前向需反量化回 bf16 做 matmul）。想换回原版只需改 `BREEZE_MODEL=breeze-tts-2` 环境变量。

### ASR — SenseVoiceSmall（FunAudioLLM）

阿里通义实验室开源多语音频理解模型（50+ 语言、富情感与声学事件标签、带标点输出），经 [funasr](https://github.com/modelscope/FunASR)（MIT）加载，权重由 ModelScope `iic/SenseVoiceSmall` 首次运行自动下载。

### 翻唱 — Seed-VC V1 + MSST

零样本歌声转换引擎 [Seed-VC](https://github.com/Plachtaa/seed-vc)（GPL-3.0）V1 44.1kHz SVC 模型，配合 MSST（BS-Roformer 人声/伴奏分离）完成「分离 → 换声 → 变调 → 混音」全链路。

## 开源组件与许可

本仓库代码以 MIT 发布。集成的第三方引擎与权重：

| 组件 | 角色 | 上游 | 许可 |
| :--- | :--- | :--- | :--- |
| **Breeze TTS 2** | TTS 引擎 | [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) | 源码 Apache-2.0；**权重非商用** |
| **Seed-VC** | 歌声转换 | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | GPL-3.0 |
| **MSST / BS-Roformer** | 人声分离 | [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | MIT |
| **SenseVoiceSmall** | ASR 模型 | [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)（ModelScope `iic/SenseVoiceSmall`） | 权重 Apache-2.0（可商用） |
| **funasr** | ASR 框架 | [modelscope/FunASR](https://github.com/modelscope/FunASR) | MIT |
| FastAPI / uvicorn / gradio / soundfile / librosa | Web 与音频栈 | PyPI | 各自开源协议 |

**商用提醒**：**Breeze TTS 2 权重为非商用许可**（源码 Apache-2.0 但权重仅限研究/非商用，见上表）；Seed-VC 为 GPL-3.0；SenseVoiceSmall 权重 Apache-2.0 可商用。本项目整体仅供个人学习研究与非商业自托管。

## License

MIT（本仓库代码）。所含各模型权重遵循其各自许可，见上表。
