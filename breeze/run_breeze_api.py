# -*- coding: utf-8 -*-
"""
Breeze TTS 2 推理服务启动器（audio-station 版）
==============================================
背景: 复用的 runtime/python.exe 是 embedded Python —— sys.path 不含当前目录、
      忽略 PYTHONPATH, 因此不能直接 `python -m breeze_infer.api`。
做法: 导入前显式插入依赖路径, 打上 int8 量化权重所需的适配补丁, 再调用官方 main()。

用法:
    python breeze/run_breeze_api.py <模型目录> --host 127.0.0.1 --port 7860
"""
import os
import sys
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))
# 隔离依赖目录(qwen_tts / transformers 4.57 / numpy 2.3 / tokenizers / onnxruntime / sox / torchao)
sys.path.insert(0, os.path.join(BASE, "pylibs"))
# Breeze 推理仓库目录(breeze_infer 包)
sys.path.insert(0, os.path.join(BASE, "breeze-tts"))

import breeze_infer.api as _api  # noqa: E402
from breeze_infer.api import main  # noqa: E402


def _patch_index_file() -> None:
    """兼容 int8 量化权重目录的分片索引文件名。

    上游 api.compute_model_digest() 只认 `model.safetensors.index.json`
    (bf16 权重目录里的名字)。torchao 量化权重必须存成 pickle 分片, 索引名为
    `pytorch_model.bin.index.json`, 直接启动会 FileNotFoundError 退出。
    这里在调用前按目录实际内容改写常量, 不修改上游代码。
    """
    original = _api.compute_model_digest

    def compute_model_digest(model_dir):
        for name in ("pytorch_model.bin.index.json", _api.MODEL_INDEX_FILE):
            if (Path(model_dir) / name).is_file():
                _api.MODEL_INDEX_FILE = name
                break
        return original(model_dir)

    _api.compute_model_digest = compute_model_digest


_patch_index_file()

if __name__ == "__main__":
    main()
