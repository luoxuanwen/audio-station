# -*- coding: utf-8 -*-
"""
Breeze TTS 2 推理服务启动器（audio-station 版）
==============================================
背景: 复用的 runtime/python.exe 是 embedded Python —— sys.path 不含当前目录、
      忽略 PYTHONPATH, 因此不能直接 `python -m breeze_infer.api`。
做法: 导入前显式插入依赖路径, 再调用官方 main()。

用法:
    python breeze/run_breeze_api.py <模型目录> --host 127.0.0.1 --port 7860
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
# 隔离依赖目录(qwen_tts / transformers 4.57 / numpy 2.3 / tokenizers / onnxruntime / sox)
sys.path.insert(0, os.path.join(BASE, "pylibs"))
# Breeze 推理仓库目录(breeze_infer 包)
sys.path.insert(0, os.path.join(BASE, "breeze-tts"))

from breeze_infer.api import main  # noqa: E402

if __name__ == "__main__":
    main()
