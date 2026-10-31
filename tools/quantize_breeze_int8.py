# -*- coding: utf-8 -*-
"""
Breeze TTS 2 本地 int8 量化（torchao weight-only）
=================================================
把 bf16 权重(约 7.2GB)转成 int8 量化权重(约 5GB), 显存从 ~7.7G 降到 ~6.1G,
在 12GB 卡上与 Seed-VC / SenseVoice 共存时能腾出约 1.6G 余量。

量化范围(与社区实践一致, transformers TorchAoConfig 自动处理):
    - 只压主干(Qwen3 28L)与深度解码器(Llama 12L)的 nn.Linear
    - 文本编码器 / codec / embedding / lm_head 保持 bf16(精度敏感)

限制:
    - torchao 必须是 0.17.0(4.57.3 的 safetensors 读路径只认 torchao<=0.14 的扁平 key,
      因此必须存成 .bin pickle 分片, 不能用 safe_serialization=True)
    - 量化不提升速度(前向要反量化), 收益只体现在显存

用法:
    runtime/python.exe tools/quantize_breeze_int8.py <bf16目录> <输出目录>
"""
import os
import shutil
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "breeze", "pylibs"))
sys.path.insert(0, os.path.join(BASE, "breeze", "breeze-tts"))

import torch  # noqa: E402
from transformers import AutoTokenizer, TorchAoConfig  # noqa: E402
from models.breeze import BreezeForConditionalGeneration  # noqa: E402

from torchao.quantization import Int8WeightOnlyConfig  # noqa: E402
from torchao.quantization.granularity import PerRow  # noqa: E402

# 与社区实践一致: version=2 + 按行(PerRow)一个 scale, 只压 nn.Linear
QUANT_CONFIG = Int8WeightOnlyConfig(
    version=2,
    group_size=None,
    granularity=PerRow(dim=-1),
    set_inductor_config=False,
)

# 精度敏感但省不了多少的部分一律不压:
# 文本编码器(每请求只跑一次)、codec(直接决定波形)、embedding/lm_head(查表)、
# codebooks_head(分类头, 输出要 argmax)
MODULES_TO_NOT_CONVERT = [
    "text_encoder",
    "text_encoder_proj",
    "codec_model",
    "embed_text_tokens",
    "lm_head",
    "backbone_model.embed_tokens",
    "depth_decoder.codebooks_head",
]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    if not os.path.isdir(src):
        print("源目录不存在:", src)
        return 2
    if os.path.isdir(dst) and os.listdir(dst):
        print("输出目录非空, 请先清理:", dst)
        return 2
    os.makedirs(dst, exist_ok=True)

    print(f"[1/4] 加载 bf16 权重: {src}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(src, fix_mistral_regex=False)
    qcfg = TorchAoConfig(QUANT_CONFIG, modules_to_not_convert=MODULES_TO_NOT_CONVERT)
    model = BreezeForConditionalGeneration.from_pretrained(
        src,
        dtype=torch.bfloat16,
        quantization_config=qcfg,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    print(f"      完成, 耗时 {time.time() - t0:.1f}s")

    print("[2/4] 保存量化权重(.bin pickle, 不能存 safetensors)")
    t1 = time.time()
    model.save_pretrained(dst, safe_serialization=False)
    tokenizer.save_pretrained(dst)
    print(f"      完成, 耗时 {time.time() - t1:.1f}s")

    print("[3/4] 复制音频解码器与许可文件")
    for name in ("audio_tokenizer", "LICENSE", "README.md"):
        s = os.path.join(src, name)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, name), dirs_exist_ok=True)
        elif os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, name))

    print("[4/4] 体积对比")
    def size(p: str) -> float:
        total = 0
        for root, _, files in os.walk(p):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / 1024 ** 3

    print(f"      bf16: {size(src):.2f} GB -> int8: {size(dst):.2f} GB")
    print("完成:", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
