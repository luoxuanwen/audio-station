# -*- coding: utf-8 -*-
"""Download Seed-VC V1 (44.1kHz singing) dependencies via hf-mirror into local HF cache."""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download, snapshot_download

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf-cache")
os.environ["HF_HOME"] = CACHE


def log(msg):
    print(f"[DL] {msg}", flush=True)


# 1. Seed-VC 44.1kHz f0 DiT model (singing)
log("Seed-VC f0 44k DiT ...")
hf_hub_download("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth")
hf_hub_download("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
log("Seed-VC f0 44k DiT done")

# 2. CAMPPlus speaker embedding
log("campplus ...")
hf_hub_download("funasr/campplus", "campplus_cn_common.bin")
log("campplus done")

# 3. BigVGAN 44k vocoder
log("bigvgan 44k ...")
snapshot_download("nvidia/bigvgan_v2_44khz_128band_512x")
log("bigvgan 44k done")

# 4. Whisper-small content tokenizer
log("whisper-small ...")
snapshot_download(
    "openai/whisper-small",
    allow_patterns=["*.json", "*.txt", "model.safetensors", "*.model"],
)
log("whisper-small done")

# 5. RMVPE f0 extractor
log("rmvpe ...")
hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt")
log("rmvpe done")

print("ALL DOWNLOADS DONE", flush=True)
