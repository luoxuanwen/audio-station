# -*- coding: utf-8 -*-
"""Seed-VC V1 44.1kHz singing conversion engine (resident in VRAM).

Adapted from Plachtaa/seed-vc app_svc.py (load_models + voice_conversion),
trimmed to the f0-conditioned 44.1k singing path and made non-streaming.
All models are loaded from local directories under D:/seed-vc/models.

convert() 增加泛化的智能音域逻辑（auto_tone）：跨性别整八度平移、
高音歌自适应步数、F0 软钳制与峰值保护。
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SEEDVC_REPO = PROJECT_ROOT / "seed-vc"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"

if str(SEEDVC_REPO) not in sys.path:
    sys.path.insert(0, str(SEEDVC_REPO))

from modules.commons import build_model, load_checkpoint, recursive_munch  # noqa: E402
from modules.campplus.DTDNN import CAMPPlus  # noqa: E402
from modules.bigvgan import bigvgan  # noqa: E402
from modules.audio import mel_spectrogram  # noqa: E402
from modules.rmvpe import RMVPE  # noqa: E402
from transformers import AutoFeatureExtractor, WhisperModel  # noqa: E402

DIT_CHECKPOINT = CHECKPOINT_DIR / "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
DIT_CONFIG = CHECKPOINT_DIR / "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
BIGVGAN_DIR = MODELS_DIR / "bigvgan_v2_44khz_128band_512x"
WHISPER_DIR = MODELS_DIR / "whisper-small"
CAMPPLUS_BIN = MODELS_DIR / "campplus_cn_common.bin"
RMVPE_PT = MODELS_DIR / "rmvpe.pt"


def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


class SeedVCEngine:
    """Singleton 44.1kHz f0-conditioned Seed-VC singing conversion engine."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: str | None = None, fp16: bool = True):
        if self._initialized:
            return
        self._initialized = True
        self.fp16 = fp16
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[SeedVC] device={self.device} fp16={self.fp16}", flush=True)
        print("[SeedVC] loading DiT f0 44k model ...", flush=True)

        config = yaml.safe_load(open(DIT_CONFIG, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = "DiT"
        self.model = build_model(model_params, stage="DiT")
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        self.sr = config["preprocess_params"]["sr"]

        self.model, _, _, _ = load_checkpoint(
            self.model, None, str(DIT_CHECKPOINT),
            load_only_params=True, ignore_modules=[], is_distributed=False,
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        print("[SeedVC] loading CAMPPlus ...", flush=True)
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(CAMPPLUS_BIN, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model.to(self.device)

        print("[SeedVC] loading BigVGAN 44k ...", flush=True)
        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(str(BIGVGAN_DIR), use_cuda_kernel=False)
        self.bigvgan_model.remove_weight_norm()
        self.bigvgan_model = self.bigvgan_model.eval().to(self.device)

        print("[SeedVC] loading Whisper-small encoder ...", flush=True)
        self.whisper_model = WhisperModel.from_pretrained(
            str(WHISPER_DIR), torch_dtype=torch.float16
        ).to(self.device)
        del self.whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(str(WHISPER_DIR))

        print("[SeedVC] loading RMVPE ...", flush=True)
        self.rmvpe = RMVPE(str(RMVPE_PT), is_half=False, device=self.device)

        mel_fn_args = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr,
            "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None" else 8000,
            "center": False,
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)  # noqa: E731

        self.max_context_window = self.sr // self.hop_length * 30
        self.overlap_frame_len = 16
        self.overlap_wave_len = self.overlap_frame_len * self.hop_length
        print("[SeedVC] all models loaded.", flush=True)

    # ------------------------------------------------------------------
    def _semantic_fn(self, waves_16k):
        ori_inputs = self.whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
        )
        ori_input_features = self.whisper_model._mask_input_features(
            ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
        ).to(self.device)
        with torch.no_grad():
            ori_outputs = self.whisper_model.encoder(
                ori_input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
        return S_ori

    # ------------------------------------------------------------------
    @torch.no_grad()
    @torch.inference_mode()
    def convert(
        self,
        source: str | Path,
        target: str | Path,
        diffusion_steps: int = 50,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.95,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
        seed: int | None = None,
        smart_f0: bool = True,
        auto_tone: bool = True,
    ) -> tuple[int, np.ndarray]:
        """Convert singing source to the target timbre. Returns (sample_rate, mono float32 waveform).

auto_tone=True 时按源/参考 F0 统计自动决策音域策略（跨性别整八度平移、
高音歌自动提升步数），泛化于任意歌曲与音色；pitch_shift!=0 表示用户手动
指定变调，此时跳过自动决策。"""
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        source_audio = librosa.load(str(source), sr=self.sr)[0]
        ref_audio = librosa.load(str(target), sr=self.sr)[0]

        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref_audio[: self.sr * 25]).unsqueeze(0).float().to(self.device)

        ref_waves_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)

        # source semantic features (long audio is chunked with 5s overlap)
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = self._semantic_fn(converted_waves_16k)
        else:
            overlapping_time = 5
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:
                    chunk = converted_waves_16k[:, traversed_time: traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat(
                        [buffer, converted_waves_16k[:, traversed_time: traversed_time + 16000 * (30 - overlapping_time)]],
                        dim=-1,
                    )
                S_alt = self._semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

        S_ori = self._semantic_fn(ref_waves_16k)

        mel = self.to_mel(source_audio.to(self.device).float())
        mel2 = self.to_mel(ref_audio.to(self.device).float())

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        F0_ori = self.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
        F0_alt = self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(self.device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        # 智能音域总决策（auto_tone）：按源与参考的中位差自动选择策略，泛化于任意歌曲/音色/性别。
        # - |中位差| <= 6 半音（同性别/相近音域）：中位微调对齐（auto_f0_adjust=True）
        # - |中位差| > 6 半音（跨性别/音域差大）：关闭中位对齐，改整八度平移——
        #   旋律相对音高与调性和声完整保留，高音句不会因非八度平移被拉出音域
        # - 用户手动指定 pitch_shift 时尊重手动，跳过自动决策
        tone_note = ""
        if auto_tone and pitch_shift == 0 and voiced_F0_alt.numel() > 0 and voiced_F0_ori.numel() > 0:
            med_src = float(torch.median(voiced_F0_alt))
            med_ref = float(torch.median(voiced_F0_ori))
            diff_st = 12.0 * math.log2(med_src / med_ref)
            if abs(diff_st) > 6:
                auto_f0_adjust = False
                # diff_st = 源比参考高出的半音数；旋律需向参考音域平移，故取负号
                pitch_shift = -int(round(diff_st / 12.0) * 12)
                tone_note = f"跨音域({diff_st:+.1f}st) -> 整八度平移{pitch_shift:+d}"
            else:
                auto_f0_adjust = True
                tone_note = f"同音域({diff_st:+.1f}st) -> 中位微调"
            # 高音自适应质量：源高音密集时增加扩散步数，缓解高音区重建失真
            p90 = float(torch.quantile(voiced_F0_alt.float(), 0.9))
            if p90 > 380.0 and diffusion_steps < 48:
                diffusion_steps = 48
                tone_note += f" | 高音歌(P90={p90:.0f}Hz) 步数提升至 48"
            print(f"[SeedVC] 音域决策: med_src={med_src:.0f}Hz med_ref={med_ref:.0f}Hz {tone_note}", flush=True)

        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = (
                log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            )
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)

        # 智能音域适配：仅当 F0 超出参考音色合理歌唱范围（中位 ±12 半音，即整个八度）
        # 时才软钳制——正常歌唱高音不干预，只防跨性别极端平移导致的超域破音。
        if smart_f0:
            ref_med = torch.median(voiced_F0_ori) if voiced_F0_ori.numel() > 0 else torch.tensor(220.0, device=self.device)
            f0_max = ref_med * 2.0    # +12 半音（高八度）
            f0_min = ref_med / 2.0    # -12 半音（低八度）
            log_f = torch.log(shifted_f0_alt.clamp(min=1e-5))
            log_cap = torch.log(f0_max)
            log_floor = torch.log(f0_min)
            soft = torch.where(
                log_f > log_cap,
                log_cap + torch.tanh((log_f - log_cap) * 1.5) / 1.5,
                log_f,
            )
            soft = torch.where(
                log_f < log_floor,
                log_floor - torch.tanh((log_floor - log_f) * 1.5) / 1.5,
                soft,
            )
            shifted_f0_alt = torch.exp(soft)

        cond, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        prompt_condition, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )
        interpolated_shifted_f0_alt = torch.nn.functional.interpolate(
            shifted_f0_alt.unsqueeze(1), size=cond.size(1), mode="nearest"
        ).squeeze(1)
        max_source_window = self.max_context_window - mel2.size(2)

        processed_frames = 0
        generated_wave_chunks = []
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames: processed_frames + max_source_window]
            chunk_f0 = interpolated_shifted_f0_alt[:, processed_frames: processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if self.fp16 else torch.float32,
            ):
                vc_target = self.model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2,
                    style2,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate,
                )
                vc_target = vc_target[:, :, mel2.size(-1):]
            vc_wave = self.bigvgan_model(vc_target.float()).squeeze().cpu()
            if vc_wave.ndim == 1:
                vc_wave = vc_wave.unsqueeze(0)
            if processed_frames == 0:
                if is_last_chunk:
                    generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                    break
                generated_wave_chunks.append(vc_wave[0, : -self.overlap_wave_len].cpu().numpy())
                previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len
            elif is_last_chunk:
                output_wave = crossfade(
                    previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), self.overlap_wave_len
                )
                generated_wave_chunks.append(output_wave)
                break
            else:
                output_wave = crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, : -self.overlap_wave_len].cpu().numpy(),
                    self.overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len

        full = np.concatenate(generated_wave_chunks)
        # 峰值保护：float 输出可能超过 ±1.0，导出 int16/MP3 时会削波破音，归一到 -1dBFS
        peak = float(np.abs(full).max())
        if peak > 0.89:
            full = full / peak * 0.89
        return self.sr, full


def get_engine() -> SeedVCEngine:
    return SeedVCEngine()


if __name__ == "__main__":
    engine = get_engine()
    print("[SeedVC] smoke test: engine loaded OK")
