from acestep.training_v2.model_loader import (
    load_vae,
    load_text_encoder,
    load_silence_latent,
    unload_models,
    _resolve_dtype,
    )
from acestep.training.dataset_builder_modules.preprocess_audio import load_audio_stereo
import torch
import gc
import math
from typing import Any, Callable, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)

_TARGET_SR = 48000
max_duration: float = 240.0,
def _empty_gpu_cache() -> None:
    """Release cached GPU memory (CUDA / MPS / XPU)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

def _tiled_vae_encode(
    vae: Any,
    audio: torch.Tensor,
    dtype: torch.dtype,
    chunk_size: Optional[int] = None,
    overlap: int = 96000,
) -> torch.Tensor:
    """Encode audio through the VAE using overlap-discard tiling.

    Processes long audio in chunks to avoid OOM on the monolithic
    ``vae.encode()`` call.  Mirrors the tiling strategy from
    ``handler.tiled_encode`` but as a standalone function with no
    ``self`` / handler dependency.

    Args:
        vae: The ``AutoencoderOobleck`` VAE model (on device, in eval mode).
        audio: Audio tensor ``[B, C, S]`` (batch, channels, samples).
        dtype: Target dtype for the output latents.
        chunk_size: Audio samples per chunk.  ``None`` = auto-select
            based on available GPU memory (30 s for >=8 GB, 15 s otherwise).
        overlap: Overlap in audio samples between adjacent chunks
            (default 2 s at 48 kHz = 96 000).

    Returns:
        Latent tensor ``[B, T, 64]`` (same format as upstream
        ``vae_encode``), cast to *dtype*.
    """
    vae_device = next(vae.parameters()).device
    vae_dtype = vae.dtype

    # Auto-select chunk size based on GPU VRAM
    if chunk_size is None:
        gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(vae_device)
                gpu_mem_gb = props.total_mem / (1024 ** 3)
            except Exception:
                pass
        chunk_size = _TARGET_SR * 15 if gpu_mem_gb <= 8 else _TARGET_SR * 30

    B, C, S = audio.shape

    # Short audio -- direct encode (no tiling needed)
    if S <= chunk_size:
        vae_input = audio.to(vae_device, dtype=vae_dtype)
        with torch.inference_mode():
            latents = vae.encode(vae_input).latent_dist.sample()
        return latents.transpose(1, 2).to(dtype)

    # Calculate stride (core region per chunk, excluding overlap)
    stride = chunk_size - 2 * overlap
    if stride <= 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be > 2 * overlap ({overlap})"
        )

    num_steps = math.ceil(S / stride)
    downsample_factor: Optional[float] = None
    latent_write_pos = 0
    final_latents: Optional[torch.Tensor] = None

    for i in range(num_steps):
        core_start = i * stride
        core_end = min(core_start + stride, S)

        # Window with overlap on both sides
        win_start = max(0, core_start - overlap)
        win_end = min(S, core_end + overlap)

        chunk = audio[:, :, win_start:win_end].to(vae_device, dtype=vae_dtype)

        with torch.inference_mode():
            latent_chunk = vae.encode(chunk).latent_dist.sample()

        # Determine downsample factor from the first chunk
        if downsample_factor is None:
            downsample_factor = chunk.shape[-1] / latent_chunk.shape[-1]
            total_latent_len = int(round(S / downsample_factor))
            final_latents = torch.zeros(
                B, latent_chunk.shape[1], total_latent_len,
                dtype=latent_chunk.dtype, device="cpu",
            )

        # Trim the overlap regions from the latent
        added_start = core_start - win_start
        trim_start = int(round(added_start / downsample_factor))

        added_end = win_end - core_end
        trim_end = int(round(added_end / downsample_factor))

        lat_len = latent_chunk.shape[-1]
        end_idx = lat_len - trim_end if trim_end > 0 else lat_len
        latent_core = latent_chunk[:, :, trim_start:end_idx]

        # Copy to pre-allocated CPU tensor
        core_len = latent_core.shape[-1]
        assert final_latents is not None
        final_latents[:, :, latent_write_pos:latent_write_pos + core_len] = latent_core.cpu()
        latent_write_pos += core_len

        del chunk, latent_chunk, latent_core

    # Trim to actual written length
    assert final_latents is not None
    final_latents = final_latents[:, :, :latent_write_pos]

    # Transpose to (B, T, 64) and cast -- matches vae_encode output format
    return final_latents.transpose(1, 2).to(dtype)


device = 'cuda'
precision = 'auto'
dtype = _resolve_dtype(precision)

logger.info("[Side-Step] Pass 1/2: Loading VAE...")
vae = load_vae(checkpoint_dir, device, precision)

intermediates: List[Path] = []
failed = 0
total = len(audio_files)

try:
    for i, af in enumerate(audio_files):

        # Skip if final .pt already exists (resumable)
        final_pt = out_path / f"{af.stem}.pt"
        if final_pt.exists():
            logger.info("Skipping (final exists): %s", af.name)
            continue

        try:
            # 1. Load audio (stereo, 48 kHz)
            audio, _sr = load_audio_stereo(str(af), _TARGET_SR, max_duration)
            audio = audio.unsqueeze(0).to(device).to(vae.dtype)

            # 2. VAE encode (tiled for long audio)
            with torch.no_grad():
                target_latents = _tiled_vae_encode(vae, audio, dtype)

            del audio  # free GPU audio tensor before text encoding
            _empty_gpu_cache()

            latent_length = target_latents.shape[1]
            attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)
