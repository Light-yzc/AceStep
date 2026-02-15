import math
import os
import tempfile
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
import torch
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig

from acestep.audio_utils import AudioSaver, generate_uuid_from_params, normalize_audio

# Initialize parameters and config
params = GenerationParams(
    caption="A catchy pop song with upbeat rhythm",
    duration=10.0,
    thinking=True
)
config = GenerationConfig(
    batch_size=1
)

# Derived logic from acestep/inference.py
chunk_size = config.batch_size
chunk_seeds = None
progress = None

top_k_value = None if not params.lm_top_k or params.lm_top_k == 0 else int(params.lm_top_k)
top_p_value = None if not params.lm_top_p or params.lm_top_p >= 1.0 else params.lm_top_p

# Audio duration
audio_duration = params.duration

# User metadata construction
user_metadata = {}
if params.bpm is not None:
    try:
        bpm_value = float(params.bpm)
        if bpm_value > 0:
            user_metadata['bpm'] = int(bpm_value)
    except (ValueError, TypeError):
        pass

if params.keyscale and params.keyscale.strip():
    if params.keyscale.strip().lower() not in ["n/a", ""]:
        user_metadata['keyscale'] = params.keyscale.strip()

if params.timesignature and params.timesignature.strip():
    if params.timesignature.strip().lower() not in ["n/a", ""]:
        user_metadata['timesignature'] = params.timesignature.strip()

if params.duration is not None:
    try:
        duration_value = float(params.duration)
        if duration_value > 0:
            user_metadata['duration'] = int(duration_value)
    except (ValueError, TypeError):
        pass

user_metadata_to_pass = user_metadata if user_metadata else None

# Determine infer_type
user_provided_audio_codes = bool(params.audio_codes and str(params.audio_codes).strip())
need_audio_codes = not user_provided_audio_codes
infer_type = "llm_dit" if need_audio_codes and params.thinking else "dit"


llm_handler = LLMHandler()

print("Initializing LLM Handler...")
# Define paths for initialization
current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(current_dir, "checkpoints")
lm_model_path = "acestep-5Hz-lm-1.7B" # Default 5Hz LM model

init_msg, success = llm_handler.initialize(checkpoint_dir=checkpoint_dir, lm_model_path=lm_model_path)
print(init_msg)
if not success:
    print("❌ Initialization failed. Exiting.")
    exit(1)

print(f"Generating with infer_type={infer_type}...")
result = llm_handler.generate_with_stop_condition(
    caption=params.caption or "",
    lyrics=params.lyrics or "",
    infer_type=infer_type,
    temperature=params.lm_temperature,
    cfg_scale=params.lm_cfg_scale,
    negative_prompt=params.lm_negative_prompt,
    top_k=top_k_value,
    top_p=top_p_value,
    target_duration=audio_duration,  # Pass duration to limit audio codes generation
    user_metadata=user_metadata_to_pass,
    use_cot_caption=params.use_cot_caption,
    use_cot_language=params.use_cot_language,
    use_cot_metas=params.use_cot_metas,
    use_constrained_decoding=params.use_constrained_decoding,
    constrained_decoding_debug=config.constrained_decoding_debug,
    batch_size=chunk_size,
    seeds=chunk_seeds,
    progress=progress,
)

if not result.get("success", False):
    error_msg = result.get("error", "Unknown LM error")
    print(f"❌ LM Error: {error_msg}")
else:
    print("✅ Generation Successful!")
    # print("Result Metadata:", result.get("metadata"))
    # print("Result Audio Codes:", result.get("audio_codes")) # Might be long

    # --- Extract parameters for DiT ---
    print("\n--- Parameters for DiT (Review) ---")
    
    lm_metadata = result.get("metadata", {})
    audio_codes = result.get("audio_codes", "")

    # 1. Audio Codes
    # If using LLM generated codes, this is the main structural input
    dit_audio_code_string = audio_codes if infer_type == "llm_dit" else params.audio_codes
    print(f"audio_code_string (Length): {len(dit_audio_code_string)} chars")
    print(f"audio_code_string (Preview): {dit_audio_code_string[:100]}...")

    # 2. Captions & Lyrics
    # LLM might rewrite caption or detect language
    dit_caption = lm_metadata.get("caption", params.caption)
    dit_lyrics = lm_metadata.get("lyrics", params.lyrics)
    dit_language = lm_metadata.get("vocal_language", params.vocal_language) # Might be 'unknown' or detected code

    print(f"captions: {dit_caption}")
    print(f"lyrics: {dit_lyrics}")
    print(f"vocal_language: {dit_language}")

    # 3. Musical Metadata
    # Prioritize LLM generated metadata, fallback to user params, then defaults
    def parse_meta(key, default_val=None):
        val = lm_metadata.get(key)
        if val in [None, "", "N/A"]:
             val = getattr(params, key, default_val) # Fallback to params if LLM didn't generate valid meta
        return val

    dit_bpm = parse_meta("bpm")
    dit_keyscale = parse_meta("keyscale") # or 'key_scale' in some contexts, but result uses 'keyscale'
    dit_timesignature = parse_meta("timesignature")
    dit_duration = parse_meta("duration", params.duration)

    print(f"bpm: {dit_bpm}")
    print(f"key_scale: {dit_keyscale}")
    print(f"time_signature: {dit_timesignature}")
    print(f"audio_duration: {dit_duration}")

    # 4. Other Generation Params (Passed directly from params/config)
    print(f"inference_steps: {params.inference_steps}")
    print(f"guidance_scale: {params.guidance_scale}")
    print(f"use_random_seed: {config.use_random_seed}")
    
    # Batch handling (simplified for this script's single batch context)
    seed_for_generation = str(config.seeds) if config.seeds is not None else str(params.seed) 
    print(f"seed: {seed_for_generation}")
    print(f"batch_size: {config.batch_size}")
    
    # Advanced Params
    print(f"cfg_interval_start: {params.cfg_interval_start}")
    print(f"cfg_interval_end: {params.cfg_interval_end}")
    print(f"shift: {params.shift}")
    print(f"infer_method: {params.infer_method}")
    
    print("-----------------------------------")