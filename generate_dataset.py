import os
import json
import time
import torch
import random
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# Import ACE-Step modules
from acestep.llm_inference import LLMHandler
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig

# ================= Configuration =================
OUTPUT_DIR = "acestep_synthetic_dataset"
NUM_SAMPLES = 5  # How many samples to generate
BATCH_SIZE = 1    
MODEL_CHECKPOINT_DIR = "acestep/checkpoints" 

# Model Configuration
LM_MODEL_PATH = "acestep-5Hz-lm-1.7B"
DIT_MODEL_PATH = "acestep-v15-turbo"

# Seed prompts (Simulate "Simple User Input")
BASE_PROMPTS = [
    "A catchy pop song about summer",
    "An emotional piano ballad",
    "Upbeat electronic dance music",
    "A smooth jazz track for relaxing",
    "Heavy metal song with intense drums",
    "Acoustic guitar folk song",
    "Lo-fi hip hop beat for studying",
    "Cinematic orchestral soundtrack",
    "R&B soul music with vocals",
    "Cyberpunk synthwave track"
]

def main():
    # 1. Setup Directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio_dir = os.path.join(OUTPUT_DIR, "audio")
    meta_dir = os.path.join(OUTPUT_DIR, "metadata") # Keep individual JSONs for safety
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # 2. Initialize Models
    logger.info("Initializing LLM Handler...")
    llm_handler = LLMHandler()
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(project_root, "acestep", "checkpoints")
    
    # Initialize LLM
    init_msg, success = llm_handler.initialize(checkpoint_dir=checkpoint_dir, lm_model_path=LM_MODEL_PATH)
    if not success:
        logger.error(f"LLM Init Failed: {init_msg}")
        return

    logger.info("Initializing DiT Handler...")
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=project_root,
        config_path=DIT_MODEL_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mlx_dit=False 
    )
    if not success:
        logger.error(f"DiT Init Failed: {status_msg}")
        return

    logger.success("Models Initialized. Starting Generation Loop...")

    # 3. Generation Loop
    generated_records = []
    
    for i in tqdm(range(NUM_SAMPLES), desc="Generating Data"):
        
        # A. Prepare Prompt
        prompt = random.choice(BASE_PROMPTS)
        uid = f"sample_{int(time.time())}_{i}"
        logger.info(f"[{i+1}/{NUM_SAMPLES}] Processing: '{prompt}'")

        # B. Run LLM (Metadata + Audio Codes)
        target_duration = 30.0 
        
        # Default Params matching your JSON structure
        params = {
            "task_type": "text2music",
            "instruction": "Fill the audio semantic mask based on the given conditions:",
            "reference_audio": None,
            "src_audio": None,
            "inference_steps": 20, 
            "guidance_scale": 7.0,
            "shift": 3.0,
            "infer_method": "ode",
            "audio_cover_strength": 1.0,
            "thinking": True,
            "lm_temperature": 0.85,
            "lm_cfg_scale": 2.0,
            "lm_top_k": 0,
            "lm_top_p": 0.9,
            "use_cot_caption": True,
            "use_constrained_decoding": True,
            "use_adg": False,
            "seed": random.randint(0, 2**32-1)
        }

        # LLM Generation
        llm_result = llm_handler.generate_with_stop_condition(
            caption=prompt,
            lyrics="",
            infer_type="llm_dit", 
            batch_size=1,
            target_duration=target_duration,
            # Pass controls
            temperature=params["lm_temperature"],
            cfg_scale=params["lm_cfg_scale"],
            top_k=params["lm_top_k"],
            top_p=params["lm_top_p"],
            use_cot_caption=params["use_cot_caption"],
            use_constrained_decoding=params["use_constrained_decoding"]
        )

        if not llm_result.get("success", False):
            logger.warning(f"LLM Failed for {prompt}: {llm_result.get('error')}")
            continue

        # Extract LLM artifacts
        audio_codes = llm_result.get("audio_codes", "")
        lm_metadata = llm_result.get("metadata", {})
        
        # Construct Full JSON Record
        record = params.copy()
        
        # Fill Dynamic Content from LLM
        record.update({
            "model_id_llm": LM_MODEL_PATH,
            "model_id_dit": DIT_MODEL_PATH,
            "audio_codes": audio_codes,
            "caption": lm_metadata.get("caption", prompt),
            "lyrics": lm_metadata.get("lyrics", ""),
            "vocal_language": lm_metadata.get("vocal_language", "en"),
            "bpm": lm_metadata.get("bpm"), 
            "keyscale": lm_metadata.get("keyscale", ""),
            "timesignature": lm_metadata.get("timesignature", "4/4"),
            "duration": float(lm_metadata.get("duration", target_duration)),
            "instrumental": False if lm_metadata.get("lyrics") else True,
            
            # Placeholders
            "timesteps": None,
            "repainting_start": 0,
            "repainting_end": -1,
            "lm_negative_prompt": "NO USER INPUT",
            "use_cot_metas": False,
            "use_cot_lyrics": False,
            "use_cot_language": True,
            "cot_bpm": None,
            "cot_keyscale": "",
            "cot_timesignature": "",
            "cot_duration": None,
            "cot_vocal_language": "unknown",
            "cot_caption": "",
            "cot_lyrics": ""
        })
        
        if record["bpm"]: 
            try: record["bpm"] = int(float(record["bpm"]))
            except: record["bpm"] = 120 
            
        # C. Run DiT (Latent Generation -> Audio)
        try:
            dit_output = dit_handler.service_generate(
                captions=[record["caption"]],
                lyrics=[record["lyrics"]],
                metas=[{
                     "bpm": record["bpm"],
                     "key": record["keyscale"],
                     "time_signature": record["timesignature"],
                }],
                vocal_languages=[record["vocal_language"]],
                audio_code_hints=[audio_codes], 
                
                # Params
                infer_steps=record["inference_steps"],
                guidance_scale=record["guidance_scale"],
                seed=record["seed"],
                infer_method=record["infer_method"],
                shift=record["shift"],
                audio_cover_strength=record["audio_cover_strength"]
            )
        except Exception as e:
            logger.error(f"DiT Generation Failed: {e}")
            continue

        generated_audio_paths = dit_output.get("audio_files", [])
        if not generated_audio_paths:
            logger.warning("No audio generated.")
            continue
            
        src_audio_path = generated_audio_paths[0]
        
        # D. Save
        target_audio_name = f"{uid}.wav"
        target_audio_path = os.path.join(audio_dir, target_audio_name)
        
        # 1. Move Audio
        os.system(f"cp '{src_audio_path}' '{target_audio_path}'")
        
        # 2. Add 'file_name' for HF Dataset compatibility
        # HF expects 'file_name' to point to the audio file relative to the metadata file
        # or relative to the dataset root if using audiofolder
        record["file_name"] = f"audio/{target_audio_name}"
        
        generated_records.append(record)
        
        # Also save individual JSON as backup
        target_json_path = os.path.join(meta_dir, f"{uid}.json")
        with open(target_json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            
        logger.success(f"Saved sample {uid}")

    # 4. Generate metadata.jsonl for HuggingFace
    logger.info("Generating metadata.jsonl for HuggingFace...")
    jsonl_path = os.path.join(OUTPUT_DIR, "metadata.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in generated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    logger.success(f"Dataset generation complete! Ready for HF upload in {OUTPUT_DIR}")
    print("\nTo upload, run:")
    print(f"huggingface-cli upload your-username/acestep-synthetic {OUTPUT_DIR} --repo-type dataset")

if __name__ == "__main__":
    main()
