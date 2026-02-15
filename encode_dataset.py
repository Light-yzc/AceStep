import os
import argparse
from acestep.handler import AceStepHandler
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Encode audio files to audio codes for DiT training.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file to encode.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda, mps, cpu).")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        logger.error(f"Audio file not found: {args.audio_path}")
        return

    # Initialize Handler
    handler = AceStepHandler()
    
    # Project root is current dir for this script context
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    logger.info("Initializing AceStepHandler...")
    # config_path can be the default model version
    status, success = handler.initialize_service(
        project_root=project_root,
        config_path="acestep-v15-turbo", 
        device=args.device
    )
    
    if not success:
        logger.error(f"Failed to initialize handler: {status}")
        return

    logger.info(f"Encoding audio: {args.audio_path}")
    
    # Use the handler built-in method to convert audio to codes
    # This handles loading, resampling to 48kHz, VAE encoding, and Tokenization/Quantization
    codes_string = handler.convert_src_audio_to_codes(args.audio_path)
    
    if codes_string.startswith("❌"):
        logger.error(f"Encoding failed: {codes_string}")
    else:
        print("\n--- Generated Audio Codes ---")
        print(codes_string)
        print(f"\nTotal codes: {len(codes_string.split('<|audio_code_')) - 1}")
        print("-----------------------------\n")

if __name__ == "__main__":
    main()
