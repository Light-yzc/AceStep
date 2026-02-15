from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music, understand_music
from acestep.inference import create_sample

# Initialize handlers
dit_handler = AceStepHandler()
llm_handler = LLMHandler()
# # # Initialize services
dit_handler.initialize_service(
    project_root="/content/ACE-Step-1.5",
    config_path="acestep-v15-turbo",
    device="cuda"
)
llm_handler.initialize(
    checkpoint_dir="/content/ACE-Step-1.5/checkpoints",
    lm_model_path="acestep-5Hz-lm-1.7B",
    # backend="vllm",
    device="cuda"
)
# print(a)
import os
files = os.listdir('/content/ACE-Step-1.5/my_audio')
codes = dit_handler.convert_src_audio_to_codes(os.path.join('/content/ACE-Step-1.5/my_audio', files[0]))
print(codes) 
print(understand_music(llm_handler, codes))