from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music, understand_music
from acestep.inference import create_sample

# Initialize handlers
dit_handler = AceStepHandler()
llm_handler = LLMHandler()
# # # Initialize services
dit_handler.initialize_service(
    project_root="/content/AceStep",
    config_path="acestep-v15-turbo",
    device="cuda"
)
print(llm_handler.initialize(
    checkpoint_dir="/content/AceStep/checkpoints",
    lm_model_path="acestep-5Hz-lm-4B",
    # backend="vllm",
    device="cuda"
))
# print(a)
import os
path = '/content/AceStep/my_song'
files = os.listdir(path)
codes = dit_handler.convert_src_audio_to_codes(os.path.join(path, files[0]))
# print(codes) 
print(understand_music(llm_handler, codes))