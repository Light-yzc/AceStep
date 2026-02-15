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
# print(codes) 

llm_handler.initialize(
    checkpoint_dir="/content/AceStep/checkpoints",
    lm_model_path="acestep-5Hz-lm-4B",
    # backend="vllm",
    device="cuda"
)


#===create sample

def gen_music(caption):
    codes = dit_handler.convert_src_audio_to_codes("/content/AceStep/my_song/obj_wo3DlMOGwrbDjj7DisKw_8282269831_1b5d_2c8b_5207_87d27fc37904ebc25bf9d70191a0d20e (1).mp3")
    params_parse = understand_music(llm_handler, codes)
    params = GenerationParams(
        caption=params_parse.caption,
        bpm=params_parse.bpm,
        duration=params_parse.duration,
        audio_codes=codes,
        vocal_language=params_parse.language
    )
    # result = create_sample(
    #     llm_handler=llm_handler,
    #     query=caption,
    #     instrumental=False,
    #     vocal_language="ja",  
    #     temperature=0.85,
    # )

#     if result.success:
#         print(f"Caption: {result.caption}")
#         print(f"Lyrics: {result.lyrics}")
#         print(f"BPM: {result.bpm}")
#         print(f"Duration: {result.duration}s")
#         print(f"Key: {result.keyscale}")
#         print(f"Is Instrumental: {result.instrumental}")
        
#         # Use with generate_music
#         params = GenerationParams(
#             caption=result.caption,
#             lyrics=result.lyrics,
#             bpm=result.bpm,
#             duration=result.duration,
#             keyscale=result.keyscale,
#             vocal_language=result.language,
#         )
#     else:
#         print(f"Error: {result.error}")

    # ====



    # Configure generation settings
    config = GenerationConfig(
        batch_size=2,
        audio_format="mp3",
    )

    # Generate music
    result = generate_music(dit_handler, llm_handler, params, config, save_dir="/content/output")

    # Access results
    if result.success:
        for audio in result.audios:
            print(f"Generated: {audio['path']}")
            print(f"Key: {audio['key']}")
            print(f"Seed: {audio['params']['seed']}")
    else:
        print(f"Error: {result.error}")

# def audio_to_text(path):
prompt = 'An explosive fusion of J-rock and speed metal, driven by hyper-fast, harmonized guitar arpeggios and relentless double-bass drumming. A high-pitched, synthesized female vocal, characteristic of Vocaloid software, delivers a rapid-fire melody over the dense instrumentation. The arrangement is packed with technical guitar solos featuring shredding and sweep-picking, alongside chiptune-esque synth leads that add a digital, video-game-like texture. The track maintains an intense, high-energy pace throughout, with a brief, slightly more melodic bridge offering a momentary shift before launching into a final, climactic guitar and synth-driven outro.'
gen_music(prompt)