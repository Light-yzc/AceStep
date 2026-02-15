import torch
import torchaudio
from acestep.handler import AceStepHandler

def test_vae_reconstruction(audio_path):
    # 1. 初始化 Handler (它会帮你加载 VAE 模型)
    handler = AceStepHandler()
    print("正在初始化 Handler...")
    # 注意：config_path 只影响 DiT 模型，VAE 是通用的，所以这里也能跑通
    handler.initialize_service(
    project_root="/content/ACE-Step-1.5",
    config_path="acestep-v15-turbo",
    device="cuda"
)
    
    # 2. 读取音频并预处理 (重采样到 48k, 转双声道)
    print(f"正在处理音频: {audio_path}")
    audio_tensor = handler.process_src_audio(audio_path)
    if audio_tensor is None:
        print("错误: 无法读取音频")
        return

    # 3. 编码 (Encode): Audio -> Latents
    print("正在编码 (Encode)...")
    # 将音频移到 GPU
    audio_tensor = audio_tensor.to(handler.device).unsqueeze(0) # [1, 2, T]
    
    with torch.no_grad():
        with handler._load_model_context("vae"):
            # 调用底层 VAE Encode
            target_dtype = handler.vae.dtype
            
            # 将输入移动到设备，并强转为正确的 dtype
            print(audio_tensor.shape)
            input_tensor = audio_tensor.to(handler.device, dtype=target_dtype)
            latents = handler.vae.encode(input_tensor).latent_dist.sample()
            
    print(f"Latents Shape: {latents.shape}")

    # 4. 解码 (Decode): Latents -> Audio (Reconstruction)
    print("正在解码 (Decode)...")
    with torch.no_grad():
        with handler._load_model_context("vae"):
            # 调用底层 VAE Decode
            recon_audio = handler.vae.decode(latents).sample

    print(f"Reconstructed Audio Shape: {recon_audio.shape}")

    # 5. 保存结果对比
    recon_audio = recon_audio.squeeze(0).cpu()
    torchaudio.save("reconstructed_test.wav", recon_audio, 48000)
    print("已保存重建后的音频到 reconstructed_test.wav")

# 使用示例 (请替换为你自己的音频文件路径)
test_vae_reconstruction("/content/ACE-Step-1.5/my_audio/obj_wo3DlMOGwrbDjj7DisKw_8282269831_1b5d_2c8b_5207_87d27fc37904ebc25bf9d70191a0d20e.mp3") 