import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- [1] 경로 및 환경 변수 설정 ---
UNI_ENGINE_DIR = "/data2/hojun/UniDepth"
if UNI_ENGINE_DIR not in sys.path:
    sys.path.append(UNI_ENGINE_DIR)

os.environ["TORCH_HOME"] = "/data2/hojun/torch_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data2/hojun/torch_cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- [2] 라이브러리 임포트 ---
try:    
    from unidepth.models import UniDepthV2
    print("✅ UniDepth 엔진 로드 성공!")
except ImportError:
    print(f"❌ UniDepth 엔진을 찾을 수 없습니다. 경로를 확인하세요: {UNI_ENGINE_DIR}")
    sys.exit(1)

import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')

# --- [3] 로컬 OXE 데이터 로드 함수 ---
def get_local_oxe_image():
    local_path = "/data2/hojun/oxe/cmu_play_fusion"
    try:
        builder = tfds.builder_from_directory(local_path)
        ds = builder.as_dataset(split='train')
        for episode in ds.take(1):
            for step in episode['steps'].take(1):
                img_array = step['observation']['image'].numpy()
                return Image.fromarray(img_array).convert("RGB")
    except Exception as e:
        print(f"❌ 데이터셋 로드 에러: {e}")
        return None

# --- [4] 메인 추론 및 차원 확인 ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "/data2/hojun/UniDepth_test/oxe_unidepth_test_result.png"

    # 1. 모델 로드
    print("🚀 UniDepthV2 (ViT-L) 모델 로딩 중...")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(DEVICE)
    model.eval()

    # 2. 이미지 준비
    raw_img = get_local_oxe_image()
    if raw_img is None: return

    # ---------------------------------------------------------
    # [추가] Backbone(Latent Vector) 차원 확인 루틴
    # ---------------------------------------------------------
    print("\n🔍 [Backbone 분석 시작]")
    # 인코더는 14의 배수 입력을 원하므로 내부와 동일하게 518로 리사이징
    input_size = (518, 518)
    img_for_backbone = raw_img.resize(input_size, Image.BILINEAR)
    
    # 텐서 변환 및 배치 차원 추가
    test_tensor = torch.from_numpy(np.array(img_for_backbone)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    test_tensor = test_tensor / 255.0 # 기본적인 스케일링
    
    with torch.no_grad():
        # 아까 구조 확인 결과에 따라 'pixel_encoder' 직접 호출
        # n=1: 마지막 블록, reshape=True: [B, C, H, W] 형태 변환
        latent_vector = model.pixel_encoder.get_intermediate_layers(test_tensor, n=1, reshape=True)[0]
    
    print(f"   - 원본 이미지 크기: {raw_img.size}")
    print(f"   - 인코더 입력 크기: {test_tensor.shape}")
    print(f"   - 최종 Latent Vector 차원: {latent_vector.shape}")
    print("====================================================\n")
    # ---------------------------------------------------------

    # 3. 기존 추론 (Inference)
    input_tensor = torch.from_numpy(np.array(raw_img)).permute(2, 0, 1).to(DEVICE)
    print("🤖 UniDepth 전체 추론 수행 중...")
    with torch.no_grad():
        predictions = model.infer(input_tensor)
        depth = predictions["depth"].squeeze().cpu().numpy()
        confidence = predictions["confidence"].squeeze().cpu().numpy()

    # 4. 결과 시각화 및 저장
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(raw_img); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 3, 2); im1 = plt.imshow(depth, cmap='magma'); plt.title("Depth"); plt.colorbar(im1); plt.axis("off")
    plt.subplot(1, 3, 3); im2 = plt.imshow(confidence, cmap='viridis'); plt.title("Confidence"); plt.colorbar(im2); plt.axis("off")
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ 시각화 완료: {SAVE_PATH}")

if __name__ == "__main__":
    main()
