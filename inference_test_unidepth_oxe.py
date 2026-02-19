import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- [1] 경로 및 환경 변수 설정 ---
# 1. 실제 UniDepth 엔진 소스 코드가 위치한 곳 (수정 금지)
UNI_ENGINE_DIR = "/data2/hojun/UniDepth"
if UNI_ENGINE_DIR not in sys.path:
    sys.path.append(UNI_ENGINE_DIR)

# 2. 모델 가중치가 이미 저장된 캐시 경로 지정 (수정 금지)
os.environ["TORCH_HOME"] = "/data2/hojun/torch_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data2/hojun/torch_cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TF 로그 끄기

# --- [2] 라이브러리 임포트 ---
try:
    from unidepth.models import UniDepthV2
    print("✅ UniDepth 엔진 로드 성공!")
except ImportError:
    print(f"❌ UniDepth 엔진을 찾을 수 없습니다. 경로를 확인하세요: {UNI_ENGINE_DIR}")
    sys.exit(1)

import tensorflow as tf
import tensorflow_datasets as tfds
# TF가 GPU를 점유하지 않도록 설정
tf.config.set_visible_devices([], 'GPU')

# --- [3] 로컬 OXE 데이터 로드 함수 ---
def get_local_oxe_image():
    local_path = "/data2/hojun/oxe/cmu_play_fusion"
    print(f"📂 로컬 OXE 데이터셋 로딩 중: {local_path}")
    
    try:
        builder = tfds.builder_from_directory(local_path)
        ds = builder.as_dataset(split='train')
        
        # 첫 번째 에피소드의 첫 번째 스텝 이미지 가져오기
        for episode in ds.take(1):
            for step in episode['steps'].take(1):
                img_array = step['observation']['image'].numpy()
                return Image.fromarray(img_array).convert("RGB")
    except Exception as e:
        print(f"❌ 데이터셋 로드 에러: {e}")
        return None

# --- [4] 메인 추론 및 시각화 ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "/data2/hojun/UniDepth_test/oxe_unidepth_test_result.png"

    # 1. 모델 로드 (캐시 경로에서 불러옴)
    print("🚀 UniDepthV2 (ViT-L) 모델 로딩 중...")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(DEVICE)
    model.eval()

    # 2. 이미지 준비
    raw_img = get_local_oxe_image()
    if raw_img is None:
        return

    # UniDepth 입력 형식: [C, H, W] 텐서
    input_tensor = torch.from_numpy(np.array(raw_img)).permute(2, 0, 1).to(DEVICE)

    # 3. 추론 (Inference)
    print("🤖 UniDepth 추론 수행 중 (Metric Depth & Confidence)...")
    with torch.no_grad():
        # model.infer는 내부적으로 전처리를 모두 수행함
        predictions = model.infer(input_tensor)
        
        # 결과값 추출 및 CPU 이동
        depth = predictions["depth"].squeeze().cpu().numpy()
        confidence = predictions["confidence"].squeeze().cpu().numpy()

    # 4. 결과 시각화 및 저장
    print("📊 결과 시각화 생성 중...")
    plt.figure(figsize=(15, 5))

    # (1) 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.title(f"Original OXE Image\n({raw_img.size[0]}x{raw_img.size[1]})")
    plt.axis("off")

    # (2) Depth Map (실제 거리)
    plt.subplot(1, 3, 2)
    # magma 컬러맵: 가까우면 밝고 멀면 어두움
    im1 = plt.imshow(depth, cmap='magma')
    plt.title("Predicted Metric Depth")
    plt.colorbar(im1, label="Distance (meters)")
    plt.axis("off")

    # (3) Confidence Map (신뢰도)
    plt.subplot(1, 3, 3)
    im2 = plt.imshow(confidence, cmap='viridis')
    plt.title("Model Confidence")
    plt.colorbar(im2, label="Confidence Score")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ 성공! 결과가 저장되었습니다:\n👉 {SAVE_PATH}")

if __name__ == "__main__":
    main()
