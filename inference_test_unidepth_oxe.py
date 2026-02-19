import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 경로 설정 (필수) ---
# 파이썬에게 UniDepth 소스 코드 위치를 직접 알려줍니다.
UNI_DIR = "/data2/hojun/UniDepth"
if UNI_DIR not in sys.path:
    sys.path.append(UNI_DIR)

# 가중치 저장소 위치 고정
os.environ["TORCH_HOME"] = "/data2/hojun/torch_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data2/hojun/torch_cache"

# 이제 import가 가능해집니다.
from unidepth.models import UniDepthV2

# --- TFDS 로컬 데이터 로드 ---
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')

def get_local_oxe_image():
    local_path = "/data2/hojun/oxe/cmu_play_fusion"
    builder = tfds.builder_from_directory(local_path)
    ds = builder.as_dataset(split='train')
    for episode in ds.take(1):
        for step in episode['steps'].take(1):
            return Image.fromarray(step['observation']['image'].numpy()).convert("RGB")

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드 (캐시된 파일을 자동으로 찾습니다)
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(DEVICE)
    model.eval()

    # 이미지 준비
    raw_img = get_local_oxe_image()
    input_tensor = torch.from_numpy(np.array(raw_img)).permute(2, 0, 1).to(DEVICE)

    # 추론
    with torch.no_grad():
        predictions = model.infer(input_tensor)
        depth = predictions["depth"].squeeze().cpu().numpy()
        conf = predictions["confidence"].squeeze().cpu().numpy()

    # 시각화 저장
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(raw_img); plt.title("Original OXE"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(depth, cmap='magma'); plt.title("Depth (m)"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(conf, cmap='viridis'); plt.title("Confidence"); plt.axis("off")
    
    save_path = "/data2/hojun/oxe_unidepth_test.png"
    plt.savefig(save_path)
    print(f"✅ 결과 저장 완료: {save_path}")

if __name__ == "__main__":
    main()
