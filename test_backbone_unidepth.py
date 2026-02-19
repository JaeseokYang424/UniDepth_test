import sys
import os
import torch
import numpy as np
from PIL import Image

# --- [1] 경로 및 환경 변수 설정 ---
UNI_ENGINE_DIR = "/data2/hojun/UniDepth"
if UNI_ENGINE_DIR not in sys.path:
    sys.path.append(UNI_ENGINE_DIR)

os.environ["TORCH_HOME"] = "/data2/hojun/torch_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data2/hojun/torch_cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from unidepth.models import UniDepthV2
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')

# --- [2] 로컬 OXE 데이터 로드 ---
def get_local_oxe_image():
    local_path = "/data2/hojun/oxe/cmu_play_fusion"
    builder = tfds.builder_from_directory(local_path)
    ds = builder.as_dataset(split='train')
    for episode in ds.take(1):
        for step in episode['steps'].take(1):
            return Image.fromarray(step['observation']['image'].numpy()).convert("RGB")

# --- [3] 메인 실행부 ---
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 로드
    print("🚀 UniDepthV2 (ViT-L) 모델 로딩 중...")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(DEVICE)
    model.eval()

    # 2. 이미지 준비
    raw_img = get_local_oxe_image()
    # 원본 크기 기록
    orig_w, orig_h = raw_img.size
    
    # 텐서 변환 (C, H, W)
    img_tensor = torch.from_numpy(np.array(raw_img)).permute(2, 0, 1).to(DEVICE)
    # Batch 차원 추가 (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)

    # 3. Backbone 출력 추출
    print("🤖 Backbone 특징 추출 중...")
    with torch.no_grad():
        # UniDepthV2는 내부적으로 model.backbone을 가지고 있습니다.
        # model.infer() 내부에서 일어나는 전처리(정규화 등)를 수동으로 적용해줍니다.
        
        # 3-1. 모델의 내부 해상도 설정에 맞춰 리사이징 (보통 512 내외)
        # UniDepthV2의 경우 내부적으로 해상도를 조정하여 backbone에 넣습니다.
        # 여기서는 모델이 사용하는 실제 입력값을 가로채기 위해 forward_features 혹은 
        # backbone을 직접 호출하는 방식을 사용합니다.
        
        # 가공된 입력 (Normalize 등 포함)
        processed_input = model.preprocess(img_tensor)
        
        # Backbone 통과
        # ViT-L의 경우, 마지막 레이어의 특징 맵을 가져옵니다.
        backbone_features = model.backbone(processed_input)
        
        # UniDepthV2의 backbone 출력은 보통 리스트 형태거나 튜플일 수 있습니다.
        # 마지막 레이어의 텐서 크기를 확인합니다.
        if isinstance(backbone_features, (list, tuple)):
            feature_tensor = backbone_features[-1]
        else:
            feature_tensor = backbone_features

    # 4. 결과 비교 출력
    print("\n" + "="*50)
    print("📊 [Shape Transformation Result]")
    print("="*50)
    print(f"1. 원본 이미지 (PIL): {orig_w} x {orig_h} (RGB)")
    print(f"2. 모델 입력 텐서:    {img_tensor.shape} (Batch, Channel, H, W)")
    print(f"3. 백본 통과 후 (Latent): {feature_tensor.shape}")
    print("-"*50)
    
    # 변화 설명
    channels = feature_tensor.shape[1]
    h_feat = feature_tensor.shape[2]
    w_feat = feature_tensor.shape[3]
    
    print(f"💡 분석 결과:")
    print(f" - 채널 수: 3 (RGB) -> {channels} (고차원 특징)")
    print(f" - 공간 해상도: {orig_h}x{orig_w} -> {h_feat}x{w_feat} (패치 단위 압축)")
    print(f" - 총 특징점 수: {channels * h_feat * w_feat} 개")
    print("="*50)

if __name__ == "__main__":
    main()