import sys
import os
import torch

# 경로 설정
UNI_ENGINE_DIR = "/data2/hojun/UniDepth"
if UNI_ENGINE_DIR not in sys.path:
    sys.path.append(UNI_ENGINE_DIR)

os.environ["TORCH_HOME"] = "/data2/hojun/torch_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data2/hojun/torch_cache"

from unidepth.models import UniDepthV2

def main():
    # 가중치 없이 구조만 빠르게 보기 위해 device 설정
    print("🚀 UniDepthV2 구조 분석 중...")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", device='cpu')
    
    print("\n" + "="*60)
    print("ARCHITECTURE SUMMARY")
    print("="*60)
    
    # 모델의 전체 레이어 구조 출력
    print(model)
    
    print("\n" + "="*60)
    print("TOP-LEVEL MODULE NAMES")
    print("="*60)
    # 최상위 모듈 이름들만 따로 깔끔하게 출력
    for name, module in model.named_children():
        print(f"Module Name: {name:15} | Type: {type(module)}")

if __name__ == "__main__":
    main()
