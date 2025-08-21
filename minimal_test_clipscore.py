#!/usr/bin/env python3
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('flow_grpo')

def test_clipscore():
    try:
        from flow_grpo.clip_scorer import ClipScorer
        
        # 初始化
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scorer = ClipScorer(device=device)
        
        # 创建测试图像
        red_img = Image.new('RGB', (512, 512), (255, 0, 0))
        green_img = Image.new('RGB', (512, 512), (0, 255, 0))
        
        # 转换为tensor
        images = [np.array(red_img), np.array(green_img)]
        images_tensor = torch.tensor(np.array(images), dtype=torch.uint8)
        images_tensor = images_tensor.permute(0, 3, 1, 2) / 255.0
        
        # 计算分数
        prompts = ["a red square", "a green square"]
        scores = scorer(images_tensor, prompts)
        
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        print("CLIPScore测试结果:")
        for prompt, score in zip(prompts, scores):
            print(f"  {prompt}: {score:.4f}")
        
        print("✓ 测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_clipscore()
