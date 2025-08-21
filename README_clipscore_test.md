# CLIPScore 测试脚本

## 简介
这是一个精简的CLIPScore测试脚本，用于验证clipscore功能是否正常工作。

## 文件说明
- `minimal_test_clipscore.py` - 最精简的CLIPScore测试脚本

## 使用方法

### 1. 直接运行测试
```bash
python minimal_test_clipscore.py
```

### 2. 预期输出
```
CLIPScore测试结果:
  a red square: 0.9219
  a green square: 0.8635
✓ 测试成功！
```

## 功能说明

### 测试内容
- 创建两个测试图像：红色和绿色
- 使用对应的文本描述计算CLIPScore
- 验证分数是否在合理范围内 (0-1)

### 核心代码逻辑
```python
# 初始化ClipScorer
scorer = ClipScorer(device=device)

# 创建测试图像
red_img = Image.new('RGB', (512, 512), (255, 0, 0))
green_img = Image.new('RGB', (512, 512), (0, 255, 0))

# 转换为tensor (NCHW格式)
images = [np.array(red_img), np.array(green_img)]
images_tensor = torch.tensor(np.array(images), dtype=torch.uint8)
images_tensor = images_tensor.permute(0, 3, 1, 2) / 255.0

# 计算分数
prompts = ["a red square", "a green square"]
scores = scorer(images_tensor, prompts)
```

## 依赖要求
- Python 3.6+
- PyTorch
- PIL (Pillow)
- numpy
- transformers (用于CLIP模型)

## 注意事项
1. 确保`flow_grpo`模块在正确路径下
2. 脚本会自动检测并使用可用的设备 (CUDA/CPU)
3. 测试图像尺寸为512x512，可根据需要调整
4. 分数范围通常在0-1之间，越高表示图像与文本越匹配

## 故障排除
如果遇到导入错误，请检查：
- `flow_grpo`模块路径是否正确
- 是否安装了所有必要的依赖包
- CLIP模型文件是否存在

