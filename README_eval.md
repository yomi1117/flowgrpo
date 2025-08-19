# FLUX模型独立评估脚本使用说明

这个脚本可以独立评估FLUX模型和LoRA微调后的模型，使用PickScore等reward模型进行图像质量评估。

## 功能特性

- ✅ 支持原始FLUX模型评估
- ✅ 支持LoRA微调后的模型评估
- ✅ 集成多种reward模型：PickScore、Aesthetic Scorer、CLIP Scorer
- ✅ 支持txt和jsonl格式的数据集
- ✅ 批量生成和评估
- ✅ 结果保存和可视化
- ✅ 可选的wandb记录

## 安装依赖

```bash
pip install torch diffusers transformers peft pillow wandb tqdm numpy
```

## 使用方法

### 基本用法

```bash
python eval_standalone.py \
    --model_path /path/to/flux/model \
    --dataset_path dataset/pickscore/test.txt \
    --dataset_type txt \
    --output_dir ./eval_results
```

### 评估LoRA模型

```bash
python eval_standalone.py \
    --model_path /path/to/flux/model \
    --lora_path /path/to/lora/weights \
    --dataset_path dataset/pickscore/test.txt \
    --dataset_type txt \
    --output_dir ./eval_results_lora
```

### 完整参数示例

```bash
python eval_standalone.py \
    --model_path /pfs/yangyuanming/code2/models/FLUX.1-dev/model/black-forest-labs__FLUX.1-dev/main \
    --lora_path logs/pickscore/flux-group24-8gpu-full-parameter/checkpoints/checkpoint-300/lora \
    --dataset_path dataset/pickscore/test.txt \
    --dataset_type txt \
    --output_dir ./eval_results_flux_lora \
    --batch_size 4 \
    --num_inference_steps 28 \
    --guidance_scale 3.5 \
    --resolution 512 \
    --device cuda \
    --dtype float16 \
    --use_wandb \
    --max_samples 20
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | 必需 | FLUX模型路径 |
| `--lora_path` | str | None | LoRA权重路径（可选） |
| `--dataset_path` | str | 必需 | 测试数据集路径 |
| `--dataset_type` | str | "txt" | 数据集类型：txt或jsonl |
| `--output_dir` | str | "./eval_results" | 输出目录 |
| `--batch_size` | int | 4 | 批处理大小 |
| `--num_inference_steps` | int | 28 | 推理步数 |
| `--guidance_scale` | float | 3.5 | 引导尺度 |
| `--resolution` | int | 512 | 图像分辨率 |
| `--device` | str | "cuda" | 设备 |
| `--dtype` | str | "float32" | 数据类型 |
| `--use_wandb` | flag | False | 是否使用wandb记录 |
| `--max_samples` | int | 15 | 保存的最大样本数 |

## 数据集格式

### TXT格式
每行一个prompt：
```
Beautiful woman in skin tight body suite working out in the gym
a jung male cyborg with white hair sitting down on a throne in a dystopian world
beautiful asian girl with huge naturals in pool
```

### JSONL格式
每行一个JSON对象：
```json
{"prompt": "Beautiful woman in skin tight body suite working out in the gym", "metadata": {}}
{"prompt": "a jung male cyborg with white hair sitting down on a throne in a dystopian world", "metadata": {}}
```

## 输出结果

脚本会在指定的输出目录中创建以下文件：

```
eval_results/
├── images/
│   ├── sample_000.jpg
│   ├── sample_001.jpg
│   └── ...
└── eval_results.json
```

### eval_results.json 内容示例

```json
{
  "prompts": [
    "Beautiful woman in skin tight body suite working out in the gym",
    "a jung male cyborg with white hair sitting down on a throne in a dystopian world"
  ],
  "rewards": {
    "pickscore": [0.8234, 0.7567],
    "aesthetic": [0.7123, 0.6891],
    "clipscore": [0.7891, 0.7456]
  },
  "average_rewards": {
    "pickscore": 0.7901,
    "aesthetic": 0.7007,
    "clipscore": 0.7674
  }
}
```

## Reward模型说明

### PickScore
- 基于CLIP的图像-文本对齐度评分
- 分数范围：0-1（归一化后）
- 越高表示图像与文本描述越匹配

### Aesthetic Scorer
- 图像美学质量评分
- 分数范围：0-10
- 越高表示图像美学质量越好

### CLIP Scorer
- CLIP模型的图像-文本相似度评分
- 分数范围：0-1
- 越高表示图像与文本语义越匹配

## 使用建议

1. **内存优化**：如果GPU内存不足，可以减小batch_size或使用float16精度
2. **推理质量**：增加num_inference_steps可以提高生成质量，但会增加推理时间
3. **批量大小**：根据GPU内存调整batch_size，通常4-8比较合适
4. **数据集选择**：建议使用专门的测试集进行评估，避免过拟合

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 使用float16精度
   - 减少num_inference_steps

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确认模型文件完整性
   - 检查依赖库版本

3. **Reward模型加载失败**
   - 检查模型路径配置
   - 确认CUDA环境
   - 查看错误日志

### 调试模式

可以添加更多的print语句来调试：

```python
# 在FLUXEvaluator中添加
print(f"Text embeddings shape: {prompt_embeds.shape}")
print(f"Pooled embeddings shape: {pooled_prompt_embeds.shape}")
```

## 扩展功能

### 添加新的Reward模型

1. 在`RewardEvaluator`中添加新的scorer类型
2. 在`setup_reward_models`中初始化新模型
3. 在`evaluate_batch`中添加评估逻辑

### 自定义评估指标

可以修改`save_eval_results`函数来添加自定义的统计指标，如：
- 分数分布统计
- 分位数分析
- 相关性分析

## 性能优化建议

1. **使用混合精度**：设置`--dtype float16`可以显著减少内存使用
2. **批量处理**：适当增加batch_size可以提高GPU利用率
3. **并行评估**：可以考虑使用多进程来并行计算reward分数
4. **缓存机制**：对于重复的prompt，可以实现缓存机制避免重复计算





