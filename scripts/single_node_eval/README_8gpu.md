# FLUX模型评估脚本使用指南

## 📝 最新更新 (2025-08-18)

### ✅ 代码精简和优化
- **移除冗余导入**: 删除了不必要的ABSL flags、ml_collections等复杂配置系统
- **简化配置加载**: 使用直接的importlib动态导入，避免复杂的fallback机制
- **统一代码风格**: 三个版本（单卡、8卡、批量）使用一致的代码结构
- **减少代码行数**: 从500+行精简到300+行，保持功能完整

### 🔧 多卡重复问题修复
- **主进程控制**: 所有wandb上传、日志打印、结果保存只在主进程执行
- **避免重复输出**: 使用`accelerator.is_main_process`条件控制，防止8张卡重复打印
- **分布式优化**: 保持分布式推理性能，避免不必要的重复操作

## 🚀 功能特性

### 1. **单卡评估** (`yym_eval.py`)
- 单GPU FLUX模型评估
- 支持pickscore、clipscore等多种reward函数
- 自动保存评估结果和图像样本

### 2. **8卡分布式评估** (`yym_eval_8gpu.py`)
- 8-GPU分布式FLUX模型评估
- 优化的分布式策略，避免重复输出
- 支持多种reward函数组合

### 3. **批量LoRA评估** (`yym_eval_batch_8gpu.py`)
- 自动扫描并评估多个LoRA检查点
- 实时上传评分到wandb
- 生成step vs score曲线图

### 4. **灵活的Reward配置切换**
- **pickscore配置**: `config/grpo.py:pickscore_flux_8gpu`
  - 单一pickscore reward，评估速度快
  - 适合快速验证和调试
  
- **multi_score配置**: `config/grpo.py:multi_score_flux_8gpu`
  - 多种reward组合：pickscore + clipscore + imagereward
  - 权重分配：各占25%
  - 适合全面评估模型性能

- **自定义配置**: 可在shell脚本中灵活调整
  - 支持命令行参数选择
  - 可调整reward权重和组合

## 概述

本项目提供了单卡、八卡和批量评估三种FLUX模型推理方式，支持分布式评估和图像生成。

## 文件说明

### 单卡推理
- `yym_eval.sh` - 单卡推理shell脚本
- `yym_eval.py` - 单卡推理Python脚本

### 八卡推理
- `yym_eval_8gpu.sh` - 八卡推理shell脚本
- `yym_eval_8gpu.py` - 八卡推理Python脚本

### 批量评估（新功能）
- `yym_eval_batch_8gpu.sh` - 八卡批量评估shell脚本
- `yym_eval_batch_8gpu.py` - 八卡批量评估Python脚本
- `yym_eval_batch_test.sh` - 批量评估测试脚本（只评估前3个checkpoint）

### 工具文件
- `utils.py` - 工具函数文件（JSON序列化等）

### 配置文件
- `scripts/accelerate_configs/deepspeed_zero2.yaml` - 八卡加速配置

## 使用方法

### 1. 单卡推理

```bash
cd /pfs/yangyuanming/code2/flow_grpo
bash scripts/single_node/yym_eval.sh
```

### 2. 八卡推理

```bash
cd /pfs/yangyuanming/code2/flow_grpo
bash scripts/single_node/yym_eval_8gpu.sh
```

### 3. 批量评估多个LoRA检查点

#### 完整批量评估
```bash
cd /pfs/yangyuanming/code2/flow_grpo
bash scripts/single_node/yym_eval_batch_8gpu.sh
```

#### 测试批量评估（只评估前3个checkpoint）
```bash
cd /pfs/yangyuanming/code2/flow_grpo
bash scripts/single_node/yym_eval_batch_test.sh
```

#### 自定义参数批量评估
```bash
cd /pfs/yangyuanming/code2/flow_grpo

accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=8 --main_process_port 29501 \
    scripts/yym_eval_batch_8gpu.py \
    --config config/grpo.py:pickscore_flux_8gpu \
    --model_path /path/to/your/flux/model \
    --checkpoints_dir /path/to/checkpoints \
    --dataset /path/to/your/dataset \
    --prompt_fn general_ocr \
    --save_dir ./eval_results_batch \
    --eval_batch_size 16 \
    --eval_num_steps 28 \
    --guidance_scale 3.5 \
    --resolution 512 \
    --use_wandb \
    --max_checkpoints 10  # 可选：限制评估数量
```

## 批量评估功能详解

### 功能特性
1. **自动扫描checkpoints**: 自动发现并排序所有LoRA检查点
2. **批量加载LoRA权重**: 为每个checkpoint加载对应的LoRA权重
3. **分布式评估**: 使用8张GPU并行评估
4. **wandb曲线图**: 自动绘制step数vs评分的曲线图
5. **结果保存**: 保存详细的评估结果和step-reward数据
6. **⭐ 实时上传**: 每个checkpoint评估完成后立即上传评分到wandb，无需等待全部完成

### 支持的checkpoint格式
- 目录结构: `checkpoint-{step}/lora/`
- 自动识别step数: 420, 480, 540, 600, ..., 6360
- 每60步保存一次checkpoint

### wandb可视化
- **横轴**: training step数
- **纵轴**: 各种reward评分（pickscore等）
- **图表类型**: 曲线图 + 数据表格
- **项目名称**: `flux-batch-eval-8gpu`
- **实时上传**: ⭐ **每个checkpoint评估完成后立即上传评分到wandb**

## 参数说明

### 通用参数
- `--config`: 配置文件路径和配置函数名
- `--model_path`: 基础FLUX模型路径
- `--dataset`: 数据集路径
- `--prompt_fn`: Prompt函数类型 (geneval 或 general_ocr)
- `--save_dir`: 结果保存目录
- `--eval_batch_size`: 评估批次大小（建议设置为GPU数量）
- `--eval_num_steps`: 推理步数
- `--guidance_scale`: 分类器引导强度
- `--resolution`: 图像分辨率
- `--use_wandb`: 是否使用wandb记录

### 批量评估特有参数
- `--checkpoints_dir`: LoRA检查点目录路径
- `--max_checkpoints`: 最大评估checkpoint数量（用于测试）

## 输出结果

### 批量评估输出
- **`batch_eval_results.json`**: 每个checkpoint的详细评估结果
- **`step_rewards.json`**: step数vs reward评分的结构化数据
- **wandb图表**: step数vs评分的曲线图和数据表格

### 数据结构示例
```json
{
  "step": 600,
  "checkpoint_path": "/path/to/checkpoint-600",
  "reward_summary": {
    "pickscore": {
      "mean": 0.8480,
      "std": 0.1234,
      "min": 0.5000,
      "max": 1.0000,
      "count": 2048
    }
  },
  "timestamp": "2025-08-17T21:32:25"
}
```

## 配置说明

### accelerate配置
- 使用 `scripts/accelerate_configs/deepspeed_zero2.yaml`
- 支持8个GPU进程
- 使用DeepSpeed ZeRO-2优化

### 模型配置
- 支持FLUX.1-dev模型
- 自动处理fp16 variant不可用的情况
- 自动将模型组件移动到正确的GPU设备
- 支持LoRA权重动态加载

## 注意事项

1. **环境要求**: 确保激活了正确的conda环境 (`flow_grpo`)
2. **GPU数量**: 确保有8张可用的GPU
3. **内存要求**: 每张GPU需要足够的显存来加载模型和LoRA权重
4. **端口配置**: 使用29501端口，确保端口未被占用
5. **数据集**: 确保数据集路径正确且包含test.txt或test_metadata.jsonl文件
6. **checkpoints**: 确保checkpoints目录存在且包含有效的LoRA权重

## 故障排除

### 常见问题

1. **CUDA设备不匹配**: 脚本会自动处理设备分配
2. **配置加载失败**: 会自动使用备用配置
3. **模型加载失败**: 会自动尝试不同的variant
4. **分布式同步问题**: 使用accelerator.gather()确保结果正确收集
5. **JSON序列化错误**: 使用安全的JSON保存函数处理numpy数据类型
6. **LoRA加载失败**: 检查checkpoint目录结构和lora子目录

### 已修复的问题

#### wandb初始化问题
- **问题**: `wandb.errors.errors.Error: You must call wandb.init() before wandb.log()`
- **原因**: 在使用wandb.log()之前没有正确初始化wandb
- **修复**: 自动检查和初始化wandb，支持离线模式

#### JSON序列化问题
- **问题**: `TypeError: Object of type float32 is not JSON serializable`
- **原因**: numpy的float32类型不能直接序列化为JSON
- **修复**: 使用安全的JSON保存函数，自动转换所有numpy数据类型

#### Reward函数参数问题
- **问题**: `ClipScorer.__init__() got an unexpected keyword argument 'dtype'`
- **原因**: `ClipScorer`类只接受`device`参数，不接受`dtype`参数
- **修复**: 修改`clip_score()`函数，移除错误的`dtype`参数
- **状态**: ✅ 已修复，clip_score和multi_score函数现在可以正常工作

### 调试建议

1. 先运行测试版本验证基本功能
2. 检查GPU使用情况: `nvidia-smi`
3. 检查进程状态: `ps aux | grep python`
4. 查看日志输出了解具体错误
5. 验证checkpoints目录结构是否正确

## 性能优化

1. **批次大小**: 建议设置为GPU数量以获得最佳性能
2. **Worker数量**: 已优化为4个worker避免资源竞争
3. **内存管理**: 使用torch.float16减少显存占用
4. **分布式策略**: 使用DeepSpeed ZeRO-2优化多GPU通信
5. **LoRA加载**: 为每个checkpoint重新加载pipeline确保权重正确应用

## 使用建议

### 1. **首次使用**: 先运行测试版本（`yym_eval_batch_test.sh`）
### 2. **生产环境**: 使用完整版本（`yym_eval_batch_8gpu.sh`）
### 3. **监控进度**: 通过日志和wandb实时监控评估进度
### 4. **结果分析**: 使用生成的JSON文件和wandb图表分析模型性能趋势

### 5. **Reward配置切换示例**

#### 快速切换配置
```bash
# 使用pickscore单一reward
bash scripts/single_node/yym_eval_batch_8gpu.sh

# 使用multi_score多种reward组合
bash scripts/single_node/yym_eval_batch_8gpu_flexible.sh

# 使用高级脚本，支持命令行参数
bash scripts/single_node/yym_eval_batch_8gpu_advanced.sh -c multi_score -b 32
```

#### 查看帮助信息
```bash
bash scripts/single_node/yym_eval_batch_8gpu_advanced.sh --help
```

#### 支持的配置选项
- `pickscore`: 单一pickscore reward，评估速度快
- `multi_score`: 多种reward组合，全面评估
- `custom`: 自定义reward权重配置

### 6. **常见配置组合**
- **快速验证**: `pickscore_flux_8gpu` + `eval_batch_size=16`
- **全面评估**: `multi_score_flux_8gpu` + `eval_batch_size=32`
- **调试模式**: `pickscore_flux_8gpu` + `--max_checkpoints 3`