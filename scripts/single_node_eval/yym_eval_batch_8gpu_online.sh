#!/bin/bash

# 在线模式的批量评估脚本 - 确保数据实时同步到wandb

# 设置环境变量
export PYTHONPATH=/pfs/yangyuanming/code2/flow_grpo:$PYTHONPATH
export WANDB_MODE=online  # 启用在线模式

echo "🌐 启用wandb在线模式，数据将实时同步到云端"
echo "📊 项目名称: flux-batch-eval-8gpu"
echo "🔗 查看地址: https://wandb.ai/[你的用户名]/flux-batch-eval-8gpu"

# 执行评估
accelerate launch \
    --num_processes=8 \
    --multi_gpu \
    --mixed_precision=fp16 \
    --use_deepspeed \
    --deepspeed_config_file=config/deepspeed_config.json \
    scripts/yym_eval_batch_8gpu.py \
    --config config/grpo.py:multi_score_flux_8gpu \
    --model_path /pfs/yangyuanming/code2/models/FLUX.1-dev/model/black-forest-labs__FLUX.1-dev/main \
    --checkpoints_dir /pfs/yangyuanming/code2/flow_grpo/logs/pickscore/flux-group24-8gpu/checkpoints \
    --dataset /pfs/yangyuanming/code2/flow_grpo/dataset/pickscore \
    --prompt_fn general_ocr \
    --save_dir ./eval_results_batch_8gpu_online \
    --use_wandb \
    --eval_batch_size 16 \
    --eval_num_steps 28 \
    --guidance_scale 3.5 \
    --resolution 512

echo "✅ 评估完成！"
echo "📊 查看结果: https://wandb.ai/[你的用户名]/flux-batch-eval-8gpu"

