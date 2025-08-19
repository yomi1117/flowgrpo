#!/bin/bash

# 灵活的批量评估脚本 - 支持多种reward配置

# 设置环境变量
export PYTHONPATH=/pfs/yangyuanming/code2/flow_grpo:$PYTHONPATH
export WANDB_OFFLINE=true

# 选择reward配置
# 选项1: pickscore (单一reward)
REWARD_CONFIG_1="config/grpo.py:pickscore_flux_8gpu"

# 选项2: multi_score (多种reward组合)
REWARD_CONFIG_2="config/grpo.py:multi_score_flux_8gpu"

# 选项3: 自定义reward权重
REWARD_CONFIG_3="config/grpo.py:pickscore_flux_8gpu"

# 当前使用的配置 (修改这里来切换)
CURRENT_CONFIG=$REWARD_CONFIG_2

echo "使用reward配置: $CURRENT_CONFIG"

# 根据配置选择不同的参数
case $CURRENT_CONFIG in
    *"pickscore_flux_8gpu"*)
        echo "使用pickscore单一reward配置"
        EVAL_BATCH_SIZE=16
        ;;
    *"multi_score_flux_8gpu"*)
        echo "使用multi_score多种reward配置"
        EVAL_BATCH_SIZE=16
        ;;
    *)
        echo "使用默认配置"
        EVAL_BATCH_SIZE=16
        ;;
esac

# 执行评估
accelerate launch \
    --num_processes=8 \
    --multi_gpu \
    --mixed_precision=fp16 \
    --use_deepspeed \
    --deepspeed_config_file=config/deepspeed_config.json \
    scripts/yym_eval_batch_8gpu.py \
    --config $CURRENT_CONFIG \
    --model_path /pfs/yangyuanming/code2/models/FLUX.1-dev/model/black-forest-labs__FLUX.1-dev/main \
    --checkpoints_dir /pfs/yangyuanming/code2/flow_grpo/logs/pickscore/flux-group24-8gpu/checkpoints \
    --dataset /pfs/yangyuanming/code2/flow_grpo/dataset/pickscore \
    --prompt_fn general_ocr \
    --save_dir ./eval_results_batch_8gpu \
    --use_wandb \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --eval_num_steps 28 \
    --guidance_scale 3.5 \
    --resolution 512

echo "评估完成！"
