#!/bin/bash
export PYTHONPATH=/pfs/yangyuanming/code2/flow_grpo:$PYTHONPATH

echo "使用简化的配置方式运行评估脚本..."

# 直接运行Python脚本，不使用accelerate launch来避免配置复杂性
cd /pfs/yangyuanming/code2/flow_grpo

python scripts/yym_eval.py \
    --config config/grpo.py:multi_score_flux_8gpu \
    --model_path /pfs/yangyuanming/code2/models/FLUX.1-dev/model/black-forest-labs__FLUX.1-dev/main \
    --dataset /pfs/yangyuanming/code2/flow_grpo/dataset/pickscore \
    --prompt_fn general_ocr \
    --save_dir ./eval_results \
    --eval_batch_size 1 \
    --eval_num_steps 28 \
    --guidance_scale 3.5 \
    --resolution 512 \
    --use_wandb