#!/bin/bash
export PYTHONPATH=/pfs/yangyuanming/code2/flow_grpo:$PYTHONPATH
export WANDB_OFFLINE=true
echo "启动八卡FLUX模型评估..."

# 使用accelerate launch启动八卡推理
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=8 --main_process_port 29501 \
    scripts/yym_eval_8gpu.py \
    --config config/grpo.py:pickscore_flux_8gpu \
    --model_path /pfs/yangyuanming/code2/models/FLUX.1-dev/model/black-forest-labs__FLUX.1-dev/main \
    --dataset /pfs/yangyuanming/code2/flow_grpo/dataset/pickscore \
    --prompt_fn general_ocr \
    --save_dir ./eval_results_8gpu \
    --eval_batch_size 16 \
    --eval_num_steps 28 \
    --guidance_scale 3.5 \
    --resolution 512 \
    --use_wandb
