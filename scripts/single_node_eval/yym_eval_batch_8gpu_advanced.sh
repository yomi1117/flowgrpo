#!/bin/bash

# 高级批量评估脚本 - 支持命令行参数选择reward配置

# 默认配置
DEFAULT_CONFIG="pickscore"
DEFAULT_BATCH_SIZE=16

# 显示使用说明
show_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --config CONFIG     选择reward配置 (pickscore|multi_score|custom)"
    echo "  -b, --batch-size SIZE   设置评估批次大小 (默认: $DEFAULT_BATCH_SIZE)"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "可用的reward配置:"
    echo "  pickscore     - 单一pickscore reward (推荐用于快速评估)"
    echo "  multi_score   - 多种reward组合 (pickscore + clipscore + imagereward)"
    echo "  custom        - 自定义reward权重"
    echo ""
    echo "示例:"
    echo "  $0 -c multi_score -b 32"
    echo "  $0 --config pickscore"
}

# 解析命令行参数
CONFIG=$DEFAULT_CONFIG
BATCH_SIZE=$DEFAULT_BATCH_SIZE

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 验证配置参数
case $CONFIG in
    "pickscore")
        CONFIG_FILE="config/grpo.py:pickscore_flux_8gpu"
        echo "✅ 使用pickscore单一reward配置"
        ;;
    "multi_score")
        CONFIG_FILE="config/grpo.py:multi_score_flux_8gpu"
        echo "✅ 使用multi_score多种reward配置"
        ;;
    "custom")
        CONFIG_FILE="config/grpo.py:pickscore_flux_8gpu"
        echo "✅ 使用自定义reward配置"
        ;;
    *)
        echo "❌ 无效的配置: $CONFIG"
        echo "支持的配置: pickscore, multi_score, custom"
        exit 1
        ;;
esac

# 设置环境变量
export PYTHONPATH=/pfs/yangyuanming/code2/flow_grpo:$PYTHONPATH
export WANDB_OFFLINE=true

echo "📋 评估配置:"
echo "  Reward配置: $CONFIG"
echo "  配置文件: $CONFIG_FILE"
echo "  批次大小: $BATCH_SIZE"
echo "  设备数量: 8 GPU"
echo ""

# 执行评估
echo "🚀 开始执行评估..."
accelerate launch \
    --num_processes=8 \
    --multi_gpu \
    --mixed_precision=fp16 \
    --use_deepspeed \
    --deepspeed_config_file=config/deepspeed_config.json \
    scripts/yym_eval_batch_8gpu.py \
    --config $CONFIG_FILE \
    --model_path /pfs/yangyuanming/code2/models/FLUX.1-dev/model/black-forest-labs__FLUX.1-dev/main \
    --checkpoints_dir /pfs/yangyuanming/code2/flow_grpo/logs/pickscore/flux-group24-8gpu/checkpoints \
    --dataset /pfs/yangyuanming/code2/flow_grpo/dataset/pickscore \
    --prompt_fn general_ocr \
    --save_dir ./eval_results_batch_8gpu_${CONFIG} \
    --use_wandb \
    --eval_batch_size $BATCH_SIZE \
    --eval_num_steps 28 \
    --guidance_scale 3.5 \
    --resolution 512

if [ $? -eq 0 ]; then
    echo "🎉 评估完成！结果保存在: ./eval_results_batch_8gpu_${CONFIG}"
else
    echo "❌ 评估失败！请检查错误信息"
    exit 1
fi

