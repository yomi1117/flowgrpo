# 科学上网
source /pfs/yangyuanming/set_proxy.sh
export WANDB_OFFLINE=true
export WANDB_DIR=../

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
# 8 GPU
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:pickscore_flux_8gpu
# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero3.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux_o3.py --config config/grpo.py:pickscore_flux_8gpu
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:clipscore_flux_8gpu


