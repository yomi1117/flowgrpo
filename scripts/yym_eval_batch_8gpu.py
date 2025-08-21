#!/usr/bin/env python3
"""
批量评估多个LoRA检查点的脚本
支持wandb绘制step数vs评分的曲线图
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from collections import defaultdict
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
import argparse
import logging
import glob
import re

# 导入工具函数
from utils import safe_json_dump

# 设置基本日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds

def get_checkpoint_steps(checkpoints_dir):
    """
    从checkpoints目录中提取所有step数
    """
    checkpoint_pattern = os.path.join(checkpoints_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    steps = []
    for checkpoint_dir in checkpoint_dirs:
        # 提取step数
        match = re.search(r'checkpoint-(\d+)$', checkpoint_dir)
        if match:
            step = int(match.group(1))
            steps.append((step, checkpoint_dir))
    
    # 按step数排序
    steps.sort(key=lambda x: x[0])
    return steps

def eval_single_checkpoint(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, reward_fn, executor, autocast, checkpoint_step):
    """
    评估单个checkpoint
    """
    all_rewards = defaultdict(list)
    
    # 设置分布式评估的进度条
    progress_bar = tqdm(
        test_dataloader,
        desc=f"Eval checkpoint-{checkpoint_step}: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    )
    i = 0
    for test_batch in progress_bar:
        prompts, prompt_metadata = test_batch
        
        # 计算文本嵌入
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        
        # 生成图像
        with autocast():
            with torch.no_grad():
                images, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )
        
        # 计算reward（分布式）
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)  # yield to make sure reward computation starts
        rewards, reward_metadata = rewards.result()

        # 收集所有GPU的结果
        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
        i += 1
        if i > 2:
            break
    
    # 合并所有GPU的结果
    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    
    # 计算平均reward
    reward_summary = {}
    for key, value in all_rewards.items():
        valid_values = value[value != -10]
        if len(valid_values) > 0:
            reward_summary[key] = {
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "count": int(len(valid_values))
            }
        else:
            reward_summary[key] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0
            }
    
    return reward_summary, all_rewards

def main():
    parser = argparse.ArgumentParser(description="批量评估多个LoRA检查点")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="基础FLUX模型路径")
    parser.add_argument("--checkpoints_dir", type=str, default="/pfs/yangyuanming/code2/flow_grpo/logs/pickscore/flux-group24-8gpu/checkpoints", help="LoRA检查点目录")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--prompt_fn", type=str, default="geneval", choices=["geneval", "general_ocr"], help="Prompt函数类型")
    parser.add_argument("--save_dir", type=str, default="./eval_results_batch", help="结果保存目录")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="评估批次大小")
    parser.add_argument("--eval_num_steps", type=int, default=28, help="评估推理步数")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="分类器引导强度")
    parser.add_argument("--resolution", type=int, default=512, help="图像分辨率")
    parser.add_argument("--max_checkpoints", type=int, default=None, help="最大评估checkpoint数量（用于测试）")
    
    args = parser.parse_args()
    
    # 正确加载配置
    try:
        logger.info("尝试直接导入配置文件...")
        config_path = args.config
        if ':' in config_path:
            file_path, func_name = config_path.split(':', 1)
        else:
            file_path, func_name = config_path, 'get_config'
        
        # 动态导入配置文件
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # 调用配置函数
        config_func = getattr(config_module, func_name)
        config = config_func()
        
        logger.info(f"成功直接导入配置文件: {args.config}")
        logger.info(f"配置对象类型: {type(config)}")
        
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return 1
    
    # 覆盖配置中的关键参数
    config.pretrained.model = args.model_path
    config.dataset = args.dataset
    config.prompt_fn = args.prompt_fn
    config.save_dir = args.save_dir
    config.sample.test_batch_size = args.eval_batch_size
    config.sample.eval_num_steps = args.eval_num_steps
    config.sample.guidance_scale = args.guidance_scale
    config.resolution = args.resolution
    config.use_wandb = args.use_wandb
    
    # 设置随机种子
    set_seed(42)
    
    # 初始化accelerator（支持多GPU）
    accelerator = Accelerator()
    
    # 只在主进程中初始化wandb和打印信息
    if accelerator.is_main_process:
        logger.info(f"分布式信息: 进程数={accelerator.num_processes}, 设备={accelerator.device}")
        
        # 初始化wandb
        if config.use_wandb:
            try:
                os.environ["WANDB_MODE"] = "offline"
                wandb.init(
                    project="flux-batch-eval-8gpu",
                    name=f"batch_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "model_path": config.pretrained.model,
                        "dataset": config.dataset,
                        "prompt_fn": config.prompt_fn,
                        "eval_num_steps": config.sample.eval_num_steps,
                        "guidance_scale": config.sample.guidance_scale,
                        "resolution": config.resolution,
                        "num_gpus": 8,
                        "checkpoints_dir": args.checkpoints_dir,
                        "real_time_upload": True,
                        "upload_frequency": "per_checkpoint"
                    },
                    settings=wandb.Settings(init_timeout=30, mode="offline")
                )
                logger.info("✓ wandb离线模式初始化成功，实时上传功能已启用")
            except Exception as e:
                logger.warning(f"wandb初始化失败: {e}")
                config.use_wandb = False
    
    # 获取所有checkpoint
    if accelerator.is_main_process:
        logger.info(f"扫描checkpoints目录: {args.checkpoints_dir}")
    
    checkpoint_steps = get_checkpoint_steps(args.checkpoints_dir)
    
    if not checkpoint_steps:
        if accelerator.is_main_process:
            logger.error("未找到任何checkpoint")
        return 1
    
    # 限制checkpoint数量（用于测试）
    if args.max_checkpoints:
        checkpoint_steps = checkpoint_steps[:args.max_checkpoints]
        if accelerator.is_main_process:
            logger.info(f"限制为前{args.max_checkpoints}个checkpoint进行测试")
    
    if accelerator.is_main_process:
        logger.info(f"找到 {len(checkpoint_steps)} 个checkpoint:")
        for step, path in checkpoint_steps:
            logger.info(f"  checkpoint-{step}: {path}")
    
    # 加载基础FLUX pipeline
    if accelerator.is_main_process:
        logger.info(f"正在加载基础FLUX模型: {config.pretrained.model}")
    
    try:
        pipeline = FluxPipeline.from_pretrained(
            config.pretrained.model,
            torch_dtype=torch.float16,
            variant="fp16"
        )
    except ValueError as e:
        if "variant=fp16" in str(e):
            if accelerator.is_main_process:
                logger.warning("fp16 variant不可用，尝试加载默认版本...")
            pipeline = FluxPipeline.from_pretrained(
                config.pretrained.model,
                torch_dtype=torch.float16
            )
        else:
            raise e
    
    # 确保所有组件都在正确的设备上
    device = accelerator.device
    pipeline = pipeline.to(device)
    
    # 确保所有子组件都在正确的设备上
    if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
        pipeline.transformer = pipeline.transformer.to(device)
        pipeline.transformer.eval()
    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        pipeline.text_encoder = pipeline.text_encoder.to(device)
        pipeline.text_encoder.eval()
    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2 = pipeline.text_encoder_2.to(device)
        pipeline.text_encoder_2.eval()
    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        pipeline.vae = pipeline.vae.to(device)
        pipeline.vae.eval()
    
    if accelerator.is_main_process:
        logger.info("模型组件已移动到正确设备并设置为评估模式")
    
    # 获取text encoders和tokenizers
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
    
    # 准备reward函数
    try:
        reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        #  print("reward_fn", reward_fn)  
    except Exception as e:
        if accelerator.is_main_process:
            logger.warning(f"无法加载reward函数: {e}")
        # 创建一个简单的默认reward函数
        reward_fn = lambda images, prompts, metadata, only_strict=False: (
            {'pickscore': torch.ones(len(images), device=accelerator.device)}, 
            {'pickscore': torch.ones(len(images), device=accelerator.device)}
        )
    
    # 准备数据集
    if config.prompt_fn == "general_ocr":
        test_dataset = TextPromptDataset(config.dataset, 'test')
    elif config.prompt_fn == "geneval":
        test_dataset = GenevalPromptDataset(config.dataset, 'test')
    else:
        raise NotImplementedError(f"不支持的prompt_fn: {config.prompt_fn}")
    
    # 创建分布式数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=4,
    )
    
    # 准备accelerator
    test_dataloader = accelerator.prepare(test_dataloader)
    
    # 创建异步执行器
    executor = futures.ThreadPoolExecutor(max_workers=4)
    
    # 设置autocast
    autocast = accelerator.autocast
    
    # 开始批量评估
    if accelerator.is_main_process:
        logger.info("开始批量评估多个checkpoint...")
    
    all_results = []
    step_rewards = defaultdict(list)
    
    for step, checkpoint_path in tqdm(checkpoint_steps, desc="评估checkpoint"):
        if accelerator.is_main_process:
            logger.info(f"\n评估 checkpoint-{step}...")
        
        try:
            # 加载LoRA权重
            lora_path = os.path.join(checkpoint_path, "lora")
            if os.path.exists(lora_path):
                if accelerator.is_main_process:
                    logger.info(f"加载LoRA权重: {lora_path}")
                
                # 重新加载pipeline以应用新的LoRA权重
                pipeline = FluxPipeline.from_pretrained(
                    config.pretrained.model,
                    torch_dtype=torch.float16
                )
                pipeline = pipeline.to(device)
                
                # 应用LoRA权重
                pipeline.transformer = PeftModel.from_pretrained(
                    pipeline.transformer,
                    lora_path
                )
                
                # 设置评估模式
                pipeline.transformer.eval()
                pipeline.transformer = pipeline.transformer.to(device)
                
                # 评估当前checkpoint
                reward_summary, all_rewards = eval_single_checkpoint(
                    pipeline, 
                    test_dataloader, 
                    text_encoders, 
                    tokenizers, 
                    config, 
                    accelerator, 
                    reward_fn, 
                    executor, 
                    autocast,
                    step
                )
                
                # 记录结果
                print("reward_summary", reward_summary)
                result = {
                    "step": step,
                    "checkpoint_path": checkpoint_path,
                    "reward_summary": reward_summary,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                all_results.append(result)
                
                # 只在主进程中上传wandb和打印日志
                if accelerator.is_main_process:
                    # 立即上传到wandb
                    if config.use_wandb:
                        upload_data = {
                            "step": step,
                            "checkpoint": f"checkpoint-{step}",
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        
                        for key, summary in reward_summary.items():
                            upload_data[f"{key}_mean"] = reward_summary[key]["mean"]
                            upload_data[f"{key}_std"] = reward_summary[key]["std"]
                            upload_data[f"{key}_min"] = reward_summary[key]["min"]
                            upload_data[f"{key}_max"] = reward_summary[key]["max"]
                            upload_data[f"{key}_count"] = reward_summary[key]["count"]
                        
                        try:
                            wandb.log(upload_data)
                            logger.info(f"✓ checkpoint-{step} 评分已成功上传到wandb")
                        except Exception as e:
                            logger.error(f"上传 checkpoint-{step} 评分到wandb失败: {e}")
                    
                    # 保存到本地数据结构
                    for key in reward_summary.keys():
                        step_rewards[key].append({
                            "step": step,
                            "mean": reward_summary[key]["mean"],
                            "std": reward_summary[key]["std"],
                            "min": reward_summary[key]["min"],
                            "max": reward_summary[key]["max"],
                            "count": reward_summary[key]["count"]
                        })
                    
                    logger.info(f"checkpoint-{step} 评估完成")
                    for key, summary in reward_summary.items():
                        logger.info(f"  {key}: mean={summary['mean']:.4f}, std={summary['std']:.4f}")
                
            else:
                if accelerator.is_main_process:
                    logger.warning(f"LoRA路径不存在: {lora_path}")
                
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"评估 checkpoint-{step} 时出错: {e}")
            continue
    
    # 只在主进程中保存结果和创建wandb表格
    if accelerator.is_main_process:
        logger.info("\n保存批量评估结果...")
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 保存详细结果
        results_file = os.path.join(config.save_dir, "batch_eval_results.json")
        safe_json_dump(all_results, results_file)
        logger.info(f"详细结果已保存到: {results_file}")
        
        # 保存step vs reward数据
        step_rewards_file = os.path.join(config.save_dir, "step_rewards.json")
        safe_json_dump(dict(step_rewards), step_rewards_file)
        logger.info(f"Step vs Reward数据已保存到: {step_rewards_file}")
        
        # 在wandb中绘制曲线图和创建数据表格
        if config.use_wandb:
            logger.info("在wandb中创建最终的数据表格和统计信息...")
            
            # 为每个reward类型创建详细的表格
            for reward_type, data in step_rewards.items():
                if data:
                    # 创建详细的step vs reward表格
                    table_data = []
                    for d in data:
                        table_data.append([
                            d["step"],           # step数
                            d["mean"],           # 平均值
                            d["std"],            # 标准差
                            d["min"],            # 最小值
                            d["max"],            # 最大值
                            d["count"]           # 样本数量
                        ])
                    
                    # 创建wandb表格
                    table = wandb.Table(
                        data=table_data, 
                        columns=["step", "mean", "std", "min", "max", "count"]
                    )
                    
                    # 上传表格和统计信息
                    wandb.log({
                        f"{reward_type}_detailed_table": table,
                        f"{reward_type}_total_checkpoints": len(data),
                        f"{reward_type}_final_step": max(d["step"] for d in data),
                        f"{reward_type}_best_score": max(d["mean"] for d in data),
                        f"{reward_type}_best_step": max(d["step"] for d in data if d["mean"] == max(d["mean"] for d in data))
                    })
                    
                    logger.info(f"✓ 已创建 {reward_type} 的详细数据表格")
                    logger.info(f"  总checkpoint数: {len(data)}")
                    logger.info(f"  最佳评分: {max(d['mean'] for d in data):.4f}")
                    logger.info(f"  最佳step: {max(d['step'] for d in data if d['mean'] == max(d['mean'] for d in data))}")
            
            # 创建总体统计信息
            total_checkpoints = len(all_results)
            if total_checkpoints > 0:
                wandb.log({
                    "total_checkpoints_evaluated": total_checkpoints,
                    "evaluation_completed": True,
                    "final_timestamp": datetime.datetime.now().isoformat()
                })
                logger.info(f"✓ 总体统计信息已上传到wandb")
                logger.info(f"  总共评估了 {total_checkpoints} 个checkpoint")
        
        logger.info("批量评估完成！")
    
    logger.info("批量评估完成!")

if __name__ == "__main__":
    main()
