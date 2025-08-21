#!/usr/bin/env python3
"""
FLUX模型评估脚本
基于train_flux.py的评估部分，专门用于模型评估
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from collections import defaultdict
import datetime
from concurrent import futures
import time
import json
from accelerate import Accelerator
from diffusers import FluxPipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse
import logging

from utils import safe_json_dump

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

def eval_flux(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, reward_fn, executor, autocast):
    """评估FLUX模型的主函数"""
    all_rewards = defaultdict(list)
    i = 0
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
        )
        
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
        
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
        i += 1
        if i > 10:
            break
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            
            for key, value in all_rewards.items():
                print(f"{key}: {value.shape}, mean: {np.mean(value[value != -10]):.4f}")
            
            if config.use_wandb:
                try:
                    if not wandb.run:
                        logger.warning("wandb未初始化，跳过记录")
                    else:
                        wandb.log(
                            {
                                "eval_images": [
                                    wandb.Image(
                                        os.path.join(tmpdir, f"{idx}.jpg"),
                                        caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                                    )
                                    for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                                ],
                                **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                            },
                            step=0,
                        )
                        logger.info("结果已记录到wandb")
                except Exception as e:
                    logger.warning(f"wandb记录失败: {e}")
            
            eval_results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model_path": config.pretrained.model,
                "dataset": config.dataset,
                "prompt_fn": config.prompt_fn,
                "eval_num_steps": config.sample.eval_num_steps,
                "guidance_scale": config.sample.guidance_scale,
                "resolution": config.resolution,
                "rewards_summary": {key: {
                    "mean": float(np.mean(value[value != -10]).item()),
                    "std": float(np.std(value[value != -10]).item()),
                    "min": float(np.min(value[value != -10]).item()),
                    "max": float(np.max(value[value != -10]).item()),
                    "count": int(np.sum(value != -10).item())
                } for key, value in all_rewards.items()},
                "sample_prompts": sampled_prompts,
                "sample_rewards": sampled_rewards
            }
            
            results_file = os.path.join(config.save_dir, "eval_results.json")
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            safe_json_dump(eval_results, results_file)
            
            print(f"评估结果已保存到: {results_file}")
            print(f"图像样本已保存到: {tmpdir}")

def main():
    parser = argparse.ArgumentParser(description="FLUX模型评估脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="本地FLUX模型路径")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--prompt_fn", type=str, default="geneval", choices=["geneval", "general_ocr"], help="Prompt函数类型")
    parser.add_argument("--save_dir", type=str, default="./eval_results", help="结果保存目录")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="评估批次大小")
    parser.add_argument("--eval_num_steps", type=int, default=28, help="评估推理步数")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="分类器引导强度")
    parser.add_argument("--resolution", type=int, default=512, help="图像分辨率")
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config_path = args.config
        if ':' in config_path:
            file_path, func_name = config_path.split(':', 1)
        else:
            file_path, func_name = config_path, 'get_config'
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        config_func = getattr(config_module, func_name)
        config = config_func()
        
        logger.info(f"成功加载配置文件: {args.config}")
        
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return 1
    
    # 覆盖配置参数
    config.pretrained.model = args.model_path
    config.dataset = args.dataset
    config.prompt_fn = args.prompt_fn
    config.save_dir = args.save_dir
    config.sample.test_batch_size = args.eval_batch_size
    config.sample.eval_num_steps = args.eval_num_steps
    config.sample.guidance_scale = args.guidance_scale
    config.resolution = args.resolution
    config.use_wandb = args.use_wandb
    
    # 初始化wandb
    if config.use_wandb:
        try:
            os.environ["WANDB_MODE"] = "offline"
            
            wandb.init(
                project="flux-eval-single",
                name=f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model_path": config.pretrained.model,
                    "dataset": config.dataset,
                    "prompt_fn": config.prompt_fn,
                    "eval_num_steps": config.sample.eval_num_steps,
                    "guidance_scale": config.sample.guidance_scale,
                    "resolution": config.resolution,
                    "num_gpus": 1
                },
                settings=wandb.Settings(init_timeout=30, mode="offline")
            )
            logger.info("wandb离线模式初始化成功")
        except Exception as e:
            logger.warning(f"wandb初始化失败: {e}")
            config.use_wandb = False
    
    # 初始化accelerator
    accelerator = Accelerator()
    
    # 加载FLUX pipeline
    logger.info(f"正在加载FLUX模型: {config.pretrained.model}")
    try:
        pipeline = FluxPipeline.from_pretrained(
            config.pretrained.model,
            torch_dtype=torch.float16,
            variant="fp16"
        )
    except ValueError as e:
        if "variant=fp16" in str(e):
            logger.warning("fp16 variant不可用，尝试加载默认版本...")
            pipeline = FluxPipeline.from_pretrained(
                config.pretrained.model,
                torch_dtype=torch.float16
            )
        else:
            raise e
    
    # 移动到设备
    device = accelerator.device
    pipeline = pipeline.to(device)
    
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
    
    logger.info("模型组件已移动到正确设备并设置为评估模式")
    
    # 准备组件
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
    
    try:
        reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    except Exception as e:
        logger.warning(f"无法加载reward函数: {e}")
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
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=4,
    )
    
    test_dataloader = accelerator.prepare(test_dataloader)
    executor = futures.ThreadPoolExecutor(max_workers=4)
    autocast = accelerator.autocast
    
    # 开始评估
    logger.info("开始评估...")
    eval_flux(
        pipeline, test_dataloader, text_encoders, tokenizers, config, 
        accelerator, reward_fn, executor, autocast
    )
    
    logger.info("评估完成!")

if __name__ == "__main__":
    main()