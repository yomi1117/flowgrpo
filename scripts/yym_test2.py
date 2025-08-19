"""
简单的Flux模型加载与推理脚本
支持批量prompt处理和带时间戳的输出文件夹
"""

import torch
from diffusers import FluxPipeline
from peft import PeftModel
import datetime
import os
import json

def load_flux_pipeline(model_path, device="cuda", dtype=torch.float16):
    """
    加载FluxPipeline模型
    """
    pipeline = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    # 引入lora
    lora_path = "/pfs/yangyuanming/code2/flow_grpo/logs/pickscore/flux-group24-8gpu/checkpoints/checkpoint-6000/lora"
    pipeline.transformer = PeftModel.from_pretrained(
        pipeline.transformer,
        lora_path
    )
    pipeline.safety_checker = None
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    pipeline = pipeline.to(device)
    pipeline.transformer.eval()
    
    return pipeline

def infer_one_prompt(pipeline, prompt, height=512, width=512, num_inference_steps=30, guidance_scale=7.5):
    """
    对单个prompt进行推理，返回生成的图片
    """
    with torch.no_grad():
        result = pipeline(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pil"
        )
    return result.images[0]

def create_output_directory(base_output_path):
    """
    创建带时间戳的输出目录
    """
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir, output_file = os.path.split(base_output_path)
    name, ext = os.path.splitext(output_file)
    
    # 创建带时间戳的文件夹
    timestamp_dir = os.path.join(output_dir, f"{name}_{now_str}")
    os.makedirs(timestamp_dir, exist_ok=True)
    
    return timestamp_dir, now_str

def save_prompt_list(prompts, output_dir):
    """
    保存prompt列表到JSON文件
    """
    prompt_file = os.path.join(output_dir, "prompts.json")
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"Prompt列表已保存到: {prompt_file}")

if __name__ == "__main__":
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/pfs/yangyuanming/code2/models/FLUX.1-dev/model/black-forest-labs__FLUX.1-dev/main", type=str, help="Flux模型路径")
    parser.add_argument("--prompt", type=str, help="单个输入的prompt")
    parser.add_argument("--prompt_file", type=str, default="/pfs/yangyuanming/code2/flow_grpo/scripts/promptlist.txt", help="包含prompt列表的文本文件路径，每行一个prompt")
    parser.add_argument("--prompt_list", nargs='+', help="直接在命令行输入多个prompt")
    parser.add_argument("--output", default="/pfs/yangyuanming/code2/flow_grpo/scripts/output.png", type=str, help="输出图片路径")
    parser.add_argument("--device", default="cuda", type=str, help="推理设备")
    parser.add_argument("--height", type=int, default=512, help="图片高度")
    parser.add_argument("--width", type=int, default=512, help="图片宽度")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="引导比例")
    args = parser.parse_args()

    # 收集所有prompts
    prompts = []
    
    if args.prompt:
        prompts.append(args.prompt)
    
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                file_prompts = [line.strip() for line in f if line.strip()]
                prompts.extend(file_prompts)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {args.prompt_file}")
            exit(1)
    
    if args.prompt_list:
        prompts.extend(args.prompt_list)
    
    if not prompts:
        print("错误: 请提供至少一个prompt (使用 --prompt, --prompt_file, 或 --prompt_list)")
        parser.print_help()
        exit(1)
    
    print(f"总共需要处理 {len(prompts)} 个prompt")
    
    # 创建输出目录
    output_dir, timestamp = create_output_directory(args.output)
    print(f"输出目录已创建: {output_dir}")
    
    # 保存prompt列表
    save_prompt_list(prompts, output_dir)
    
    print("加载Flux模型中...")
    pipeline = load_flux_pipeline(args.model_path, device=args.device)
    print("模型加载完成。")
    
    # 处理每个prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n处理第 {i}/{len(prompts)} 个prompt: {prompt[:50]}...")
        
        try:
            image = infer_one_prompt(
                pipeline, 
                prompt, 
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            )
            
            # 保存图片，使用prompt的前几个字符作为文件名
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            image_filename = f"{i:03d}_{safe_prompt}.png"
            image_path = os.path.join(output_dir, image_filename)
            
            image.save(image_path)
            print(f"图片已保存到: {image_path}")
            
        except Exception as e:
            print(f"处理prompt '{prompt[:30]}...' 时出错: {str(e)}")
            continue
    
    print(f"\n所有处理完成！结果保存在: {output_dir}")
    print(f"时间戳: {timestamp}")
