#!/usr/bin/env python3
"""
实时监控wandb数据的脚本
"""

import os
import glob
import json
import time
from datetime import datetime
import argparse

def find_latest_wandb_run():
    """找到最新的wandb运行目录"""
    wandb_dir = "wandb"
    if not os.path.exists(wandb_dir):
        return None
    
    # 查找所有offline运行
    offline_runs = glob.glob(os.path.join(wandb_dir, "offline-run-*"))
    if not offline_runs:
        return None
    
    # 按修改时间排序，返回最新的
    latest_run = max(offline_runs, key=os.path.getmtime)
    return latest_run

def monitor_wandb_data(run_dir, project_name="flux-batch-eval-8gpu"):
    """监控wandb数据"""
    print(f"🔍 监控wandb运行: {os.path.basename(run_dir)}")
    print(f"📊 项目名称: {project_name}")
    print(f"📁 本地目录: {run_dir}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 监控文件
    events_file = os.path.join(run_dir, "files", "wandb-events.jsonl")
    log_file = os.path.join(run_dir, "files", "output.log")
    
    if not os.path.exists(events_file):
        print(f"❌ 找不到事件文件: {events_file}")
        return
    
    # 记录已处理的行数
    processed_lines = 0
    
    try:
        while True:
            # 读取新的事件数据
            with open(events_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > processed_lines:
                # 处理新行
                for i in range(processed_lines, len(lines)):
                    try:
                        data = json.loads(lines[i].strip())
                        if 'step' in data:
                            print(f"📈 Step {data['step']}: {data.get('checkpoint', 'N/A')}")
                            # 显示reward数据
                            for key, value in data.items():
                                if key.endswith('_mean') and isinstance(value, (int, float)):
                                    print(f"   {key}: {value:.4f}")
                        elif 'timestamp' in data:
                            print(f"⏰ 时间戳: {data['timestamp']}")
                    except json.JSONDecodeError:
                        continue
                
                processed_lines = len(lines)
            
            # 检查日志文件
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_lines = f.readlines()
                    if log_lines:
                        last_line = log_lines[-1].strip()
                        if last_line and "checkpoint" in last_line.lower():
                            print(f"📝 日志: {last_line}")
            
            time.sleep(5)  # 每5秒检查一次
            
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")
        print(f"📊 总共处理了 {processed_lines} 行数据")

def main():
    parser = argparse.ArgumentParser(description="监控wandb数据")
    parser.add_argument("--run-dir", type=str, help="指定wandb运行目录")
    parser.add_argument("--project", type=str, default="flux-batch-eval-8gpu", help="项目名称")
    
    args = parser.parse_args()
    
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_wandb_run()
    
    if not run_dir:
        print("❌ 找不到wandb运行目录")
        print("请确保已经运行过评估脚本，或者手动指定目录")
        return
    
    monitor_wandb_data(run_dir, args.project)

if __name__ == "__main__":
    main()



