#!/usr/bin/env python3
"""
工具函数文件
"""

import numpy as np
import torch
import json
from typing import Any, Dict, List, Union

def convert_to_serializable(obj: Any) -> Any:
    """
    将对象转换为JSON可序列化的格式
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj.item())
    elif isinstance(obj, np.floating):
        return float(obj.item())
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def safe_json_dump(obj: Any, file_path: str, **kwargs) -> None:
    """
    安全地保存对象到JSON文件，自动处理序列化问题
    
    Args:
        obj: 要保存的对象
        file_path: 文件路径
        **kwargs: 传递给json.dump的其他参数
    """
    # 转换为可序列化格式
    serializable_obj = convert_to_serializable(obj)
    
    # 保存到文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_obj, f, indent=2, ensure_ascii=False, **kwargs)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    安全地将对象转换为JSON字符串
    
    Args:
        obj: 要转换的对象
        **kwargs: 传递给json.dumps的其他参数
        
    Returns:
        JSON字符串
    """
    serializable_obj = convert_to_serializable(obj)
    return json.dumps(serializable_obj, indent=2, ensure_ascii=False, **kwargs)


