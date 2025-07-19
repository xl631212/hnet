#!/usr/bin/env python3
"""
主训练脚本 - H-Net 场景图生成

使用方法:
    python train.py --config configs/hnet_scene_graph.json
    python train.py --config configs/hnet_scene_graph.json --resume outputs/checkpoint_latest.pth
"""

import argparse
import json
import logging
import os
import sys
import torch
import warnings
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))  # 添加 hnet 根目录

from src.train import Trainer
from src.utils.logger import setup_logger, log_system_info

# 忽略一些常见的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="H-Net 场景图生成训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (JSON 格式)"
    )
    
    # 可选参数
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练的路径"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据集目录路径"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载器的工作进程数"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="干运行模式，只验证配置不实际训练"
    )
    
    # wandb参数已移除
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件格式错误: {e}")


def validate_config(config):
    """验证配置文件的完整性"""
    required_sections = ['model', 'training', 'data']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必需的部分: {section}")
    
    # 验证模型配置
    model_config = config['model']
    required_model_keys = [
        'arch_layout', 'd_model', 'image_size', 'patch_size',
        'num_object_classes', 'num_predicate_classes', 'num_attribute_classes'
    ]
    
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"模型配置缺少必需的键: {key}")
    
    # 验证训练配置
    training_config = config['training']
    required_training_keys = ['num_epochs', 'batch_size', 'learning_rate']
    
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"训练配置缺少必需的键: {key}")
    
    # 验证数据配置
    data_config = config['data']
    required_data_keys = ['dataset_name']
    
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"数据配置缺少必需的键: {key}")


def setup_device(device_arg):
    """设置训练设备"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"自动选择设备: {device} (GPU: {torch.cuda.get_device_name()})")
        else:
            device = "cpu"
            print(f"自动选择设备: {device}")
    else:
        device = device_arg
        if device == "cuda" and not torch.cuda.is_available():
            print("警告: 指定了 CUDA 但不可用，回退到 CPU")
            device = "cpu"
    
    return device


def setup_directories(output_dir, data_dir):
    """创建必要的目录"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")
        print("请先运行数据准备脚本: python src/data/prepare_data.py")
    
    return output_dir, data_dir


def set_random_seed(seed):
    """设置随机种子以确保可重现性"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDA 操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建输出目录
    output_dir, data_dir = setup_directories(args.output_dir, args.data_dir)
    
    # 设置日志
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(
        name="train",
        log_dir=os.path.join(output_dir, "logs"),
        level=log_level
    )
    
    logger.info("=" * 80)
    logger.info("开始 H-Net 场景图生成训练")
    logger.info("=" * 80)
    
    # 记录系统信息
    log_system_info(logger)
    
    # 记录命令行参数
    logger.info("命令行参数:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    try:
        # 加载和验证配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        validate_config(config)
        logger.info("配置文件验证通过")
        
        # 更新配置中的路径
        config['data']['data_dir'] = data_dir
        config['training']['output_dir'] = output_dir
        config['training']['num_workers'] = args.num_workers
        
        # 设置日志配置
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['log_dir'] = os.path.join(output_dir, "logs")
        
        # wandb配置已移除
        config['logging'] = config.get('logging', {})
        config['logging']['use_wandb'] = False
        
        # 设置设备
        device = setup_device(args.device)
        config['training']['device'] = device
        
        # 保存使用的配置
        config_save_path = os.path.join(output_dir, "config_used.json")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"保存使用的配置到: {config_save_path}")
        
        if args.dry_run:
            logger.info("干运行模式: 配置验证完成，退出")
            return
        
        # 创建训练器
        logger.info("初始化训练器...")
        trainer = Trainer(config)
        
        # 从检查点恢复（如果指定）
        if args.resume:
            logger.info(f"从检查点恢复训练: {args.resume}")
            trainer.resume_from_checkpoint(args.resume)
            logger.info("从检查点恢复训练")
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        logger.info("训练完成!")
        logger.info(f"模型和日志保存在: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        if args.debug:
            import traceback
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()