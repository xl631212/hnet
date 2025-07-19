#!/usr/bin/env python3
"""
主评估脚本 - H-Net 场景图生成

使用方法:
    python evaluate.py --config configs/hnet_scene_graph.json --checkpoint outputs/checkpoint_best.pth
    python evaluate.py --config configs/hnet_scene_graph.json --checkpoint outputs/checkpoint_best.pth --split test
"""

import argparse
import json
import os
import sys
import torch
import warnings
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))  # 添加 hnet 根目录

from src.evaluate import Evaluator
from src.utils.logger import setup_logger, log_system_info

# 忽略一些常见的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="H-Net 场景图生成评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (JSON 格式)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    
    # 可选参数
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="评估的数据集分割"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="评估结果输出目录"
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
        help="评估设备"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="评估批大小（默认使用配置文件中的值）"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载器的工作进程数"
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="保存预测结果到文件"
    )
    
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="保存可视化结果"
    )
    
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=50,
        help="保存的可视化样本数量"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="运行推理速度基准测试"
    )
    
    parser.add_argument(
        "--benchmark_iterations",
        type=int,
        default=100,
        help="基准测试的迭代次数"
    )
    
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="预测置信度阈值"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    
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


def setup_device(device_arg):
    """设置评估设备"""
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


def setup_directories(output_dir):
    """创建必要的目录"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    return output_dir


def print_metrics_summary(metrics, logger):
    """打印评估指标摘要"""
    logger.info("=" * 60)
    logger.info("评估结果摘要")
    logger.info("=" * 60)
    
    # 对象检测指标
    if 'object_detection' in metrics:
        obj_metrics = metrics['object_detection']
        logger.info("对象检测:")
        logger.info(f"  mAP: {obj_metrics.get('mAP', 0.0):.4f}")
        logger.info(f"  精确率: {obj_metrics.get('precision', 0.0):.4f}")
        logger.info(f"  召回率: {obj_metrics.get('recall', 0.0):.4f}")
        logger.info(f"  F1分数: {obj_metrics.get('f1', 0.0):.4f}")
    
    # 属性预测指标
    if 'attribute_prediction' in metrics:
        attr_metrics = metrics['attribute_prediction']
        logger.info("属性预测:")
        logger.info(f"  mAP: {attr_metrics.get('mAP', 0.0):.4f}")
        logger.info(f"  精确率: {attr_metrics.get('precision', 0.0):.4f}")
        logger.info(f"  召回率: {attr_metrics.get('recall', 0.0):.4f}")
        logger.info(f"  F1分数: {attr_metrics.get('f1', 0.0):.4f}")
    
    # 关系预测指标
    if 'relationship_prediction' in metrics:
        rel_metrics = metrics['relationship_prediction']
        logger.info("关系预测:")
        logger.info(f"  mAP: {rel_metrics.get('mAP', 0.0):.4f}")
        logger.info(f"  精确率: {rel_metrics.get('precision', 0.0):.4f}")
        logger.info(f"  召回率: {rel_metrics.get('recall', 0.0):.4f}")
        logger.info(f"  F1分数: {rel_metrics.get('f1', 0.0):.4f}")
    
    # 场景图整体指标
    if 'scene_graph' in metrics:
        sg_metrics = metrics['scene_graph']
        logger.info("场景图整体:")
        logger.info(f"  图准确率: {sg_metrics.get('graph_accuracy', 0.0):.4f}")
        logger.info(f"  三元组召回率: {sg_metrics.get('triplet_recall', 0.0):.4f}")
    
    logger.info("=" * 60)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = setup_directories(args.output_dir)
    
    # 设置日志
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(
        name="evaluate",
        log_file=os.path.join(output_dir, "logs", "evaluate.log"),
        level=log_level
    )
    
    logger.info("=" * 80)
    logger.info("开始 H-Net 场景图生成评估")
    logger.info("=" * 80)
    
    # 记录系统信息
    if args.verbose:
        log_system_info(logger)
    
    # 记录命令行参数
    logger.info("命令行参数:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    try:
        # 检查检查点文件
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"检查点文件未找到: {args.checkpoint}")
        
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 更新配置
        config['data']['data_dir'] = args.data_dir
        config['evaluation'] = config.get('evaluation', {})
        config['evaluation']['output_dir'] = output_dir
        config['evaluation']['split'] = args.split
        config['evaluation']['save_predictions'] = args.save_predictions
        config['evaluation']['save_visualizations'] = args.save_visualizations
        config['evaluation']['num_visualizations'] = args.num_visualizations
        config['evaluation']['confidence_threshold'] = args.confidence_threshold
        config['evaluation']['num_workers'] = args.num_workers
        
        if args.batch_size is not None:
            config['evaluation']['batch_size'] = args.batch_size
        
        # 设置设备
        device = setup_device(args.device)
        config['evaluation']['device'] = device
        
        # 保存评估配置
        config_save_path = os.path.join(output_dir, "evaluation_config.json")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"保存评估配置到: {config_save_path}")
        
        # 创建评估器
        logger.info("初始化评估器...")
        evaluator = Evaluator(config)
        
        # 加载模型
        logger.info(f"加载模型检查点: {args.checkpoint}")
        evaluator.load_model(args.checkpoint)
        
        # 运行评估
        logger.info(f"开始评估 {args.split} 数据集...")
        metrics = evaluator.evaluate_dataset(args.split)
        
        # 打印结果摘要
        print_metrics_summary(metrics, logger)
        
        # 保存详细指标
        metrics_file = os.path.join(output_dir, "metrics", f"metrics_{args.split}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"详细指标保存到: {metrics_file}")
        
        # 运行基准测试（如果指定）
        if args.benchmark:
            logger.info("运行推理速度基准测试...")
            benchmark_results = evaluator.benchmark_inference(
                num_iterations=args.benchmark_iterations
            )
            
            logger.info("基准测试结果:")
            logger.info(f"  平均推理时间: {benchmark_results['avg_inference_time']:.4f} 秒")
            logger.info(f"  推理速度: {benchmark_results['fps']:.2f} FPS")
            logger.info(f"  内存使用: {benchmark_results['memory_usage']:.2f} MB")
            
            # 保存基准测试结果
            benchmark_file = os.path.join(output_dir, "metrics", "benchmark_results.json")
            with open(benchmark_file, 'w', encoding='utf-8') as f:
                json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
            logger.info(f"基准测试结果保存到: {benchmark_file}")
        
        logger.info("评估完成!")
        logger.info(f"结果保存在: {output_dir}")
        
        # 返回主要指标（用于脚本集成）
        if 'scene_graph' in metrics:
            main_metric = metrics['scene_graph'].get('graph_accuracy', 0.0)
            print(f"\n主要指标 (图准确率): {main_metric:.4f}")
            return main_metric
        
    except KeyboardInterrupt:
        logger.info("评估被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"评估过程中发生错误: {str(e)}")
        if args.debug:
            import traceback
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()