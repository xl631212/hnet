#!/usr/bin/env python3
"""
数据准备主脚本 - H-Net 场景图生成

使用方法:
    python prepare_data.py --data_dir ./data --download --preprocess
    python prepare_data.py --data_dir ./data --preprocess --min_objects 3 --min_relationships 2
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))  # 添加 hnet 根目录

from src.data.prepare_data import (
    download_visual_genome,
    build_vocabulary,
    filter_scene_graphs,
    create_data_splits
)
from src.utils.logger import setup_logger

# 忽略一些常见的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="H-Net 场景图生成数据准备脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据集目录路径"
    )
    
    # 操作选项
    parser.add_argument(
        "--download",
        action="store_true",
        help="下载 Visual Genome 数据集"
    )
    
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="预处理数据集"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证数据完整性"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理（覆盖现有文件）"
    )
    
    # 预处理参数
    parser.add_argument(
        "--min_objects",
        type=int,
        default=3,
        help="每张图像的最小对象数量"
    )
    
    parser.add_argument(
        "--min_relationships",
        type=int,
        default=2,
        help="每张图像的最小关系数量"
    )
    
    parser.add_argument(
        "--min_object_freq",
        type=int,
        default=100,
        help="对象类别的最小频次"
    )
    
    parser.add_argument(
        "--min_predicate_freq",
        type=int,
        default=50,
        help="谓词的最小频次"
    )
    
    parser.add_argument(
        "--min_attribute_freq",
        type=int,
        default=50,
        help="属性的最小频次"
    )
    
    parser.add_argument(
        "--max_objects_per_image",
        type=int,
        default=30,
        help="每张图像的最大对象数量"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="训练集比例"
    )
    
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="验证集比例"
    )
    
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="测试集比例"
    )
    
    # 其他选项
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行处理的工作进程数"
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
        "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    
    return parser.parse_args()


def setup_directories(data_dir):
    """创建必要的目录"""
    directories = [
        data_dir,
        os.path.join(data_dir, "raw"),
        os.path.join(data_dir, "processed"),
        os.path.join(data_dir, "vocabularies"),
        os.path.join(data_dir, "splits"),
        os.path.join(data_dir, "logs")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return data_dir


def set_random_seed(seed):
    """设置随机种子以确保可重现性"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)


def check_data_splits_ratio(train_ratio, val_ratio, test_ratio):
    """检查数据分割比例"""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"数据分割比例之和必须等于 1.0，当前为 {total_ratio:.6f}"
        )


def print_progress_summary(step, total_steps, description, logger):
    """打印进度摘要"""
    progress = (step / total_steps) * 100
    logger.info(f"[{step}/{total_steps}] ({progress:.1f}%) {description}")


def verify_data_integrity(data_dir, logger):
    """验证数据完整性"""
    logger.info("验证数据完整性...")
    
    # 检查必要的文件是否存在
    required_files = [
        os.path.join(data_dir, "vocabularies", "vocab.json"),
        os.path.join(data_dir, "processed", "filtered_scene_graphs.json"),
        os.path.join(data_dir, "splits", "splits.json")
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"缺少必要文件: {file_path}")
            return False
        logger.info(f"✓ 文件存在: {file_path}")
    
    # 检查文件内容是否有效
    try:
        # 检查词汇表
        vocab_file = os.path.join(data_dir, "vocabularies", "vocab.json")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        required_keys = ['object_classes', 'predicate_classes', 'attribute_classes']
        for key in required_keys:
            if key not in vocab:
                logger.error(f"词汇表缺少键: {key}")
                return False
        
        logger.info(f"✓ 词汇表有效: {len(vocab['object_classes'])} 对象, {len(vocab['predicate_classes'])} 谓词, {len(vocab['attribute_classes'])} 属性")
        
        # 检查场景图
        scene_graphs_file = os.path.join(data_dir, "processed", "filtered_scene_graphs.json")
        with open(scene_graphs_file, 'r', encoding='utf-8') as f:
            scene_graphs = json.load(f)
        
        logger.info(f"✓ 场景图有效: {len(scene_graphs)} 张图像")
        
        # 检查数据分割
        splits_file = os.path.join(data_dir, "splits", "splits.json")
        with open(splits_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        
        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            if split not in splits:
                logger.error(f"数据分割缺少: {split}")
                return False
        
        logger.info(f"✓ 数据分割有效: 训练集 {len(splits['train'])}, 验证集 {len(splits['val'])}, 测试集 {len(splits['test'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"验证过程中发生错误: {str(e)}")
        return False


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 验证参数
    if not (args.download or args.preprocess or args.verify):
        print("错误: 必须指定至少一个操作 (--download, --preprocess, --verify)")
        sys.exit(1)
    
    check_data_splits_ratio(args.train_ratio, args.val_ratio, args.test_ratio)
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建数据目录
    data_dir = setup_directories(args.data_dir)
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(
        name="prepare_data",
        log_dir=os.path.join(data_dir, "logs"),
        level=log_level
    )
    
    logger.info("=" * 80)
    logger.info("开始 H-Net 场景图生成数据准备")
    logger.info("=" * 80)
    
    # 记录命令行参数
    if args.verbose:
        logger.info("命令行参数:")
        for key, value in vars(args).items():
            logger.info(f"  {key}: {value}")
    
    try:
        total_steps = sum([args.download, args.preprocess, args.verify])
        current_step = 0
        
        # 步骤 1: 下载数据
        if args.download:
            current_step += 1
            print_progress_summary(
                current_step, total_steps, "下载 Visual Genome 数据集", logger
            )
            
            raw_data_dir = os.path.join(data_dir, "raw")
            
            if not args.force and os.path.exists(os.path.join(raw_data_dir, "image_data.json")):
                logger.info("数据集已存在，跳过下载（使用 --force 强制重新下载）")
            else:
                logger.info("开始下载 Visual Genome 数据集...")
                download_visual_genome(raw_data_dir)
                logger.info("数据集下载完成")
        
        # 步骤 2: 预处理数据
        if args.preprocess:
            current_step += 1
            print_progress_summary(
                current_step, total_steps, "预处理数据集", logger
            )
            
            raw_data_dir = os.path.join(data_dir, "raw")
            processed_data_dir = os.path.join(data_dir, "processed")
            vocab_dir = os.path.join(data_dir, "vocabularies")
            splits_dir = os.path.join(data_dir, "splits")
            
            # 检查原始数据是否存在
            required_files = [
                "image_data.json",
                "objects.json",
                "relationships.json",
                "attributes.json"
            ]
            
            for file_name in required_files:
                file_path = os.path.join(raw_data_dir, file_name)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"原始数据文件未找到: {file_path}\n"
                        "请先运行 --download 下载数据集"
                    )
            
            # 子步骤 2.1: 构建词汇表
            logger.info("构建词汇表...")
            scene_graphs_path = os.path.join(raw_data_dir, "scene_graphs.json")
            vocabularies = build_vocabulary(
                scene_graphs_path=scene_graphs_path,
                min_object_freq=args.min_object_freq,
                min_predicate_freq=args.min_predicate_freq,
                min_attribute_freq=args.min_attribute_freq
            )
            
            # 保存词汇表
            vocab_file = os.path.join(vocab_dir, "vocab.json")
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocabularies, f, indent=2, ensure_ascii=False)
            
            logger.info(f"词汇表构建完成:")
            logger.info(f"  对象类别: {len(vocabularies['object_classes'])}")
            logger.info(f"  谓词类别: {len(vocabularies['predicate_classes'])}")
            logger.info(f"  属性类别: {len(vocabularies['attribute_classes'])}")
            
            # 子步骤 2.2: 过滤场景图
            logger.info("过滤场景图...")
            filtered_scene_graphs_path = os.path.join(processed_data_dir, "filtered_scene_graphs.json")
            filter_scene_graphs(
                scene_graphs_path=scene_graphs_path,
                vocab=vocabularies,
                output_path=filtered_scene_graphs_path,
                min_objects=args.min_objects,
                min_relationships=args.min_relationships
            )
            
            # 读取过滤后的数据
            with open(filtered_scene_graphs_path, 'r', encoding='utf-8') as f:
                filtered_data = json.load(f)
            
            logger.info(f"场景图过滤完成:")
            logger.info(f"  保留图像数量: {len(filtered_data)}")
            
            # 子步骤 2.3: 创建数据分割
            logger.info("创建数据分割...")
            splits = create_data_splits(
                scene_graphs_path=filtered_scene_graphs_path,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed
            )
            
            # 保存数据分割
            splits_file = os.path.join(splits_dir, "splits.json")
            with open(splits_file, 'w', encoding='utf-8') as f:
                json.dump(splits, f, indent=2, ensure_ascii=False)
            
            logger.info(f"数据分割完成:")
            logger.info(f"  训练集: {len(splits['train'])} 张图像")
            logger.info(f"  验证集: {len(splits['val'])} 张图像")
            logger.info(f"  测试集: {len(splits['test'])} 张图像")
            
            logger.info("数据预处理完成")
        
        # 步骤 3: 验证数据完整性
        if args.verify:
            current_step += 1
            print_progress_summary(
                current_step, total_steps, "验证数据完整性", logger
            )
            
            logger.info("验证数据完整性...")
            is_valid = verify_data_integrity(
                data_dir=data_dir,
                logger=logger
            )
            
            if is_valid:
                logger.info("数据完整性验证通过")
            else:
                logger.error("数据完整性验证失败")
                sys.exit(1)
        
        logger.info("=" * 80)
        logger.info("数据准备完成!")
        logger.info(f"数据保存在: {data_dir}")
        logger.info("=" * 80)
        
        # 打印使用说明
        logger.info("\n下一步操作:")
        logger.info("1. 训练模型:")
        logger.info(f"   python train.py --config configs/hnet_scene_graph.json")
        logger.info("2. 评估模型:")
        logger.info(f"   python evaluate.py --config configs/hnet_scene_graph.json --checkpoint outputs/checkpoint_best.pth")
        logger.info("3. 运行演示:")
        logger.info(f"   python demo.py --config configs/hnet_scene_graph.json --checkpoint outputs/checkpoint_best.pth --image path/to/image.jpg")
        
    except KeyboardInterrupt:
        logger.info("数据准备被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"数据准备过程中发生错误: {str(e)}")
        if args.debug:
            import traceback
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()