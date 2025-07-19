#!/usr/bin/env python3
"""
快速开始脚本 - H-Net 场景图生成

一键完成从环境配置到训练的整个流程

使用方法:
    python quick_start.py                    # 完整流程
    python quick_start.py --skip-download    # 跳过数据下载
    python quick_start.py --demo-only        # 仅运行演示
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))  # 添加 hnet 根目录

from src.utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="H-Net 场景图生成快速开始脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过数据下载步骤"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="跳过训练步骤"
    )
    
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="仅运行演示（需要已有训练好的模型）"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="数据目录"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="输出目录"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hnet_scene_graph.json",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--demo-image",
        type=str,
        default=None,
        help="演示图像路径（如果不指定，将使用示例图像）"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（覆盖配置文件中的设置）"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小（覆盖配置文件中的设置）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练/推理设备"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式（减少训练轮数和数据量）"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    
    return parser.parse_args()


def run_command(command, description, logger, check=True, cwd=None):
    """运行命令并记录输出"""
    logger.info(f"开始: {description}")
    logger.info(f"命令: {command}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        
        elapsed_time = time.time() - start_time
        
        if result.stdout:
            logger.info(f"输出:\n{result.stdout}")
        
        if result.stderr and result.returncode == 0:
            logger.warning(f"警告:\n{result.stderr}")
        
        logger.info(f"完成: {description} (耗时: {elapsed_time:.2f}秒)")
        return result
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"失败: {description} (耗时: {elapsed_time:.2f}秒)")
        logger.error(f"错误代码: {e.returncode}")
        if e.stdout:
            logger.error(f"标准输出:\n{e.stdout}")
        if e.stderr:
            logger.error(f"错误输出:\n{e.stderr}")
        raise


def check_dependencies(logger):
    """检查依赖项"""
    logger.info("检查依赖项...")
    
    # 检查 Python 版本
    python_version = sys.version_info
    if python_version < (3, 8):
        raise RuntimeError(f"需要 Python 3.8+，当前版本: {python_version.major}.{python_version.minor}")
    
    logger.info(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查关键依赖
    required_packages = ['torch', 'torchvision', 'numpy', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} 未安装")
    
    if missing_packages:
        logger.info("安装缺失的依赖项...")
        run_command(
            f"pip install {' '.join(missing_packages)}",
            "安装依赖项",
            logger
        )


def setup_environment(logger):
    """设置环境"""
    logger.info("设置环境...")
    
    # 检查并安装依赖
    if os.path.exists("requirements.txt"):
        run_command(
            "pip install -r requirements.txt",
            "安装项目依赖",
            logger
        )
    else:
        check_dependencies(logger)
    
    # 设置 PYTHONPATH
    hnet_root = str(Path(__file__).parent.parent)
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if hnet_root not in current_pythonpath:
        os.environ['PYTHONPATH'] = f"{hnet_root}:{current_pythonpath}"
        logger.info(f"设置 PYTHONPATH: {os.environ['PYTHONPATH']}")


def prepare_data(args, logger):
    """准备数据"""
    if args.skip_download:
        logger.info("跳过数据下载")
        command = f"python prepare_data.py --data_dir {args.data_dir} --preprocess --verify"
    else:
        command = f"python prepare_data.py --data_dir {args.data_dir} --download --preprocess --verify"
    
    if args.quick:
        command += " --min_objects 2 --min_relationships 1 --min_object_freq 50 --min_predicate_freq 25"
    
    run_command(command, "数据准备", logger)


def modify_config_for_quick_mode(config_path, args, logger):
    """为快速模式修改配置"""
    if not (args.quick or args.epochs or args.batch_size):
        return
    
    logger.info("修改配置文件...")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 快速模式设置
    if args.quick:
        config['training']['num_epochs'] = 5
        config['training']['batch_size'] = 8
        config['training']['save_every'] = 1
        config['training']['eval_every'] = 1
        config['model']['max_objects_per_image'] = 15
        logger.info("应用快速模式设置")
    
    # 覆盖特定设置
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
        logger.info(f"设置训练轮数: {args.epochs}")
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        logger.info(f"设置批大小: {args.batch_size}")
    
    # 保存修改后的配置
    config_backup = config_path + '.backup'
    if not os.path.exists(config_backup):
        os.rename(config_path, config_backup)
        logger.info(f"备份原配置文件: {config_backup}")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"保存修改后的配置: {config_path}")


def train_model(args, logger):
    """训练模型"""
    command = f"python train.py --config {args.config} --output_dir {args.output_dir} --device {args.device}"
    
    if args.verbose:
        command += " --debug"
    
    run_command(command, "模型训练", logger)


def evaluate_model(args, logger):
    """评估模型"""
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_best.pth")
    
    if not os.path.exists(checkpoint_path):
        # 尝试使用最新的检查点
        checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
        if not os.path.exists(checkpoint_path):
            logger.warning("未找到训练好的模型，跳过评估")
            return
    
    command = f"python evaluate.py --config {args.config} --checkpoint {checkpoint_path} --device {args.device}"
    
    run_command(command, "模型评估", logger)


def run_demo(args, logger):
    """运行演示"""
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_best.pth")
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
        if not os.path.exists(checkpoint_path):
            logger.warning("未找到训练好的模型，跳过演示")
            return
    
    # 确定演示图像
    demo_image = args.demo_image
    if not demo_image:
        # 尝试从测试数据中找一张图像
        test_images_dir = os.path.join(args.data_dir, "raw", "images")
        if os.path.exists(test_images_dir):
            image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                demo_image = os.path.join(test_images_dir, image_files[0])
                logger.info(f"使用示例图像: {demo_image}")
    
    if not demo_image or not os.path.exists(demo_image):
        logger.warning("未找到演示图像，跳过演示")
        logger.info("您可以使用以下命令手动运行演示:")
        logger.info(f"python demo.py --config {args.config} --checkpoint {checkpoint_path} --image YOUR_IMAGE_PATH")
        return
    
    command = f"python demo.py --config {args.config} --checkpoint {checkpoint_path} --image {demo_image} --save_visualization --save_json --device {args.device}"
    
    run_command(command, "演示运行", logger)


def print_summary(args, logger):
    """打印总结"""
    logger.info("=" * 80)
    logger.info("快速开始完成!")
    logger.info("=" * 80)
    
    logger.info("生成的文件:")
    
    # 数据文件
    if os.path.exists(args.data_dir):
        logger.info(f"📁 数据目录: {args.data_dir}")
    
    # 模型文件
    if os.path.exists(args.output_dir):
        logger.info(f"📁 输出目录: {args.output_dir}")
        
        checkpoint_files = [
            "checkpoint_best.pth",
            "checkpoint_latest.pth"
        ]
        
        for checkpoint in checkpoint_files:
            checkpoint_path = os.path.join(args.output_dir, checkpoint)
            if os.path.exists(checkpoint_path):
                logger.info(f"🤖 模型检查点: {checkpoint_path}")
    
    # 演示结果
    demo_output = "demo_output"
    if os.path.exists(demo_output):
        logger.info(f"🎨 演示结果: {demo_output}")
    
    logger.info("\n下一步操作:")
    logger.info("1. 查看训练日志:")
    logger.info(f"   cat {args.output_dir}/logs/train.log")
    
    logger.info("2. 运行自定义演示:")
    logger.info(f"   python demo.py --config {args.config} --checkpoint {args.output_dir}/checkpoint_best.pth --image YOUR_IMAGE.jpg")
    
    logger.info("3. 进一步训练:")
    logger.info(f"   python train.py --config {args.config} --resume {args.output_dir}/checkpoint_latest.pth")
    
    logger.info("4. 详细评估:")
    logger.info(f"   python evaluate.py --config {args.config} --checkpoint {args.output_dir}/checkpoint_best.pth --save_predictions --save_visualizations")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    os.makedirs("logs", exist_ok=True)
    logger = setup_logger(
        name="quick_start",
        log_dir="logs",
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    logger.info("=" * 80)
    logger.info("H-Net 场景图生成 - 快速开始")
    logger.info("=" * 80)
    
    # 记录参数
    logger.info("运行参数:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    try:
        start_time = time.time()
        
        # 步骤 1: 设置环境
        setup_environment(logger)
        
        if args.demo_only:
            # 仅运行演示
            logger.info("仅运行演示模式")
            run_demo(args, logger)
        else:
            # 完整流程
            
            # 步骤 2: 准备数据
            prepare_data(args, logger)
            
            # 步骤 3: 修改配置（如果需要）
            modify_config_for_quick_mode(args.config, args, logger)
            
            # 步骤 4: 训练模型
            if not args.skip_training:
                train_model(args, logger)
            
            # 步骤 5: 评估模型
            evaluate_model(args, logger)
            
            # 步骤 6: 运行演示
            run_demo(args, logger)
        
        # 打印总结
        total_time = time.time() - start_time
        logger.info(f"\n总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
        
        print_summary(args, logger)
        
    except KeyboardInterrupt:
        logger.info("快速开始被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"快速开始过程中发生错误: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()