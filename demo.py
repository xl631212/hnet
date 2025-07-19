#!/usr/bin/env python3
"""
主演示脚本 - H-Net 场景图生成

使用方法:
    python demo.py --config configs/hnet_scene_graph.json --checkpoint outputs/checkpoint_best.pth --image path/to/image.jpg
    python demo.py --config configs/hnet_scene_graph.json --checkpoint outputs/checkpoint_best.pth --image path/to/image.jpg --output demo_results
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))  # 添加 hnet 根目录

from src.demo import SceneGraphDemo
from src.utils.logger import setup_logger

# 忽略一些常见的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="H-Net 场景图生成演示脚本",
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
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图像路径"
    )
    
    # 可选参数
    parser.add_argument(
        "--output",
        type=str,
        default="demo_output",
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据集目录路径（用于加载词汇表）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备"
    )
    
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="预测置信度阈值"
    )
    
    parser.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="最大对象数量（默认使用配置文件中的值）"
    )
    
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="保存可视化结果"
    )
    
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="保存 JSON 格式的场景图"
    )
    
    parser.add_argument(
        "--show_attributes",
        action="store_true",
        help="在可视化中显示属性"
    )
    
    parser.add_argument(
        "--show_confidence",
        action="store_true",
        help="在可视化中显示置信度"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式，可以调整阈值"
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
    """设置推理设备"""
    import torch
    
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
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "json_outputs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    return output_dir


def print_scene_graph_summary(scene_graph, logger):
    """打印场景图摘要"""
    logger.info("=" * 60)
    logger.info("场景图生成结果")
    logger.info("=" * 60)
    
    # 对象信息
    objects = scene_graph.get('objects', [])
    logger.info(f"检测到 {len(objects)} 个对象:")
    for obj in objects:
        obj_info = f"  {obj['id']}: {obj['class']}"
        if obj.get('confidence'):
            obj_info += f" (置信度: {obj['confidence']:.3f})"
        if obj.get('attributes'):
            obj_info += f" - 属性: {', '.join(obj['attributes'])}"
        logger.info(obj_info)
    
    # 关系信息
    relationships = scene_graph.get('relationships', [])
    logger.info(f"\n检测到 {len(relationships)} 个关系:")
    for rel in relationships:
        rel_info = f"  {rel['subject']} -> {rel['predicate']} -> {rel['object']}"
        if rel.get('confidence'):
            rel_info += f" (置信度: {rel['confidence']:.3f})"
        logger.info(rel_info)
    
    logger.info("=" * 60)


def interactive_mode(demo, image_path, output_dir, logger):
    """交互式模式"""
    logger.info("进入交互式模式...")
    logger.info("您可以调整置信度阈值来查看不同的结果")
    logger.info("输入 'quit' 或 'exit' 退出")
    
    while True:
        try:
            threshold_input = input("\n请输入置信度阈值 (0.0-1.0, 默认 0.5): ").strip()
            
            if threshold_input.lower() in ['quit', 'exit', 'q']:
                logger.info("退出交互式模式")
                break
            
            if threshold_input == "":
                threshold = 0.5
            else:
                threshold = float(threshold_input)
                if not 0.0 <= threshold <= 1.0:
                    print("阈值必须在 0.0 到 1.0 之间")
                    continue
            
            print(f"\n使用置信度阈值: {threshold}")
            
            # 运行演示
            result = demo.run_demo(
                image_path=image_path,
                output_dir=output_dir,
                confidence_threshold=threshold,
                save_visualization=True,
                save_json=True
            )
            
            # 打印结果摘要
            print_scene_graph_summary(result['scene_graph'], logger)
            
            print(f"结果已保存到: {output_dir}")
            
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            logger.info("\n交互式模式被中断")
            break
        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = setup_directories(args.output)
    
    # 设置日志
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(
        name="demo",
        log_file=os.path.join(output_dir, "logs", "demo.log"),
        level=log_level
    )
    
    logger.info("=" * 80)
    logger.info("H-Net 场景图生成演示")
    logger.info("=" * 80)
    
    # 记录命令行参数
    if args.verbose:
        logger.info("命令行参数:")
        for key, value in vars(args).items():
            logger.info(f"  {key}: {value}")
    
    try:
        # 检查输入文件
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"输入图像未找到: {args.image}")
        
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"检查点文件未找到: {args.checkpoint}")
        
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 更新配置
        config['data']['data_dir'] = args.data_dir
        config['demo'] = config.get('demo', {})
        config['demo']['confidence_threshold'] = args.confidence_threshold
        config['demo']['show_attributes'] = args.show_attributes
        config['demo']['show_confidence'] = args.show_confidence
        
        if args.max_objects is not None:
            config['model']['max_objects_per_image'] = args.max_objects
        
        # 设置设备
        device = setup_device(args.device)
        config['demo']['device'] = device
        
        # 创建演示器
        logger.info("初始化演示器...")
        demo = SceneGraphDemo(config)
        
        # 加载模型
        logger.info(f"加载模型检查点: {args.checkpoint}")
        demo.load_model(args.checkpoint)
        
        logger.info(f"处理图像: {args.image}")
        
        if args.interactive:
            # 交互式模式
            interactive_mode(demo, args.image, output_dir, logger)
        else:
            # 单次运行模式
            result = demo.run_demo(
                image_path=args.image,
                output_dir=output_dir,
                confidence_threshold=args.confidence_threshold,
                save_visualization=args.save_visualization,
                save_json=args.save_json
            )
            
            # 打印结果摘要
            print_scene_graph_summary(result['scene_graph'], logger)
            
            # 保存结果信息
            if args.save_json or args.save_visualization:
                logger.info(f"结果已保存到: {output_dir}")
            
            # 打印 JSON 结果（如果不保存到文件）
            if not args.save_json:
                print("\nJSON 格式结果:")
                print(json.dumps(result['scene_graph'], indent=2, ensure_ascii=False))
        
        logger.info("演示完成!")
        
    except KeyboardInterrupt:
        logger.info("演示被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"演示过程中发生错误: {str(e)}")
        if args.debug:
            import traceback
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()