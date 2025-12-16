"""
训练命令 - 提供模型训练相关的CLI接口
"""

import argparse
from src.adalab.core.trainer import TrainingPipeline


def train_command(args):
    """
    执行训练命令
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    print(f"📦 配置文件: {args.config}")
    
    # 创建训练流程
    pipeline = TrainingPipeline(config_path=args.config)
    
    # 执行训练
    results = pipeline.run()
    
    print("\n✅ 训练完成！")
    return results


def add_train_parser(subparsers):
    """
    添加训练子命令解析器
    
    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        子命令解析器容器
    """
    parser = subparsers.add_parser(
        'train',
        help='训练AdaBoost模型',
        description='使用配置文件训练AdaBoost模型，并在MNIST测试集上评估'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径 (JSON格式)',
        metavar='PATH'
    )
    
    parser.set_defaults(func=train_command)
    
    return parser


