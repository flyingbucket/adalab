"""
可视化命令 - 提供训练结果可视化的CLI接口
"""

import argparse
import os
import sys

# 动态添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def visualize_command(args):
    """
    执行可视化命令
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    # 导入可视化模块（延迟导入避免循环依赖）
    from scripts.visualization.visualize_from_results import (
        load_from_csv,
        load_from_joblib,
        visualize_training_data,
        print_summary
    )
    
    print("=" * 60)
    print("训练结果可视化".center(60))
    print("=" * 60)
    
    # 加载数据
    if args.csv:
        print(f"\n📂 从CSV加载: {args.csv}")
        data = load_from_csv(args.csv)
    elif args.joblib:
        print(f"\n📂 从Joblib加载: {args.joblib}")
        data = load_from_joblib(args.joblib)
    else:
        raise ValueError("必须指定 --csv 或 --joblib 参数")
    
    print("✅ 数据加载成功")
    
    # 打印摘要
    print_summary(data)
    
    # 可视化
    print("\n🎨 生成可视化图表...")
    fig = visualize_training_data(data)
    
    # 保存
    if args.save:
        output_path = args.save
    else:
        # 默认保存路径
        if args.csv:
            base_name = os.path.splitext(os.path.basename(args.csv))[0]
        else:
            base_name = os.path.splitext(os.path.basename(args.joblib))[0]
        output_path = f"outputs/figures/{base_name}_visualization.png"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")
    
    if args.show:
        import matplotlib.pyplot as plt
        print("📊 显示图表...")
        plt.show()
    
    return output_path


def add_visualize_parser(subparsers):
    """
    添加可视化子命令解析器
    
    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        子命令解析器容器
    """
    parser = subparsers.add_parser(
        'visualize',
        help='可视化训练结果',
        description='从CSV或Joblib文件加载训练结果并生成可视化图表'
    )
    
    # 数据源（互斥）
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--csv',
        type=str,
        help='CSV结果文件路径',
        metavar='PATH'
    )
    source_group.add_argument(
        '--joblib',
        type=str,
        help='Joblib monitor文件路径',
        metavar='PATH'
    )
    
    # 输出选项
    parser.add_argument(
        '--save',
        type=str,
        help='输出图片路径（默认: outputs/figures/<name>_visualization.png）',
        metavar='PATH'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='显示图表窗口'
    )
    
    parser.set_defaults(func=visualize_command)
    
    return parser

