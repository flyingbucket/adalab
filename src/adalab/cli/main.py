"""
主CLI入口 - 统一的命令行接口
"""

import argparse
import sys
from .train import add_train_parser
from .evaluate import add_evaluate_parser
from .visualize import add_visualize_parser


def main():
    """
    主CLI入口函数
    """
    parser = argparse.ArgumentParser(
        prog='adalab',
        description='AdaLab - AdaBoost实验平台命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练模型
  python main.py train --config configs/baseline.json
  
  # 评估模型
  python main.py evaluate --model model.joblib --data test.npz
  
  # 可视化训练结果
  python main.py visualize --joblib monitor.joblib --save output.png
  python main.py visualize --csv results.csv --show
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # 创建子命令
    subparsers = parser.add_subparsers(
        title='可用命令',
        description='使用 `adalab <command> --help` 查看详细帮助',
        dest='command',
        required=True
    )
    
    # 添加各个子命令
    add_train_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_visualize_parser(subparsers)
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行对应的命令
    try:
        result = args.func(args)
        return result
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


