"""
评估命令 - 提供模型评估相关的CLI接口
"""

import argparse
import joblib
import numpy as np
from src.adalab.core.evaluator import evaluate, evaluate_detailed


def evaluate_command(args):
    """
    执行评估命令
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    print(f"📦 模型文件: {args.model}")
    print(f"📊 数据文件: {args.data}")
    
    # 加载模型
    clf = joblib.load(args.model)
    print("✅ 模型加载成功")
    
    # 加载数据
    data = np.load(args.data)
    X_test = data['X']
    y_test = data['y']
    print(f"✅ 数据加载成功 ({len(X_test)} 个样本)")
    
    # 预测
    y_pred = clf.predict(X_test)
    
    # 评估
    if args.detailed:
        scores = evaluate_detailed(y_test, y_pred, title="Model Evaluation")
    else:
        scores = evaluate(y_test, y_pred, title="Model Evaluation")
    
    return scores


def add_evaluate_parser(subparsers):
    """
    添加评估子命令解析器
    
    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        子命令解析器容器
    """
    parser = subparsers.add_parser(
        'evaluate',
        help='评估训练好的模型',
        description='加载训练好的模型并在测试数据上进行评估'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型文件路径 (.joblib)',
        metavar='PATH'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='测试数据文件路径 (.npz)',
        metavar='PATH'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='显示详细评估结果（包含混淆矩阵）'
    )
    
    parser.set_defaults(func=evaluate_command)
    
    return parser


