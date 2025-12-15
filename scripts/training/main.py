#!/usr/bin/env python3
"""
[已废弃] 旧版训练脚本

⚠️ 此脚本已废弃，请使用项目根目录的 main.py：

新的使用方式:
    cd /Users/frederick/Documents/ML
    python main.py train --config configs/baseline.json

或者直接使用这个兼容性包装器（不推荐）:
    python scripts/training/main.py --config_path configs/baseline.json
"""

import os
import sys
import argparse

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.adalab.core.trainer import TrainingPipeline


if __name__ == "__main__":
    print("⚠️  警告: 此脚本已废弃，建议使用根目录的 main.py")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(
        description="[兼容性包装器] 训练AdaBoost模型"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True,
        help="配置文件路径 (JSON格式)"
    )
    args = parser.parse_args()
    
    # 使用新的训练流程
    pipeline = TrainingPipeline(config_path=args.config_path)
    results = pipeline.run()
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("💡 下次请使用: python main.py train --config", args.config_path)
    print("=" * 60)
