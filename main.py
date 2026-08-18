#!/usr/bin/env python3
"""
AdaLab 主入口脚本

这是一个薄封装入口，实际CLI逻辑在 src.adalab.cli 模块中实现。

使用方法:
    python main.py train --config configs/baseline.json
    python main.py evaluate --model model.joblib --data test.npz
    python main.py visualize --joblib monitor.joblib --save output.png
"""

import os
import sys

# 确保项目根目录在Python路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.adalab.cli import main

if __name__ == "__main__":
    main()
