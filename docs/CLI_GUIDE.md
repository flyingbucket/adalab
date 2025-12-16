# AdaLab CLI 使用指南

## 概述

AdaLab提供统一的命令行接口(CLI)，用于训练、评估和可视化AdaBoost模型。

## 架构设计

### 目录结构

```
ML/
├── main.py                      # 薄封装入口脚本
├── src/
│   ├── adalab/                  # AdaLab核心模块
│   │   ├── __init__.py
│   │   ├── cli/                 # CLI命令模块
│   │   │   ├── __init__.py
│   │   │   ├── main.py          # CLI主入口
│   │   │   ├── train.py         # 训练命令
│   │   │   ├── evaluate.py      # 评估命令
│   │   │   └── visualize.py     # 可视化命令
│   │   └── core/                # 核心功能模块
│   │       ├── __init__.py
│   │       ├── evaluator.py     # 评估器
│   │       └── trainer.py       # 训练流程管理器
│   ├── monitor.py
│   ├── patch.py
│   ├── evaluation.py
│   └── utils.py
└── scripts/
    └── training/
        ├── main.py              # [已废弃] 兼容性包装器
        └── main_hog.py          # [已废弃] 兼容性包装器
```

### 设计原则

1. **关注点分离**
   - `src/adalab/core/`: 核心业务逻辑（评估、训练）
   - `src/adalab/cli/`: CLI命令解析和入口
   - `main.py`: 薄封装，仅负责启动CLI

2. **单一职责**
   - 每个CLI命令独立模块
   - 核心功能可被CLI和脚本复用

3. **向后兼容**
   - 旧脚本保留为兼容性包装器
   - 建议迁移到新CLI

## 命令参考

### 1. 训练命令 (train)

训练AdaBoost模型并在测试集上评估。

#### 基本用法

```bash
python main.py train --config <配置文件路径>
```

#### 示例

```bash
# 使用baseline配置训练
python main.py train --config configs/baseline.json

# 使用噪声配置训练
python main.py train --config configs/noise5_est500_depth2.json

# 使用HOG特征训练
python main.py train --config configs/hog_config.json
```

#### 输出

- 训练完成后自动评估MNIST测试集
- 如果存在`data/test_images/`，也会评估课程数据
- 结果保存到配置指定的输出目录：
  - `model.joblib`: 训练好的模型
  - `monitor.joblib`: 训练监控数据
  - `results.csv`: 训练历史CSV
  - `scores.json`: 评估分数

---

### 2. 评估命令 (evaluate)

加载训练好的模型，在指定数据上评估。

#### 基本用法

```bash
python main.py evaluate --model <模型文件> --data <数据文件>
```

#### 参数

- `--model PATH`: 模型文件路径（`.joblib`格式）
- `--data PATH`: 测试数据文件路径（`.npz`格式）
- `--detailed`: 显示详细评估结果（包含混淆矩阵）

#### 示例

```bash
# 基础评估
python main.py evaluate \
    --model experiments/baseline/model.joblib \
    --data data/test_data.npz

# 详细评估（包含混淆矩阵）
python main.py evaluate \
    --model experiments/baseline/model.joblib \
    --data data/test_data.npz \
    --detailed
```

#### 输出

```
=== Model Evaluation ===
Accuracy:       0.9652
Precision_macro:0.9648
Recall_macro:   0.9651
F1_macro:       0.9649
```

---

### 3. 可视化命令 (visualize)

从训练结果文件生成可视化图表。

#### 基本用法

```bash
# 从Joblib文件加载
python main.py visualize --joblib <monitor文件> [选项]

# 从CSV文件加载
python main.py visualize --csv <结果CSV文件> [选项]
```

#### 参数

**数据源（二选一）：**
- `--joblib PATH`: 从monitor.joblib文件加载
- `--csv PATH`: 从results.csv文件加载

**输出选项：**
- `--save PATH`: 输出图片路径（默认：`outputs/figures/<name>_visualization.png`）
- `--show`: 显示图表窗口

#### 示例

```bash
# 从joblib加载，保存到指定路径
python main.py visualize \
    --joblib experiments/baseline/monitor.joblib \
    --save outputs/baseline_viz.png

# 从CSV加载，显示窗口
python main.py visualize \
    --csv experiments/noise_exp/results.csv \
    --show

# 使用默认保存路径
python main.py visualize \
    --joblib experiments/baseline/monitor.joblib
# 自动保存到: outputs/figures/monitor_visualization.png
```

#### 输出

生成包含以下内容的6子图可视化：
1. 样本权重分布（噪声vs干净样本）
2. 权重演化趋势
3. 准确率演化曲线
4. 弱学习器误差率
5. F1分数演化曲线
6. 学习器权重(alpha)分布

---

## 快速开始

### 完整工作流示例

```bash
# 1. 训练模型
python main.py train --config configs/baseline.json

# 2. 可视化训练过程
python main.py visualize \
    --joblib experiments/baseline/monitor.joblib \
    --save outputs/baseline_training.png

# 3. 评估模型（如果有额外测试数据）
python main.py evaluate \
    --model experiments/baseline/model.joblib \
    --data data/my_test_data.npz \
    --detailed
```

### 查看帮助

```bash
# 查看主帮助
python main.py --help

# 查看子命令帮助
python main.py train --help
python main.py evaluate --help
python main.py visualize --help

# 查看版本
python main.py --version
```

---

## 高级用法

### 1. val-after-train模式可视化

当配置文件中设置了`val_after_train`和`validation_interval`时，可视化会自动适配：

```bash
python main.py visualize \
    --joblib experiments/long_train/monitor.joblib \
    --save outputs/val_after_train_viz.png
```

可视化图表会：
- 使用实际验证轮次作为横轴（如：[10, 20, 30, ...]）
- 正确显示稀疏验证点

### 2. 批量可视化

```bash
# 可视化所有实验结果
for exp in experiments/*/monitor.joblib; do
    exp_name=$(basename $(dirname $exp))
    python main.py visualize \
        --joblib $exp \
        --save outputs/figures/${exp_name}_viz.png
done
```

### 3. 与其他脚本集成

CLI模块的核心功能可以被其他Python脚本导入：

```python
from src.adalab.core import TrainingPipeline, evaluate

# 训练
pipeline = TrainingPipeline(config_path="configs/baseline.json")
results = pipeline.run()

# 评估
scores = evaluate(y_true, y_pred, title="Custom Evaluation")
```

---

## 迁移指南

### 从旧脚本迁移

**旧方式:**

```bash
python scripts/training/main.py --config_path configs/baseline.json
```

**新方式:**

```bash
python main.py train --config configs/baseline.json
```

### 兼容性说明

旧脚本（`scripts/training/main.py`、`scripts/training/main_hog.py`）已改为兼容性包装器：
- ✅ 仍然可以运行
- ⚠️  会显示废弃警告
- 💡 建议迁移到新CLI

---

## 常见问题

### Q: 为什么要重构CLI？

**A:** 
- **代码重复**: 旧脚本中评估逻辑重复多次
- **难以维护**: 功能分散在多个独立脚本中
- **扩展困难**: 添加新功能需要修改多个文件
- **不够优雅**: main.py臃肿，混杂业务逻辑

### Q: CSV和Joblib可视化有什么区别？

**A:**
- **Joblib**: 包含完整的monitor对象，支持所有可视化功能
- **CSV**: 只包含基本训练历史，功能有限
- **推荐**: 优先使用`--joblib`选项

### Q: 如何处理val_idx缺失错误？

**A:**
如果看到"val_idx字段缺失"错误：
1. 确保使用最新版本的`BoostMonitor`训练模型
2. 使用`--joblib`而非`--csv`（CSV可能不包含val_idx）
3. 如果是旧数据，需要重新训练

### Q: 可以在脚本中直接导入CLI功能吗？

**A:**
可以！核心功能在`src.adalab.core`中：

```python
from src.adalab.core import TrainingPipeline, evaluate, evaluate_detailed

# 使用TrainingPipeline
pipeline = TrainingPipeline("configs/baseline.json")
results = pipeline.run()

# 使用评估函数
scores = evaluate(y_true, y_pred)
```

---

## 技术细节

### CLI架构

```
main.py (薄入口)
    ↓
src.adalab.cli.main.main()
    ↓
argparse解析命令
    ↓
    ├─→ train → TrainingPipeline
    ├─→ evaluate → evaluate/evaluate_detailed
    └─→ visualize → visualize_from_results
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `main.py` | 入口脚本，设置Python路径 |
| `adalab.cli.main` | 主CLI入口，命令路由 |
| `adalab.cli.train` | 训练命令解析和执行 |
| `adalab.cli.evaluate` | 评估命令解析和执行 |
| `adalab.cli.visualize` | 可视化命令解析和执行 |
| `adalab.core.trainer` | 训练流程封装 |
| `adalab.core.evaluator` | 评估函数封装 |

---

## 更新日志

### v1.0.0 (当前版本)

- ✅ 创建统一CLI接口
- ✅ 重构main.py为薄入口
- ✅ 实现train/evaluate/visualize命令
- ✅ 封装核心功能到adalab.core
- ✅ 保留旧脚本兼容性
- ✅ 支持val-after-train模式可视化

---

## 相关文档

- [项目README](../README.md)
- [评估系统指南](./EVALUATION_SYSTEM_GUIDE.md)
- [Val-After-Train模式](./val_after_train_mode.md)
- [Robust AdaBoost指南](./ROBUST_ADABOOST_GUIDE.md)


