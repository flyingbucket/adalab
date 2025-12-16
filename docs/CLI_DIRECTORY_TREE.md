# CLI重构后的目录结构

## 完整目录树

```
ML/
├── main.py                              # 统一CLI入口（24行薄封装）✨
│
├── src/
│   ├── adalab/                          # AdaLab统一模块 ✨新增
│   │   ├── __init__.py                  # 模块初始化
│   │   ├── cli/                         # CLI命令层
│   │   │   ├── __init__.py              # CLI导出
│   │   │   ├── main.py                  # CLI主入口（argparse配置）
│   │   │   ├── train.py                 # 训练命令（train_command）
│   │   │   ├── evaluate.py              # 评估命令（evaluate_command）
│   │   │   └── visualize.py             # 可视化命令（visualize_command）
│   │   └── core/                        # 核心业务层
│   │       ├── __init__.py              # 核心模块导出
│   │       ├── evaluator.py             # 统一评估器（evaluate/evaluate_detailed）
│   │       └── trainer.py               # 训练流程管理器（TrainingPipeline）
│   │
│   ├── __init__.py                      # src包初始化
│   ├── evaluation.py                    # 完整评估系统
│   ├── monitor.py                       # 训练监控器（含val_idx支持）
│   ├── patch.py                         # AdaBoost补丁
│   ├── utils.py                         # 数据准备工具
│   └── robust_adaboost.py               # 鲁棒AdaBoost
│
├── scripts/
│   ├── training/
│   │   ├── main.py                      # [已废弃] → adalab.core.trainer
│   │   ├── main_hog.py                  # [已废弃] → adalab.core.trainer
│   │   ├── train_with_clean_data.py     # 专用训练脚本
│   │   └── train_with_noise_track.py    # 专用训练脚本
│   ├── evaluation/
│   │   ├── test_generalization.py
│   │   └── compare_robust_methods.py
│   ├── visualization/
│   │   ├── visualize_from_results.py    # 可被CLI调用
│   │   └── visualize_overfitting.py
│   └── demo/
│       └── demo_robust.py
│
├── configs/
│   └── *.json                           # 实验配置
│
├── experiments/
│   └── [experiment_name]/
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── data/
│   └── test_images/
│
├── docs/
│   ├── CLI_GUIDE.md                     # CLI使用指南 ✨新增
│   ├── CLI_REFACTORING_SUMMARY.md       # CLI重构总结 ✨新增
│   ├── CLI_DIRECTORY_TREE.md            # 本文档 ✨新增
│   ├── val_after_train_mode.md
│   ├── PROJECT_STRUCTURE.md
│   └── *.md
│
├── test_cli.sh                          # CLI测试脚本 ✨新增
├── environment.yaml
├── requirements.txt
└── README.md
```

## 模块关系图

```
┌─────────────────────────────────────────────────┐
│                   main.py                       │
│              (24行薄封装入口)                    │
└────────────────────┬────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────┐
│          src.adalab.cli.main                    │
│         (CLI主入口，命令路由)                    │
└──┬──────────────┬──────────────┬────────────────┘
   │              │              │
   ↓              ↓              ↓
┌──────────┐ ┌──────────┐ ┌──────────────┐
│  train   │ │ evaluate │ │  visualize   │
│ command  │ │ command  │ │   command    │
└────┬─────┘ └────┬─────┘ └──────┬───────┘
     │            │               │
     ↓            ↓               ↓
┌─────────────────────────────────────────────────┐
│          src.adalab.core (核心业务层)            │
│  ┌────────────────┐  ┌─────────────────┐       │
│  │ TrainingPipeline│  │   evaluator     │       │
│  │    .run()      │  │ evaluate()      │       │
│  └────────────────┘  │ evaluate_detail()│       │
│                      └─────────────────┘       │
└─────────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────┐
│        现有模块（monitor/utils/evaluation）      │
│  - src.monitor.BoostMonitor                     │
│  - src.utils.train_and_save                     │
│  - src.evaluation.quick_evaluate                │
└─────────────────────────────────────────────────┘
```

## 数据流图

### 训练流程

```
用户命令
  ↓
python main.py train --config configs/baseline.json
  ↓
main.py (设置Python路径)
  ↓
adalab.cli.main.main() (解析命令)
  ↓
adalab.cli.train.train_command()
  ↓
adalab.core.trainer.TrainingPipeline
  ↓
TrainingPipeline.run()
  ├─→ src.utils.train_and_save() (训练)
  ├─→ adalab.core.evaluator.evaluate() (MNIST评估)
  ├─→ adalab.core.evaluator.evaluate() (课程数据评估)
  └─→ 保存结果 (scores.json)
  ↓
输出：experiments/[name]/results/
  - model.joblib
  - monitor.joblib
  - results.csv
  - scores.json
```

### 可视化流程

```
用户命令
  ↓
python main.py visualize --joblib monitor.joblib --save output.png
  ↓
main.py
  ↓
adalab.cli.main.main()
  ↓
adalab.cli.visualize.visualize_command()
  ↓
scripts.visualization.visualize_from_results
  ├─→ load_from_joblib() (加载数据)
  ├─→ print_summary() (打印摘要)
  └─→ visualize_training_data() (生成图表)
  ↓
输出：outputs/figures/output.png
```

## 文件大小统计

### 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `main.py` | 24 | 入口脚本（原63行，减少62%）|
| `src/adalab/__init__.py` | 5 | 模块初始化 |
| `src/adalab/cli/__init__.py` | 10 | CLI导出 |
| `src/adalab/cli/main.py` | 65 | CLI主入口 |
| `src/adalab/cli/train.py` | 55 | 训练命令 |
| `src/adalab/cli/evaluate.py` | 85 | 评估命令 |
| `src/adalab/cli/visualize.py` | 125 | 可视化命令 |
| `src/adalab/core/__init__.py` | 10 | 核心模块导出 |
| `src/adalab/core/evaluator.py` | 105 | 评估器 |
| `src/adalab/core/trainer.py` | 105 | 训练流程管理器 |
| `test_cli.sh` | 40 | CLI测试脚本 |
| **总计（代码）** | **629行** | 新架构代码 |

### 文档文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `docs/CLI_GUIDE.md` | 500+ | CLI使用指南 |
| `docs/CLI_REFACTORING_SUMMARY.md` | 450+ | CLI重构总结 |
| `docs/CLI_DIRECTORY_TREE.md` | 本文档 | 目录结构说明 |
| **总计（文档）** | **1000+行** | 完整文档 |

## 代码分层

### 第1层：入口层（Entry Layer）

```
main.py (24行)
  ↓
设置Python路径 + 启动CLI
```

**职责：** 
- 设置项目根目录到sys.path
- 导入并启动CLI主函数
- 无业务逻辑

### 第2层：CLI层（CLI Layer）

```
src/adalab/cli/
  ├── main.py      (65行) - 命令解析和路由
  ├── train.py     (55行) - 训练命令
  ├── evaluate.py  (85行) - 评估命令
  └── visualize.py (125行) - 可视化命令
```

**职责：**
- argparse命令解析
- 参数验证
- 调用核心层功能
- 输出格式化

### 第3层：核心层（Core Layer）

```
src/adalab/core/
  ├── evaluator.py (105行) - 评估逻辑
  └── trainer.py   (105行) - 训练流程
```

**职责：**
- 核心业务逻辑
- 可被CLI和脚本复用
- 独立测试

### 第4层：基础层（Foundation Layer）

```
src/
  ├── monitor.py         - 训练监控
  ├── utils.py           - 数据准备
  ├── evaluation.py      - 完整评估
  ├── patch.py           - AdaBoost补丁
  └── robust_adaboost.py - 鲁棒实现
```

**职责：**
- 底层工具函数
- 被所有上层模块使用

## 依赖关系

### 模块依赖

```
main.py
  └─→ src.adalab.cli

src.adalab.cli
  ├─→ src.adalab.core (核心功能)
  └─→ scripts.visualization (可视化)

src.adalab.core
  ├─→ src.monitor
  ├─→ src.utils
  ├─→ src.evaluation
  └─→ sklearn

src.monitor, src.utils, src.evaluation
  └─→ numpy, pandas, sklearn
```

### 无循环依赖

✅ **设计原则：**
- 上层依赖下层
- 下层不知道上层存在
- CLI层不被其他模块导入
- 核心层可被任意模块导入

## 接口总结

### CLI接口（用户层）

```bash
# 训练
python main.py train --config PATH

# 评估
python main.py evaluate --model PATH --data PATH [--detailed]

# 可视化
python main.py visualize (--csv|--joblib) PATH [--save PATH] [--show]
```

### Python接口（开发者层）

```python
# 导入核心功能
from src.adalab.core import TrainingPipeline, evaluate, evaluate_detailed

# 训练流程
pipeline = TrainingPipeline("configs/baseline.json")
results = pipeline.run()

# 评估
scores = evaluate(y_true, y_pred)
detailed_scores = evaluate_detailed(y_true, y_pred)
```

## 扩展指南

### 添加新CLI命令

1. 在 `src/adalab/cli/` 创建新文件（如 `compare.py`）
2. 实现命令函数和解析器
3. 在 `src/adalab/cli/main.py` 中注册
4. 更新 `src/adalab/cli/__init__.py`

示例：

```python
# src/adalab/cli/compare.py
def compare_command(args):
    # 实现比较逻辑
    pass

def add_compare_parser(subparsers):
    parser = subparsers.add_parser('compare', help='比较多个模型')
    # 添加参数
    parser.set_defaults(func=compare_command)
```

### 添加新核心功能

1. 在 `src/adalab/core/` 创建新模块
2. 实现纯业务逻辑（不依赖CLI）
3. 导出到 `src/adalab/core/__init__.py`
4. 在CLI命令中调用

## 总结

### 架构特点

- ✅ **薄入口**：main.py仅24行
- ✅ **分层清晰**：入口→CLI→核心→基础
- ✅ **职责单一**：每个模块一个职责
- ✅ **易于扩展**：添加新命令只需2个文件
- ✅ **向后兼容**：旧脚本改为包装器

### 代码统计

- **新增代码**：~630行
- **新增文档**：~1000行
- **main.py减少**：62% (63→24行)
- **消除重复**：3处evaluate函数合并为1处

### 相关文档

- [CLI使用指南](./CLI_GUIDE.md) - 详细使用说明
- [CLI重构总结](./CLI_REFACTORING_SUMMARY.md) - 重构过程和收益
- [项目结构说明](./PROJECT_STRUCTURE.md) - 整体项目结构


