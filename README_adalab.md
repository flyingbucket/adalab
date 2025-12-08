# 🚀 **adalab: A Lightweight Framework for Analyzing AdaBoost Behavior**

**adalab** 是一个专注于 *AdaBoost 行为分析、训练监控、噪声鲁棒性研究* 的轻量级 Python 框架。
项目包含：

* **adalab**：训练、监控、数据处理与工作流管理
* **adalab_viz**：可选安装的可视化模块，用于分析 AdaBoost 的动态行为

该框架旨在为研究者提供一个结构清晰、可扩展的实验工具库，用于深入理解 AdaBoost 的权重更新机制、噪声放大效应、过拟合行为与泛化表现。

---

# ✨ Features

## 🔧 **1. Training Backend（adalab）**

核心功能包括：

### ✔ 自定义 AdaBoostClassifier（带监控）

* 自动记录每一轮的：

  * 加权误差（weighted error）
  * 无权误差（unweighted error）
  * α 系数（弱学习器权重）
  * 样本权重直方图信息
  * 训练 / 验证集 Accuracy & F1
* 可控噪声注入（MNIST 或课程数据）
* 支持保存：

  * monitor.joblib（监控对象）
  * final_results.csv
  * model.joblib.xz
  * checkpoint（可选）

### ✔ 数据处理（MNIST + 课程数据）

* 自动划分 train/test
* 噪声注入：对指定比例样本随机翻转标签
* 统一接口：`DataPreparation.prepare_mnist()`, `prepare_course_data()`

### ✔ 实验工作流管理

* 通过 JSON 配置文件完成整个训练流程：

  ```
  train_and_save(config_path)
  ```

* 自动创建实验目录：

  ```
  experiments/<exp_name>/
  ├── results/
  ├── checkpoints/
  └── config.json
  ```

---

## 📊 **2. Visualization Module（adalab_viz）**

可视化模块为可选安装：

```
pip install "adalab[viz]"
```

提供：

### ✔ 支持从 CSV 或 joblib 加载监控数据

* `load_from_csv()`
* `load_from_joblib()`
* `load_from_experiment()`

自动识别 monitor.joblib / final_results.csv。

### ✔ 高质量分析图（6×1 summary)

包含：

1. 噪声 vs 干净样本权重变化
2. 样本权重分布（箱线图）
3. 错误率曲线
4. α 系数曲线
5. Train vs Val Accuracy
6. Train vs Val F1

用于展示 AdaBoost 的噪声放大效应与过拟合过程。

### ✔ 保存单独子图（支持论文绘图）

自动按比例缩放 linewidth / markersize，确保美观一致。

---

# 📦 Installation

## 基础功能（训练+数据处理）

```
pip install adalab
```

## 启用完整可视化（推荐）

```
pip install "adalab[viz]"
```

## 开发模式（本地源码）

```
pip install -e .
```

---

# ⚙ Usage

## 1. 训练（从 JSON 配置启动）

```bash
python main.py --config_path configs/your_exp.json
```

典型 config：

```json
{
  "experiment": { "name": "noise10_depth2_500" },
  "data": {
    "noise_ratio": 0.1,
    "test_size": 0.2,
    "random_state": 42
  },
  "monitor": { "use_monitor": true },
  "model": {
    "n_estimators": 500,
    "learning_rate": 1.0
  }
}
```

---

## 2. 训练完成后自动可视化

```
python main.py --config_path configs/exp.json --viz
```

效果：

```
experiments/<exp_name>/results/
├── training_viz.png
├── monitor.joblib
├── final_results.csv
└── scores.json
```

---

## 3. 不训练，仅可视化已有结果

```
python main.py --config_path configs/exp.json --viz-only
```

自动加载：

```
experiments/<exp_name>/results/{monitor.joblib | final_results.csv}
```

并生成：

```
training_viz.png
single_plots/
    noisy_vs_clean.png
    sample_weight_distribution.png
    error_evolution.png
    alpha_evolution.png
    accuracy_evolution.png
    f1_evolution.png
```

---

# 📁 Project Structure

```
adalab/
├── data.py              # MNIST + 课程数据准备
├── monitor.py           # BoostMonitor（训练过程记录器）
├── patch.py             # 自定义 AdaBoostClassifier
├── utils.py             # train_and_save 工作流
└── robust_adaboost.py   # 噪声鲁棒变体（可选）

adalab_viz/
├── loader.py            # 加载 CSV / joblib / experiment 目录
├── plotter.py           # 可视化主逻辑（6 个子图）
└── cli.py               # 命令行可视化接口（可选）
```

---

# 📊 Example Visualization

（可插入示意图）

---

# 🧪 Testing

项目包含基本单测：

```
pytest tests/
```

---

# 📝 Roadmap

* [ ] 增加多种弱学习器可视化（决策树形状、节点分裂统计）
* [ ] 支持多次实验结果对比（多曲线模式）
* [ ] 加入 robustness benchmark（Flip Noise / Label Noise）
* [ ] 丰富可视化主题（Seaborn / LaTeX theme）
* [ ] 提供 Jupyter Notebook 教程

---

# 🤝 Contributing

欢迎 PR！
推荐分支：

```
feat/<feature-name>
fix/<bug-name>
recon/<backend-refactor>
```

---

# 📜 License

MIT License

---

# 🎯 Summary

**adalab** 旨在提供一个简洁、可扩展的 AdaBoost 行为研究框架，结合 **训练监控 + 数据处理 + 可视化分析**，帮助研究者深入理解：

* AdaBoost 如何放大样本权重
* 噪声如何导致过拟合
* 弱学习器强度（α 系数）的变化
* 训练/验证性能分歧
* 样本权重分布在过拟合前后的变化

适用于课程项目、论文实验、核查 AdaBoost 行为、研究鲁棒性与泛化表现等场景。
