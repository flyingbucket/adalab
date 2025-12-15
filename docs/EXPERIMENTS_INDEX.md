# 📁 实验结果索引

所有已完成的实验及其可视化方法。

---

## 🎯 快速可视化任一实验

```bash
python visualize_from_results.py -e [实验名]
```

---

## 📊 可用实验列表

### 1. `baseline_est500_depth2`

**配置：**
- 噪声比例：0%（干净数据）
- 弱学习器：500
- 树深度：2

**可视化：**
```bash
python visualize_from_results.py -e baseline_est500_depth2
```

---

### 2. `noise5_est500_depth2`

**配置：**
- 噪声比例：5%
- 弱学习器：500
- 树深度：2

**可视化：**
```bash
python visualize_from_results.py -e noise5_est500_depth2
```

---

### 3. `train_val_500rounds`

**配置：**
- 噪声比例：5%
- 训练轮次：500
- 包含验证集监控

**可视化：**
```bash
python visualize_from_results.py -e train_val_500rounds
```

**特点：** 完整的训练监控数据，适合深度分析

---

### 4. `test_experiment_wrapper`

**配置：**
- 测试实验
- 弱学习器：10

**可视化：**
```bash
python visualize_from_results.py -e test_experiment_wrapper
```

---

### 5. `main_hog_v2`

**配置：**
- 特征：HOG（Histogram of Oriented Gradients）
- 用途：图像特征提取测试

**可视化：**
```bash
python visualize_from_results.py -e main_hog_v2
```

---

## 🔄 批量对比实验

### 对比干净 vs 噪声数据

```bash
# 生成两个图表
python visualize_from_results.py -e baseline_est500_depth2 -s baseline.png
python visualize_from_results.py -e noise5_est500_depth2 -s noise5.png

# 查看对比
open baseline.png noise5.png  # macOS
# 或
eog baseline.png noise5.png  # Linux
```

---

### 批量查看所有实验摘要

```bash
for exp in baseline_est500_depth2 noise5_est500_depth2 train_val_500rounds; do
    echo "========== $exp =========="
    python visualize_from_results.py -e $exp --no-plot
    echo ""
done
```

---

## 📋 实验文件结构

每个实验包含：

```
experiments/[实验名]/
├── checkpoints/          # 训练中间结果
│   ├── round_0050.csv
│   ├── round_0100.csv
│   └── ...
├── results/              # 最终结果
│   ├── final_results.csv     # ⭐ CSV格式（推荐）
│   └── monitor.joblib        # joblib格式（完整）
└── config.json           # 实验配置（可能存在）
```

---

## 🎨 可视化示例

### 示例1：快速查看

```bash
$ python visualize_from_results.py -e train_val_500rounds --no-plot

============================================================
                      Training Summary                      
============================================================

📊 Basic Info:
   - Total Rounds: 500
   - Data Type: Noisy

📈 Final Metrics:
   - Final Val Accuracy: 0.8321
   - Best Val Accuracy:  0.8321 (round 500)

💡 Noise Analysis:
   - Final Noisy Weight:   0.5554
   - Final Clean Weight:   0.4446
   - Weight Ratio (noisy/clean): 1.249
   ⚠️  Noisy samples slightly over-weighted
============================================================
```

---

### 示例2：生成图表

```bash
$ python visualize_from_results.py -e train_val_500rounds -s result.png

✓ Figure saved to: result.png
```

生成包含6个子图的专业可视化！

---

## 🔍 查看特定 Checkpoint

```bash
# 查看第250轮的训练状态
python visualize_from_results.py \
    --csv experiments/train_val_500rounds/checkpoints/round_0250.csv

# 对比不同轮次
python visualize_from_results.py -c experiments/train_val_500rounds/checkpoints/round_0100.csv -s round100.png
python visualize_from_results.py -c experiments/train_val_500rounds/checkpoints/round_0500.csv -s round500.png
```

---

## 📊 实验对比矩阵

| 实验 | 噪声 | 弱学习器 | 最佳准确率* | 过拟合程度* | 推荐用途 |
|-----|------|---------|-----------|-----------|---------|
| baseline_est500_depth2 | 0% | 500 | - | - | 基线对比 |
| noise5_est500_depth2 | 5% | 500 | - | - | 噪声影响研究 |
| train_val_500rounds | 5% | 500 | 0.8321 | 低 | 完整分析 |
| test_experiment_wrapper | - | 10 | - | - | 测试用 |

`*` 运行可视化工具获取具体数值

---

## 🚀 推荐工作流

### 步骤1：列出所有实验

```bash
ls experiments/
```

### 步骤2：快速查看摘要

```bash
python visualize_from_results.py -e [实验名] --no-plot
```

### 步骤3：生成详细图表

```bash
python visualize_from_results.py -e [实验名] -s output.png
```

### 步骤4：对比分析

将多个实验的图表并排查看，找出最佳配置。

---

## 📚 相关文档

- `docs/visualize_from_results_guide.md` - 详细使用指南
- `VISUALIZATION_METHODS.md` - 三种可视化方式对比
- `docs/monitor.md` - 数据结构说明

---

## 💡 小技巧

### 技巧1：快速对比命令

创建别名（添加到 `~/.bashrc` 或 `~/.zshrc`）：

```bash
alias viz="python visualize_from_results.py -e"
alias viz-save="python visualize_from_results.py -e"
```

使用：
```bash
viz train_val_500rounds
viz-save baseline_est500_depth2 -s baseline.png
```

---

### 技巧2：自动对比脚本

创建 `compare_all.sh`：

```bash
#!/bin/bash
for exp in baseline_est500_depth2 noise5_est500_depth2 train_val_500rounds; do
    echo "Processing $exp..."
    python visualize_from_results.py -e $exp -s ${exp}_analysis.png
done
echo "✓ All done! Check *_analysis.png files."
```

运行：
```bash
chmod +x compare_all.sh
./compare_all.sh
```

---

### 技巧3：生成 PDF 报告

```bash
# 生成高质量 PDF
python visualize_from_results.py -e train_val_500rounds -s report.pdf

# PDF 适合打印和分享
```

---

## 🎯 常见问题

### Q: 如何查看实验是否完成？

```bash
ls experiments/[实验名]/results/final_results.csv
```

如果文件存在，实验已完成。

### Q: CSV vs joblib，哪个更好？

- **CSV**: 轻量、快速、易分享（推荐日常使用）
- **joblib**: 完整数据（需要完整样本权重时使用）

### Q: 如何添加新实验？

运行训练脚本后，结果会自动保存到 `experiments/` 文件夹。

---

## 📞 获取帮助

```bash
python visualize_from_results.py --help
```

---

**最后更新：** 2024年  
**可用实验数：** 5  
**推荐工具：** `visualize_from_results.py` ⭐

🎉 **秒级可视化任何已完成的实验！**





