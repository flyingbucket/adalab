# 从已保存结果可视化训练过程

## 🎯 核心功能

**不需要重新训练**，直接从已保存的结果文件（CSV 或 joblib）生成可视化！

---

## 🚀 快速开始

### 方式1：从实验文件夹加载（推荐）

```bash
python visualize_from_results.py --experiment train_val_500rounds
```

**自动查找：**
1. 优先 `experiments/train_val_500rounds/results/monitor.joblib`
2. 备选 `experiments/train_val_500rounds/results/final_results.csv`

---

### 方式2：直接指定 CSV 文件

```bash
python visualize_from_results.py --csv experiments/train_val_500rounds/results/final_results.csv
```

---

### 方式3：直接指定 joblib 文件

```bash
python visualize_from_results.py --joblib experiments/my_exp/results/monitor.joblib
```

---

## 📊 输出内容

### 1. **文本摘要**

```
============================================================
                      Training Summary                      
============================================================

📊 Basic Info:
   - Total Rounds: 500
   - Data Type: Noisy

📈 Final Metrics:
   - Final Val Accuracy: 0.8321
   - Best Val Accuracy:  0.8321 (round 500)

🔍 Error Analysis:
   - Initial Error: 0.6700
   - Final Error:   0.8794

⚖️ Alpha Analysis:
   - Mean Alpha: 0.098
   - Std Alpha:  0.113

💡 Noise Analysis:
   - Final Noisy Weight:   0.5554
   - Final Clean Weight:   0.4446
   - Weight Ratio (noisy/clean): 1.249
   ⚠️  Noisy samples slightly over-weighted
============================================================
```

### 2. **可视化图表（6个子图）**

1. **错误率演化** - 加权 vs 未加权
2. **Alpha 系数** - 弱学习器权重
3. **准确率曲线** - 训练 vs 验证
4. **噪声影响** - 噪声样本 vs 干净样本权重
5. **F1 分数** - 训练 vs 验证
6. **样本权重分布** - 关键轮次对比

---

## 🔧 命令选项

```bash
python visualize_from_results.py [选项]

必选（三选一）：
  --experiment, -e    实验名称（如 train_val_500rounds）
  --csv, -c          CSV 文件路径
  --joblib, -j       joblib 文件路径

可选：
  --save, -s         保存图表到指定路径
  --no-plot          只显示摘要，不生成图表
```

---

## 📝 使用示例

### 示例1：快速查看摘要

```bash
python visualize_from_results.py --experiment train_val_500rounds --no-plot
```

只显示文本摘要，不生成图表（速度快）

---

### 示例2：生成并保存图表

```bash
python visualize_from_results.py --experiment train_val_500rounds --save my_analysis.png
```

生成图表并保存为 `my_analysis.png`

---

### 示例3：对比多个实验

```bash
# 实验1
python visualize_from_results.py -e baseline_est500_depth2 -s baseline.png

# 实验2
python visualize_from_results.py -e noise5_est500_depth2 -s noise5.png

# 对比两张图
```

---

### 示例4：分析特定 checkpoint

```bash
python visualize_from_results.py --csv experiments/train_val_500rounds/checkpoints/round_0250.csv
```

加载第250轮的 checkpoint 数据

---

## 📊 CSV vs Joblib 对比

| 数据字段 | CSV | Joblib | 说明 |
|---------|-----|--------|------|
| 错误率（加权/未加权） | ✅ | ✅ | `error_history` |
| Alpha 系数 | ✅ | ✅ | `alpha_history` |
| 验证准确率/F1 | ✅ | ✅ | `val_acc_history` |
| 训练准确率/F1 | ❌ | ✅ | `acc_on_train_data` |
| 噪声/干净样本权重和 | ✅ | ✅ | `noisy_weight_history` |
| 完整样本权重向量 | ❌ | ✅ | `sample_weights_history` |

**建议：**
- ✅ **CSV**: 轻量、易读、适合快速查看
- ✅ **joblib**: 完整数据、适合深度分析

---

## 🔍 可视化详解

### 子图1：错误率演化
- **蓝色实线**: 加权错误率（AdaBoost 实际使用）
- **红色虚线**: 未加权错误率（原始错误率）
- **趋势**: 加权错误率上升 → 难分类样本权重增加

### 子图2：Alpha 系数
- **绿色曲线**: 每个弱学习器的贡献权重
- **橙色虚线**: 平均值
- **分析**: 
  - α 大 → 强学习器
  - α 小 → 弱学习器
  - 后期 α 持续减小 → 收益递减

### 子图3：准确率曲线
- **蓝色**: 训练准确率（仅 joblib）
- **红色**: 验证准确率
- **警告**: 
  - 两曲线分离 → 过拟合
  - 验证下降 → 严重过拟合

### 子图4：噪声影响（仅噪声数据）
- **红色**: 噪声样本总权重
- **绿色**: 干净样本总权重
- **黑色虚线**: 初始均衡线（0.5）
- **警告**: 
  - 红线 > 绿线 → 噪声被过度关注
  - 比值 > 1.5 → 建议用鲁棒方法

### 子图5：F1 分数
- 类似准确率，但对类别不平衡更敏感

### 子图6：样本权重分布
- **箱型图**: 显示4个关键轮次
- **趋势**: 权重方差增大 → 模型聚焦难样本
- **注意**: 仅 joblib 格式包含此数据

---

## 🎨 自定义分析

### 读取数据进行自定义分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
df = pd.read_csv('experiments/train_val_500rounds/results/final_results.csv')

# 自定义绘图
plt.figure(figsize=(10, 6))
plt.plot(df['round'], df['val_acc'], label='Validation Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Custom Analysis')
plt.legend()
plt.grid(True)
plt.savefig('custom_plot.png', dpi=300)
plt.show()
```

---

## 📂 项目中可用的实验

运行以下命令查看所有实验：

```bash
ls experiments/
```

常见实验：
- `baseline_est500_depth2` - 基线模型
- `noise5_est500_depth2` - 5% 噪声
- `train_val_500rounds` - 500轮训练
- `test_experiment_wrapper` - 测试实验

---

## 🔥 典型工作流

### 场景1：训练完成后分析

```bash
# 第1步：训练模型（生成结果文件）
python train_with_noise_track.py

# 第2步：可视化分析
python visualize_from_results.py --experiment my_experiment --save analysis.png
```

---

### 场景2：对比不同配置

```bash
# 加载多个实验
for exp in baseline_est500_depth2 noise5_est500_depth2; do
    python visualize_from_results.py -e $exp -s ${exp}.png
done

# 对比生成的图表
```

---

### 场景3：查看训练进展

```bash
# 查看 checkpoint（训练中途）
python visualize_from_results.py --csv experiments/train_val_500rounds/checkpoints/round_0100.csv

# 查看最终结果
python visualize_from_results.py --csv experiments/train_val_500rounds/results/final_results.csv
```

---

## ⚠️ 注意事项

### CSV 数据限制

CSV 文件**不包含**：
1. 训练集准确率 (`acc_on_train_data`)
2. 训练集 F1 (`f1_on_training_data`)
3. 完整样本权重 (`sample_weights_history`)

这些字段在可视化时会显示 "N/A" 或提示信息。

**解决方案：** 使用 joblib 格式获取完整数据。

---

### Joblib 依赖

如果使用 joblib 格式，需要确保：
1. `src.monitor.BoostMonitor` 类定义未变
2. Python 环境一致

---

## 🎉 优势总结

| 对比项 | 重新训练 | 从结果加载 |
|-------|---------|-----------|
| ⏱️ 时间 | 5-10分钟 | < 5秒 |
| 💾 资源 | 需要数据集 | 只需结果文件 |
| 🔄 灵活性 | 一次一个配置 | 快速对比多个 |
| 📊 数据完整性 | 100% | CSV 80% / joblib 100% |

**推荐使用场景：**
- ✅ 快速回顾历史实验
- ✅ 生成论文图表
- ✅ 对比不同配置
- ✅ 分享实验结果（只需发送CSV）

---

## 📚 相关文档

- `docs/monitor.md` - BoostMonitor 数据结构
- `docs/visualization_guide.md` - 完整可视化指南
- `docs/VISUALIZATION_ENHANCEMENT.md` - 增强功能说明

---

**创建时间：** 2024年  
**工具文件：** `visualize_from_results.py`  
**支持格式：** CSV, joblib

---

## 快速参考

```bash
# 最常用命令
python visualize_from_results.py -e train_val_500rounds

# 保存图表
python visualize_from_results.py -e train_val_500rounds -s result.png

# 只看摘要
python visualize_from_results.py -e train_val_500rounds --no-plot
```

🎯 **一键从已保存结果生成专业可视化！**

