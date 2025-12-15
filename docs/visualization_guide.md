# 模型拟合过程可视化完整指南

本指南详细说明如何在项目中可视化 AdaBoost 模型的拟合过程。

## 📊 可视化方法总览

项目提供三种主要的可视化方式：

| 方法 | 文件 | 主要功能 | 适用场景 |
|-----|-----|---------|---------|
| 过拟合可视化 | `visualize_overfitting.py` | 学习曲线、过拟合分析 | 快速诊断、参数选择 |
| 训练监控 | `train_with_noise_track.py` | 实时追踪训练过程 | 深入分析、样本权重研究 |
| 鲁棒方法对比 | `compare_robust_methods.py` | 多方法性能对比 | 方法选择、效果验证 |

---

## 🌟 方法1：过拟合可视化（推荐）

### 快速开始

```bash
python visualize_overfitting.py
```

### 详细用法

```python
from sklearn.tree import DecisionTreeClassifier
from src.utils import prepare_data
from src.evaluation import visualize_overfitting_process

# 1. 准备数据
X_train, X_test, y_train, y_test, _, _ = prepare_data(noise_ratio=0.05)

# 2. 运行可视化
results = visualize_overfitting_process(
    X_train, y_train, X_test, y_test,
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators_list=[1, 5, 10, 20, 30, 40, 50, 75, 100],
    learning_rate=0.5,
    random_state=42,
    save_path='overfitting_analysis.png'  # None则直接显示
)
```

### 参数说明

#### `n_estimators_list` - 测试点配置

```python
# 配置1: 快速测试（推荐）
n_estimators_list=[1, 10, 30, 50, 100]  # 5个点，约3分钟

# 配置2: 标准测试（默认）
n_estimators_list=[1, 5, 10, 20, 30, 40, 50, 75, 100]  # 9个点，约5-10分钟

# 配置3: 精细分析
n_estimators_list=list(range(1, 51, 2))  # 25个点，约15-20分钟

# 配置4: 扩展范围
n_estimators_list=[1, 10, 20, 50, 100, 150, 200]  # 测试更多弱学习器
```

#### `base_estimator` - 基学习器配置

```python
# 决策树桩（最常用）
base_estimator=DecisionTreeClassifier(max_depth=1)

# 深度3的树（更容易过拟合）
base_estimator=DecisionTreeClassifier(max_depth=3)

# 深度5的树（观察严重过拟合）
base_estimator=DecisionTreeClassifier(max_depth=5)
```

#### `learning_rate` - 学习率

```python
# 高学习率（收敛快，容易过拟合）
learning_rate=1.0

# 标准学习率（推荐）
learning_rate=0.5

# 低学习率（收敛慢，泛化好）
learning_rate=0.1
```

### 输出解读

#### 图表1：学习曲线

```
准确率
  │
1.0├─────────────────  蓝色：训练准确率（持续上升）
  │            ╱
0.9│          ╱  ──── 红色：测试准确率（可能平稳或下降）
  │        ╱   ╱
0.8│      ╱  ╱    ★   绿色星标：最佳模型点
  │    ╱  ╱     
0.7│  ╱ ╱          🟠 橙色区域：过拟合差距
  │╱ ╱
  └──────────────────> 弱学习器数量
```

**关键信息：**
- 两曲线差距 = 过拟合程度
- 测试准确率峰值 = 最佳弱学习器数量
- 测试准确率下降 = 严重过拟合警告

#### 图表2：过拟合程度曲线

```
过拟合度
  │
0.2├─     ╱──╲
  │    ╱      ╲     🟥 红色：过拟合区域
0.1├─ ╱    ★   ──╲  ★ 最小过拟合点
  │ ╱
0.0├─────────────────  黑色虚线：完美拟合
  └──────────────────> 弱学习器数量
```

**评价标准：**

| 过拟合程度 | 评价 | 说明 |
|----------|------|------|
| < 0.05 | ✅ 优秀 | 模型泛化良好 |
| 0.05-0.10 | ✅ 良好 | 轻微过拟合，可接受 |
| 0.10-0.15 | ⚠️ 一般 | 中度过拟合 |
| 0.15-0.20 | ⚠️ 较差 | 明显过拟合 |
| > 0.20 | ❌ 差 | 严重过拟合 |

#### 文本报告

```text
============================================================
            AdaBoost 过拟合分析总结
============================================================

最佳模型:
  弱学习器数量: 40
  测试集准确率: 0.8156 (81.56%)
  训练集准确率: 0.9234 (92.34%)
  过拟合程度: 0.1078 (10.78%)

最小过拟合模型:
  弱学习器数量: 20
  过拟合程度: 0.0645 (6.45%)
  测试集准确率: 0.7923

趋势分析:
  初始 (n=1): 测试准确率 = 0.6234, 过拟合 = 0.0156
  最终 (n=100): 测试准确率 = 0.8034, 过拟合 = 0.1534
  ⚠️ 警告: 测试准确率在 n=40 后开始下降，建议使用早停
============================================================
```

---

## 📝 方法2：训练监控

### 快速开始

```bash
# 噪声数据训练（推荐，更能展示问题）
python train_with_noise_track.py

# 干净数据训练
python train_with_clean_data.py
```

### 实时输出示例

```text
[BOOST] 5/50 | error=0.0234 | alpha=1.2345 | noisy_w=0.023456
[VAL]   round=005 | acc=  0.7234 | f1=  0.7156
[TRAIN] round=005 | acc=  0.8456 | f1=  0.8389

[BOOST] 10/50 | error=0.0189 | alpha=1.3456 | noisy_w=0.034567
[VAL]   round=010 | acc=  0.7456 | f1=  0.7389
[TRAIN] round=010 | acc=  0.8789 | f1=  0.8712

[CHECKPOINT] Saved 'experiments/my_exp/checkpoints/round_0050.csv' (round=50)
```

### 检查点数据

每隔50轮自动保存CSV文件，包含：

```csv
round,weighted_error,alpha,acc_on_training_data,val_acc_history,noisy_weight,clean_weight
1,0.0245,1.2134,0.7234,0.6789,0.0123,0.9877
2,0.0234,1.2345,0.7456,0.7012,0.0156,0.9844
...
```

### 使用监控数据进行自定义可视化

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取检查点数据
df = pd.read_csv('experiments/my_exp/results/final_results.csv')

# 绘制训练曲线
plt.figure(figsize=(12, 5))

# 子图1：准确率演化
plt.subplot(1, 2, 1)
plt.plot(df['round'], df['acc_on_training_data'], label='训练准确率')
plt.plot(df['round'], df['val_acc_history'], label='验证准确率')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

# 子图2：样本权重演化
plt.subplot(1, 2, 2)
plt.plot(df['round'], df['noisy_weight'], label='噪声样本权重', color='red')
plt.plot(df['round'], df['clean_weight'], label='干净样本权重', color='green')
plt.xlabel('训练轮次')
plt.ylabel('权重总和')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('custom_monitoring.png', dpi=300)
plt.show()
```

---

## 🛡️ 方法3：鲁棒方法对比

### 快速演示

```bash
python demo_robust.py
```

**输出：**
- 标准AdaBoost vs 鲁棒改进方法的性能对比
- 过拟合程度对比
- 建议使用的方法

### 完整实验对比

```bash
python compare_robust_methods.py
```

**测试的方法：**
1. 标准AdaBoost（baseline）
2. 权重裁剪（weight_clipping）
3. 早停（early_stopping）
4. 权重平滑（weight_smoothing）
5. 保守学习率（conservative）
6. 平衡配置（balanced）

**输出示例：**

```text
============================================================
            鲁棒方法性能对比
============================================================

方法排名（按测试准确率）:
1. balanced:           测试=0.8234, 过拟合=0.0845
2. early_stopping:     测试=0.8189, 过拟合=0.0789
3. weight_clipping:    测试=0.8156, 过拟合=0.0912
4. baseline:           测试=0.7789, 过拟合=0.1234

推荐方法: balanced
理由: 最高测试准确率 + 较低过拟合程度
============================================================
```

---

## 🔬 实验场景

### 场景1：确定最佳弱学习器数量

**目标：** 找到性能最优的弱学习器数量

```python
# 步骤1：运行过拟合可视化
results = visualize_overfitting_process(
    X_train, y_train, X_test, y_test,
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators_list=[1, 5, 10, 20, 30, 40, 50, 75, 100],
    learning_rate=0.5
)

# 步骤2：提取最佳配置
best_idx = results["test_accuracy"].index(max(results["test_accuracy"]))
best_n_estimators = results["n_estimators"][best_idx]
print(f"最佳弱学习器数量: {best_n_estimators}")

# 步骤3：使用最佳配置训练最终模型
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=best_n_estimators,
    learning_rate=0.5
)
clf.fit(X_train, y_train)
```

### 场景2：对比干净数据 vs 噪声数据

**目标：** 研究噪声对过拟合的影响

```python
noise_levels = [0, 0.05, 0.10]

for noise in noise_levels:
    print(f"\n测试噪声水平: {noise*100}%")
    
    X_train, X_test, y_train, y_test, _, _ = prepare_data(noise_ratio=noise)
    
    results = visualize_overfitting_process(
        X_train, y_train, X_test, y_test,
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators_list=[1, 5, 10, 20, 30, 40, 50, 75, 100],
        learning_rate=0.5,
        save_path=f'results/noise_{int(noise*100)}.png'
    )
```

**预期发现：**
- 噪声数据最佳弱学习器数量更少
- 噪声数据过拟合更严重
- 噪声数据测试准确率峰值更低

### 场景3：对比不同树深度

**目标：** 研究基学习器复杂度的影响

```python
depths = [1, 3, 5]

for depth in depths:
    print(f"\n测试树深度: {depth}")
    
    results = visualize_overfitting_process(
        X_train, y_train, X_test, y_test,
        base_estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators_list=[1, 5, 10, 20, 30, 40, 50],
        learning_rate=0.5,
        save_path=f'results/depth_{depth}.png'
    )
```

**预期发现：**
- 深树收敛更快（需要更少弱学习器）
- 深树更容易过拟合
- 树桩（depth=1）泛化最好

### 场景4：对比不同学习率

**目标：** 找到最优学习率

```python
learning_rates = [0.1, 0.3, 0.5, 1.0]

for lr in learning_rates:
    print(f"\n测试学习率: {lr}")
    
    results = visualize_overfitting_process(
        X_train, y_train, X_test, y_test,
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators_list=[1, 10, 20, 30, 50, 75, 100, 150, 200],
        learning_rate=lr,
        save_path=f'results/lr_{lr}.png'
    )
```

**预期发现：**
- 低学习率需要更多弱学习器
- 高学习率收敛快但容易过拟合
- 0.3-0.5通常是平衡点

---

## 📈 高级可视化技巧

### 技巧1：对比多个配置

```python
import matplotlib.pyplot as plt
import numpy as np

configs = [
    {"depth": 1, "lr": 0.5, "label": "树桩+标准LR"},
    {"depth": 3, "lr": 0.5, "label": "深树+标准LR"},
    {"depth": 1, "lr": 0.1, "label": "树桩+低LR"},
]

plt.figure(figsize=(12, 5))

# 子图1：测试准确率对比
plt.subplot(1, 2, 1)
for config in configs:
    results = visualize_overfitting_process(
        X_train, y_train, X_test, y_test,
        base_estimator=DecisionTreeClassifier(max_depth=config["depth"]),
        n_estimators_list=[1, 10, 20, 30, 50, 75, 100],
        learning_rate=config["lr"],
        save_path=None  # 不保存
    )
    plt.plot(results["n_estimators"], results["test_accuracy"], 
             label=config["label"], marker='o')

plt.xlabel('弱学习器数量')
plt.ylabel('测试准确率')
plt.legend()
plt.grid(True)
plt.title('测试准确率对比')

# 子图2：过拟合程度对比
plt.subplot(1, 2, 2)
for config in configs:
    results = visualize_overfitting_process(
        X_train, y_train, X_test, y_test,
        base_estimator=DecisionTreeClassifier(max_depth=config["depth"]),
        n_estimators_list=[1, 10, 20, 30, 50, 75, 100],
        learning_rate=config["lr"],
        save_path=None
    )
    plt.plot(results["n_estimators"], results["overfitting_degree"], 
             label=config["label"], marker='s')

plt.xlabel('弱学习器数量')
plt.ylabel('过拟合程度')
plt.legend()
plt.grid(True)
plt.title('过拟合程度对比')

plt.tight_layout()
plt.savefig('multi_config_comparison.png', dpi=300)
plt.show()
```

### 技巧2：制作动画演示训练过程

```python
import matplotlib.animation as animation

# 读取检查点数据
df = pd.read_csv('experiments/my_exp/results/final_results.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

def update(frame):
    ax1.clear()
    ax2.clear()
    
    # 子图1：准确率演化
    ax1.plot(df['round'][:frame], df['acc_on_training_data'][:frame], 
             label='训练', color='blue')
    ax1.plot(df['round'][:frame], df['val_acc_history'][:frame], 
             label='验证', color='red')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0.6, 1.0)
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'准确率演化 (轮次: {frame})')
    
    # 子图2：样本权重演化
    if 'noisy_weight' in df.columns:
        ax2.plot(df['round'][:frame], df['noisy_weight'][:frame], 
                 label='噪声样本', color='red')
        ax2.plot(df['round'][:frame], df['clean_weight'][:frame], 
                 label='干净样本', color='green')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('权重总和')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title(f'样本权重演化 (轮次: {frame})')

ani = animation.FuncAnimation(fig, update, frames=len(df), interval=100)
ani.save('training_animation.gif', writer='pillow', fps=10)
```

### 技巧3：生成论文级图表

```python
# 设置论文风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# 生成高质量图表
results = visualize_overfitting_process(
    X_train, y_train, X_test, y_test,
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators_list=[1, 5, 10, 20, 30, 40, 50, 75, 100],
    learning_rate=0.5,
    save_path='paper_figure1.pdf'  # PDF格式，矢量图
)
```

---

## ⚠️ 常见问题

### Q1: 为什么测试准确率会下降？

**A:** 这是严重过拟合的信号。原因：
- 弱学习器数量过多
- 学习率过高
- 基学习器过于复杂
- 数据中有噪声

**解决方案：**
1. 使用早停，在峰值点停止
2. 降低学习率（0.5 → 0.1）
3. 使用更简单的基学习器（树桩）
4. 使用鲁棒方法（见 `demo_robust.py`）

### Q2: 如何确定是否过拟合？

**A:** 看两个指标：
1. **过拟合程度** = 训练准确率 - 测试准确率
   - < 10%: 正常
   - 10-15%: 可接受
   - > 15%: 需要改进

2. **测试准确率趋势**
   - 持续上升: 良好
   - 平稳: 可接受
   - 下降: ⚠️ 警告

### Q3: 训练时间太长怎么办？

**A:** 减少测试点：

```python
# 从这个（9个点，10分钟）
n_estimators_list=[1, 5, 10, 20, 30, 40, 50, 75, 100]

# 改为这个（5个点，5分钟）
n_estimators_list=[1, 10, 30, 50, 100]
```

### Q4: 如何保存图表？

**A:** 设置 `save_path` 参数：

```python
# PNG格式（屏幕展示）
save_path='my_result.png'

# PDF格式（论文/打印）
save_path='my_result.pdf'

# None（只显示不保存）
save_path=None
```

### Q5: 如何对比多个实验结果？

**A:** 三种方法：

**方法1：保存多个图表**
```python
for noise in [0, 0.05, 0.10]:
    visualize_overfitting_process(
        ...,
        save_path=f'noise_{int(noise*100)}.png'
    )
```

**方法2：使用监控数据**
```python
# 读取多个实验的CSV
df1 = pd.read_csv('exp1/final_results.csv')
df2 = pd.read_csv('exp2/final_results.csv')

# 绘制对比
plt.plot(df1['round'], df1['val_acc_history'], label='实验1')
plt.plot(df2['round'], df2['val_acc_history'], label='实验2')
plt.legend()
plt.show()
```

**方法3：使用对比脚本**
```bash
python compare_robust_methods.py
```

### Q6: 图表中文显示异常？

**A:** 运行字体初始化：

```python
from mplfonts.bin.cli import init
init()  # 首次运行，自动下载中文字体
```

如果还不行：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# 或
plt.rcParams['font.sans-serif'] = ['PingFang SC']  # Mac
```

---

## 📚 相关文档

- [过拟合可视化指南](overfitting_visualization_guide.md) - 详细的API文档
- [鲁棒AdaBoost指南](robust_adaboost_guide.md) - 改进方法说明
- [特征重要性指南](feature_importance_guide.md) - 特征分析
- [评估指南](evaluation_guide.md) - 性能评估详解

---

## 🎯 最佳实践总结

### 推荐工作流

```bash
# 第1步：快速诊断（5分钟）
python visualize_overfitting.py

# 第2步：查看报告，确定问题
# - 是否过拟合？
# - 最佳弱学习器数量？
# - 是否需要改进？

# 第3步：如果过拟合严重，尝试鲁棒方法（10分钟）
python demo_robust.py

# 第4步：详细分析（可选）
python train_with_noise_track.py
```

### 论文/报告撰写

```python
# 生成所有需要的图表
configs = {
    "baseline": {"noise": 0},
    "noise_5": {"noise": 0.05},
    "noise_10": {"noise": 0.10},
}

for name, config in configs.items():
    X_train, X_test, y_train, y_test, _, _ = prepare_data(
        noise_ratio=config["noise"]
    )
    
    visualize_overfitting_process(
        X_train, y_train, X_test, y_test,
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators_list=[1, 5, 10, 20, 30, 40, 50, 75, 100],
        learning_rate=0.5,
        save_path=f'paper_figures/{name}.pdf'
    )
```

### 模型调优

```python
# 第1步：找最佳配置
results = visualize_overfitting_process(...)
best_n = results['n_estimators'][np.argmax(results['test_accuracy'])]

# 第2步：使用最佳配置
clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=best_n,
    learning_rate=0.5
)
clf.fit(X_train, y_train)

# 第3步：最终评估
test_score = clf.score(X_test, y_test)
print(f"最终测试准确率: {test_score:.4f}")
```

---

**最后更新：** 2024年  
**维护者：** ML项目组






