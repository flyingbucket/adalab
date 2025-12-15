# Val-After-Train 模式适配说明

## 📋 概述

本文档说明 `visualize_from_results.py` 对 **val-after-train（训练后验证）** 模式的适配实现。

---

## 🔧 问题背景

### 两种验证模式

#### 模式1：val-every-round（每轮验证）
```python
for i in range(n_estimators):
    train_one_round(i)
    validate(i)  # 每轮都验证
```
- **特点**：验证数据长度 = 训练轮数
- **横轴**：验证曲线的横轴就是训练轮次（1, 2, 3, ..., n）

#### 模式2：val-after-train（训练后采样验证）
```python
val_rounds = [10, 20, 50, 100, 200, 500]  # 指定验证轮次
for i in range(n_estimators):
    train_one_round(i)
    if (i+1) in val_rounds:
        validate(i)  # 仅在指定轮次验证
```
- **特点**：验证数据长度 < 训练轮数
- **横轴**：需要记录验证发生的实际轮次索引

---

## ⚠️ 可视化挑战

### 错误绘图示例（未适配）

如果直接使用 `range(len(val_acc_history))` 作为横轴：

```python
# ❌ 错误：横轴变成了0, 1, 2, 3, 4, 5
plt.plot(range(len(val_acc)), val_acc)
```

**结果**：
- 实际验证轮次：[10, 20, 50, 100, 200, 500]
- 图表显示轮次：[0, 1, 2, 3, 4, 5]
- **问题**：完全失去了"在哪个训练阶段进行验证"的语义

### 正确绘图（已适配）

```python
# ✅ 正确：使用实际验证轮次作为横轴
plt.plot(data['val_idx'], val_acc)  # [10, 20, 50, 100, 200, 500]
```

**效果**：
- 验证曲线正确反映了训练进度
- 可以与训练曲线对齐比较

---

## ✅ 实现方案

### 1. BoostMonitor 类增强

#### 新增字段
```python
class BoostMonitor:
    def __init__(self, ...):
        self.val_acc_history = []
        self.val_f1_history = []
        self.val_idx = []  # ✅ 新增：记录验证轮次索引
```

#### 修改记录方法
```python
def record_validation(self, iboost, acc, f1):
    """
    记录验证集指标
    
    Parameters
    ----------
    iboost : int
        当前轮次索引（0-based）
    acc : float
        验证集准确率
    f1 : float
        验证集F1分数
    """
    self.val_acc_history.append(acc)
    self.val_f1_history.append(f1)
    self.val_idx.append(iboost + 1)  # ✅ 记录1-based轮次
```

---

### 2. 数据加载器适配

#### load_from_joblib（完整支持）✅

```python
def load_from_joblib(joblib_path):
    monitor = joblib.load(joblib_path)
    data = {
        'val_acc_history': monitor.val_acc_history,
        'val_f1_history': monitor.val_f1_history,
        'val_idx': monitor.val_idx if hasattr(monitor, 'val_idx') else [],  # ✅
        # ... 其他字段
    }
    return data
```

**特点**：
- ✅ 完整恢复 `val_idx` 信息
- ✅ 向后兼容（旧版 monitor 没有 val_idx 时使用空列表）

#### load_from_csv（部分支持）⚠️

```python
def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    data = {
        'val_acc_history': df['val_acc'].tolist() if 'val_acc' in df.columns else [],
        'val_f1_history': df['val_f1'].tolist() if 'val_f1' in df.columns else [],
        'val_idx': df['val_idx'].tolist() if 'val_idx' in df.columns else [],  # ⚠️
        # ... 其他字段
    }
    return data
```

**限制**：
- ⚠️ CSV 格式通常不包含 `val_idx` 列
- ⚠️ 无法从 CSV 恢复 val-after-train 的完整语义
- ✅ 可视化时会自动降级为顺序索引

---

### 3. 可视化函数适配

#### visualize_training_data 函数

**ax3（准确率曲线）和 ax5（F1曲线）的适配：**

```python
# ========== 准确率演化 ==========
if len(data['val_acc_history']) > 0:
    # ✅ 智能选择横轴：优先使用 val_idx
    val_x = data['val_idx'] if len(data['val_idx']) == len(data['val_acc_history']) \
            else rounds[:len(data['val_acc_history'])]
    
    ax.plot(val_x, data['val_acc_history'], 'r-', 
            label='Val Accuracy', marker='s')
```

**逻辑说明：**
1. **条件判断**：`len(val_idx) == len(val_acc_history)`
   - ✅ 长度匹配 → 使用 `val_idx`（完整语义）
   - ❌ 长度不匹配或为空 → 降级为顺序索引

2. **降级策略**：`rounds[:len(val_acc_history)]`
   - 取前 N 个训练轮次作为横轴
   - 适用于 val-every-round 或数据不完整的情况

---

## 📊 可视化对比

### 场景示例

假设训练 500 轮，仅在特定轮次验证：

```python
# 训练配置
n_estimators = 500
val_rounds = [50, 100, 200, 300, 400, 500]

# 记录的数据
val_acc_history = [0.85, 0.88, 0.90, 0.91, 0.915, 0.92]  # 6个值
val_idx = [50, 100, 200, 300, 400, 500]  # 对应轮次
```

### 未适配效果（错误）❌

```
横轴：[0, 1, 2, 3, 4, 5]
纵轴：[0.85, 0.88, 0.90, 0.91, 0.915, 0.92]

问题：
- 看不出在哪个训练阶段验证
- 无法与训练曲线对齐
- 失去了时间轴语义
```

### 适配后效果（正确）✅

```
横轴：[50, 100, 200, 300, 400, 500]
纵轴：[0.85, 0.88, 0.90, 0.91, 0.915, 0.92]

优势：
- ✅ 清晰显示验证发生在哪些训练阶段
- ✅ 可以与训练曲线（1-500）直接对比
- ✅ 保留完整的时间轴语义
```

---

## 🚀 使用指南

### 推荐流程

#### 1. 训练时使用 BoostMonitor 记录
```python
from src.monitor import BoostMonitor

monitor = BoostMonitor(...)

# 训练循环
for i in range(n_estimators):
    # 训练...
    
    # 验证（仅在特定轮次）
    if (i + 1) in [50, 100, 200, 500]:
        val_acc, val_f1 = evaluate_on_val_set(model)
        monitor.record_validation(i, val_acc, val_f1)  # ✅ 自动记录 val_idx

# 保存 monitor
import joblib
joblib.dump(monitor, 'experiments/my_exp/results/monitor.joblib')
```

#### 2. 可视化时加载 joblib
```bash
# ✅ 推荐：从 joblib 加载（完整信息）
python scripts/visualization/visualize_from_results.py \
    --joblib experiments/my_exp/results/monitor.joblib
```

**输出示例：**
```
📂 Loading from joblib: experiments/my_exp/results/monitor.joblib
✓ Loaded BoostMonitor object
✓ Data fields available:
  ✓ val_idx                      # ✅ 可用
  ✓ val_acc_history
  ✓ val_f1_history

📊 Basic Info:
   - Total Rounds: 500
   - Validation Mode: val-after-train (sampled at 6 rounds)
   - Val Rounds: [50, 100, 200, 300, 400, 500]
```

#### 3. 从 CSV 加载（受限）
```bash
# ⚠️  受限：CSV 不包含 val_idx
python scripts/visualization/visualize_from_results.py \
    --csv experiments/my_exp/results/final_results.csv
```

**输出示例：**
```
📂 Loading from CSV: experiments/my_exp/results/final_results.csv
✓ Loaded 500 rounds of training data
✓ Data fields available:
  ✗ val_idx (⚠️  CSV format limitation - validation curves will use sequential indexing)
  ✓ val_acc_history
  ✓ val_f1_history

📊 Basic Info:
   - Total Rounds: 500
   - Validation Mode: val-every-round (or val_idx not recorded)
```

---

## ⚠️ 当前限制

### CSV 格式的局限性

#### 为什么 CSV 不支持 val_idx？

**原因1：数据结构不匹配**
- CSV 是行对齐的表格格式
- 每行对应一个训练轮次
- 但 val-after-train 模式下，验证数据是稀疏的

**示例冲突：**
```csv
round,train_acc,val_acc
1,0.80,        # ← val_acc 为空（未验证）
2,0.82,
...
50,0.85,0.85   # ← 第50轮有验证
...
100,0.88,0.88  # ← 第100轮有验证
```

**问题**：
- 如果 val_acc 列有空值，读取后无法区分"第50轮的验证"和"第1个验证点"
- 需要额外的 val_idx 列来记录索引，但这会让 CSV 变得复杂

**原因2：BoostMonitor.dump() 的设计**
```python
def dump(self, filename="monitor_log.csv"):
    # 以 error_history 为主轴（每轮一条）
    rounds = len(self.error_history)
    
    # 验证数据长度可能 < rounds
    if len(self.val_acc_history) == rounds:
        data["val_acc"] = self.val_acc_history
    else:
        data["val_acc"] = [None] * rounds  # ⚠️  填充 None
```

- CSV 导出时，验证数据被填充为 `None` 以对齐行数
- 丢失了"在哪些轮次验证"的信息

---

### 解决方案对比

| 方案 | val_idx 支持 | 数据完整性 | 推荐度 |
|------|-------------|-----------|--------|
| **joblib 格式** | ✅ 完整 | ✅ 100% | ⭐⭐⭐⭐⭐ |
| **CSV 格式** | ❌ 不支持 | ⚠️  部分丢失 | ⭐⭐⭐ |

**建议**：
- ✅ **优先使用 joblib** 保存和加载 BoostMonitor
- ⚠️  CSV 仅作为快速预览或与其他工具兼容的备选

---

## 📖 相关文档

- **BoostMonitor 类文档**：`docs/monitor.md`
- **可视化工具指南**：`docs/visualize_from_results_guide.md`
- **项目结构说明**：`PROJECT_STRUCTURE.md`

---

## 🔍 技术细节

### 子图适配清单

| 子图 | 位置 | 内容 | val_idx 适配 |
|------|------|------|-------------|
| ax1 | [0,0] | 错误率演化 | ❌ 不需要（训练数据） |
| ax2 | [0,1] | Alpha系数 | ❌ 不需要（训练数据） |
| ax3 | [0,2] | 训练vs验证准确率 | ✅ **验证曲线适配** |
| ax4 | [1,0] | 噪声vs干净样本权重 | ❌ 不需要（训练数据） |
| ax5 | [1,1] | 训练vs验证F1 | ✅ **验证曲线适配** |
| ax6 | [1,2] | 样本权重分布 | ❌ 不需要（箱型图） |

**适配重点**：
- ✅ ax3 和 ax5 的**验证曲线**使用 `val_idx` 作为横轴
- ✅ 训练曲线仍使用 `rounds`（每轮都有）
- ✅ 两条曲线在同一坐标系下可正确对比

---

## 🎓 最佳实践

### 1. 训练脚本模板
```python
import joblib
from src.monitor import BoostMonitor

# 初始化监控器
monitor = BoostMonitor(
    noise_indices=noise_idx,
    clean_indices=clean_idx,
    is_data_noisy=True
)

# 训练
for i in range(n_estimators):
    # 每轮训练...
    
    # val-after-train 模式：仅在特定轮次验证
    if (i + 1) % 50 == 0:  # 每50轮验证一次
        val_acc, val_f1 = evaluate_on_val_set(clf)
        monitor.record_validation(i, val_acc, val_f1)  # ✅ 自动记录 val_idx

# 保存完整监控数据
os.makedirs('experiments/my_exp/results', exist_ok=True)
joblib.dump(monitor, 'experiments/my_exp/results/monitor.joblib')  # ✅ 推荐
monitor.dump('experiments/my_exp/results/final_results.csv')      # 可选
```

### 2. 可视化脚本
```bash
# ✅ 推荐：完整语义
python scripts/visualization/visualize_from_results.py \
    --joblib experiments/my_exp/results/monitor.joblib \
    --save outputs/figures/my_exp.png

# ⚠️  降级：CSV格式
python scripts/visualization/visualize_from_results.py \
    --csv experiments/my_exp/results/final_results.csv
```

---

## ✅ 适配状态总结

### 已完成 ✅

1. **BoostMonitor 类增强**
   - ✅ 新增 `val_idx` 字段
   - ✅ `record_validation()` 自动记录轮次索引
   - ✅ 向后兼容旧版代码

2. **数据加载器适配**
   - ✅ `load_from_joblib()` 完整支持 val_idx
   - ✅ `load_from_csv()` 添加占位和说明
   - ✅ 智能降级策略

3. **可视化函数适配**
   - ✅ ax3（准确率）使用 val_idx 作为横轴
   - ✅ ax5（F1分数）使用 val_idx 作为横轴
   - ✅ 自动检测和降级处理

4. **用户提示优化**
   - ✅ 数据加载时显示 val_idx 状态
   - ✅ CSV 限制的明确警告
   - ✅ 验证模式的自动识别

### 限制说明 ⚠️

- ⚠️  **CSV 格式不支持 val_idx**
  - 原因：数据结构不匹配
  - 影响：无法完整恢复 val-after-train 语义
  - 降级：使用顺序索引作为横轴

---

**最后更新：** 2025-12-15  
**维护者：** ML项目团队

