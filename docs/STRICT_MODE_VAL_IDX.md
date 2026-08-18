# 🔒 严格模式：val_idx 必需字段说明

**更新日期：** 2025-12-15  
**版本：** 2.0 - 严格模式

---

## ⚠️ 重要变更

**从即日起，`visualize_from_results.py` 采用严格模式：**

- ❌ **不再支持降级和回退机制**
- ❌ **不再兼容没有 val_idx 的旧数据**
- ✅ **必须使用包含 val_idx 的数据源**

---

## 🔒 严格模式要求

### 1. BoostMonitor 必须有 val_idx 字段

**加载 joblib 时的检查：**
```python
if not hasattr(monitor, "val_idx"):
    raise AttributeError(
        "BoostMonitor 对象缺少 'val_idx' 字段！\n"
        "请使用更新后的 BoostMonitor 类重新训练模型。"
    )
```

**后果：**
- ❌ 旧版本的 monitor.joblib（没有 val_idx）将无法加载
- ✅ 必须使用新版 BoostMonitor 重新训练

---

### 2. CSV 必须包含 val_idx 列

**加载 CSV 时的检查：**
```python
if "val_idx" not in df.columns:
    raise ValueError(
        "CSV 文件缺少 'val_idx' 列！\n"
        "CSV 格式无法完整支持 val-after-train 模式。\n"
        "请使用 joblib 格式加载数据。"
    )
```

**后果：**
- ❌ 旧版本的 CSV（没有 val_idx 列）将无法加载
- ✅ 强烈推荐使用 joblib 格式而不是 CSV

---

### 3. val_idx 必须与验证数据长度匹配

**可视化时的检查：**
```python
if len(data["val_idx"]) != len(data["val_acc_history"]):
    raise ValueError(
        f"val_idx 长度 ({len(data['val_idx'])}) 与 val_acc_history 长度不匹配！\n"
        "数据完整性检查失败。"
    )
```

**后果：**
- ❌ 数据不一致将直接报错
- ✅ 确保数据完整性和可靠性

---

## ✅ 如何符合严格模式要求

### 方法1：使用更新后的 BoostMonitor（推荐）

```python
from src.monitor import BoostMonitor
import joblib

# 1. 初始化监控器
monitor = BoostMonitor(
    noise_indices=noise_idx, clean_indices=clean_idx, is_data_noisy=True
)

# 2. 训练循环
for i in range(n_estimators):
    # 训练...

    # 验证（每N轮或指定轮次）
    if should_validate(i):
        val_acc, val_f1 = evaluate_on_val_set(clf)
        monitor.record_validation(i, val_acc, val_f1)  # ✅ 自动记录 val_idx

# 3. 保存（必须使用 joblib）
joblib.dump(monitor, "experiments/my_exp/results/monitor.joblib")  # ✅ 推荐
```

**关键点：**
- ✅ `monitor.record_validation()` 会自动记录 `val_idx`
- ✅ 使用 `joblib.dump()` 保存完整对象
- ❌ 不推荐使用 `monitor.dump()` 导出 CSV（会丢失信息）

---

### 方法2：手动构造包含 val_idx 的 CSV（不推荐）

如果必须使用 CSV 格式：

```python
import pandas as pd

# 确保包含 val_idx 列
data = {
    "round": [1, 2, 3, ..., 500],
    "weighted_error": [...],
    "alpha": [...],
    "val_acc": [None, None, ..., 0.85, None, ...],  # 稀疏
    "val_idx": [None, None, ..., 50, None, ...],  # ✅ 必须有此列
    # ... 其他列
}

df = pd.DataFrame(data)
df.to_csv("monitor.csv", index=False)
```

**问题：**
- ⚠️  CSV 中大量 None 值，不优雅
- ⚠️  难以维护和理解
- ⚠️  不如 joblib 格式直观

---

## 🚫 不再支持的情况

### ❌ 情况1：旧版 BoostMonitor（没有 val_idx）

```python
# 旧版 monitor.joblib
python scripts/visualization/visualize_from_results.py \
    --joblib old_monitor.joblib

# 报错：
# AttributeError: BoostMonitor 对象缺少 'val_idx' 字段！
```

**解决方案：**
- 使用新版 BoostMonitor 重新训练模型

---

### ❌ 情况2：旧版 CSV（没有 val_idx 列）

```python
# 旧版 CSV
python scripts/visualization/visualize_from_results.py \
    --csv old_results.csv

# 报错：
# ValueError: CSV 文件缺少 'val_idx' 列！
```

**解决方案：**
- 使用新版 BoostMonitor 重新训练并导出
- 或使用 joblib 格式

---

### ❌ 情况3：val_idx 长度不匹配

```python
# 数据不一致
data = {
    "val_acc_history": [0.85, 0.88, 0.90],  # 3个值
    "val_idx": [50, 100],  # 2个值 ❌
}

# 报错：
# ValueError: val_idx 长度 (2) 与 val_acc_history 长度 (3) 不匹配！
```

**解决方案：**
- 检查训练代码，确保每次调用 `record_validation()` 都正确

---

## 📊 严格模式的优势

### 1. 数据完整性保证 ✅
- 不会出现"降级"导致的语义丢失
- 验证曲线横轴始终正确反映训练进度

### 2. 错误提前发现 ✅
- 数据问题在加载时立即报错
- 不会在生成图表后才发现问题

### 3. 强制最佳实践 ✅
- 鼓励使用 joblib 格式（完整、可靠）
- 避免使用 CSV 格式的局限性

### 4. 代码简化 ✅
- 不需要复杂的降级逻辑
- 减少边界情况处理

---

## 🔄 迁移指南

### 从旧版本迁移

#### 步骤1：更新 BoostMonitor 类
```bash
# 确保使用最新版本的 src/monitor.py
git pull origin main
# 或手动检查 BoostMonitor.__init__() 中是否有 self.val_idx = []
```

#### 步骤2：重新训练模型
```python
# 使用新版 BoostMonitor 训练
python scripts/training/main_experiment.py --config configs/my_config.json
```

#### 步骤3：验证数据格式
```python
import joblib

# 检查 monitor 是否有 val_idx
monitor = joblib.load("experiments/my_exp/results/monitor.joblib")
print(f"val_idx 可用: {hasattr(monitor, 'val_idx')}")
print(f"val_idx 长度: {len(monitor.val_idx)}")
print(f"val_acc_history 长度: {len(monitor.val_acc_history)}")
```

#### 步骤4：可视化
```bash
# 现在可以正常使用
python scripts/visualization/visualize_from_results.py \
    --joblib experiments/my_exp/results/monitor.joblib
```

---

## 🆚 版本对比

| 特性 | 旧版本（兼容模式） | 新版本（严格模式） |
|------|------------------|------------------|
| 缺少 val_idx | ⚠️  降级为顺序索引 | ❌ 报错 |
| CSV 无 val_idx | ⚠️  使用空列表 | ❌ 报错 |
| 长度不匹配 | ⚠️  截断或填充 | ❌ 报错 |
| 向后兼容 | ✅ 支持旧数据 | ❌ 不支持 |
| 数据完整性 | ⚠️  部分保证 | ✅ 完全保证 |
| 错误检测 | ⚠️  延迟发现 | ✅ 立即报错 |
| 代码复杂度 | 🔴 高（降级逻辑） | 🟢 低（简洁明了） |

---

## ❓ 常见问题

### Q1: 我的旧实验数据怎么办？

**A:** 有两个选择：
1. **推荐**：使用新版 BoostMonitor 重新训练
2. **临时方案**：手动添加 val_idx 到 CSV（不推荐）

### Q2: 为什么要采用严格模式？

**A:** 严格模式的优势：
- ✅ 保证数据完整性和语义正确性
- ✅ 提前发现数据问题
- ✅ 简化代码逻辑
- ✅ 强制最佳实践

### Q3: 可以关闭严格模式吗？

**A:** 不建议关闭。如果确实需要兼容旧数据：
1. 使用旧版本的 `visualize_from_results.py`
2. 或在加载时手动处理异常

### Q4: CSV 格式还能用吗？

**A:** 可以，但有限制：
- ✅ 必须包含 val_idx 列
- ⚠️  需要手动维护列的对齐
- 🔴 不推荐，优先使用 joblib

---

## 📝 错误信息速查

### 错误1：缺少 val_idx 字段
```
AttributeError: BoostMonitor 对象缺少 'val_idx' 字段！
请使用更新后的 BoostMonitor 类重新训练模型。
```
**解决**：使用新版 BoostMonitor 重新训练

### 错误2：CSV 缺少 val_idx 列
```
ValueError: CSV 文件缺少 'val_idx' 列！
CSV 格式无法完整支持 val-after-train 模式。
请使用 joblib 格式加载数据。
```
**解决**：使用 joblib 格式，或手动添加 val_idx 列到 CSV

### 错误3：长度不匹配
```
ValueError: val_idx 长度 (5) 与 val_acc_history 长度 (6) 不匹配！
数据完整性检查失败。请确保 BoostMonitor.record_validation() 正确调用。
```
**解决**：检查训练代码，确保每次验证都调用 `record_validation()`

---

## 🎯 最佳实践总结

### ✅ 推荐做法

1. **使用 joblib 格式**
   ```python
   joblib.dump(monitor, "monitor.joblib")
   ```

2. **每次验证都记录**
   ```python
   monitor.record_validation(i, val_acc, val_f1)
   ```

3. **加载前验证**
   ```python
   monitor = joblib.load("monitor.joblib")
   assert hasattr(monitor, "val_idx"), "缺少 val_idx 字段"
   assert len(monitor.val_idx) == len(monitor.val_acc_history), "长度不匹配"
   ```

### ❌ 避免做法

1. **不要使用旧版 BoostMonitor**
2. **不要使用 CSV 格式保存监控数据**（除非有特殊需求）
3. **不要跳过 record_validation() 调用**

---

**严格模式已启用！确保数据完整性和可靠性。** 🔒

**相关文档：**
- `docs/val_after_train_mode.md` - 详细技术文档
- `src/monitor.py` - BoostMonitor 类实现
- `scripts/visualization/visualize_from_results.py` - 可视化脚本

