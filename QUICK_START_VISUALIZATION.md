# 🎨 可视化快速开始

## 📊 两种方式

### 方式1：基础版（默认，5-10分钟）

```bash
python visualize_overfitting.py
```

**生成：** 2个子图
- 学习曲线
- 过拟合程度

---

### 方式2：增强版（详细监控，6-12分钟）⭐

**第1步：** 编辑 `visualize_overfitting.py` 第138行

```python
enable_detailed_monitoring = True  # 改为 True
```

**第2步：** 运行

```bash
python visualize_overfitting.py
```

**生成：** 2 + 6 = 8个子图
- 过拟合分析（2个）
- **详细训练监控（6个）** ⭐
  1. 错误率演化
  2. Alpha 系数
  3. 准确率曲线
  4. 噪声 vs 干净样本权重
  5. F1 分数演化
  6. 样本权重分布

---

## 📋 可视化清单（基于 monitor.md）

| 可视化内容 | 数据来源 | 启用条件 |
|----------|---------|---------|
| ✅ 学习曲线 | 基础训练 | 总是显示 |
| ✅ 过拟合程度 | 基础训练 | 总是显示 |
| ⭐ 错误率演化 | `monitor.error_history` | 监控启用 |
| ⭐ Alpha系数 | `monitor.alpha_history` | 监控启用 |
| ⭐ 准确率曲线 | `monitor.acc_on_train_data` | 监控启用 |
| ⭐ 噪声影响 | `monitor.noisy_weight_history` | 监控启用 + 噪声数据 |
| ⭐ F1演化 | `monitor.f1_on_training_data` | 监控启用 |
| ⭐ 权重分布 | `monitor.sample_weights_history` | 监控启用 |

---

## 🎯 推荐工作流

```bash
# 第1次：快速诊断
python visualize_overfitting.py
# → 找到最佳 n_estimators

# 第2次：详细分析（可选）
# 编辑: enable_detailed_monitoring = True
python visualize_overfitting.py
# → 深入理解训练动态
```

---

## 📚 相关文档

- `docs/monitor.md` - 数据结构说明
- `docs/VISUALIZATION_ENHANCEMENT.md` - 详细修改说明
- `docs/visualization_guide.md` - 完整使用指南

---

**快速提示：** 修改一行代码即可启用全部6个详细监控图！




