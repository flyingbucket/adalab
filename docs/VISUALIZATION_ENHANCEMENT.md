# 可视化增强说明

## ✅ 修改内容

参考 `docs/monitor.md` 中的数据结构，增强了 `visualize_overfitting.py` 的功能。

---

## 📊 新增功能

### 1. **基础功能（保持不变）**
- ✅ 过拟合分析（学习曲线 + 过拟合程度）
- ✅ 自动识别最佳弱学习器数量
- ✅ 智能建议

### 2. **新增功能：详细训练监控** ⭐

通过设置 `enable_detailed_monitoring=True` 启用，生成 **6个子图** 的详细训练过程可视化：

#### 子图1：错误率演化
- **蓝色实线**：加权错误率（Weighted Error）
- **红色虚线**：未加权错误率（Unweighted Error）
- **数据来源**：`monitor.error_history`, `monitor.error_without_weight_history`

#### 子图2：Alpha 系数演化
- **绿色曲线**：每个弱学习器的权重系数 α
- **橙色虚线**：平均 alpha 值
- **数据来源**：`monitor.alpha_history`

#### 子图3：训练 vs 验证准确率
- **蓝色曲线**：训练集准确率
- **红色曲线**：验证集准确率
- **数据来源**：`monitor.acc_on_train_data`, `monitor.val_acc_history`

#### 子图4：噪声样本 vs 干净样本权重
- **红色曲线**：噪声样本总权重
- **绿色曲线**：干净样本总权重
- **黑色虚线**：初始均衡线（0.5）
- **数据来源**：`monitor.noisy_weight_history`, `monitor.clean_weight_history`
- **注意**：仅噪声数据时显示

#### 子图5：F1 分数演化
- **蓝色曲线**：训练集 F1
- **红色曲线**：验证集 F1
- **数据来源**：`monitor.f1_on_training_data`, `monitor.val_f1_history`

#### 子图6：样本权重分布变化
- **箱型图**：显示4个关键轮次的权重分布
  - 第1轮
  - 1/3处
  - 2/3处
  - 最后一轮
- **数据来源**：`monitor.sample_weights_history`

---

## 🚀 使用方法

### 方式1：基础使用（默认）

```bash
python visualize_overfitting.py
```

**生成内容：**
- ✅ 过拟合分析图（2个子图）
- ✅ 文本分析报告

### 方式2：启用详细监控

**修改脚本第138行：**
```python
enable_detailed_monitoring = True  # 改为 True
```

**运行：**
```bash
python visualize_overfitting.py
```

**生成内容：**
- ✅ 过拟合分析图（2个子图）
- ✅ 详细训练监控图（6个子图）⭐ 新增
- ✅ 增强的文本分析报告

### 方式3：使用增强版脚本

```bash
python visualize_overfitting_enhanced.py
```

这是完全独立的增强版本，默认启用所有监控功能。

---

## 📋 对比表

| 特性 | 原版本 | 修改后（默认） | 修改后（监控启用） | 增强版脚本 |
|-----|-------|--------------|-----------------|-----------|
| 过拟合分析 | ✅ | ✅ | ✅ | ✅ |
| 错误率演化 | ❌ | ❌ | ✅ | ✅ |
| Alpha系数 | ❌ | ❌ | ✅ | ✅ |
| 准确率曲线 | ❌ | ❌ | ✅ | ✅ |
| 噪声影响分析 | ❌ | ❌ | ✅ | ✅ |
| F1分数演化 | ❌ | ❌ | ✅ | ✅ |
| 权重分布 | ❌ | ❌ | ✅ | ✅ |
| 运行时间 | 5-10分钟 | 5-10分钟 | 6-12分钟 | 6-12分钟 |

---

## 🔍 数据来源映射

根据 `docs/monitor.md`，所有可视化数据来自 `BoostMonitor` 对象：

```python
# 错误率相关（2.2节）
monitor.error_history              # 加权错误率 → 子图1
monitor.error_without_weight_history  # 未加权错误率 → 子图1
monitor.alpha_history              # Alpha系数 → 子图2

# 验证集指标（2.3节）
monitor.val_acc_history            # 验证准确率 → 子图3
monitor.val_f1_history             # 验证F1 → 子图5

# 训练集指标（2.4节）
monitor.acc_on_train_data          # 训练准确率 → 子图3
monitor.f1_on_training_data        # 训练F1 → 子图5

# 样本权重相关（2.1节）
monitor.sample_weights_history     # 完整权重向量 → 子图6
monitor.noisy_weight_history       # 噪声样本权重和 → 子图4
monitor.clean_weight_history       # 干净样本权重和 → 子图4
```

---

## 💡 使用建议

### 场景1：快速诊断
```python
# 使用默认配置
enable_detailed_monitoring = False  # 默认
```
- ⏱️ 运行时间：5-10分钟
- 📊 获得：过拟合分析
- 🎯 适合：快速找到最佳配置

### 场景2：深入分析
```python
# 启用详细监控
enable_detailed_monitoring = True
```
- ⏱️ 运行时间：6-12分钟
- 📊 获得：过拟合分析 + 详细训练动态
- 🎯 适合：研究训练过程、论文撰写

### 场景3：噪声研究
```python
# 启用监控 + 使用噪声数据
enable_detailed_monitoring = True
choice = 2  # 或 3（5%或10%噪声）
```
- 📊 获得：噪声样本权重演化分析
- 🎯 适合：研究 AdaBoost 对噪声的敏感性

---

## 📈 可视化示例说明

### 正常训练模式
```
错误率下降 → Alpha稳定 → 准确率上升 → 权重分布合理
```

### 过拟合模式
```
训练错误率持续下降 → 测试准确率开始下降 → 过拟合程度增加
```

### 噪声敏感模式
```
噪声样本权重持续上升 → 超过干净样本 → 模型过度关注噪声
```

---

## 🎯 关键洞察

通过详细监控可以发现：

1. **Alpha 系数变化**
   - 如果后期 alpha 值很小 → 弱学习器贡献减少
   - 如果 alpha 波动大 → 训练不稳定

2. **样本权重演化**
   - 正常：权重逐渐集中到难分类样本
   - 异常：少数样本权重爆炸

3. **噪声影响**
   - 噪声样本权重 > 干净样本 → 模型被噪声误导
   - 比值 > 1.5 → 建议使用鲁棒方法

4. **过拟合信号**
   - 训练准确率持续上升，测试准确率平稳/下降
   - 加权错误率和未加权错误率差距拉大

---

## 🔧 自定义可视化

### 添加新的子图

在 `visualize_monitor_data()` 函数中添加：

```python
# 示例：添加第7个子图 - 错误率差异
ax7 = plt.subplot(3, 3, 7)  # 改为3x3布局
error_diff = np.array(monitor.error_history) - np.array(monitor.error_without_weight_history)
ax7.plot(rounds, error_diff, 'purple', linewidth=2)
ax7.set_title('Error Rate Difference')
ax7.grid(True, alpha=0.3)
```

### 保存图表

修改函数调用：
```python
visualize_monitor_data(monitor, best_n, noise_ratio > 0, save_path='monitoring.png')
```

---

## 📚 相关文档

- [monitor.md](monitor.md) - BoostMonitor 数据结构详解
- [visualization_guide.md](visualization_guide.md) - 可视化使用指南
- [overfitting_visualization_guide.md](overfitting_visualization_guide.md) - 过拟合分析API

---

## ⚙️ 配置选项

### 可调整参数

```python
# visualize_overfitting.py 第138行
enable_detailed_monitoring = True  # 启用/禁用详细监控

# 第25行
choice = 2  # 1=干净数据, 2=5%噪声, 3=10%噪声

# 第53-58行
config = {
    "base_estimator": DecisionTreeClassifier(max_depth=1),
    "n_estimators_list": [1, 5, 10, 20, 30, 40, 50, 75, 100],
    "learning_rate": 0.5,
    "random_state": 42,
}
```

---

## 🎉 总结

修改后的 `visualize_overfitting.py`：

✅ **向后兼容**：默认行为不变  
✅ **可选增强**：一键启用详细监控  
✅ **数据完整**：利用 monitor.md 中所有字段  
✅ **易于使用**：只需改一个变量  
✅ **灵活扩展**：易于添加新的可视化  

**推荐使用场景：**
- 研究 AdaBoost 训练动态
- 分析噪声影响
- 论文实验可视化
- 模型调优和诊断

---

**创建时间：** 2024年  
**参考文档：** docs/monitor.md  
**修改文件：** visualize_overfitting.py  
**新增文件：** visualize_overfitting_enhanced.py






