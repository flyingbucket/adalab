# 泛化能力测试指南

本文档说明如何测试AdaBoost模型对视觉扰动的鲁棒性。

## 测试原理

### 什么是泛化能力？

**泛化能力**是指模型在**未见过的数据**上的表现能力。

### 为什么要测试视觉扰动？

在实际应用中，输入数据可能存在：
- 🌓 **亮度变化**：不同光照条件
- 🌫️ **噪声**：传感器噪声、压缩噪声
- 🔲 **模糊**：失焦、运动模糊
- 📐 **几何变换**：轻微旋转、平移
- 🎨 **对比度**：显示设备差异

**好的模型应该对这些扰动具有鲁棒性。**

---

## 测试方案

### 测试流程

```
1. 在标准MNIST上训练模型
   ↓
2. 生成带扰动的测试集
   ↓
3. 在扰动测试集上评估
   ↓
4. 对比分析性能下降
```

### 扰动类型

我们实现了7种扰动：

#### 1. 亮度偏移 🌓

**效果：** 整体变亮或变暗

**参数：** `shift_range` - 偏移范围

```python
# 亮度 ± 30%
X_perturbed = X + shift  # shift ∈ [-0.3, 0.3]
```

**模拟场景：**
- 不同光照条件
- 曝光差异
- 扫描仪设置不同

#### 2. 高斯噪声 📡

**效果：** 添加随机噪声

**参数：** `noise_std` - 噪声标准差

```python
# 高斯噪声 σ=0.1
X_perturbed = X + N(0, 0.1²)
```

**模拟场景：**
- 传感器噪声
- 数字化噪声
- 通信干扰

#### 3. 椒盐噪声 ⚫⚪

**效果：** 随机黑点和白点

**参数：** `amount` - 噪声比例

```python
# 5% 椒盐噪声
# 2.5% 像素变为黑色
# 2.5% 像素变为白色
```

**模拟场景：**
- 图像采集错误
- 传输错误
- 损坏的像素

#### 4. 模糊 🌫️

**效果：** 边缘模糊

**参数：** `kernel_size` - 模糊核大小

```python
# 3x3 平均滤波
X_perturbed = uniform_filter(X, size=3)
```

**模拟场景：**
- 失焦
- 运动模糊
- 低分辨率

#### 5. 对比度调整 🎨

**效果：** 增强或减弱对比度

**参数：** `factor_range` - 对比度因子范围

```python
# 对比度 ± 50%
X_perturbed = mean + factor * (X - mean)
# factor ∈ [0.5, 1.5]
```

**模拟场景：**
- 显示设备差异
- 图像处理差异
- 扫描质量

#### 6. 旋转 🔄

**效果：** 轻微旋转

**参数：** `angle_range` - 旋转角度范围（度）

```python
# 旋转 ± 15°
X_perturbed = rotate(X, angle)
# angle ∈ [-15°, 15°]
```

**模拟场景：**
- 手写倾斜
- 拍照角度
- 扫描未对齐

---

## 使用方法

### 快速开始

```bash
python test_generalization.py
```

**运行时间：** 约10-15分钟

**输出：**
1. `results/perturbation_examples.png` - 扰动示例展示
2. `results/generalization_test.png` - 泛化测试结果
3. 控制台详细报告

### 测试流程

```
步骤1: 准备数据
步骤2: 创建数据扰动器
步骤3: 训练模型
  - 标准AdaBoost
  - 鲁棒AdaBoost
步骤4: 测试泛化能力（17种扰动配置）
步骤5: 生成可视化
步骤6: 打印总结报告
```

---

## 结果解读

### 输出示例

```
标准AdaBoost:
  基线准确率（无扰动）: 0.8200 (82.00%)
  平均准确率（有扰动）: 0.7450 (74.50%)
  平均准确率下降: 0.0750 (7.50%)
  最大准确率下降: 0.1500 (15.00%)
  最难扰动: 旋转±15° (准确率: 0.6700)

鲁棒AdaBoost:
  基线准确率（无扰动）: 0.8100 (81.00%)
  平均准确率（有扰动）: 0.7600 (76.00%)
  平均准确率下降: 0.0500 (5.00%)
  最大准确率下降: 0.1000 (10.00%)
  最难扰动: 旋转±15° (准确率: 0.7100)
```

### 关键指标

#### 1. 基线准确率

**定义：** 在无扰动测试集上的准确率

**意义：** 模型的基本性能

#### 2. 平均准确率（有扰动）

**定义：** 在所有扰动测试集上的平均准确率

**意义：** 整体泛化能力

**评价标准：**
- > 基线-5%：✅ 优秀
- 基线-5%至-10%：✅ 良好
- 基线-10%至-15%：⚠️ 一般
- < 基线-15%：❌ 较差

#### 3. 平均准确率下降

**定义：** 基线准确率 - 平均准确率

**意义：** 对扰动的平均敏感度

**评价标准：**
- < 5%：✅ 鲁棒性优秀
- 5-10%：✅ 鲁棒性良好
- 10-15%：⚠️ 鲁棒性一般
- > 15%：❌ 鲁棒性较差

#### 4. 最大准确率下降

**定义：** 所有扰动中最大的准确率下降

**意义：** 最脆弱的情况

**用途：** 发现模型的弱点

#### 5. 最难扰动

**定义：** 准确率下降最多的扰动类型

**意义：** 需要重点改进的方向

---

## 可视化说明

### 图1：扰动示例展示

**内容：** 7种扰动类型 × 5个样本

**用途：** 
- 直观理解每种扰动的效果
- 验证扰动强度是否合适

### 图2：泛化测试结果

包含两个子图：

#### 子图1：不同扰动下的准确率

**横轴：** 扰动类型（17种配置）  
**纵轴：** 准确率

**解读：**
- 柱子越高越好
- 对比不同模型的柱子高度
- 找出哪些扰动导致性能下降最多

#### 子图2：准确率下降幅度

**横轴：** 扰动类型  
**纵轴：** 准确率下降

**颜色：**
- 🟢 绿色：负值（性能提升，罕见）
- 🔴 红色：正值（性能下降，常见）

**解读：**
- 柱子越低（接近0）越好
- 红色区域越小越好
- 对比不同模型的鲁棒性

---

## 典型结果分析

### 场景1：标准MNIST数据

**预期结果：**
```
基线准确率: 75-85%
平均下降: 5-10%
最难扰动: 旋转±15°
```

**原因：**
- MNIST训练数据已经是标准化的
- 对简单扰动（亮度、噪声）有一定抗性
- 对几何变换（旋转）较敏感

### 场景2：对比标准vs鲁棒AdaBoost

**预期差异：**

| 指标 | 标准AdaBoost | 鲁棒AdaBoost | 改进 |
|------|-------------|-------------|------|
| 基线准确率 | 82% | 81% | -1% |
| 平均准确率 | 75% | 77% | +2% |
| 平均下降 | 7% | 4% | -3% |

**结论：**
- 鲁棒方法可能牺牲1-2%基线性能
- 但在扰动数据上表现更好
- 整体鲁棒性提升30-40%

---

## 改进建议

### 如果泛化能力差：

#### 1. 数据增强

```python
# 训练时添加扰动
from sklearn.utils import shuffle

X_train_augmented = []
y_train_augmented = []

for X_batch, y_batch in batches:
    # 原始数据
    X_train_augmented.append(X_batch)
    y_train_augmented.append(y_batch)
    
    # 添加扰动版本
    X_perturbed = perturber.add_gaussian_noise(X_batch, noise_std=0.05)
    X_train_augmented.append(X_perturbed)
    y_train_augmented.append(y_batch)

# 合并并训练
X_train_aug = np.vstack(X_train_augmented)
y_train_aug = np.hstack(y_train_augmented)
clf.fit(X_train_aug, y_train_aug)
```

#### 2. 使用鲁棒方法

```python
from src.robust_adaboost import create_robust_adaboost

# 使用鲁棒AdaBoost
clf = create_robust_adaboost(strategy="balanced")
```

#### 3. 正则化

```python
# 使用更保守的配置
clf = AdaBoostClassifier(
    n_estimators=30,  # 减少弱学习器
    learning_rate=0.3,  # 降低学习率
    base_estimator=DecisionTreeClassifier(max_depth=1),  # 简单基学习器
)
```

#### 4. 集成方法

```python
from sklearn.ensemble import VotingClassifier

# 多模型投票
clf = VotingClassifier(
    [
        ("adaboost", clf_adaboost),
        ("rf", clf_random_forest),
        ("svm", clf_svm),
    ],
    voting="soft",
)
```

---

## 扩展测试

### 自定义扰动

```python
from test_generalization import MNISTPerturber

perturber = MNISTPerturber()


# 自定义扰动
def my_perturbation(X):
    # 你的扰动逻辑
    return X_perturbed


# 测试
X_test_perturbed = my_perturbation(X_test)
acc = clf.score(X_test_perturbed, y_test)
```

### 组合扰动

```python
# 同时应用多种扰动
X_perturbed = X_test
X_perturbed = perturber.add_gaussian_noise(X_perturbed, noise_std=0.05)
X_perturbed = perturber.add_brightness_shift(X_perturbed, shift_range=0.1)
X_perturbed = perturber.add_blur(X_perturbed, kernel_size=3)

# 测试
acc = clf.score(X_perturbed, y_test)
print(f"组合扰动准确率: {acc:.4f}")
```

### 逐步增强扰动

```python
import matplotlib.pyplot as plt

# 测试不同强度
noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
accuracies = []

for noise_std in noise_levels:
    X_perturbed = perturber.add_gaussian_noise(X_test, noise_std=noise_std)
    acc = clf.score(X_perturbed, y_test)
    accuracies.append(acc)

# 绘制
plt.plot(noise_levels, accuracies, "o-")
plt.xlabel("噪声标准差")
plt.ylabel("准确率")
plt.title("准确率 vs 噪声强度")
plt.grid(True)
plt.show()
```

---

## 常见问题

### Q1: 为什么某些扰动准确率反而提升？

**A:** 可能的原因：
1. 扰动起到了"正则化"作用
2. 去除了过拟合的细节
3. 统计波动（样本量不够）

### Q2: 如何选择扰动强度？

**A:** 建议：
- 从实际应用场景出发
- 参考现有研究
- 逐步增强测试

### Q3: 泛化测试需要多长时间？

**A:** 
- 完整测试（17种扰动）：约10-15分钟
- 可以减少测试配置加快速度

### Q4: 如何提升泛化能力？

**A:** 优先级：
1. 数据增强（最有效）
2. 使用鲁棒方法
3. 正则化
4. 集成方法

### Q5: 测试结果如何用于论文？

**A:** 可以：
1. 展示扰动示例图
2. 绘制准确率曲线
3. 制作对比表格
4. 分析鲁棒性改进

---

## 参考资料

1. **数据增强**: https://arxiv.org/abs/1712.04621
2. **对抗鲁棒性**: https://arxiv.org/abs/1706.06083
3. **测试集扰动**: https://arxiv.org/abs/1903.12261

---

**最后更新：** 2024年  
**维护者：** ML项目组




