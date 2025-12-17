# 配置文件说明

AdaLab 的所有实验均通过 **JSON 配置文件**进行驱动。
该配置文件定义了一次完整实验的 **实验元信息、数据处理方式、噪声注入策略、监控机制以及模型参数**。

实验运行过程中不会修改配置文件本身，而是将其作为**实验的不可变输入**，与实验结果一同保存，用于实验复现与对比分析。

---

## 1. 配置文件总体结构

一个标准的配置文件由以下几个顶层字段组成：

```json
{
  "experiment": { ... },
  "data": { ... },
  "monitor": { ... },
  "model": { ... }
}
```

各模块职责如下：

* `experiment`：实验的元信息与目录组织
* `data`：数据划分、特征提取与噪声注入策略
* `monitor`：AdaBoost 训练过程监控与 checkpoint 设置
* `model`：AdaBoost 及其基学习器的参数

---

## 2. experiment：实验基本信息

```json
"experiment": {
  "name": "your_experiment_name"
}
```

### 字段说明

* `name`

  * 类型：`string`
  * 含义：实验名称
  * 作用：

    * 自动创建实验目录 `experiments/<name>/`
    * 所有模型、日志、评估结果与可视化文件都会保存在该目录下

### 设计说明

实验名称在 AdaLab 中具有**唯一标识作用**。
建议使用能反映实验关键信息的命名方式，例如：

```
baseline_est500_depth2
noise20_hog_entropy
robust_flip_vs_gaussian
```

---

## 3. data：数据与特征配置

```json
"data": {
  "noise_config": { ... },
  "test_size": 0.2,
  "random_state": 42,
  "use_feature": "hog",
  "hog_params": { ... },
  "hu_params": { ... }
}
```

---

### 3.1 噪声配置（noise_config）

```json
"noise_config": {
  "ratio": 0.2,
  "label_flip": true,
  "gaussian": {"std": 0.05},
  "salt_pepper": {"amount": 0.02},
  "contrast": {"factor_range": [0.7, 1.3]},
  "rotate": {"angle_range": 10},
  "blur": {"kernel_size": 3},
  "brightness": {"shift_range": 0.2}
}
```

#### 字段说明

* `ratio`

  * 类型：`float`
  * 含义：噪声样本在训练集中的比例
  * 例如 `0.2` 表示 20% 的训练样本会被选为噪声样本

* `label_flip`

  * 类型：`bool`
  * 含义：是否进行标签翻转噪声
  * 常用于研究 boosting 对标签噪声的敏感性

* 图像扰动类噪声（可组合使用）：

  * `gaussian`：高斯噪声
  * `salt_pepper`：椒盐噪声
  * `contrast`：对比度变化
  * `rotate`：随机旋转
  * `blur`：高斯模糊
  * `brightness`：亮度扰动

#### 设计说明

* 噪声仅作用于 **训练集**
* 噪声样本索引会被显式记录，用于后续：

  * 权重演化分析
  * 噪声样本与干净样本的对比统计
* 多种噪声可以同时启用，用于构造复杂扰动场景

---

### 3.2 数据划分与随机性控制

```json
"test_size": 0.2,
"random_state": 42
```

* `test_size`

  * 测试集比例
* `random_state`

  * 控制数据划分、噪声采样和模型随机性
  * 强烈建议在对比实验中固定该值

---

### 3.3 特征提取方式（use_feature）

```json
"use_feature": "hog"
```

可选值：

* `"original"`：直接使用原始像素
* `"hog"`：Histogram of Oriented Gradients
* `"hu"`：Hu 不变矩

不同特征会显著影响 AdaBoost 的：

* 过拟合速度
* 对噪声的敏感性
* 泛化行为

---

### 3.4 HOG 特征参数（hog_params）

```json
"hog_params": {
  "orientations": 9,
  "pixels_per_cell": [2, 2],
  "cells_per_block": [2, 2]
}
```

仅在 `use_feature = "hog"` 时生效。

这些参数直接传递给 HOG 特征提取器，用于控制特征分辨率与局部统计方式。

---

### 3.5 Hu 矩特征参数（hu_params）

```json
"hu_params": {
  "log_scale": true
}
```

* `log_scale`

  * 是否对 Hu 矩进行对数缩放
  * 有助于数值稳定性与尺度对齐

---

## 4. monitor：训练监控配置

```json
"monitor": {
  "use_monitor": true,
  "is_data_noisy": true,
  "checkpoint_interval": 10
}
```

### 字段说明

* `use_monitor`

  * 是否启用 BoostMonitor
  * 若为 `false`，则不记录权重与 alpha 历史

* `is_data_noisy`

  * 是否启用噪声样本相关统计
  * 需要与 `data.noise_config` 配合使用

* `checkpoint_interval`

  * 每 N 轮 boosting 保存一次 checkpoint
  * checkpoint 可用于中断恢复与后处理分析

### 注意事项

* 使用 `--viz` 或 `--viz-only` 模式时，必须启用 `use_monitor`
* 监控信息会显著增加存储与内存开销，但对于研究型实验是必要的

---

## 5. model：模型与算法参数

```json
"model": {
  "estimator": { ... },
  "n_estimators": 500,
  "learning_rate": 0.5,
  "random_state": 42
}
```

---

### 5.1 基学习器配置（estimator）

```json
"estimator": {
  "max_depth": 2,
  "criterion": "entropy",
  "max_features": 0.25,
  "random_state": 42
}
```

该部分定义 AdaBoost 中使用的弱分类器（决策树）。

* `max_depth`

  * 决策树深度
  * 深度越大，单个弱分类器拟合能力越强，但更易过拟合

* `criterion`

  * `"entropy"` 或 `"gini"`
  * 在高维稀疏特征下，`entropy` 往往更稳定

* `max_features`

  * 每棵树使用的特征比例
  * 可用于引入随机性，缓解过拟合

---

### 5.2 AdaBoost 参数

```json
"n_estimators": 500,
"learning_rate": 0.5,
"random_state": 42
```

* `n_estimators`

  * boosting 轮数
  * 是研究过拟合行为时的重要变量

* `learning_rate`

  * 控制每轮 alpha 的缩放
  * 较小的值通常带来更平滑的权重演化

---

## 6. 实验可复现性建议

为了保证实验的可复现性，建议：

1. 固定 `random_state`为42
2. 稍微修改参数时可不修改实验名称，adalab会自动按之间戳创建新的实验文目录
3. 大量修改时最好使用新的实验名和配置文件
