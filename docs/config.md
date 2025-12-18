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

    * 自动创建实验目录 `<experiments-dir>/<name>/`
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
  "test_size": 0.2,
  "random_state": 42,
  "use_feature": "hog",
  "feature_config": {
    "hog_params": { ... },
    "hu_params": { ... }
  },
  "training_noise_config": { ... },
  "test_shift_config": { ... }
}
```

### 3.1 噪声配置（training_noise_config）

```json
"training_noise_config": {
  "ratio": 0.2,
  "label_flip": true,
  "gaussian": { "std": 0.05 },
  "salt_pepper": { "amount": 0.02 },
  "blur": { "kernel_size": 3 },
  "contrast": { "factor_range": [0.7, 1.3] },
  "rotate": { "angle_range": 10 },
  "brightness": { "shift_range": 0.2 }
}
```

#### 字段说明

* `ratio`：噪声样本在训练集中的比例。
* `label_flip`：是否进行标签翻转噪声。
* `gaussian`：高斯噪声的标准差。
* `salt_pepper`：椒盐噪声的占比。
* `blur`：高斯模糊的内核大小。
* `contrast`：对比度变化的范围。
* `rotate`：随机旋转的角度范围。
* `brightness`：亮度扰动的范围。

这些噪声配置可用于模拟不同的干扰情况，从而帮助研究算法对噪声的鲁棒性。

---

### 3.2 数据划分与随机性控制

```json
"test_size": 0.2,
"random_state": 42
```

* `test_size`：测试集占比。
* `random_state`：控制数据划分、噪声采样和模型随机性。

建议在对比实验中固定该值，确保实验的可复现性。

---

### 3.3 特征提取方式（use_feature）

```json
"use_feature": "hog"
```

可选值：

* `"original"`：直接使用原始像素。
* `"hog"`：Histogram of Oriented Gradients（HOG 特征）。
* `"hu"`：Hu 不变矩。

特征选择会显著影响 AdaBoost 的训练效果及性能表现。

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
这些参数控制 HOG 特征的分辨率与局部统计方式。

---

### 3.5 Hu 矩特征参数（hu_params）

```json
"hu_params": {
  "log_scale": true
}
```

* `log_scale`：是否对 Hu 矩进行对数缩放，增强数值稳定性。

---

### 3.6 测试集风格构造（test_shift_config）
在测试集中引入特定扰动以评估模型的泛化能力。test_shift_config下可以引入任意数量的键值对，每个键值对对应一种扰动方式及其参数，键名即为该扰动测试集在实验中的名称。
```json
"test_shift_config": {
  "contrast": {
    "factor_range": [0.7, 1.3]
  },
  "rotate": {
    "angle_range": 10
  },
  "brightness": {
    "shift_range": 0.2
  },
  "multiple_disturbances_test": {
    "contrast": {
      "factor_range": [0.5, 1.5]
    },
    "rotate": {
      "angle_range": 15
    },
    "brightness": {
      "shift_range": 0.3
    }
  }
}
```
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

* `use_monitor`：是否启用 BoostMonitor。
* `is_data_noisy`：是否记录噪声样本的指标。
* `checkpoint_interval`：每 N 轮保存一次 checkpoint。

### 注意事项

* 使用 `--viz` 或 `--viz-only` 模式时，必须启用 `use_monitor`。
* 监控信息会显著增加存储与内存开销，但对于研究型实验是必要的。

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

### 5.1 基学习器配置（estimator）

```json
"estimator": {
  "max_depth": 2,
  "criterion": "entropy",
  "max_features": 0.25,
  "random_state": 42
}
```

该部分定义了 AdaBoost 中使用的弱分类器（决策树）参数。

* `max_depth`：决策树的最大深度，较小的值有助于防止过拟合。
* `criterion`：选择分裂标准 `"entropy"` 或 `"gini"`。
* `max_features`：每棵树使用的特征比例，用于引入随机性。

---

### 5.2 AdaBoost 参数

```json
"n_estimators": 500,
"learning_rate": 0.5,
"random_state": 42
```

* `n_estimators`：AdaBoost 中弱分类器的数量。
* `learning_rate`：控制每轮学习的步长，较小的值通常带来更平滑的权重演化。

---

## 6. 实验可复现性建议

为了保证实验的可复现性，建议：

1. 固定 `random_state` 为 42。
2. 稍微修改参数时可不修改实验名称，AdaLab 会自动按时间戳创建新的实验目录。
3. 大量修改时最好使用新的实验名称和配置文件。
