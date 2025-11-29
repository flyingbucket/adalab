# Experiment Pipeline Documentation

本项目提供了统一的实验构建与训练流程包装器，使得实验目录结构、数据准备、模型构建、训练过程与结果保存完全自动化。

核心函数包括：

* `build_experiment(config_path)`
* `train_and_save(config_path)`

本文档主要介绍参数说明与返回值格式。

---

## 1. 函数：`build_experiment(config_path)`

负责：

* 读取配置文件
* 创建实验目录
* 加载数据、构建 HOG / HU 特征
* 返回模型、Monitor、数据集与保存结果的目录

通常不直接被用户调用；建议使用外层函数 `train_and_save()`。

---

## 2. 函数：`train_and_save(config_path)`

**这是推荐用户直接调用的包装器。**

功能：

1. 调用 `build_experiment()` 构建数据、模型、Monitor
2. 模型训练
3. 自动保存（未压缩 + 压缩）的模型和监控文件
4. 返回训练对象、数据对象、路径字典

---

## 3. `train_and_save()` 返回值说明

```python
(
    clf,        # 训练好的 AdaBoostClassifier / AdaBoostClfWithMonitor
    monitor,    # BoostMonitor 对象（如果启用）否则 None
    prep,       # DataPreparation 对象（包含噪声设置、特征方式等）
    data,       # (X_train, X_test, y_train, y_test, noise_idx, clean_idx)
    paths       # 输出文件路径字典
)
```

### 3.1 `data` 内容

`data` 是一个六元组：

| 项目          | 内容         |
| ----------- | ---------- |
| `X_train`   | 训练特征矩阵     |
| `X_test`    | 测试特征矩阵     |
| `y_train`   | 训练标签       |
| `y_test`    | 测试标签       |
| `noise_idx` | 加噪声的样本索引列表 |
| `clean_idx` | 干净样本索引列表   |

---

### 3.2 `paths` 内容

`paths` 是一个字典：

| 键                      | 含义                                                  |
| ---------------------- | --------------------------------------------------- |
| `"raw_clf"`            | 未压缩模型文件 `model.joblib`                              |
| `"raw_monitor"`        | 未压缩 Monitor（如果启用，否则 None）                           |
| `"compressed_clf"`     | 压缩后的模型 `model.joblib.xz`                            |
| `"compressed_monitor"` | 压缩后的监控文件（如果启用）                                      |
| `"monitor_csv"`        | Monitor 保存的 CSV（如果 monitor 启用）                      |
| `"result_dir"`         | 实验的结果results目录=`f"paths["experiment_dir"]/results"` |
| "`experiment_dir`"     | 实验结果一级目录                                            |

---

## 4. 配置文件字段说明（config JSON）

一个典型的配置包含 4 个一级字段：

```json
{
  "experiment": { ... },
  "data": { ... },
  "monitor": { ... },
  "model": { ... }
}
```

以下逐个说明。

---

## 4.1 字段：`experiment`

| key    | 类型  | 含义                                 |
| ------ | --- | ---------------------------------- |
| `name` | str | 实验名称（将自动创建目录 `experiments/<name>`） |

---

## 4.2 字段：`data`

涉及数据处理、特征构造、噪声注入。

| key            | 类型                              | 含义           |
| -------------- | ------------------------------- | ------------ |
| `noise_ratio`  | float                           | 噪声比例（0–1）    |
| `test_size`    | float                           | 测试集比例        |
| `random_state` | int                             | 随机种子         |
| `use_feature`  | `"original"` / `"hog"` / `"hu"` | 使用特征类型,三选一   |
| `hog_params`   | dict                            | HOG 相关参数（可选） |
| `hu_params`    | dict                            | HU 矩参数（可选）   |

### HOG 参数

| key               | 默认值    | 含义           |
| ----------------- | ------ | ------------ |
| `orientations`    | 9      | 梯度方向数        |
| `pixels_per_cell` | [4, 4] | 每 cell 的像素大小 |
| `cells_per_block` | [2, 2] | block 结构     |

### HU 参数

| key         | 默认值  | 含义                         |
| ----------- | ---- | -------------------------- |
| `log_scale` | true | 是否对 Hu moments 做 log-scale |

---

## 4.3 字段：`monitor`

控制 BoostMonitor。

| key                   | 类型   | 默认值  | 含义                   |
| --------------------- | ---- | ---- | -------------------- |
| `use_monitor`         | bool | true | 是否启用 monitor。选择false则会回退至sklearn的原始AdaBoostClassifier,false模式应在不需要深入研究过拟合问题时使用，可以在一定程度上加快训练         |
| `is_data_noisy`       | bool | true | 是否记录噪声样本表现           |
| `checkpoint_interval` | int  | 10   | 每 N 轮保存一次 checkpoint |
| `checkpoint_prefix`   | str  | 自动填充 | checkpoint 路径        |

---

## 4.4 字段：`model`

构造 AdaBoost / AdaBoostClfWithMonitor 的参数。

| key             | 类型    | 含义                                 |
| --------------- | ----- | ---------------------------------- |
| `estimator`     | dict  | 基学习器的参数（用于 DecisionTreeClassifier） |
| `n_estimators`  | int   | AdaBoost 的弱分类器数量                   |
| `learning_rate` | float | AdaBoost 学习率                       |
| `random_state`  | int   | 随机数种子                              |

示例：

```json
"model": {
  "estimator": { "max_depth": 2 },
  "n_estimators": 500,
  "learning_rate": 0.5,
  "random_state": 42
}
```

---

## 5. 配置文件模板

```json
{
  "experiment": {
    "name": "your_experiment_name"  // 实验名称，将创建 experiments/<name>/
  },

  "data": {
    "noise_ratio": 0.0,             // 加噪声比例（0~1）
    "test_size": 0.2,               // 测试集划分比例
    "random_state": 42,             // 随机种子
    "use_feature": "hog",           // 可选： "original" / "hog" / "hu"

    // HOG 特征参数（use_feature = "hog" 时生效）
    "hog_params": {
      "orientations": 9,
      "pixels_per_cell": [2, 2],
      "cells_per_block": [2, 2]
    },

    // HU 矩特征参数（use_feature = "hu" 时生效）
    "hu_params": {
      "log_scale": true
    }
  },

  "monitor": {
    "use_monitor": true,            // 是否启用 BoostMonitor
    "is_data_noisy": true,          // 是否记录噪声样本的指标
    "checkpoint_interval": 10       // 每 N 轮保存一次 checkpoint
  },

  "model": {
    "estimator": {
      "max_depth": 2                // 基学习器（决策树）配置
    },
    "n_estimators": 500,            // AdaBoost 的弱分类器数
    "learning_rate": 0.5,           // AdaBoost 学习率
    "random_state": 42              // 随机种子
  }
}
```

---

## 6. 使用方式示例

```python
from src.utils import train_and_save

clf, monitor, prep, data, paths = train_and_save("./configs/hog_trial_v1.json")

print(paths["result_dir"])
print(paths["compressed_clf"])
```
