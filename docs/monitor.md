# BoostMonitor 数据结构与可视化说明

`BoostMonitor` 用于记录 AdaBoost 训练过程中的各种动态指标，monitor实例会在训练结束后以joblib的形式存储到实验结果文件夹，所有信息也会同时以csv的形式存储到训练结果文件夹，便于后续可视化分析。可视化模块从 `BoostMonitor` 实例中直接读取属性或读取结果csv文件均可。

下面介绍所有字段、数据格式，以及应该如何使用它们。

---

## 1. 基本信息

BoostMonitor 的结构如下：

```python
class BoostMonitor:
    def __init__(self,
        noise_indices,        # 噪声样本 index 列表
        clean_indices,        # 干净样本 index 列表
        is_data_noisy=False,  # 是否使用噪声分析
        checkpoint_interval=50,
        checkpoint_prefix="monitor_checkpoint",
    ):
```

### 1.1 数据划分

- `noise_indices: List[int]`
    
- `clean_indices: List[int]`
    

当 `is_data_noisy=True` 时，这两个列表用于区分样本类别。

---

# 2. BoostMonitor 的所有监控属性与含义

以下所有属性均在训练过程中按 boosting iteration 打点记录：

## 2.1 **样本权重相关（核心 AdaBoost 机制）**

| 属性名                      | 类型                 | 内容说明                                                                    |
| ------------------------ | ------------------ | ----------------------------------------------------------------------- |
| `sample_weights_history` | `List[np.ndarray]` | 每一轮完整的样本权重向量（长度 = n_samples），此项用于后续计算noisy_weight和clean_weight,不应用于可视化， |
| `noisy_weight_history`   | `List[float]`      | 每一轮噪声样本权重之和                                                             |
| `clean_weight_history`   | `List[float]`      | 每一轮干净样本权重之和                                                             |

### 用法示例

#### 画样本权重分布随时间变化：

```python
noisy = monitor.noisy_weight_history
clean = monitor.clean_weight_history
plt.plot(noisy, label="Noisy weights")
plt.plot(clean, label="Clean weights")
plt.legend()
```

#### 查看第 k 轮的完整权重向量：

```python
w_k = monitor.sample_weights_history[k]
```

---

## 2.2 **错误与 alpht 系数（AdaBoost 基础公式）**

|属性名|类型|内容说明|
|---|---|---|
|`error_without_weight_history`|List[float]|每一轮弱分类器的未加权错误率|
|`error_history`|List[float]|每一轮弱分类器的加权错误率（AdaBoost 真正使用的 ε_t）|
|`alpha_history`|List[float]|每一轮弱分类器的系数 α_t|

### 画 alpha_t：

```python
plt.plot(monitor.alpha_history)
plt.title("Alpha values of weak learners")
```

---

## 2.3 **验证集指标（每轮验证）**

|属性名|类型|内容说明|
|---|---|---|
|`val_acc_history`|List[float]|每一轮验证集 accuracy|
|`val_f1_history`|List[float]|每一轮验证集 F1 score|

### 用法（画 val-acc 曲线）：

```python
plt.plot(monitor.val_acc_history)
plt.title("Validation Accuracy Over Boosting Rounds")
```

---

## 2.4 **训练集指标（每轮）**

|属性名|类型|内容说明|
|---|---|---|
|`acc_on_train_data`|List[float]|每一轮训练集 accuracy|
|`f1_on_training_data`|List[float]|每一轮训练集 F1 score|

---

## 3. 属性的数据结构总结表

|属性名|形状/类型|每一轮记录一次？|用途|
|---|---|---|---|
|`sample_weights_history`|List[np.ndarray(shape=(n_samples,))]|是|全部样本权重|
|`noisy_weight_history`|List[float]|是|噪声样本权重和|
|`clean_weight_history`|List[float]|是|干净样本权重和|
|`error_without_weight_history`|List[float]|是|未加权错误率|
|`error_history`|List[float]|是|加权错误率 ε_t|
|`alpha_history`|List[float]|是|弱分类器系数 α_t|
|`val_acc_history`|List[float]|是|val acc|
|`val_f1_history`|List[float]|是|val f1|
|`acc_on_train_data`|List[float]|是|train acc|
|`f1_on_training_data`|List[float]|是|train f1|

