# adalab.workflow.train_and_save使用说明

`train_and_save` 是一个从 config 文件到结果 joblib 等文件落盘的全流程训练 pipeline。

在cli和ExperimentPipeline中解析和使用它的返回值来完成评估与可视化的调度。

---

## 返回值总览

`train_and_save(...)` 返回一个五元组：

1. `clf`
2. `monitor`
3. `split`（`DataSplit`）
4. `layout`（`ExperimentPaths`）
5. `result_paths`（`ArtifactPaths`）

DataSplit ExperimentPaths ArtifactPaths是三个在workflow.py中定义的dataclass,下面仅解释较为关键的内容，详情可查看其docstring

---

## 1) `clf`：训练完成的分类模型对象

* 含义：**已经训练完成**的分类器实例（“内存态模型”）。
* 类型：二选一

  * `sklearn.ensemble.AdaBoostClassifier`（未启用 monitor 时）
  * `adalab.patch.AdaBoostClfWithMonitor`（启用 monitor 时）
* 与落盘关系：对应 `result_paths.raw_clf` / `result_paths.compressed_clf` 指向的文件内容（同一份模型的不同序列化形式）。

---

## 2) `monitor`：`BoostMonitor` 或 `None`

* 含义：训练过程的监控器（“内存态监控数据”）。
* 取值：

  * 启用 monitor：返回 `BoostMonitor` 实例
  * 未启用 monitor：返回 `None`
* 关键点（对前端很重要）：

  * 当 `monitor is None` 时，**所有与 monitor 相关的产物路径也会在 `ArtifactPaths` 中体现为 `None`**（详见下面 `ArtifactPaths`）。

---

## 3) `split`：`DataSplit`（一次实验的数据划分结果与元信息）

`DataSplit` 是一次实验在 workflow 内部各阶段传递的 **统一数据载体**，它把“这次实验到底用的哪份数据”明确冻结下来，便于前端/分析模块对齐后续操作。

字段语义如下：

* `X_train: np.ndarray`
  训练集特征矩阵，形状 `(n_train, d)`。

* `X_test: np.ndarray`
  测试集特征矩阵，形状 `(n_test, d)`。

* `y_train: np.ndarray`
  训练集标签向量。

* `y_test: np.ndarray`
  测试集标签向量。

* `noise_idx: np.ndarray`
  训练集中被标记为噪声样本的索引（用于区分 noisy/clean 子集）。

* `clean_idx: np.ndarray`
  训练集中被标记为干净样本的索引。

* `prep: DataPreparation`
  生成该次划分的 `DataPreparation` 实例（携带这次划分的内部状态/参数，便于追溯）,**处理潘老师数据的方法在这个实例当中: `DataPreparation.prepare_course_data`**

---

## 4) `layout`：`ExperimentPaths`（实验目录布局协议，前端读取实验产物的“入口坐标系”）

`ExperimentPaths` 是一次实验在磁盘上的 **目录布局** 的只读描述对象（frozen dataclass）。它的意义是：前端/Runner/可视化层不需要推断路径，只需要读取这个对象即可定位实验目录结构。

字段语义：

* `exp_dir: Path`
  实验根目录。

* `ckpt_dir: Path`
  训练过程 checkpoint 存放目录（monitor 的 checkpoint_prefix 指向这里）。

* `result_dir: Path`
  最终结果与模型文件存放目录。

* `config_path: Path`
  保存“本次实际运行所用配置”的路径（workflow 会把解析到的 config 冻结写入这里）。

* `result_csv: Path`
  最终导出的监控结果 CSV 文件路径（有 monitor 时会写入；无 monitor 时仍在 layout 中给出约定位置，但是否存在由 `ArtifactPaths.monitor_csv` 负责表达）。

注意：`ExperimentPaths` 表达的是“布局约定”,最主要的作用是为`train_and_save`内部提供一个方便的目录访问机制，但考虑到这里面包含了实验根目录的相关信息，
于是也返回出来，是否真的生成某个文件，应以 `ArtifactPaths` 为准。

---

## 5) `result_paths`：`ArtifactPaths`（本次实验“产物清单”，前端最应该依赖的对象）

`ArtifactPaths` 是**文件级产物**的统一描述：告诉你这次实验到底落盘了什么，以及每个产物在哪里；并且用 `Optional[Path]` 明确区分“没有启用 monitor，因此不会有这些文件”。

字段语义：

* `raw_clf: Path`
  未压缩的模型 joblib 文件路径（一定存在）。

* `compressed_clf: Path`
  压缩后的模型 joblib.xz 文件路径（一定存在）。

* `raw_monitor: Optional[Path]`
  未压缩的 BoostMonitor joblib 路径：

  * 启用 monitor：存在
  * 未启用 monitor：为 `None`

* `compressed_monitor: Optional[Path]`
  压缩后的 BoostMonitor joblib.xz 路径：

  * 启用 monitor：存在
  * 未启用 monitor：为 `None`

* `monitor_csv: Optional[Path]`
  监控结果导出的 CSV 文件路径：

  * 启用 monitor：存在（与 `layout.result_csv` 对齐）
  * 未启用 monitor：为 `None`

**前端依赖建议（协议层面）**

* 读取模型：优先看 `compressed_clf` 或 `raw_clf`
* 读取监控与曲线：只看 `raw_monitor/compressed_monitor/monitor_csv` 是否为 `None` 来决定功能是否可用
* csv文件记录的数据缺失严重，在csv保存数据这一feature完成之前，从monitor的joblib读取数据进行可视化
