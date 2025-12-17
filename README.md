# AdaLab

**AdaLab** 是一个面向研究的 **AdaBoost 实验园地**，以手写数字图像分类任务为背景，系统性地研究 **AdaBoost 算法的过拟合行为、泛化能力与鲁棒性特征**。

项目提供了一个成熟的 Python 实验框架与 CLI 工具，支持通过 **JSON 配置文件** 驱动端到端实验流程，实现**零代码运行实验**，适用于算法分析、实验复现与对比研究。

---

## 项目特点

### 1. 端到端实验框架

* 提供完整的 **训练 → 评估 →（可选）可视化** 实验流程
* 实验参数统一由 JSON 配置文件管理
* 实验结果自动保存，支持断点评估与后处理

### 2. AdaBoost 核心机制可观测

* 扩展 `sklearn.ensemble.AdaBoostClassifier`
* 内置 **BoostMonitor**，可精细监控：

  * 样本权重分布演化
  * 噪声样本与干净样本权重变化
  * 每一轮弱分类器的 `alpha`
* 支持传入 **验证集**，实时跟踪验证性能，用于分析过拟合现象

### 3. 面向鲁棒性研究的数据处理模块

除标准的数据流程外，AdaLab 特别关注 **数据扰动与鲁棒性验证**：

* 标准数据集划分（训练 / 测试 / 验证）
* 特征提取支持：

  * HOG
  * Hu 不变矩
* 提供多种 **图像扰动方式**（噪声、破坏、变形等），用于系统性评估 AdaBoost 在非理想条件下的表现

### 4. CLI 驱动，零代码运行

* 提供统一的 `adalab` 命令行工具
* 支持多种运行模式（仅训练、训练+评估+可视化、仅评估已有实验）
* 适合批量实验与实验脚本化管理

### 5. 可选配套可视化模块

* 提供独立的可视化包 **adalab_viz**
* 对实验结果进行统一、结构化的可视化
* 与核心实验逻辑解耦，保持后端简洁

---

## ⚙️ 环境配置

### 1. 创建并激活 Conda 环境

```bash
conda create -f environment.yaml
conda activate adalab
```

### 2. 验证安装是否成功

```bash
which adalab
adalab -h
```

若能正常输出 CLI 帮助信息，则环境配置成功。

---

## 使用方法

### 查看 CLI 帮助

```bash
adalab -h
```

输出如下：

```text
usage: adalab [-h] --config CONFIG
              [--experiments-dir EXPERIMENTS_DIR]
              [--course-folder COURSE_FOLDER]
              [--viz | --viz-only]

AdaLab experiment runner (CLI)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to json config file
  --experiments-dir EXPERIMENTS_DIR
                        Base directory that stores experiment runs
                        (default: experiments/)
  --course-folder COURSE_FOLDER
                        Course test folder used in evaluation
                        (default: ./data/test_images)
  --viz                 Train + eval + visualize after training
                        (requires use_monitor=true)
  --viz-only            Skip training; load existing experiment results
                        then eval + visualize
```

---

### 常见运行模式

#### 1. 仅训练与评估

```bash
adalab --config configs/exp1.json
```

#### 2. 训练 + 评估 + 可视化

```bash
adalab --config configs/exp1.json --viz
```

> 需要在配置文件中设置 `use_monitor = true`

#### 3. 仅对已有实验进行评估与可视化

```bash
adalab --config configs/exp1.json --viz-only
```

---

## 配置文件说明

所有实验参数均通过 JSON 配置文件控制，包括但不限于：

* 数据集与特征设置
* AdaBoost 超参数
* 训练与验证策略
* 是否启用监控与可视化

 **配置文件的详细说明请参考：**

```
docs/config.md
```
